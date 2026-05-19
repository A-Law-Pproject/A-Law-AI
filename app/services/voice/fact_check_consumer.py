"""
RabbitMQ consumer for Spring-integrated async voice contract fact-check jobs.
"""
import asyncio
import json
import time
from typing import Optional

import aio_pika
from aio_pika import ExchangeType, Message
from loguru import logger
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.voice_contract_fact_check import (
    VoiceContractFactCheckRequest,
    VoiceContractFactCheckResult,
)
from app.services.voice.contract_fact_check_service import (
    resolve_contract_text,
    run_fact_check,
    run_voice_only_analysis,
    transcribe_audio_from_request,
)
from app.services.voice.hash_service import save_voice_fact_check_result


class VoiceContractFactCheckConsumer:
    """Consumes voice fact-check jobs and publishes results back to Spring Boot."""

    def __init__(self):
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.result_exchange: Optional[aio_pika.Exchange] = None
        self.dead_letter_exchange: Optional[aio_pika.Exchange] = None
        self._running = False
        self._consume_task: Optional[asyncio.Task] = None

    async def connect(self):
        """Connect to RabbitMQ and declare the result exchange."""
        if self.connection and not self.connection.is_closed:
            try:
                await self.connection.close()
            except Exception:
                pass

        self.connection = None
        self.channel = None
        self.result_exchange = None
        self.dead_letter_exchange = None

        self.connection = await aio_pika.connect_robust(settings.RABBITMQ_URL, heartbeat=120)
        self.channel = await self.connection.channel()
        await self.channel.set_qos(prefetch_count=1)

        self.result_exchange = await self.channel.declare_exchange(
            settings.VOICE_RESULT_EXCHANGE,
            ExchangeType.DIRECT,
            durable=True,
        )

        result_queue = await self.channel.declare_queue(
            settings.VOICE_RESULT_QUEUE,
            durable=True,
        )
        await result_queue.bind(
            self.result_exchange,
            routing_key=settings.VOICE_RESULT_ROUTING_KEY,
        )

        logger.info("[VoiceContractFactCheckConsumer] RabbitMQ connected")

    async def _declare_voice_topology(self) -> aio_pika.Queue:
        """Declare the existing main queue shape plus a separate DLQ topology."""
        if self.channel is None:
            raise RuntimeError("RabbitMQ channel not initialized")

        voice_exchange = await self.channel.declare_exchange(
            settings.VOICE_ANALYSIS_EXCHANGE,
            ExchangeType.DIRECT,
            durable=True,
        )

        # Keep the main queue compatible with the already-existing broker shape.
        voice_queue = await self.channel.declare_queue(
            settings.VOICE_ANALYSIS_QUEUE,
            durable=True,
        )
        await voice_queue.bind(
            voice_exchange,
            routing_key=settings.VOICE_ANALYSIS_ROUTING_KEY,
        )

        self.dead_letter_exchange = await self.channel.declare_exchange(
            settings.VOICE_ANALYSIS_EXCHANGE + ".dlx",
            ExchangeType.DIRECT,
            durable=True,
        )
        voice_dlq = await self.channel.declare_queue(
            settings.VOICE_ANALYSIS_QUEUE + ".dlq",
            durable=True,
        )
        await voice_dlq.bind(
            self.dead_letter_exchange,
            routing_key=settings.VOICE_ANALYSIS_ROUTING_KEY + ".failed",
        )
        return voice_queue

    async def setup_topology(self):
        """Declare the voice analysis queue topology."""
        voice_queue = await self._declare_voice_topology()
        logger.info(
            "[VoiceContractFactCheckConsumer] Topology ready: "
            f"queue={settings.VOICE_ANALYSIS_QUEUE}"
        )
        return voice_queue

    async def ensure_ready(self):
        await self.connect()
        await self.setup_topology()

    def is_healthy(self) -> bool:
        return bool(
            self.connection
            and not self.connection.is_closed
            and self.channel
            and not self.channel.is_closed
            and self.result_exchange is not None
        )

    async def start_consuming(self):
        """Start the consume loop with reconnect/backoff."""
        self._running = True
        retry_delay = 5

        while self._running:
            try:
                await self.connect()
                voice_queue = await self._declare_voice_topology()

                retry_delay = 5
                logger.info(
                    "[VoiceContractFactCheckConsumer] Start consuming: "
                    f"queue={settings.VOICE_ANALYSIS_QUEUE}"
                )

                async with voice_queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        if not self._running:
                            return
                        await self._handle_message(message)

            except asyncio.CancelledError:
                logger.info("[VoiceContractFactCheckConsumer] Consuming cancelled")
                return
            except Exception as exc:
                if not self._running:
                    return
                logger.error(
                    "[VoiceContractFactCheckConsumer] Connection lost, "
                    f"retrying in {retry_delay}s: {exc}"
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)

    async def _handle_message(self, message: aio_pika.IncomingMessage):
        job_id = "unknown"
        voice_record_id = 0
        contract_id = 0
        start_ms = int(time.time() * 1000)

        try:
            body = json.loads(message.body.decode())
            job_id = body.get("jobId") or body.get("job_id") or "unknown"
            voice_record_id = body.get("voiceRecordId") or body.get("voice_record_id") or 0
            contract_id = body.get("contractId") or body.get("contract_id") or 0
            logger.info(
                "[VoiceContractFactCheckConsumer] Received: "
                f"voiceRecordId={voice_record_id}, jobId={job_id}"
            )
        except json.JSONDecodeError as exc:
            logger.error(f"[VoiceContractFactCheckConsumer] JSON parse error: {exc}")
            await self._publish_terminal_error(
                message,
                voice_record_id,
                contract_id,
                job_id,
                f"JSON parse error: {exc}",
                start_ms,
            )
            return

        try:
            request = VoiceContractFactCheckRequest(**body)
        except ValidationError as exc:
            logger.error(f"[VoiceContractFactCheckConsumer] Payload validation error: {exc}")
            await self._publish_terminal_error(
                message,
                voice_record_id,
                contract_id,
                job_id,
                f"ValidationError: {exc}",
                start_ms,
            )
            return

        try:
            result = await self._process(request, start_ms)
            await self._publish_result(result)
            await save_voice_fact_check_result({
                "voiceRecordId": result.voiceRecordId,
                "contractId": result.contractId,
                "jobId": result.jobId,
                "status": result.status,
                "transcript": result.transcript,
                "factCheckItems": [item.model_dump(mode="json") for item in (result.factCheckItems or [])],
                "processingTimeMs": result.processingTimeMs,
            })
            await message.ack()
            logger.info(
                "[VoiceContractFactCheckConsumer] Completed: "
                f"voiceRecordId={voice_record_id}, jobId={job_id}"
            )
        except Exception as exc:
            logger.exception(
                "[VoiceContractFactCheckConsumer] Processing error: "
                f"{type(exc).__name__}: {exc}"
            )
            await self._publish_retryable_failure(
                message,
                voice_record_id=voice_record_id,
                contract_id=contract_id,
                job_id=job_id,
                error_message=f"{type(exc).__name__}: {exc}",
            )

    async def _publish_terminal_error(
        self,
        message: aio_pika.IncomingMessage,
        voice_record_id: int,
        contract_id: int,
        job_id: str,
        error_message: str,
        start_ms: int,
    ) -> None:
        try:
            await self._publish_error(
                voice_record_id,
                contract_id,
                job_id,
                error_message,
                start_ms,
            )
            await message.ack()
        except Exception as exc:
            logger.error(
                "[VoiceContractFactCheckConsumer] Failed to publish terminal error, "
                f"sending original message to DLQ: {exc}"
            )
            await self._publish_retryable_failure(
                message,
                voice_record_id=voice_record_id,
                contract_id=contract_id,
                job_id=job_id,
                error_message=error_message,
                failure_type="terminal",
            )

    async def _publish_retryable_failure(
        self,
        message: aio_pika.IncomingMessage,
        *,
        voice_record_id: int,
        contract_id: int,
        job_id: str,
        error_message: str,
        failure_type: str = "processing",
    ) -> None:
        if self.dead_letter_exchange is None:
            await self._declare_voice_topology()

        headers = dict(message.headers or {})
        headers.update(
            {
                "x-original-routing-key": settings.VOICE_ANALYSIS_ROUTING_KEY,
                "x-job-id": job_id,
                "x-contract-id": contract_id,
                "x-voice-record-id": voice_record_id,
                "x-failure-type": failure_type,
                "x-error-message": error_message[:500],
            }
        )

        try:
            await self.dead_letter_exchange.publish(
                Message(
                    body=message.body,
                    content_type=message.content_type or "application/json",
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    headers=headers,
                    correlation_id=message.correlation_id,
                    message_id=message.message_id,
                ),
                routing_key=settings.VOICE_ANALYSIS_ROUTING_KEY + ".failed",
            )
            await message.ack()
            logger.warning(
                "[VoiceContractFactCheckConsumer] Routed message to DLQ: "
                f"voiceRecordId={voice_record_id}, contractId={contract_id}, jobId={job_id}"
            )
        except Exception as exc:
            logger.error(
                "[VoiceContractFactCheckConsumer] Failed to publish to DLQ, requeueing message: "
                f"voiceRecordId={voice_record_id}, contractId={contract_id}, jobId={job_id}, error={exc}"
            )
            await message.reject(requeue=True)

    async def _process(
        self,
        request: VoiceContractFactCheckRequest,
        start_ms: int,
    ) -> VoiceContractFactCheckResult:
        contract_text, transcript = await asyncio.gather(
            resolve_contract_text(request),
            transcribe_audio_from_request(request),
        )

        logger.info(
            "[VoiceContractFactCheckConsumer] Analysis start: "
            f"voiceRecordId={request.voiceRecordId}, "
            f"mode={'fact-check' if contract_text else 'voice-only'}"
        )

        fact_check_items = (
            await run_fact_check(transcript, contract_text)
            if contract_text
            else await run_voice_only_analysis(transcript)
        )

        logger.info(f"[VoiceContractFactCheckConsumer] Analysis completed: {len(fact_check_items)} items")

        return VoiceContractFactCheckResult(
            voiceRecordId=request.voiceRecordId,
            contractId=request.contractId,
            jobId=request.jobId,
            status="COMPLETED",
            transcript=transcript,
            factCheckItems=fact_check_items,
            processingTimeMs=int(time.time() * 1000) - start_ms,
        )

    async def _publish_result(self, result: VoiceContractFactCheckResult):
        if not self.result_exchange:
            raise RuntimeError("[VoiceContractFactCheckConsumer] Result exchange not initialized")

        await self.result_exchange.publish(
            Message(
                body=json.dumps(result.model_dump(mode="json"), ensure_ascii=False).encode(),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=settings.VOICE_RESULT_ROUTING_KEY,
        )
        logger.info(
            "[VoiceContractFactCheckConsumer] Published result: "
            f"voiceRecordId={result.voiceRecordId}, jobId={result.jobId}"
        )

    async def _publish_error(
        self,
        voice_record_id: int,
        contract_id: int,
        job_id: str,
        error_message: str,
        start_ms: int,
    ):
        error_result = VoiceContractFactCheckResult(
            voiceRecordId=voice_record_id,
            contractId=contract_id,
            jobId=job_id,
            status="FAILED",
            processingTimeMs=int(time.time() * 1000) - start_ms,
            errorMessage=error_message,
        )
        await self._publish_result(error_result)

    async def stop(self):
        self._running = False
        if self._consume_task and not self._consume_task.done():
            self._consume_task.cancel()
            try:
                await self._consume_task
            except asyncio.CancelledError:
                pass
        self._consume_task = None
        if self.connection:
            await self.connection.close()
            logger.info("[VoiceContractFactCheckConsumer] RabbitMQ disconnected")


voice_contract_fact_check_consumer = VoiceContractFactCheckConsumer()


async def start_voice_contract_fact_check_consumer():
    await voice_contract_fact_check_consumer.ensure_ready()
    voice_contract_fact_check_consumer._running = True
    if (
        voice_contract_fact_check_consumer._consume_task is None
        or voice_contract_fact_check_consumer._consume_task.done()
    ):
        voice_contract_fact_check_consumer._consume_task = asyncio.create_task(
            voice_contract_fact_check_consumer.start_consuming()
        )
    logger.info("[VoiceContractFactCheckConsumer] Started")


async def stop_voice_contract_fact_check_consumer():
    await voice_contract_fact_check_consumer.stop()
