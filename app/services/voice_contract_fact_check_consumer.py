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


class VoiceContractFactCheckConsumer:
    """Consumes voice fact-check jobs and publishes results back to Spring Boot."""

    def __init__(self):
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.result_exchange: Optional[aio_pika.Exchange] = None
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

        self.connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)
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

    async def setup_topology(self):
        """Declare the voice analysis queue topology."""
        if self.channel is None:
            raise RuntimeError("RabbitMQ channel not initialized")

        voice_exchange = await self.channel.declare_exchange(
            settings.VOICE_ANALYSIS_EXCHANGE,
            ExchangeType.DIRECT,
            durable=True,
        )
        voice_queue = await self.channel.declare_queue(
            settings.VOICE_ANALYSIS_QUEUE,
            durable=True,
        )
        await voice_queue.bind(
            voice_exchange,
            routing_key=settings.VOICE_ANALYSIS_ROUTING_KEY,
        )
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

                voice_exchange = await self.channel.declare_exchange(
                    settings.VOICE_ANALYSIS_EXCHANGE,
                    ExchangeType.DIRECT,
                    durable=True,
                )
                voice_queue = await self.channel.declare_queue(
                    settings.VOICE_ANALYSIS_QUEUE,
                    durable=True,
                )
                await voice_queue.bind(
                    voice_exchange,
                    routing_key=settings.VOICE_ANALYSIS_ROUTING_KEY,
                )

                retry_delay = 5
                logger.info(
                    "[VoiceContractFactCheckConsumer] Start consuming: "
                    f"queue={settings.VOICE_ANALYSIS_QUEUE}"
                )

                async with voice_queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        if not self._running:
                            return
                        async with message.process():
                            await self._handle_message(message)

            except asyncio.CancelledError:
                logger.info("[VoiceContractFactCheckConsumer] Consuming cancelled")
                return
            except Exception as e:
                if not self._running:
                    return
                logger.error(
                    "[VoiceContractFactCheckConsumer] Connection lost, "
                    f"retrying in {retry_delay}s: {e}"
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
            job_id = body.get("jobId", "unknown")
            voice_record_id = body.get("voiceRecordId", 0)
            contract_id = body.get("contractId", 0)
            logger.info(
                "[VoiceContractFactCheckConsumer] Received: "
                f"voiceRecordId={voice_record_id}, jobId={job_id}"
            )

            request = VoiceContractFactCheckRequest(**body)
            result = await self._process(request, start_ms)
            await self._publish_result(result)
            logger.info(
                "[VoiceContractFactCheckConsumer] Completed: "
                f"voiceRecordId={voice_record_id}, jobId={job_id}"
            )

        except json.JSONDecodeError as e:
            logger.error(f"[VoiceContractFactCheckConsumer] JSON parse error: {e}")
            await self._publish_error(
                voice_record_id,
                contract_id,
                job_id,
                f"JSON parse error: {e}",
                start_ms,
            )
        except Exception as e:
            logger.exception(
                "[VoiceContractFactCheckConsumer] Processing error: "
                f"{type(e).__name__}: {e}"
            )
            await self._publish_error(
                voice_record_id,
                contract_id,
                job_id,
                f"{type(e).__name__}: {e}",
                start_ms,
            )

    async def _process(
        self,
        request: VoiceContractFactCheckRequest,
        start_ms: int,
    ) -> VoiceContractFactCheckResult:
        """Run the async fact-check pipeline for one message."""
        contract_text, transcript = await asyncio.gather(
            resolve_contract_text(request),
            transcribe_audio_from_request(request),
        )

        logger.info(
            "[VoiceContractFactCheckConsumer] Analysis start: "
            f"voiceRecordId={request.voiceRecordId}, "
            f"mode={'fact-check' if contract_text else 'voice-only'}"
        )

        if contract_text:
            fact_check_items = await run_fact_check(transcript, contract_text)
        else:
            fact_check_items = await run_voice_only_analysis(transcript)

        logger.info(
            "[VoiceContractFactCheckConsumer] Analysis completed: "
            f"{len(fact_check_items)} items"
        )

        elapsed_ms = int(time.time() * 1000) - start_ms

        return VoiceContractFactCheckResult(
            voiceRecordId=request.voiceRecordId,
            contractId=request.contractId,
            jobId=request.jobId,
            status="COMPLETED",
            transcript=transcript,
            factCheckItems=fact_check_items,
            processingTimeMs=elapsed_ms,
        )

    async def _publish_result(self, result: VoiceContractFactCheckResult):
        if not self.result_exchange:
            raise RuntimeError(
                "[VoiceContractFactCheckConsumer] Result exchange not initialized"
            )

        body = result.model_dump(mode="json")
        await self.result_exchange.publish(
            Message(
                body=json.dumps(body, ensure_ascii=False).encode(),
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
        elapsed_ms = int(time.time() * 1000) - start_ms
        error_result = VoiceContractFactCheckResult(
            voiceRecordId=voice_record_id,
            contractId=contract_id,
            jobId=job_id,
            status="FAILED",
            processingTimeMs=elapsed_ms,
            errorMessage=error_message,
        )
        try:
            await self._publish_result(error_result)
        except Exception as e:
            logger.error(
                "[VoiceContractFactCheckConsumer] Failed to publish error result: "
                f"{e}"
            )

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


# Compatibility aliases for the previous mixed naming.
VoiceRabbitMQConsumer = VoiceContractFactCheckConsumer
voice_consumer = voice_contract_fact_check_consumer
start_voice_consumer = start_voice_contract_fact_check_consumer
stop_voice_consumer = stop_voice_contract_fact_check_consumer
