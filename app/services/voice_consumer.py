"""
Voice RabbitMQ Consumer
- voice-record-queue 구독
- OpenAI Whisper API STT + LLM 팩트체크
- voice-result-queue 발행
"""
import asyncio
import json
import time
from typing import Optional

import aio_pika
from aio_pika import ExchangeType, Message
from loguru import logger

from app.core.config import settings
from app.schemas.voice_dto import VoiceAnalysisMessage, VoiceFactCheckResultMessage
from app.services.voice_service import VoiceService
from app.util.s3_client import S3Client


class VoiceConsumer:
    """음성 분석 RabbitMQ Consumer"""

    def __init__(self):
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.result_exchange: Optional[aio_pika.Exchange] = None
        self._running = False
        self._voice_service = VoiceService()
        self._s3_client = S3Client()

    async def connect(self):
        """RabbitMQ 연결"""
        try:
            self.connection = await aio_pika.connect_robust(
                settings.RABBITMQ_URL
            )
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=1)

            # 결과 발행용 Exchange 선언
            self.result_exchange = await self.channel.declare_exchange(
                settings.VOICE_RESULT_EXCHANGE,
                ExchangeType.DIRECT,
                durable=True
            )

            # 결과 Queue 선언 및 바인딩
            result_queue = await self.channel.declare_queue(
                settings.VOICE_RESULT_QUEUE,
                durable=True
            )
            await result_queue.bind(self.result_exchange, routing_key=settings.VOICE_RESULT_ROUTING_KEY)

            logger.info("VoiceConsumer: RabbitMQ connected")

        except Exception as e:
            logger.error(f"VoiceConsumer: connection failed: {e}")
            raise

    async def start_consuming(self):
        """메시지 소비 시작"""
        if not self.channel:
            await self.connect()

        # 입력 Exchange 선언
        voice_exchange = await self.channel.declare_exchange(
            settings.VOICE_EXCHANGE,
            ExchangeType.DIRECT,
            durable=True
        )

        # 입력 Queue 선언 및 바인딩
        voice_queue = await self.channel.declare_queue(
            settings.VOICE_QUEUE,
            durable=True
        )
        await voice_queue.bind(voice_exchange, routing_key=settings.VOICE_ROUTING_KEY)

        self._running = True
        logger.info(f"VoiceConsumer: consuming from {settings.VOICE_QUEUE}")

        async with voice_queue.iterator() as queue_iter:
            async for message in queue_iter:
                if not self._running:
                    break
                async with message.process():
                    await self._handle_message(message)

    async def _handle_message(self, message: aio_pika.IncomingMessage):
        """메시지 처리"""
        job_id = "unknown"
        voice_record_id = 0
        contract_id = 0
        start_time = time.time()

        try:
            body = json.loads(message.body.decode())
            job_id = body.get("jobId", "unknown")
            voice_record_id = body.get("voiceRecordId", 0)
            contract_id = body.get("contractId", 0)
            logger.info(f"[VoiceConsumer] Received: job_id={job_id}")

            request = VoiceAnalysisMessage(**body)

            # 1. STT — 이미 전사된 경우 Whisper 스킵
            if request.transcript:
                transcript = request.transcript
                logger.info(f"[VoiceConsumer] STT 재사용 - jobId={job_id}, chars={len(transcript)}")
            else:
                audio_bytes = self._s3_client.get_image(request.s3_key)
                if len(audio_bytes) > 25 * 1024 * 1024:
                    raise ValueError(f"파일 크기 초과: {len(audio_bytes) / 1024 / 1024:.1f}MB (최대 25MB)")
                transcript = await self._voice_service.transcribe(audio_bytes, request.s3_key)
                if not transcript:
                    raise ValueError("음성에서 텍스트를 인식할 수 없습니다. 파일을 확인해주세요.")

            # 3. LLM 팩트체크
            fact_check_items = await self._voice_service.fact_check(transcript, request.raw_text)

            processing_time_ms = int((time.time() - start_time) * 1000)

            result = VoiceFactCheckResultMessage(
                voiceRecordId=request.voice_record_id,
                contractId=request.contract_id,
                jobId=request.job_id,
                status="COMPLETED",
                transcript=transcript,
                factCheckItems=fact_check_items,
                processingTimeMs=processing_time_ms
            )

            await self._publish_result(result)
            logger.info(f"[VoiceConsumer] Completed: job_id={job_id}, time={processing_time_ms}ms")

        except Exception as e:
            logger.error(f"[VoiceConsumer] Failed: job_id={job_id}, error={e}")
            processing_time_ms = int((time.time() - start_time) * 1000)
            await self._publish_error(job_id, voice_record_id, contract_id, str(e), processing_time_ms)

    async def _publish_result(self, result: VoiceFactCheckResultMessage):
        """결과 메시지 발행"""
        if not self.result_exchange:
            raise RuntimeError("Result exchange not initialized")

        message = Message(
            body=json.dumps(result.to_rabbitmq_message()).encode(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        )

        await self.result_exchange.publish(
            message,
            routing_key=settings.VOICE_RESULT_ROUTING_KEY
        )
        logger.info(f"[VoiceConsumer] Published result: job_id={result.job_id}")

    async def _publish_error(self, job_id: str, voice_record_id: int, contract_id: int, error_message: str, processing_time_ms: int):
        """에러 결과 발행"""
        error_result = VoiceFactCheckResultMessage(
            voiceRecordId=voice_record_id,
            contractId=contract_id,
            jobId=job_id,
            status="FAILED",
            processingTimeMs=processing_time_ms,
            errorMessage=error_message
        )
        await self._publish_result(error_result)

    async def stop(self):
        """Consumer 종료"""
        self._running = False
        if self.connection:
            await self.connection.close()
            logger.info("[VoiceConsumer] Disconnected")


voice_consumer = VoiceConsumer()


async def start_voice_consumer():
    """VoiceConsumer 시작 (FastAPI startup에서 호출)"""
    await voice_consumer.connect()
    asyncio.create_task(voice_consumer.start_consuming())
    logger.info("[VoiceConsumer] Started in background")


async def stop_voice_consumer():
    """VoiceConsumer 종료 (FastAPI shutdown에서 호출)"""
    await voice_consumer.stop()
