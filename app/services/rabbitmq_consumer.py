"""
RabbitMQ Consumer - Spring Boot 연동
비동기로 메시지를 소비하여 병렬 분석 수행
"""
import asyncio
import json
import time
from datetime import datetime
from typing import Optional, Callable

import aio_pika
from aio_pika import ExchangeType, Message
from loguru import logger

from app.core.config import settings
from app.schemas.contract_analysis_dto import (
    ContractAnalysisRequest,
    ContractAnalysisResult,
    ContractSummary,
    RiskAnalysisResult,
    ClauseRiskResult,
    AnalysisStatus
)


class RabbitMQConsumer:
    """
    비동기 RabbitMQ Consumer

    Spring Boot에서 발행한 메시지를 소비하여:
    1. AI 요약 + Risk 분석 병렬 처리
    2. 결과를 ai.result.queue로 발행
    """

    def __init__(self):
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.result_exchange: Optional[aio_pika.Exchange] = None
        self._running = False

    async def connect(self):
        """RabbitMQ 연결"""
        try:
            self.connection = await aio_pika.connect_robust(
                settings.RABBITMQ_URL,
                loop=asyncio.get_event_loop()
            )
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=1)

            # 결과 발행용 Exchange 선언
            self.result_exchange = await self.channel.declare_exchange(
                settings.RESULT_EXCHANGE,
                ExchangeType.DIRECT,
                durable=True
            )

            # 결과 Queue 선언
            result_queue = await self.channel.declare_queue(
                'ai.result.queue',
                durable=True
            )
            await result_queue.bind(self.result_exchange, routing_key='ai.result')

            logger.info("RabbitMQ connected successfully")

        except Exception as e:
            logger.error(f"RabbitMQ connection failed: {e}")
            raise

    async def start_consuming(self):
        """메시지 소비 시작"""
        if not self.channel:
            await self.connect()

        # 분석 요청 Queue 선언
        analysis_queue = await self.channel.declare_queue(
            settings.ANALYSIS_QUEUE,
            durable=True
        )

        self._running = True
        logger.info(f"Starting to consume from: {settings.ANALYSIS_QUEUE}")

        async with analysis_queue.iterator() as queue_iter:
            async for message in queue_iter:
                if not self._running:
                    break
                async with message.process():
                    await self._handle_message(message)

    async def _handle_message(self, message: aio_pika.IncomingMessage):
        """
        메시지 처리

        Args:
            message: RabbitMQ 메시지
        """
        task_id = "unknown"
        try:
            # 메시지 파싱
            body = json.loads(message.body.decode())
            task_id = body.get('taskId', 'unknown')

            logger.info(f"[Consumer] Received message: task_id={task_id}")

            # 요청 객체 생성
            request = ContractAnalysisRequest(**body)

            # 병렬 분석 수행
            result = await self._process_analysis(request)

            # 결과 발행
            await self._publish_result(result)

            logger.info(f"[Consumer] Completed: task_id={task_id}")

        except json.JSONDecodeError as e:
            logger.error(f"[Consumer] JSON decode error: {e}")
            await self._publish_error(task_id, f"Invalid JSON: {e}")

        except Exception as e:
            logger.error(f"[Consumer] Processing error: {e}")
            await self._publish_error(task_id, str(e))


    def _convert_risk_assessments(self, assessments: list) -> RiskAnalysisResult:
        """ClauseRiskAssessment 리스트를 RiskAnalysisResult로 변환"""
        if not assessments:
            return RiskAnalysisResult(
                total_clauses=0,
                clause_results=[]
            )

        clause_results = []
        risk_count = 0
        caution_count = 0
        safety_count = 0

        for assessment in assessments:
            if assessment.risk_level == "Risk":
                risk_count += 1
            elif assessment.risk_level == "Caution":
                caution_count += 1
            else:
                safety_count += 1

            reasoning_summary = " → ".join([
                step.thought for step in assessment.reasoning[:2]
            ]) if assessment.reasoning else ""

            clause_results.append(ClauseRiskResult(
                clause_title=assessment.clause_title,
                clause_content=assessment.clause_content,
                risk_level=assessment.risk_level,
                legal_reference=assessment.legal_reference,
                recommendation=assessment.recommendation,
                reasoning_summary=reasoning_summary
            ))

        total = len(assessments)
        risk_percentage = round(risk_count / total * 100, 1) if total > 0 else 0

        return RiskAnalysisResult(
            total_clauses=total,
            risk_count=risk_count,
            caution_count=caution_count,
            safety_count=safety_count,
            risk_percentage=risk_percentage,
            clause_results=clause_results
        )

    async def _publish_result(self, result: ContractAnalysisResult):
        """결과 메시지 발행"""
        if not self.result_exchange:
            logger.error("[Consumer] Result exchange not initialized")
            return

        message = Message(
            body=json.dumps(result.to_rabbitmq_message()).encode(),
            content_type='application/json',
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        )

        await self.result_exchange.publish(
            message,
            routing_key='ai.result'
        )
        logger.info(f"[Consumer] Published result: task_id={result.task_id}")

    async def _publish_error(self, task_id: str, error_message: str):
        """에러 결과 발행"""
        error_result = ContractAnalysisResult(
            task_id=task_id,
            status=AnalysisStatus.FAILED,
            error_message=error_message
        )
        await self._publish_result(error_result)

    async def stop(self):
        """Consumer 종료"""
        self._running = False
        if self.connection:
            await self.connection.close()
            logger.info("[Consumer] Disconnected from RabbitMQ")


# 전역 Consumer 인스턴스
consumer = RabbitMQConsumer()


async def start_consumer():
    """Consumer 시작 (FastAPI startup에서 호출)"""
    await consumer.connect()
    asyncio.create_task(consumer.start_consuming())
    logger.info("[Consumer] Started in background")


async def stop_consumer():
    """Consumer 종료 (FastAPI shutdown에서 호출)"""
    await consumer.stop()
