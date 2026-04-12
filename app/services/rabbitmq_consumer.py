"""
RabbitMQ Consumer - Spring Boot 연동
비동기로 메시지를 소비하여 병렬 분석 수행
분석 결과는 RabbitMQ를 통해 Spring Boot로 전송
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
from app.core.dependencies import get_vector_db, get_embeddings, get_llm, fetch_ocr_text
from app.schemas.contract_analysis_dto import (
    ContractAnalysisRequest,
    ContractAnalysisResult,
    ContractSummary,
    RiskAnalysisResult,
    ClauseRiskResult,
    AnalysisStatus
)
from app.rag.chain.chain import rag_query, detect_risk_contract


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
        self._consume_task: Optional[asyncio.Task] = None

    @staticmethod
    def _overall_risk_level(score: float) -> str:
        if score >= 70:
            return "위험"
        if score >= 40:
            return "주의"
        return "안전"

    async def connect(self):
        """RabbitMQ 연결 (기존 연결 정리 후 재연결)"""
        if self.connection and not self.connection.is_closed:
            try:
                await self.connection.close()
            except Exception:
                pass
        self.connection = None
        self.channel = None
        self.result_exchange = None

        try:
            self.connection = await aio_pika.connect_robust(
                settings.RABBITMQ_URL,
            )
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=1)

            # 결과 발행용 Exchange 선언
            self.result_exchange = await self.channel.declare_exchange(
                settings.RESULT_EXCHANGE,
                ExchangeType.DIRECT,
                durable=True
            )

            # 결과 Queue 선언 + 바인딩 (TTL 없음 - Spring Boot 선언과 동일)
            result_queue = await self.channel.declare_queue(
                settings.RESULT_QUEUE,
                durable=True,
            )
            await result_queue.bind(self.result_exchange, routing_key=settings.RESULT_ROUTING_KEY)

            logger.info("RabbitMQ connected successfully")

        except Exception as e:
            logger.error(f"RabbitMQ connection failed: {e}")
            raise

    async def setup_topology(self):
        """필수 Exchange/Queue 선언 및 바인딩."""
        if self.channel is None:
            raise RuntimeError("RabbitMQ channel is not initialized")

        analysis_exchange = await self.channel.declare_exchange(
            settings.ANALYSIS_EXCHANGE,
            ExchangeType.DIRECT,
            durable=True
        )
        analysis_queue = await self.channel.declare_queue(
            settings.ANALYSIS_QUEUE,
            durable=True,
            arguments={
                "x-message-ttl": 86400000,
                "x-dead-letter-exchange": settings.ANALYSIS_EXCHANGE + ".dlx",
                "x-dead-letter-routing-key": settings.ANALYSIS_ROUTING_KEY + ".failed",
            },
        )
        await analysis_queue.bind(analysis_exchange, routing_key=settings.ANALYSIS_ROUTING_KEY)
        logger.info(f"RabbitMQ topology is ready: queue={settings.ANALYSIS_QUEUE}")
        return analysis_queue

    async def ensure_ready(self):
        """부팅 시점 readiness 검증."""
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
        """메시지 소비 시작 (재연결 루프 포함)"""
        self._running = True
        retry_delay = 5

        while self._running:
            try:
                await self.connect()

                # Spring Boot가 publish하는 Exchange 선언 (Spring Boot 설정과 동일해야 함)
                analysis_exchange = await self.channel.declare_exchange(
                    settings.ANALYSIS_EXCHANGE,
                    ExchangeType.DIRECT,
                    durable=True
                )

                # 분석 요청 Queue 선언 + Exchange 바인딩
                # Spring Boot QueueBuilder 인자와 완전히 동일해야 함 (PRECONDITION_FAILED 방지)
                analysis_queue = await self.channel.declare_queue(
                    settings.ANALYSIS_QUEUE,
                    durable=True,
                    arguments={
                        "x-message-ttl": 86400000,
                        "x-dead-letter-exchange": settings.ANALYSIS_EXCHANGE + ".dlx",
                        "x-dead-letter-routing-key": settings.ANALYSIS_ROUTING_KEY + ".failed",
                    },
                )
                await analysis_queue.bind(analysis_exchange, routing_key=settings.ANALYSIS_ROUTING_KEY)

                retry_delay = 5  # 연결 성공 시 재시도 딜레이 초기화
                logger.info(f"Starting to consume from: {settings.ANALYSIS_QUEUE}")

                async with analysis_queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        if not self._running:
                            return
                        async with message.process():
                            await self._handle_message(message)

            except asyncio.CancelledError:
                logger.info("[Consumer] Consuming cancelled")
                return
            except Exception as e:
                if not self._running:
                    return
                logger.error(f"[Consumer] Connection lost, retrying in {retry_delay}s: {e}")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)  # 최대 60초까지 exponential backoff

    async def _handle_message(self, message: aio_pika.IncomingMessage):
        job_id = "unknown"
        contract_id = 0
        try:
            body = json.loads(message.body.decode())
            # Spring Boot는 camelCase(jobId)로 발행, snake_case(job_id)도 허용
            job_id = body.get("jobId") or "unknown"
            contract_id = body.get("contractId") or 0
            logger.info(f"[Consumer] 수신: job_id={job_id}, contract_id={contract_id}")

            request = ContractAnalysisRequest(**body)
            result = await self._process_analysis(request)
            await self._publish_result(result)

            logger.info(f"[Consumer] 완료: job_id={job_id}")

        except json.JSONDecodeError as e:
            logger.error(f"[Consumer] JSON decode error: {e}")
            await self._publish_error(job_id, contract_id, f"Invalid JSON: {e}")

        except Exception as e:
            logger.error(f"[Consumer] Processing error: {e}")
            await self._publish_error(job_id, contract_id, str(e))


    async def _process_analysis(self, request: ContractAnalysisRequest) -> ContractAnalysisResult:
        """
        MongoDB에서 OCR 텍스트 조회 후 요약 + Risk 분석을 병렬 실행
        """
        start_time = time.time()
        try:
            # 1. MongoDB에서 OCR 텍스트 조회
            logger.info(f"[Consumer] MongoDB 텍스트 조회: s3_key={request.s3_key}")
            text = await fetch_ocr_text(request.s3_key)
            logger.info(f"[Consumer] 텍스트 조회 완료: {len(text)}자")

            # 2. 요약 + Risk 분석 병렬 실행 (타임아웃 적용)
            logger.info(f"[Consumer] 병렬 분석 시작 (요약 + Risk)")
            summary_result, risk_analysis_result = await asyncio.wait_for(
                asyncio.gather(
                    self._perform_summary(text),
                    self._perform_risk_analysis(text),
                ),
                timeout=settings.ANALYSIS_TIMEOUT,
            )
            logger.info(f"[Consumer] 병렬 분석 완료")

            elapsed_ms = int((time.time() - start_time) * 1000)

            # 3. 결과 객체 조립
            result = ContractAnalysisResult(
                job_id=request.job_id,
                contract_id=request.contract_id,
                status=AnalysisStatus.COMPLETED,
                processing_time_ms=elapsed_ms,
            )

            # 요약 결과 추가
            if summary_result:
                summary_text_parts = []
                basic_info = summary_result.get("basic_info", {})
                if basic_info:
                    info_parts = []
                    if "deposit" in basic_info:
                        info_parts.append(f"보증금: {basic_info['deposit']}원")
                    if "monthly_rent" in basic_info:
                        info_parts.append(f"월세: {basic_info['monthly_rent']}원")
                    if "contract_period" in basic_info:
                        info_parts.append(f"계약기간: {basic_info['contract_period']}")
                    if info_parts:
                        summary_text_parts.append("■ 주요 조건\n" + ", ".join(info_parts))

                for point in summary_result.get("key_points", []):
                    if isinstance(point, dict) and "answer" in point:
                        summary_text_parts.append(f"• {point['answer']}")

                summary_text = "\n\n".join(summary_text_parts) if summary_text_parts else "계약서 요약 정보가 없습니다."
                basic_info = summary_result.get("basic_info", {})
                key_terms = []
                if basic_info.get("deposit"):
                    key_terms.append(f"보증금 {basic_info['deposit']}원")
                if basic_info.get("monthly_rent"):
                    key_terms.append(f"월세 {basic_info['monthly_rent']}원")
                if basic_info.get("contract_period"):
                    key_terms.append(f"계약기간 {basic_info['contract_period']}")

                result.summary = ContractSummary(
                    title="임대차 계약서",
                    summary_text=summary_text,
                    key_terms=key_terms,
                )

            # Risk 분석 결과 추가
            if risk_analysis_result and risk_analysis_result.get("clauses"):
                risk_summary = risk_analysis_result.get("risk_summary", {})
                clauses = risk_analysis_result["clauses"]
                total = len(clauses)
                result.risk_analysis = RiskAnalysisResult(
                    total_clauses=total,
                    overall_risk_score=float(risk_analysis_result.get("overall_risk_score", 0)),
                    overall_risk_level=self._overall_risk_level(float(risk_analysis_result.get("overall_risk_score", 0))),
                    risk_count=risk_summary.get("Risk", 0),
                    caution_count=risk_summary.get("Caution", 0),
                    safety_count=risk_summary.get("Safety", 0),
                    risk_percentage=round(risk_summary.get("Risk", 0) / total * 100, 1) if total > 0 else 0.0,
                    detected_clause_count=risk_summary.get("Risk", 0) + risk_summary.get("Caution", 0),
                    clause_results=[
                        ClauseRiskResult(
                            clause_title=clause.get("category") or clause.get("title") or clause.get("text", "")[:40],
                            clause_content=clause.get("content") or clause.get("text", ""),
                            risk_level=clause["risk_level"],
                            category=clause.get("category", ""),
                            score=int(clause.get("score", 0)),
                            legal_reference=clause.get("related_law", ""),
                            recommendation=clause.get("recommendation") or clause.get("analysis", ""),
                            reasoning_summary=clause.get("analysis", ""),
                            related_law=clause.get("related_law", ""),
                        )
                        for clause in clauses
                    ],
                )

            return result

        except Exception as e:
            logger.error(f"[Consumer] Analysis error: {e}")
            raise

    async def _perform_summary(self, text: str) -> dict:
        """
        RAG 기반 계약서 요약

        법률 문서, 약관, 특약사항 DB를 참조하여 계약서의 핵심 내용을 요약
        """
        try:
            try:
                db = get_vector_db()
                embeddings = get_embeddings()
                llm = get_llm()
            except Exception as e:
                logger.warning(f"[Consumer] RAG 미초기화 - 기본 요약 사용: {e}")
                return self._basic_summary(text)

            # RAG 질의를 통한 요약
            summary_questions = [
                "이 계약서의 주요 조건은 무엇인가요? (임대료, 보증금, 계약기간 등)",
                "이 계약서에서 임차인이 주의해야 할 중요한 사항은 무엇인가요?",
                "특약사항이나 특이사항이 있다면 무엇인가요?"
            ]

            # RAG 질의 3개 병렬 실행
            async def _query(question: str):
                return await asyncio.to_thread(
                    rag_query,
                    question=f"{question}\n\n계약서 내용:\n{text[:2000]}",
                    client=db,
                    embeddings=embeddings,
                    llm=llm,
                    collections=["law_database"],
                    k_per_collection=2,
                )

            results = await asyncio.gather(
                *[_query(q) for q in summary_questions],
                return_exceptions=True
            )
            key_points = []
            for question, result in zip(summary_questions, results):
                if isinstance(result, Exception):
                    logger.error(f"[Consumer] RAG 질의 실패: {result}")
                    continue
                key_points.append({"question": question, "answer": result["answer"]})

            # 기본 정보 추출
            basic_info = self._extract_basic_info(text)

            summary = {
                "key_points": key_points,
                "basic_info": basic_info,
                "total_length": len(text),
                "estimated_read_time": max(1, len(text) // 200),  # 분 단위
                "summary_type": "rag_based"
            }

            logger.info(f"[Consumer] RAG 기반 요약 완료 - {len(key_points)}개 포인트")
            return summary

        except Exception as e:
            logger.error(f"[Consumer] Summary error: {e}")
            return self._basic_summary(text)

    def _basic_summary(self, text: str) -> dict:
        """RAG 실패 시 기본 요약"""
        return {
            "key_points": [{"answer": p} for p in self._extract_key_points(text)],
            "basic_info": self._extract_basic_info(text),
            "total_length": len(text),
            "estimated_read_time": max(1, len(text) // 200),
            "summary_type": "basic"
        }

    def _extract_key_points(self, text: str, max_points: int = 5) -> list:
        """주요 포인트 추출 (키워드 기반)"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        # 중요 키워드가 포함된 문장 우선
        important_keywords = ["보증금", "임대료", "계약기간", "특약", "해지", "갱신"]
        important_sentences = [
            s for s in sentences
            if any(keyword in s for keyword in important_keywords)
        ]
        return (important_sentences + sentences)[:max_points]

    def _extract_basic_info(self, text: str) -> dict:
        """계약서 기본 정보 추출"""
        import re

        info = {}

        # 보증금 추출
        deposit_match = re.search(r'보증금[:\s]*금?\s*([\d,]+)\s*원', text)
        if deposit_match:
            info["deposit"] = deposit_match.group(1)

        # 월세 추출
        rent_match = re.search(r'(월세|차임|임대료)[:\s]*금?\s*([\d,]+)\s*원', text)
        if rent_match:
            info["monthly_rent"] = rent_match.group(2)

        # 계약기간 추출
        period_match = re.search(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일.*?(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', text)
        if period_match:
            info["contract_period"] = f"{period_match.group(1)}.{period_match.group(2)}.{period_match.group(3)} ~ {period_match.group(4)}.{period_match.group(5)}.{period_match.group(6)}"

        return info

    async def _perform_risk_analysis(self, text: str) -> dict:
        """RAG 기반 계약서 위험 분석 (detect_risk_contract 위임)."""
        try:
            db = get_vector_db()
            embeddings = get_embeddings()
            llm = get_llm()
        except Exception as e:
            logger.warning(f"[Consumer] RAG 미초기화 - 빈 결과 반환: {e}")
            return await self._basic_risk_analysis(text)

        try:
            result = await detect_risk_contract(
                user_clause=text,
                client=db,
                embeddings=embeddings,
                llm=llm,
            )
            rs = result["risk_summary"]
            logger.info(
                f"[Consumer] RAG 위험 분석 완료 - "
                f"Risk: {rs['Risk']}, Caution: {rs['Caution']}, Safety: {rs['Safety']}"
            )
            # Spring Boot 쪽이 참조하는 'recommendation' 키 추가 (analysis는 유지)
            for clause in result["clauses"]:
                clause["recommendation"] = clause.get("analysis", "")
            return result
        except Exception as e:
            logger.error(f"[Consumer] Risk analysis error: {e}")
            return await self._basic_risk_analysis(text)

    async def _basic_risk_analysis(self, text: str) -> dict:
        """RAG 실패 시 빈 결과 반환"""
        return {
            "total_clauses": 0,
            "risk_summary": {"Risk": 0, "Caution": 0, "Safety": 0},
            "overall_risk_score": 0,
            "clauses": []
        }

    def _generate_recommendations(self, risk_result: dict) -> list[str]:
        """위험 분석 결과를 바탕으로 권장사항 생성"""
        recommendations = []

        risk_summary = risk_result.get("risk_summary", {})
        risk_count = risk_summary.get("Risk", 0)
        caution_count = risk_summary.get("Caution", 0)
        overall_score = risk_result.get("overall_risk_score", 0)

        # 위험도별 권장사항
        if risk_count > 0:
            recommendations.append(
                f"⚠️ {risk_count}개의 고위험 조항이 발견되었습니다. 반드시 전문가 검토를 받으시기 바랍니다."
            )
        if caution_count > 0:
            recommendations.append(
                f"⚡ {caution_count}개의 주의 조항이 있습니다. 계약 전 꼼꼼히 확인하세요."
            )

        # 전체 위험도에 따른 권장사항
        if overall_score >= 70:
            recommendations.append("이 계약서는 위험도가 매우 높습니다. 계약을 재검토하거나 전문가와 상담하세요.")
        elif overall_score >= 40:
            recommendations.append("일부 조항에 문제가 있을 수 있습니다. 신중한 검토가 필요합니다.")
        else:
            recommendations.append("전반적으로 안전한 계약서입니다.")

        # 필수 확인사항
        recommendations.append("📋 확정일자를 받았는지 확인하세요.")
        recommendations.append("📋 전입신고를 완료하여 대항력을 확보하세요.")

        return recommendations


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
            routing_key=settings.RESULT_ROUTING_KEY
        )
        logger.info(f"[Consumer] Published result: job_id={result.job_id}")

    async def _publish_error(self, job_id: str, contract_id: int, error_message: str):
        """에러 결과 발행"""
        error_result = ContractAnalysisResult(
            job_id=job_id,
            contract_id=contract_id,
            status=AnalysisStatus.FAILED,
            error_message=error_message,
        )
        await self._publish_result(error_result)

    async def stop(self):
        """Consumer 종료"""
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
            logger.info("[Consumer] Disconnected from RabbitMQ")


# 전역 Consumer 인스턴스
consumer = RabbitMQConsumer()


async def start_consumer():
    """Consumer 시작 (FastAPI startup에서 호출)"""
    await consumer.ensure_ready()
    consumer._running = True
    if consumer._consume_task is None or consumer._consume_task.done():
        consumer._consume_task = asyncio.create_task(consumer.start_consuming())
    logger.info("[Consumer] Started after RabbitMQ readiness check")


async def stop_consumer():
    """Consumer 종료 (FastAPI shutdown에서 호출)"""
    await consumer.stop()
