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
from pydantic import ValidationError

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
from app.rag.chain.chain import detect_risk_contract, build_context
from app.rag.chain.prompts import CONTRACT_QA_PROMPT
from app.rag.retriever.multi_retriever import async_search_multi_index

# 재시도 정책
MAX_RETRY_COUNT: int = 3          # 최대 재시도 횟수 (초과 시 DLQ)
RETRY_DELAY_MS: int = 30_000      # 재시도 대기 시간 (밀리초, TTL 방식)

# 재시도해도 의미 없는 오류 타입 — 잘못된 페이로드는 아무리 재시도해도 실패
_NON_RETRYABLE = (json.JSONDecodeError, ValidationError, ValueError)


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
        self.retry_exchange: Optional[aio_pika.Exchange] = None
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
        self.retry_exchange = None

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

            # 결과 Queue 선언 + 바인딩
            # Spring Boot RabbitMQConfig.contractResultQueue() 인자와 정확히 일치해야 함
            # (불일치 시 PRECONDITION_FAILED 발생)
            result_queue = await self.channel.declare_queue(
                settings.RESULT_QUEUE,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": f"{settings.RESULT_QUEUE}.dlx",
                    "x-dead-letter-routing-key": "dead-letter",
                },
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
        analysis_dlx = await self.channel.declare_exchange(
            settings.ANALYSIS_EXCHANGE + ".dlx",
            ExchangeType.DIRECT,
            durable=True,
        )
        analysis_dlq = await self.channel.declare_queue(
            settings.ANALYSIS_QUEUE + ".dlq",
            durable=True,
        )
        await analysis_dlq.bind(
            analysis_dlx,
            routing_key=settings.ANALYSIS_ROUTING_KEY + ".failed",
        )

        # 재시도 Exchange/Queue — TTL 만료 후 메인 큐로 자동 복귀
        self.retry_exchange = await self.channel.declare_exchange(
            settings.ANALYSIS_EXCHANGE + ".retry",
            ExchangeType.DIRECT,
            durable=True,
        )
        retry_queue = await self.channel.declare_queue(
            settings.ANALYSIS_QUEUE + ".retry",
            durable=True,
            arguments={
                "x-message-ttl": RETRY_DELAY_MS,
                "x-dead-letter-exchange": settings.ANALYSIS_EXCHANGE,
                "x-dead-letter-routing-key": settings.ANALYSIS_ROUTING_KEY,
            },
        )
        await retry_queue.bind(
            self.retry_exchange,
            routing_key=settings.ANALYSIS_ROUTING_KEY + ".retry",
        )

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
            and self.retry_exchange is not None
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
                analysis_dlx = await self.channel.declare_exchange(
                    settings.ANALYSIS_EXCHANGE + ".dlx",
                    ExchangeType.DIRECT,
                    durable=True
                )
                analysis_dlq = await self.channel.declare_queue(
                    settings.ANALYSIS_QUEUE + ".dlq",
                    durable=True,
                )
                await analysis_dlq.bind(
                    analysis_dlx,
                    routing_key=settings.ANALYSIS_ROUTING_KEY + ".failed",
                )

                # 재시도 Exchange/Queue
                self.retry_exchange = await self.channel.declare_exchange(
                    settings.ANALYSIS_EXCHANGE + ".retry",
                    ExchangeType.DIRECT,
                    durable=True,
                )
                retry_queue = await self.channel.declare_queue(
                    settings.ANALYSIS_QUEUE + ".retry",
                    durable=True,
                    arguments={
                        "x-message-ttl": RETRY_DELAY_MS,
                        "x-dead-letter-exchange": settings.ANALYSIS_EXCHANGE,
                        "x-dead-letter-routing-key": settings.ANALYSIS_ROUTING_KEY,
                    },
                )
                await retry_queue.bind(
                    self.retry_exchange,
                    routing_key=settings.ANALYSIS_ROUTING_KEY + ".retry",
                )

                retry_delay = 5  # 연결 성공 시 재시도 딜레이 초기화
                logger.info(f"Starting to consume from: {settings.ANALYSIS_QUEUE}")

                async with analysis_queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        if not self._running:
                            return
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

        # ── 1. JSON 파싱 — 실패 시 재시도 불가 ──────────────────────────────
        try:
            body = json.loads(message.body.decode())
            job_id = body.get("jobId") or body.get("job_id") or "unknown"
            contract_id = body.get("contractId") or body.get("contract_id") or 0
            logger.info(f"[Consumer] Received: job_id={job_id}, contract_id={contract_id}")
        except json.JSONDecodeError as exc:
            await self._handle_non_retryable(message, job_id, contract_id, f"Invalid JSON: {exc}")
            return

        # ── 2. 페이로드 검증 — 실패 시 재시도 불가 ──────────────────────────
        try:
            request = ContractAnalysisRequest(**body)
        except ValidationError as exc:
            await self._handle_non_retryable(message, job_id, contract_id, f"ValidationError: {exc}")
            return

        # ── 3. 분석 처리 ──────────────────────────────────────────────────────
        retry_count = self._get_retry_count(message)
        try:
            result = await self._process_analysis(request)
            await self._publish_result(result)
            await message.ack()
            logger.info(f"[Consumer] Completed: job_id={job_id} (retry={retry_count})")

        except _NON_RETRYABLE as exc:
            # 재시도해도 같은 결과 — FAILED 발행 + ACK
            await self._handle_non_retryable(message, job_id, contract_id, str(exc))

        except Exception as exc:
            if retry_count < MAX_RETRY_COUNT:
                # 재시도 가능 — retry queue로 이동
                await self._retry_message(message, job_id, contract_id, retry_count, exc)
            else:
                # 재시도 횟수 초과 — FAILED 발행 + DLQ
                await self._handle_retry_exceeded(message, job_id, contract_id, exc)

    def _get_retry_count(self, message: aio_pika.IncomingMessage) -> int:
        """메시지 헤더에서 현재까지의 재시도 횟수를 읽는다."""
        return int((message.headers or {}).get("x-retry-count", 0))

    async def _handle_non_retryable(
        self,
        message: aio_pika.IncomingMessage,
        job_id: str,
        contract_id: int,
        error_msg: str,
    ) -> None:
        """재시도 불가능한 오류: Spring Boot에 FAILED 결과 발행 후 ACK."""
        logger.error(f"[Consumer] Non-retryable error: job_id={job_id} — {error_msg}")
        try:
            await self._publish_error(job_id, contract_id, error_msg)
            await message.ack()
        except Exception as exc:
            logger.error(f"[Consumer] Failed to publish FAILED result, dead-lettering: {exc}")
            await self._dead_letter_message(message, job_id, contract_id)

    async def _retry_message(
        self,
        message: aio_pika.IncomingMessage,
        job_id: str,
        contract_id: int,
        retry_count: int,
        exc: Exception,
    ) -> None:
        """재시도 가능한 오류: retry queue에 재발행(TTL 대기 후 메인 큐 복귀) 후 ACK."""
        next_count = retry_count + 1
        logger.warning(
            f"[Consumer] Retryable error ({next_count}/{MAX_RETRY_COUNT}): "
            f"job_id={job_id} — {type(exc).__name__}: {exc}"
        )
        try:
            headers = dict(message.headers or {})
            headers["x-retry-count"] = next_count
            headers["x-last-error"] = str(exc)[:200]

            await self.retry_exchange.publish(
                Message(
                    body=message.body,
                    content_type=message.content_type or "application/json",
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    headers=headers,
                ),
                routing_key=settings.ANALYSIS_ROUTING_KEY + ".retry",
            )
            await message.ack()
            logger.info(
                f"[Consumer] Queued for retry: job_id={job_id} attempt={next_count}/{MAX_RETRY_COUNT}"
            )
        except Exception as pub_exc:
            logger.error(f"[Consumer] Failed to queue retry, dead-lettering: {pub_exc}")
            await self._dead_letter_message(message, job_id, contract_id)

    async def _handle_retry_exceeded(
        self,
        message: aio_pika.IncomingMessage,
        job_id: str,
        contract_id: int,
        exc: Exception,
    ) -> None:
        """재시도 횟수 초과: Spring Boot에 FAILED 결과 발행 후 DLQ로 reject."""
        error_msg = f"재시도 {MAX_RETRY_COUNT}회 초과: {type(exc).__name__}: {exc}"
        logger.error(f"[Consumer] Retry limit exceeded: job_id={job_id} — {error_msg}")
        try:
            await self._publish_error(job_id, contract_id, error_msg)
        except Exception as pub_exc:
            logger.error(f"[Consumer] Failed to publish FAILED result after retry exceeded: {pub_exc}")
        # FAILED 발행 여부와 무관하게 DLQ로 이동
        await self._dead_letter_message(message, job_id, contract_id)

    async def _dead_letter_message(
        self,
        message: aio_pika.IncomingMessage,
        job_id: str,
        contract_id: int,
    ) -> None:
        try:
            await message.reject(requeue=False)
            logger.warning(
                f"[Consumer] Dead-lettered message: job_id={job_id}, contract_id={contract_id}"
            )
        except Exception as exc:
            logger.error(
                f"[Consumer] Failed to dead-letter message: job_id={job_id}, contract_id={contract_id}, error={exc}"
            )

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

            # 2. 요약 + Risk 분석 병렬 실행
            # wait_for를 사용하지 않는 이유:
            # asyncio.to_thread 작업은 타임아웃으로 코루틴이 CancelledError를 받아도
            # 내부 OS 스레드가 멈추지 않아 스레드 풀이 고갈됨.
            # 대신 ChatOpenAI(timeout=...) 로 OpenAI 호출 단위에서 타임아웃을 제어함.
            logger.info(f"[Consumer] 병렬 분석 시작 (요약 + Risk)")
            summary_result, risk_analysis_result = await asyncio.gather(
                self._perform_summary(text),
                self._perform_risk_analysis(text),
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

            # Risk 분석 결과 추가 — if 가드 없이 항상 설정해야 SSE analysis_result 이벤트가 발행됨.
            # risk_analysis_result가 None/빈 dict이면 safe defaults로 처리.
            _ra = risk_analysis_result or {}
            risk_summary = _ra.get("risk_summary", {})
            clauses = _ra.get("clauses") or []
            total = len(clauses)
            raw_score = float(_ra.get("overall_risk_score") or 0)
            result.risk_analysis = RiskAnalysisResult(
                total_clauses=total,
                overall_risk_score=raw_score,
                overall_risk_level=self._overall_risk_level(raw_score),
                risk_count=risk_summary.get("Risk", 0),
                caution_count=risk_summary.get("Caution", 0),
                safety_count=risk_summary.get("Safety", 0),
                risk_percentage=round(risk_summary.get("Risk", 0) / total * 100, 1) if total > 0 else 0.0,
                detected_clause_count=risk_summary.get("Risk", 0) + risk_summary.get("Caution", 0),
                clause_results=[
                    ClauseRiskResult(
                        clause_title=clause.get("category") or clause.get("title") or (clause.get("text") or "")[:40],
                        clause_content=clause.get("content") or clause.get("text") or "",
                        risk_level=clause.get("risk_level", "안전"),
                        category=clause.get("category", ""),
                        score=int(clause.get("score") or 0),
                        legal_reference=clause.get("legal_reference", ""),
                        reasoning_summary=clause.get("analysis", ""),
                    )
                    for clause in clauses
                ],
            )

            return result

        except Exception as e:
            logger.exception(f"[Consumer] Analysis error: {type(e).__name__}: {e}")
            raise

    async def _perform_summary(self, text: str) -> dict:
        """
        RAG 기반 계약서 요약

        asyncio.to_thread(rag_query) 대신 async_search_multi_index + llm.ainvoke 사용:
        - 임베딩/Pinecone은 스레드, LLM 대기는 이벤트 루프 → 스레드 풀 점유 없음
        """
        try:
            db = get_vector_db()
            embeddings = get_embeddings()
            llm = get_llm()
        except Exception as e:
            logger.warning(f"[Consumer] RAG 미초기화 - 기본 요약 사용: {e}")
            return self._basic_summary(text)

        combined_question = (
            "아래 계약서를 분석하여 다음 세 가지를 답하세요.\n"
            "1. 주요 조건 (임대료, 보증금, 계약기간 등)\n"
            "2. 임차인이 주의해야 할 사항\n"
            "3. 특약사항 및 특이사항\n\n"
            f"계약서 내용:\n{text[:2000]}"
        )

        try:
            # 1. Pinecone 검색 (내부적으로 asyncio.to_thread 사용)
            docs = await async_search_multi_index(
                db, embeddings, combined_question,
                collections=["law_database"],
                k_per_collection=3,
            )

            # 2. LLM 비동기 호출 (스레드 불필요, 이벤트 루프에서 대기)
            context = build_context(docs)
            prompt_text = CONTRACT_QA_PROMPT.format(
                context=context,
                question=combined_question,
            )
            response = await asyncio.wait_for(
                llm.ainvoke(prompt_text),
                timeout=settings.ANALYSIS_TIMEOUT,
            )
            key_points = [{"answer": response.content}]

        except asyncio.TimeoutError:
            logger.warning(f"[Consumer] 요약 타임아웃 ({settings.ANALYSIS_TIMEOUT}s) - 기본 요약으로 폴백")
            return self._basic_summary(text)
        except Exception as e:
            logger.error(f"[Consumer] Summary error: {e}")
            return self._basic_summary(text)

        basic_info = self._extract_basic_info(text)
        logger.info("[Consumer] RAG 기반 요약 완료")
        return {
            "key_points": key_points,
            "basic_info": basic_info,
            "total_length": len(text),
            "estimated_read_time": max(1, len(text) // 200),
            "summary_type": "rag_based"
        }

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
            result = await asyncio.wait_for(
                detect_risk_contract(
                    user_clause=text,
                    client=db,
                    embeddings=embeddings,
                    llm=llm,
                ),
                timeout=settings.ANALYSIS_TIMEOUT,
            )
            rs = result["risk_summary"]
            logger.info(
                f"[Consumer] RAG 위험 분석 완료 - "
                f"Risk: {rs['Risk']}, Caution: {rs['Caution']}, Safety: {rs['Safety']}"
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"[Consumer] 위험 분석 타임아웃 ({settings.ANALYSIS_TIMEOUT}s) - 기본 결과 반환")
            return await self._basic_risk_analysis(text)
        except Exception as e:
            logger.error(f"[Consumer] Risk analysis error: {type(e).__name__}: {e}")
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
            # return 대신 raise — 메시지를 NACK해서 재시도 가능하게 함
            raise RuntimeError("[Consumer] Result exchange not initialized — cannot publish result")

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
