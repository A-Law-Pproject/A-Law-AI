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
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.schemas.contract_analysis_dto import (
    ContractAnalysisRequest,
    ContractAnalysisResult,
    ContractSummary,
    RiskAnalysisResult,
    ClauseRiskResult,
    AnalysisStatus
)
from app.services.analyzer import ContractAnalysisService
from app.util.s3_client import S3Client
from app.rag.embedding.kure import KUREEmbeddings
from app.rag.chain.chain import rag_query, detect_risk


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

        # RAG 컴포넌트 초기화
        try:
            self.qdrant_client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY
            )
            self.embeddings = KUREEmbeddings()
            self.llm = ChatOpenAI(
                model=settings.MODEL_NAME,
                api_key=settings.OPENAI_API_KEY,
                temperature=0
            )
            logger.info("RAG 컴포넌트 초기화 완료 (Qdrant, Embeddings, LLM)")
        except Exception as e:
            logger.error(f"RAG 초기화 실패: {e}")
            self.qdrant_client = None
            self.embeddings = None
            self.llm = None

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
            s3_key = body.get('s3Key')

            logger.info(f"[Consumer] Received message: task_id={task_id}, s3_key={s3_key}")
            logger.info(f"[Consumer] 메시지 파싱 완료")

            # 요청 객체 생성
            request = ContractAnalysisRequest(**body)

            # 병렬 분석 수행
            result = await self._process_analysis(request)

            # 결과 발행 (RabbitMQ → Spring Boot)
            await self._publish_result(result)

            logger.info(f"[Consumer] Completed: task_id={task_id}")

        except json.JSONDecodeError as e:
            logger.error(f"[Consumer] JSON decode error: {e}")
            await self._publish_error(task_id, f"Invalid JSON: {e}")

        except Exception as e:
            logger.error(f"[Consumer] Processing error: {e}")
            await self._publish_error(task_id, str(e))


    async def _process_analysis(self, request: ContractAnalysisRequest) -> ContractAnalysisResult:
        """
        계약서 분석 처리 (OCR → 요약 → Risk 분석)

        Args:
            request: 분석 요청

        Returns:
            분석 결과
        """
        try:
            # S3에서 파일 다운로드
            logger.info(f"[Consumer] S3에서 파일 다운로드 중: {request.s3Key}")
            file_path = await self._download_from_s3(request.s3Key)

            # ============================================
            # 1️⃣ OCR 수행
            # ============================================
            logger.info(f"[Consumer] OCR 처리 시작")
            from app.services.ocr.ocr_service import OCRService
            ocr_service = OCRService()
            ocr_result = await self._perform_ocr(file_path)

            text = ocr_result.get("text", "")
            if not text:
                raise ValueError("OCR에서 텍스트를 추출할 수 없습니다")

            logger.info(f"[Consumer] OCR 완료: {len(text)} 글자")

            # ============================================
            # 2️⃣ 요약 분석
            # ============================================
            logger.info(f"[Consumer] 요약 분석 시작")
            summary_result = await self._perform_summary(text)
            logger.info(f"[Consumer] 요약 완료")

            # ============================================
            # 3️⃣ Risk 분석 (RAG 기반 독소조항 탐지)
            # ============================================
            logger.info(f"[Consumer] RAG 기반 Risk 분석 시작")
            risk_analysis_result = await self._perform_risk_analysis(text)
            logger.info(f"[Consumer] Risk 분석 완료")

            # 결과 객체 생성
            result = ContractAnalysisResult(
                task_id=request.taskId,
                status=AnalysisStatus.COMPLETED
            )

            # 요약 결과 추가
            if summary_result:
                # 요약 텍스트 생성
                summary_text_parts = []

                # 기본 정보
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

                # RAG 기반 요약 포인트
                key_points = summary_result.get("key_points", [])
                if key_points:
                    for point in key_points:
                        if isinstance(point, dict) and "answer" in point:
                            summary_text_parts.append(f"• {point['answer']}")

                summary_text = "\n\n".join(summary_text_parts) if summary_text_parts else "계약서 요약 정보가 없습니다."

                result.summary = ContractSummary(
                    title="임대차 계약서",
                    summary_text=summary_text,
                    key_terms=self._generate_recommendations(risk_analysis_result)
                )

            # Risk 분석 결과 추가
            if risk_analysis_result and risk_analysis_result.get("clauses"):
                result.risk_analysis = RiskAnalysisResult(
                    total_clauses=len(risk_analysis_result["clauses"]),
                    clause_results=[
                        ClauseRiskResult(
                            clause_title=clause["title"],
                            clause_content=clause["content"],
                            risk_level=clause["risk_level"],
                            legal_reference=f"독소조항 유사도: {clause.get('illegal_similarity', 0):.2f}",
                            recommendation=clause["recommendation"]
                        )
                        for clause in risk_analysis_result["clauses"]
                    ]
                )

            return result

        except Exception as e:
            logger.error(f"[Consumer] Analysis error: {e}")
            raise

    async def _download_from_s3(self, s3_key: str) -> str:
        """S3에서 파일 다운로드"""
        try:
            s3_client = S3Client()
            file_path = await s3_client.download_file(s3_key)
            logger.info(f"[Consumer] Downloaded from S3: {s3_key}")
            return file_path
        except Exception as e:
            logger.error(f"[Consumer] S3 download error: {e}")
            raise

    async def _perform_ocr(self, file_path: str) -> dict:
        """OCR 수행"""
        try:
            from app.services.ocr.ocr_service import OCRService
            ocr_service = OCRService()
            
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
            
            # OCR 처리
            ocr_result = ocr_service.process_and_map(
                image_bytes=image_bytes,
                structurize=False,
                include_overlay=False
            )
            
            logger.info(f"[Consumer] OCR completed: {len(ocr_result.text)} chars")
            
            return {
                "text": ocr_result.text,
                "words": ocr_result.words or [],
                "lines": ocr_result.lines or [],
                "paragraphs": ocr_result.paragraphs or [],
                "confidence": getattr(ocr_result, 'confidence', 0)
            }
        except Exception as e:
            logger.error(f"[Consumer] OCR error: {e}")
            raise

    async def _perform_summary(self, text: str) -> dict:
        """
        RAG 기반 계약서 요약

        법률 문서, 약관, 특약사항 DB를 참조하여 계약서의 핵심 내용을 요약
        """
        try:
            if not all([self.qdrant_client, self.embeddings, self.llm]):
                logger.warning("[Consumer] RAG 미초기화 - 기본 요약 사용")
                return self._basic_summary(text)

            # RAG 질의를 통한 요약
            summary_questions = [
                "이 계약서의 주요 조건은 무엇인가요? (임대료, 보증금, 계약기간 등)",
                "이 계약서에서 임차인이 주의해야 할 중요한 사항은 무엇인가요?",
                "특약사항이나 특이사항이 있다면 무엇인가요?"
            ]

            key_points = []
            for question in summary_questions:
                try:
                    # 계약서 텍스트를 컨텍스트로 사용하여 RAG 질의
                    result = rag_query(
                        question=f"{question}\n\n계약서 내용:\n{text[:2000]}",  # 첫 2000자만 사용
                        client=self.qdrant_client,
                        embeddings=self.embeddings,
                        llm=self.llm,
                        collections=["law_database"],
                        k_per_collection=2
                    )
                    key_points.append({
                        "question": question,
                        "answer": result["answer"]
                    })
                except Exception as e:
                    logger.error(f"[Consumer] RAG 질의 실패: {e}")
                    continue

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
        """
        RAG 기반 계약서 위험 분석

        독소조항 DB를 활용하여 각 조항의 위험도를 분석
        """
        try:
            if not all([self.qdrant_client, self.embeddings, self.llm]):
                logger.warning("[Consumer] RAG 미초기화 - 기본 분석 사용")
                return await self._basic_risk_analysis(text)

            # 계약서를 조항으로 분리
            clauses = self._split_into_clauses(text)
            logger.info(f"[Consumer] 계약서를 {len(clauses)}개 조항으로 분리")

            analyzed_clauses = []
            risk_count = 0
            caution_count = 0
            safety_count = 0

            for idx, clause in enumerate(clauses, 1):
                if len(clause["content"].strip()) < 20:  # 너무 짧은 조항은 스킵
                    continue

                try:
                    logger.debug(f"[Consumer] 조항 {idx}/{len(clauses)} 분석 중...")

                    # RAG 기반 독소조항 탐지
                    risk_result = detect_risk(
                        user_clause=clause["content"],
                        client=self.qdrant_client,
                        embeddings=self.embeddings,
                        llm=self.llm
                    )

                    # 위험도 판정
                    risk_level = self._determine_risk_level(
                        risk_result["illegal_similarity"],
                        risk_result["normal_similarity"],
                        risk_result["risk_delta"]
                    )

                    if risk_level == "Risk":
                        risk_count += 1
                    elif risk_level == "Caution":
                        caution_count += 1
                    else:
                        safety_count += 1

                    analyzed_clauses.append({
                        "title": clause["title"],
                        "content": clause["content"],
                        "risk_level": risk_level,
                        "recommendation": risk_result["analysis"],
                        "illegal_similarity": risk_result["illegal_similarity"],
                        "normal_similarity": risk_result["normal_similarity"],
                        "risk_delta": risk_result["risk_delta"]
                    })

                    # 최대 10개 조항만 분석 (시간 절약)
                    if len(analyzed_clauses) >= 10:
                        logger.info("[Consumer] 최대 분석 조항 수 도달 (10개)")
                        break

                except Exception as e:
                    logger.error(f"[Consumer] 조항 {idx} 분석 실패: {e}")
                    continue

            total_analyzed = len(analyzed_clauses)
            overall_risk_score = (
                (risk_count * 100 + caution_count * 50) / total_analyzed
                if total_analyzed > 0 else 0
            )

            result = {
                "total_clauses": total_analyzed,
                "risk_summary": {
                    "Risk": risk_count,
                    "Caution": caution_count,
                    "Safety": safety_count
                },
                "overall_risk_score": round(overall_risk_score, 1),
                "clauses": analyzed_clauses[:10]  # 상위 10개만 반환
            }

            logger.info(
                f"[Consumer] RAG 기반 위험 분석 완료 - "
                f"Risk: {risk_count}, Caution: {caution_count}, Safety: {safety_count}"
            )
            return result

        except Exception as e:
            logger.error(f"[Consumer] Risk analysis error: {e}")
            return await self._basic_risk_analysis(text)

    def _split_into_clauses(self, text: str) -> list[dict]:
        """
        계약서를 조항으로 분리

        조항 구분 기준:
        - 제X조, 제X항 패턴
        - 특약사항
        - 번호 매김 (1., 2., ...)
        """
        import re

        clauses = []

        # 제X조 패턴으로 분리
        article_pattern = r'(제\s*\d+\s*조[^\n]*)\n([^제]*)'
        matches = re.finditer(article_pattern, text)

        for match in matches:
            title = match.group(1).strip()
            content = match.group(2).strip()
            if content:
                clauses.append({"title": title, "content": content})

        # 특약사항 분리
        special_pattern = r'(특약\s*사항|특약|기타\s*사항)[:\s]*\n([^제]*)'
        special_match = re.search(special_pattern, text, re.IGNORECASE)
        if special_match:
            clauses.append({
                "title": "특약사항",
                "content": special_match.group(2).strip()
            })

        # 조항이 없으면 문단으로 분리
        if len(clauses) == 0:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            for i, para in enumerate(paragraphs[:5], 1):  # 최대 5개 문단
                clauses.append({
                    "title": f"문단 {i}",
                    "content": para
                })

        return clauses

    def _determine_risk_level(
        self,
        illegal_similarity: float,
        normal_similarity: float,
        risk_delta: float
    ) -> str:
        """
        위험도 판정

        - Risk: 독소조항 유사도가 높고 정상조항 유사도가 낮음
        - Caution: 중간 정도
        - Safety: 안전
        """
        if illegal_similarity > 0.7 and risk_delta > 0.1:
            return "Risk"
        elif illegal_similarity > 0.5 or risk_delta > 0:
            return "Caution"
        else:
            return "Safety"

    async def _basic_risk_analysis(self, text: str) -> dict:
        """RAG 실패 시 기본 위험 분석 (ContractAnalysisService 사용)"""
        try:
            analyzer = ContractAnalysisService()
            result = await analyzer.analyze_contract(text, "unknown")

            return {
                "total_clauses": result.total_clauses,
                "risk_summary": result.risk_summary,
                "overall_risk_score": result.overall_risk_score,
                "clauses": [
                    {
                        "title": clause.title,
                        "content": clause.content,
                        "risk_level": clause.risk_level,
                        "recommendation": clause.recommendation
                    }
                    for clause in result.clauses
                ] if result.clauses else []
            }
        except Exception as e:
            logger.error(f"[Consumer] Basic risk analysis failed: {e}")
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
