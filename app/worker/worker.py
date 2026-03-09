"""
분석 워커 - Celery + RabbitMQ
GPT-4o를 사용하여 계약서를 분석하고 결과를 저장합니다.

실행 방법:
    celery -A app.core.celery_config worker --loglevel=info --queues=contract.analysis.queue

병렬 처리:
    - AI 요약 (5-10초)
    - Risk 분석 (3-7초)
"""
import json
from datetime import datetime
from typing import Dict, Optional

import openai
import psycopg2
import psycopg2.extras
import redis
from kombu import Connection, Exchange, Queue, Producer
from loguru import logger

from app.core.celery_config import celery_app
from app.core.config import settings
from app.core.dependencies import get_qdrant_client, get_embeddings, get_llm
from app.rag.chunking.legal_chunker import create_legal_splitter
from app.rag.retriever.multi_retriever import search_collection
from app.rag.chain.chain import detect_risk
from app.schemas.contract_analysis_dto import (
    ContractAnalysisRequest,
    ContractAnalysisResult,
    ContractSummary,
    RiskAnalysisResult,
    ClauseRiskResult,
    AnalysisStatus
)


# ===========================================
# Redis 클라이언트 (상태 관리용)
# ===========================================
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


# ===========================================
# RabbitMQ Result Publisher
# ===========================================
class ResultPublisher:
    """분석 결과를 RabbitMQ로 발행 (Spring Boot가 소비)"""

    def __init__(self):
        self.exchange = Exchange(
            settings.RESULT_EXCHANGE,
            type='direct',
            durable=True
        )
        self.result_queue = Queue(
            'ai.result.queue',
            exchange=self.exchange,
            routing_key='ai.result',
            durable=True
        )

    def publish(self, result: ContractAnalysisResult):
        """결과 메시지 발행"""
        try:
            with Connection(settings.RABBITMQ_URL) as conn:
                producer = Producer(conn)
                producer.publish(
                    result.to_rabbitmq_message(),
                    exchange=self.exchange,
                    routing_key='ai.result',
                    declare=[self.result_queue],
                    serializer='json',
                    content_type='application/json'
                )
            logger.info(f"Published result to ai.result.queue: task_id={result.task_id}")
        except Exception as e:
            logger.error(f"Failed to publish result: {e}")
            raise


# 전역 Publisher 인스턴스
result_publisher = ResultPublisher()


# ===========================================
# 데이터베이스 헬퍼 함수
# ===========================================

def get_db_connection():
    """PostgreSQL DB 커넥션 생성"""
    return psycopg2.connect(
        dbname=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        host=settings.DB_HOST,
        port=settings.DB_PORT
    )


def update_job_status_db(
    job_id: str,
    status: str,
    result: Optional[Dict] = None,
    error: Optional[str] = None
):
    """DB에 작업 상태 업데이트"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            UPDATE contract_analysis_jobs
            SET status = %s,
                result = %s,
                error_message = %s,
                updated_at = %s
            WHERE job_id = %s
        """, (
            status,
            psycopg2.extras.Json(result) if result else None,
            error,
            datetime.now(),
            job_id
        ))

        conn.commit()
        cur.close()
        conn.close()
        logger.debug(f"DB status updated: {job_id} -> {status}")
    except Exception as e:
        logger.error(f"DB update failed: {e}")


def update_status_redis(
    job_id: str,
    status: str,
    error: Optional[str] = None
):
    """Redis에 작업 상태 업데이트 (빠른 조회용)"""
    status_data = {
        "jobId": job_id,
        "status": status,
        "updatedAt": datetime.now().isoformat()
    }
    if error:
        status_data["error"] = error

    redis_client.setex(
        f"status:{job_id}",
        3600,  # 1시간 TTL
        json.dumps(status_data)
    )


def save_result_redis(job_id: str, result: Dict):
    """Redis에 분석 결과 저장"""
    redis_client.setex(
        f"result:{job_id}",
        3600,  # 1시간 TTL
        json.dumps(result)
    )


# ===========================================
# RAG 독소조항 탐지 헬퍼
# ===========================================

def _rag_detect_risky_clauses(
    ocr_text: str,
    risk_delta_threshold: float = 0.1,
    max_clauses: int = 5,
) -> list[dict]:
    """LegalChunker로 조항 분할 후 RAG 기반 독소조항 스크리닝 + LLM 분석.

    1단계: 모든 조항에 대해 임베딩 + Qdrant 검색만 수행 (LLM 없음)
    2단계: risk_delta(불법 유사도 - 일반 유사도) > threshold인 조항만 LLM 분석

    Args:
        ocr_text: OCR로 추출된 전체 계약서 텍스트.
        risk_delta_threshold: LLM 분석 대상 임계값.
        max_clauses: 비용 제한용 최대 LLM 분석 조항 수.

    Returns:
        [{"clause_text", "risk_delta", "analysis", "illegal_similarity", "normal_similarity"}, ...]
    """
    qdrant = get_qdrant_client()
    embeddings = get_embeddings()
    llm = get_llm()

    # 1. 조항 단위 분할
    splitter = create_legal_splitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(ocr_text)
    if not chunks:
        return []

    # 2. 스크리닝: 임베딩 1회 + Qdrant 검색으로 risk_delta 계산
    risky_chunks: list[tuple[str, float]] = []
    for chunk in chunks:
        try:
            query_vector = embeddings.embed_query(chunk)
            illegal_results = search_collection(
                qdrant, embeddings, chunk, "special_clauses_illegal",
                k=2, query_vector=query_vector,
            )
            normal_results = search_collection(
                qdrant, embeddings, chunk, "special_clauses_normal",
                k=2, query_vector=query_vector,
            )
            illegal_score = max(
                (d.metadata.get("score", 0) for d in illegal_results), default=0.0
            )
            normal_score = max(
                (d.metadata.get("score", 0) for d in normal_results), default=0.0
            )
            risk_delta = illegal_score - normal_score
            if risk_delta > risk_delta_threshold:
                risky_chunks.append((chunk, risk_delta))
        except Exception as e:
            logger.warning(f"RAG 스크리닝 실패 (chunk 건너뜀): {e}")

    if not risky_chunks:
        return []

    # 3. risk_delta 내림차순 정렬 후 상위 max_clauses만 LLM 분석
    risky_chunks.sort(key=lambda x: x[1], reverse=True)
    risky_chunks = risky_chunks[:max_clauses]

    rag_results: list[dict] = []
    for chunk_text, delta in risky_chunks:
        try:
            result = detect_risk(chunk_text, qdrant, embeddings, llm)
            rag_results.append({
                "clauseText": chunk_text,
                "riskDelta": round(delta, 4),
                "analysis": result.get("analysis", ""),
                "illegalSimilarity": round(result.get("illegal_similarity", 0), 4),
                "normalSimilarity": round(result.get("normal_similarity", 0), 4),
            })
        except Exception as e:
            logger.warning(f"RAG detect_risk LLM 분석 실패: {e}")

    return rag_results


# ===========================================
# Celery 태스크
# ===========================================

@celery_app.task(
    name="tasks.analyze_contract",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    autoretry_for=(openai.APIError, openai.APITimeoutError),
    retry_backoff=True
)
def analyze_contract(
    self,
    job_id: str,
    contract_id: str,
    ocr_text: str,
    callback_url: Optional[str] = None
):
    """
    GPT-4o를 호출하여 계약서 분석

    Args:
        job_id: 분석 작업 ID
        contract_id: 계약서 ID
        ocr_text: OCR로 추출된 텍스트
        callback_url: 완료 콜백 URL (선택)

    Returns:
        분석 결과 딕셔너리
    """
    logger.info(f"Starting analysis: job_id={job_id}, contract_id={contract_id}")

    try:
        # 1. 상태 업데이트: ANALYZING
        update_status_redis(job_id, "ANALYZING")
        update_job_status_db(job_id, "ANALYZING")

        # 2. GPT-4o 호출
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

        response = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": """당신은 부동산 임대차 계약서 분석 전문가입니다.

다음 항목을 분석하세요:
1. 사기 위험 요소 (fraudRisks): 계약금 과다 요구, 이중 계약, 허위 등기 등
2. 누락된 필수 조항 (missingClauses): 확정일자, 중도 해지, 수리 책임 등
3. 불법/불공정 조항 (illegalClauses): 주택임대차보호법 위반 조항

각 항목은 다음 형식으로 응답하세요:
{
    "fraudRisks": [{"pattern": "...", "severity": "high|medium|low", "description": "..."}],
    "missingClauses": [{"clause_name": "...", "importance": "critical|important|recommended", "description": "..."}],
    "illegalClauses": [{"clause_text": "...", "violation": "...", "legal_reference": "...", "recommendation": "..."}],
    "summary": "전체 요약",
    "recommendations": ["권고사항1", "권고사항2"]
}"""
                },
                {
                    "role": "user",
                    "content": f"다음 계약서를 분석하세요:\n\n{ocr_text}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=4000
        )

        # 3. 응답 파싱
        analysis_raw = response.choices[0].message.content
        analysis_data = json.loads(analysis_raw)

        # 4. 위험도 점수 계산
        risk_score = calculate_risk_score(
            analysis_data.get("fraudRisks", []),
            analysis_data.get("missingClauses", []),
            analysis_data.get("illegalClauses", [])
        )

        # 5. RAG 기반 독소조항 탐지 (조항 분할 → 스크리닝 → LLM)
        try:
            rag_risky_clauses = _rag_detect_risky_clauses(ocr_text)
            logger.info(f"RAG 독소조항 탐지 완료: {len(rag_risky_clauses)}개 발견")
        except Exception as e:
            logger.warning(f"RAG 독소조항 탐지 실패 (GPT-4o 결과만 사용): {e}")
            rag_risky_clauses = []

        # 6. 최종 결과 구성
        result = {
            "jobId": job_id,
            "contractId": contract_id,
            "fraudRisks": analysis_data.get("fraudRisks", []),
            "missingClauses": analysis_data.get("missingClauses", []),
            "illegalClauses": analysis_data.get("illegalClauses", []),
            "ragRiskyClauses": rag_risky_clauses,
            "summary": analysis_data.get("summary", ""),
            "recommendations": analysis_data.get("recommendations", []),
            "riskScore": risk_score,
            "completedAt": datetime.now().isoformat()
        }

        # 7. 결과 저장
        save_result_redis(job_id, result)
        update_status_redis(job_id, "COMPLETED")
        update_job_status_db(job_id, "COMPLETED", result=result)

        # 8. 콜백 호출 (있는 경우)
        if callback_url:
            send_callback(callback_url, result)

        logger.info(f"Analysis completed: job_id={job_id}, risk_score={risk_score}")
        return result

    except json.JSONDecodeError as e:
        error_msg = f"GPT 응답 파싱 실패: {e}"
        logger.error(error_msg)
        update_status_redis(job_id, "FAILED", error_msg)
        update_job_status_db(job_id, "FAILED", error=error_msg)
        raise

    except openai.APIError as e:
        error_msg = f"OpenAI API 오류: {e}"
        logger.warning(f"{error_msg}, retrying... (attempt {self.request.retries + 1})")
        update_status_redis(job_id, "RETRYING", error_msg)
        update_job_status_db(job_id, "RETRYING", error=error_msg)
        raise  # Celery가 자동 재시도

    except Exception as e:
        error_msg = f"분석 실패: {str(e)}"
        logger.error(error_msg)
        update_status_redis(job_id, "FAILED", error_msg)
        update_job_status_db(job_id, "FAILED", error=error_msg)
        raise


def calculate_risk_score(
    fraud_risks: list,
    missing_clauses: list,
    illegal_clauses: list
) -> float:
    """
    종합 위험도 점수 계산 (0-100)

    - 사기 위험: high=30, medium=15, low=5
    - 누락 조항: critical=20, important=10, recommended=5
    - 불법 조항: 각 25점
    """
    score = 0.0

    severity_scores = {"high": 30, "medium": 15, "low": 5}
    for risk in fraud_risks:
        severity = risk.get("severity", "low")
        score += severity_scores.get(severity, 5)

    importance_scores = {"critical": 20, "important": 10, "recommended": 5}
    for clause in missing_clauses:
        importance = clause.get("importance", "recommended")
        score += importance_scores.get(importance, 5)

    score += len(illegal_clauses) * 25

    return min(score, 100.0)


def send_callback(callback_url: str, result: Dict):
    """콜백 URL로 결과 전송"""
    import httpx

    try:
        with httpx.Client(timeout=10) as client:
            response = client.post(
                callback_url,
                json=result,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            logger.info(f"Callback sent to {callback_url}")
    except Exception as e:
        logger.warning(f"Callback failed: {callback_url}, error: {e}")


# ===========================================
# 유틸리티 태스크
# ===========================================

@celery_app.task(name="tasks.health_check")
def health_check():
    """워커 헬스체크"""
    return {
        "status": "healthy",
        "worker": "celery",
        "timestamp": datetime.now().isoformat()
    }


# ===========================================
# 병렬 처리 태스크 (AI 요약 + Risk 분석)
# ===========================================

# @celery_app.task(
#     name="tasks.analyze_contract_parallel",
#     bind=True,
#     max_retries=3,
#     default_retry_delay=30,
#     acks_late=True
# )
# def analyze_contract_parallel(self, message: dict):
#     """
#     계약서 병렬 분석 태스크 (Spring Boot 연동용)

#     병렬 처리:
#         - AI 요약 (5-10초)
#         - Risk 분석 (3-7초)

#     Args:
#         message: ContractAnalysisRequest JSON (Spring Boot에서 발행)

#     Returns:
#         분석 결과 dict
#     """
#     task_id = message.get('taskId', 'unknown')
#     logger.info(f"[Parallel] Received analysis task: {task_id}")

#     try:
#         # 메시지 파싱
#         request = ContractAnalysisRequest(**message)

#         # 상태 업데이트
#         update_status_redis(task_id, "PROCESSING")

#         # 비동기 분석 실행
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         try:
#             result = loop.run_until_complete(
#                 _process_parallel_analysis(request)
#             )
#         finally:
#             loop.close()

#         # 결과 발행 (ai.result.queue로)
#         result_publisher.publish(result)

#         # Redis/DB 상태 업데이트
#         update_status_redis(task_id, "COMPLETED")
#         save_result_redis(task_id, result.model_dump(by_alias=True))

#         logger.info(f"[Parallel] Analysis completed: {task_id}")
#         return result.model_dump(by_alias=True)

#     except Exception as e:
#         logger.error(f"[Parallel] Task failed: {e}")

#         # 실패 결과 발행
#         error_result = ContractAnalysisResult(
#             task_id=task_id,
#             status=AnalysisStatus.FAILED,
#             error_message=str(e)
#         )
#         result_publisher.publish(error_result)
#         update_status_redis(task_id, "FAILED", str(e))

#         raise self.retry(exc=e)



# def _convert_risk_assessments(assessments: list) -> RiskAnalysisResult:
#     """ClauseRiskAssessment 리스트를 RiskAnalysisResult로 변환"""
#     if not assessments:
#         return RiskAnalysisResult(
#             total_clauses=0,
#             clause_results=[]
#         )

#     clause_results = []
#     risk_count = 0
#     caution_count = 0
#     safety_count = 0

#     for assessment in assessments:
#         if assessment.risk_level == "Risk":
#             risk_count += 1
#         elif assessment.risk_level == "Caution":
#             caution_count += 1
#         else:
#             safety_count += 1

#         # Reasoning 요약 (처음 2개 step만)
#         reasoning_summary = " → ".join([
#             step.thought for step in assessment.reasoning[:2]
#         ]) if assessment.reasoning else ""

#         clause_results.append(ClauseRiskResult(
#             clause_title=assessment.clause_title,
#             clause_content=assessment.clause_content,
#             risk_level=assessment.risk_level,
#             legal_reference=assessment.legal_reference,
#             recommendation=assessment.recommendation,
#             reasoning_summary=reasoning_summary
#         ))

#     total = len(assessments)
#     risk_percentage = round(risk_count / total * 100, 1) if total > 0 else 0

#     return RiskAnalysisResult(
#         total_clauses=total,
#         risk_count=risk_count,
#         caution_count=caution_count,
#         safety_count=safety_count,
#         risk_percentage=risk_percentage,
#         clause_results=clause_results
#     )
