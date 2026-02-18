"""
계약서 분석 API 엔드포인트
- Spring Boot에서 OCR 완료 후 비동기 분석 요청
- 분석 상태/결과 조회
"""
import json
import uuid
from datetime import datetime
from typing import Optional

import redis
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.celery_config import celery_app
from app.worker.worker import analyze_contract


router = APIRouter()

# Redis 클라이언트 (상태 조회용)
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


# ===========================================
# Request/Response 스키마
# ===========================================

class AnalysisSubmitRequest(BaseModel):
    """분석 제출 요청"""
    contract_id: str = Field(..., description="계약서 ID (Spring Boot에서 발급)")
    ocr_text: str = Field(..., description="OCR로 추출된 계약서 텍스트")
    priority: int = Field(default=1, ge=1, le=3, description="우선순위 (1=낮음, 3=높음)")
    callback_url: Optional[str] = Field(None, description="완료 시 콜백 URL (선택)")


class AnalysisSubmitResponse(BaseModel):
    """분석 제출 응답"""
    job_id: str = Field(..., description="분석 작업 ID")
    contract_id: str = Field(..., description="계약서 ID")
    status: str = Field(..., description="상태: QUEUED")


class AnalysisStatusResponse(BaseModel):
    """분석 상태 응답"""
    job_id: str
    status: str  # QUEUED, ANALYZING, COMPLETED, FAILED, RETRYING
    updated_at: str
    error: Optional[str] = None


class AnalysisResultResponse(BaseModel):
    """분석 결과 응답"""
    job_id: str
    contract_id: str
    fraud_risks: list
    missing_clauses: list
    illegal_clauses: list
    summary: Optional[str] = None
    recommendations: Optional[list] = None
    risk_score: float
    completed_at: str


# ===========================================
# API 엔드포인트
# ===========================================

@router.post("/submit", response_model=AnalysisSubmitResponse)
async def submit_analysis(request: AnalysisSubmitRequest):
    """
    분석 작업을 Celery 큐에 제출 (비동기)

    Spring Boot에서 OCR 완료 후 호출합니다.
    Celery 태스크를 생성하고 즉시 job_id를 반환합니다.
    """
    job_id = str(uuid.uuid4())

    try:
        # 1. 초기 상태 저장 (Redis)
        status_data = {
            "jobId": job_id,
            "contractId": request.contract_id,
            "status": "QUEUED",
            "updatedAt": datetime.now().isoformat()
        }
        redis_client.setex(
            f"status:{job_id}",
            3600,  # 1시간 TTL
            json.dumps(status_data)
        )

        # 2. Celery 태스크 제출
        task = analyze_contract.apply_async(
            kwargs={
                "job_id": job_id,
                "contract_id": request.contract_id,
                "ocr_text": request.ocr_text,
                "callback_url": request.callback_url
            },
            task_id=job_id,  # job_id를 task_id로 사용
            priority=request.priority,
            queue=settings.ANALYSIS_QUEUE
        )

        logger.info(f"Analysis job submitted: job_id={job_id}, task_id={task.id}")

        return AnalysisSubmitResponse(
            job_id=job_id,
            contract_id=request.contract_id,
            status="QUEUED"
        )

    except redis.ConnectionError as e:
        logger.error(f"Redis connection error: {e}")
        raise HTTPException(503, "Redis 연결 실패")
    except Exception as e:
        logger.error(f"Submit analysis error: {e}")
        raise HTTPException(500, f"분석 요청 실패: {str(e)}")


@router.get("/status/{job_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(job_id: str):
    """
    분석 상태 조회

    Spring Boot에서 폴링으로 상태를 확인할 때 사용합니다.
    """
    try:
        # 1. Redis에서 상태 조회 (우선)
        status_raw = redis_client.get(f"status:{job_id}")
        if status_raw:
            status_data = json.loads(status_raw)
            return AnalysisStatusResponse(
                job_id=status_data["jobId"],
                status=status_data["status"],
                updated_at=status_data["updatedAt"],
                error=status_data.get("error")
            )

        # 2. Celery 태스크 상태 확인 (폴백)
        task_result = AsyncResult(job_id, app=celery_app)
        celery_status_map = {
            "PENDING": "QUEUED",
            "STARTED": "ANALYZING",
            "SUCCESS": "COMPLETED",
            "FAILURE": "FAILED",
            "RETRY": "RETRYING"
        }

        status = celery_status_map.get(task_result.status, task_result.status)
        error = str(task_result.result) if task_result.failed() else None

        return AnalysisStatusResponse(
            job_id=job_id,
            status=status,
            updated_at=datetime.now().isoformat(),
            error=error
        )

    except redis.ConnectionError as e:
        logger.error(f"Redis connection error: {e}")
        raise HTTPException(503, "Redis 연결 실패")


@router.get("/result/{job_id}", response_model=AnalysisResultResponse)
async def get_analysis_result(job_id: str):
    """
    분석 결과 조회

    분석이 완료된 후 결과를 가져올 때 사용합니다.
    """
    try:
        # 1. Redis에서 결과 조회 (캐시)
        result_raw = redis_client.get(f"result:{job_id}")
        if result_raw:
            result = json.loads(result_raw)
            return AnalysisResultResponse(
                job_id=result["jobId"],
                contract_id=result["contractId"],
                fraud_risks=result.get("fraudRisks", []),
                missing_clauses=result.get("missingClauses", []),
                illegal_clauses=result.get("illegalClauses", []),
                summary=result.get("summary"),
                recommendations=result.get("recommendations"),
                risk_score=result.get("riskScore", 0.0),
                completed_at=result.get("completedAt", "")
            )

        # 2. Celery 태스크 결과 조회 (폴백)
        task_result = AsyncResult(job_id, app=celery_app)

        if task_result.successful():
            result = task_result.result
            return AnalysisResultResponse(
                job_id=result["jobId"],
                contract_id=result["contractId"],
                fraud_risks=result.get("fraudRisks", []),
                missing_clauses=result.get("missingClauses", []),
                illegal_clauses=result.get("illegalClauses", []),
                summary=result.get("summary"),
                recommendations=result.get("recommendations"),
                risk_score=result.get("riskScore", 0.0),
                completed_at=result.get("completedAt", "")
            )
        elif task_result.failed():
            raise HTTPException(500, f"분석 실패: {task_result.result}")
        else:
            raise HTTPException(400, f"분석 미완료. 현재 상태: {task_result.status}")

    except redis.ConnectionError as e:
        logger.error(f"Redis connection error: {e}")
        raise HTTPException(503, "Redis 연결 실패")


@router.delete("/cancel/{job_id}")
async def cancel_analysis(job_id: str):
    """
    분석 작업 취소

    QUEUED 상태인 작업만 취소 가능합니다.
    """
    try:
        # Celery 태스크 취소
        task_result = AsyncResult(job_id, app=celery_app)

        if task_result.status == "PENDING":
            task_result.revoke(terminate=True)

            # 상태 업데이트
            status_data = {
                "jobId": job_id,
                "status": "CANCELLED",
                "updatedAt": datetime.now().isoformat()
            }
            redis_client.setex(
                f"status:{job_id}",
                3600,
                json.dumps(status_data)
            )

            logger.info(f"Analysis job cancelled: {job_id}")
            return {"job_id": job_id, "status": "CANCELLED"}
        else:
            raise HTTPException(
                400,
                f"취소 불가. 현재 상태: {task_result.status}"
            )

    except redis.ConnectionError as e:
        logger.error(f"Redis connection error: {e}")
        raise HTTPException(503, "Redis 연결 실패")
