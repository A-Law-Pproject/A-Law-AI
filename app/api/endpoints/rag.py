"""
계약서 RAG 분석 엔드포인트
- 벡터 DB 상태 확인
- 독소조항 위험 탐지
- 법률 용어 해설
"""
import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from app.rag.chain.chain import detect_risk
from app.core.config import settings
from app.core.dependencies import get_vector_db, get_embeddings, get_llm

router = APIRouter()


class RiskDetectionRequest(BaseModel):
    """독소조항 분석 요청"""
    clause_text: str = Field(
        ...,
        description="분석할 계약 조항 텍스트",
        examples=[
            "보증금 반환은 임대인의 확인 후 30일 이내에 이루어지며, 이 기간 동안의 이자는 지급하지 않는다.",
            "임차인은 퇴거 시 다음 세입자를 직접 구해야 하며, 구하지 못할 경우 보증금 반환이 3개월 유예된다."
        ]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "clause_text": "보증금 반환은 임대인의 확인 후 30일 이내에 이루어지며, 이 기간 동안의 이자는 지급하지 않는다."
                },
                {
                    "clause_text": "임차인은 퇴거 시 다음 세입자를 직접 구해야 하며, 구하지 못할 경우 보증금 반환이 3개월 유예된다."
                },
                {
                    "clause_text": "계약기간 종료 후 보증금은 14일 이내에 반환하며, 지연 시 연 5% 이자를 가산하여 지급한다."
                }
            ]
        }
    }


class RiskDetectionResponse(BaseModel):
    """독소조항 위험 탐지 응답"""
    analysis: str = Field(..., description="AI 분석 결과")
    illegal_similarity: float = Field(..., description="독소조항 유사도 (0-1)")
    normal_similarity: float = Field(..., description="정상조항 유사도 (0-1)")
    risk_delta: float = Field(..., description="위험도 차이 (illegal - normal)")
    illegal_matches_count: int = Field(..., description="유사 독소조항 개수")
    normal_matches_count: int = Field(..., description="유사 정상조항 개수")
    law_matches_count: int = Field(..., description="관련 법률 개수")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "analysis": "이 조항은 임차인에게 매우 불리한 독소조항입니다. 주택임대차보호법 제7조에서는 보증금을 14일 이내에 반환하도록 규정하고 있으나, 이 조항은 30일로 연장하고 이자까지 미지급하여 임차인의 권리를 침해합니다.",
                    "illegal_similarity": 0.82,
                    "normal_similarity": 0.45,
                    "risk_delta": 0.37,
                    "illegal_matches_count": 3,
                    "normal_matches_count": 2,
                    "law_matches_count": 2
                }
            ]
        }
    }


@router.get("/health")
async def health_check():
    """RAG 시스템 상태 확인"""
    try:
        get_vector_db()
        return {
            "status": "healthy",
            "vector_db": settings.VECTOR_DB,
            "embedding_model": "KURE",
            "llm_model": settings.MODEL_NAME
        }
    except Exception as e:
        logger.error(f"Health check 실패: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/detect-risk", response_model=RiskDetectionResponse)
async def detect_clause_risk(request: RiskDetectionRequest):
    """
    독소조항 위험 탐지 (테스트용)

    계약 조항의 위험도를 분석하고 법률 근거 제시
    """
    db = get_vector_db()
    embeddings = get_embeddings()
    llm = get_llm()

    try:
        result = await asyncio.to_thread(
            detect_risk,
            user_clause=request.clause_text,
            client=db,
            embeddings=embeddings,
            llm=llm,
        )

        return {
            "analysis": result["analysis"],
            "illegal_similarity": result["illegal_similarity"],
            "normal_similarity": result["normal_similarity"],
            "risk_delta": result["risk_delta"],
            "illegal_matches_count": len(result["illegal_matches"]),
            "normal_matches_count": len(result["normal_matches"]),
            "law_matches_count": len(result["law_matches"])
        }
    except Exception as e:
        logger.error(f"독소조항 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
