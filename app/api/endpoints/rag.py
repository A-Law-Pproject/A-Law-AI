"""
RAG 관리 API 엔드포인트
- 벡터 DB 상태 확인
- RAG 기반 질의/분석 테스트
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger

from app.rag.chain.chain import rag_query, detect_risk
from app.rag.embedding.kure import KUREEmbeddings
from app.core.config import settings
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

router = APIRouter()

# Qdrant 클라이언트 초기화 (전역)
try:
    qdrant_client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY
    )
    embeddings = KUREEmbeddings()
    llm = ChatOpenAI(
        model=settings.MODEL_NAME,
        api_key=settings.OPENAI_API_KEY,
        temperature=0
    )
    logger.info("RAG 컴포넌트 초기화 완료")
except Exception as e:
    logger.error(f"RAG 초기화 실패: {e}")
    qdrant_client = None
    embeddings = None
    llm = None


class RagQueryRequest(BaseModel):
    """RAG 질의 요청"""
    question: str = Field(
        ...,
        description="질문 내용",
        examples=["보증금 반환 시기는 언제인가요?", "임대차 계약 갱신은 어떻게 하나요?"]
    )
    collections: Optional[list[str]] = Field(
        default=["law_database"],
        description="검색할 컬렉션 목록 (law_database, contracts, special_clauses_illegal, special_clauses_normal)",
        examples=[["law_database"], ["law_database", "contracts"]]
    )
    k_per_collection: Optional[int] = Field(
        default=3,
        description="컬렉션당 검색할 문서 수",
        examples=[3, 5]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "보증금 반환 시기는 언제인가요?",
                    "collections": ["law_database"],
                    "k_per_collection": 3
                },
                {
                    "question": "임차인이 계약을 해지할 수 있는 조건은 무엇인가요?",
                    "collections": ["law_database", "contracts"],
                    "k_per_collection": 5
                }
            ]
        }
    }


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


class RagQueryResponse(BaseModel):
    """RAG 질의 응답"""
    answer: str = Field(..., description="RAG 기반 답변")
    context: str = Field(..., description="검색된 참고 문서")
    source_count: int = Field(..., description="참고한 문서 수")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "주택임대차보호법 제7조에 따르면 임대차가 종료된 경우 임대인은 임차인으로부터 목적물을 반환받은 날로부터 14일 이내에 보증금을 반환해야 합니다.",
                    "context": "제7조(보증금의 반환) 임대인은 임대차가 종료되거나 해지되어 임차인이 주택을 명도할 때에는...",
                    "source_count": 3
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
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant 클라이언트 초기화 실패")

    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        return {
            "status": "healthy",
            "qdrant_url": settings.QDRANT_URL,
            "collections": collection_names,
            "embedding_model": "KURE",
            "llm_model": settings.MODEL_NAME
        }
    except Exception as e:
        logger.error(f"Health check 실패: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/query", response_model=RagQueryResponse)
async def query_rag(request: RagQueryRequest):
    """
    RAG 기반 질의 (테스트용)

    법률 문서, 약관, 특약사항을 검색하여 답변 생성
    """
    if not all([qdrant_client, embeddings, llm]):
        raise HTTPException(status_code=503, detail="RAG 시스템 초기화 실패")

    try:
        result = rag_query(
            question=request.question,
            client=qdrant_client,
            embeddings=embeddings,
            llm=llm,
            collections=request.collections,
            k_per_collection=request.k_per_collection
        )

        return {
            "answer": result["answer"],
            "context": result["context"],
            "source_count": len(result["source_documents"])
        }
    except Exception as e:
        logger.error(f"RAG 질의 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-risk", response_model=RiskDetectionResponse)
async def detect_clause_risk(request: RiskDetectionRequest):
    """
    독소조항 위험 탐지 (테스트용)

    계약 조항의 위험도를 분석하고 법률 근거 제시
    """
    if not all([qdrant_client, embeddings, llm]):
        raise HTTPException(status_code=503, detail="RAG 시스템 초기화 실패")

    try:
        result = detect_risk(
            user_clause=request.clause_text,
            client=qdrant_client,
            embeddings=embeddings,
            llm=llm
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
