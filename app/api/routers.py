"""
API 라우터 등록
"""
from fastapi import APIRouter
from app.api.endpoints import contract, ocr, rag, contract_analysis, chat

api_router = APIRouter()

# 계약서 분석
api_router.include_router(
    contract.router,
    prefix="/contracts",
    tags=["계약서 분석"],
)

# ocr
api_router.include_router(
    ocr.router,
    prefix="/contracts",
    tags=["계약서 OCR"],
)
# 독소조항 탐지 (계약서 분석 통합)
api_router.include_router(
    rag.router,
    prefix="/contracts",
    tags=["계약서 분석"],
)

# 비동기 분석 (Spring Boot 연동)
api_router.include_router(
    contract_analysis.router,
    prefix="/analysis",
    tags=["비동기 분석"],
)

# RAG 챗봇
api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["챗봇"],
)
