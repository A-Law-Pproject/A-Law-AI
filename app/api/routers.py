"""
API 라우터 등록
"""
from fastapi import APIRouter
from app.api.endpoints import contract, ocr, rag, contract_analysis

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
# RAG 관리
api_router.include_router(
    rag.router,
    prefix="/rag",
    tags=["RAG 관리"],
)

# 비동기 분석 (Spring Boot 연동)
api_router.include_router(
    contract_analysis.router,
    prefix="/analysis",
    tags=["비동기 분석"],
)
