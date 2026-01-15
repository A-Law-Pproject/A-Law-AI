"""
API 라우터 등록
"""
from fastapi import APIRouter
from app.api.endpoints import contract, rag

# 메인 API 라우터
api_router = APIRouter()

# 계약서 분석 엔드포인트 등록
api_router.include_router(
    contract.router,
    prefix="/contracts",
    tags=["contracts"]
)

# RAG 관리 엔드포인트 등록
api_router.include_router(
    rag.router,
    prefix="/rag",
    tags=["rag"]
)
