"""
RAG 관리 API 엔드포인트
- app/rag/ 모듈 기반으로 재구현 예정
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

router = APIRouter()


# ============================================
# Request/Response 스키마
# ============================================

class IndexRequest(BaseModel):
    directory: Optional[str] = Field(None, description="특정 디렉토리만 인덱싱")
    document_type: Optional[str] = Field(None, description="특정 문서 타입만 인덱싱")
    limit: Optional[int] = Field(None, description="최대 문서 개수 (테스트용)", gt=0, le=10000)
    force_recreate: bool = Field(False, description="기존 컬렉션 삭제 후 재생성")


class IndexResponse(BaseModel):
    success: bool
    message: str
    documents_loaded: int
    documents_indexed: int
    collection_info: dict


class SearchRequest(BaseModel):
    query: str = Field(..., description="검색 쿼리")
    k: int = Field(4, description="반환할 문서 수", gt=0, le=20)
    document_type: Optional[str] = Field(None, description="문서 타입 필터")


class SearchResult(BaseModel):
    content: str
    metadata: dict
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int


class StatsResponse(BaseModel):
    collection_name: str
    total_documents: int
    total_vectors: int
    document_types: dict
    status: str


# ============================================
# API 엔드포인트
# ============================================

@router.post("/index", response_model=IndexResponse, summary="[미구현] 법률 문서 인덱싱")
async def index_documents(request: IndexRequest):
    """
    **[미구현]** app/rag/ 모듈 기반으로 재구현 예정.
    현재 노트북(rag.ipynb)에서 인덱싱 테스트 가능합니다.
    """
    raise HTTPException(status_code=501, detail="미구현: app/rag/ 모듈 연동 후 활성화 예정")


@router.post("/search", response_model=SearchResponse, summary="[미구현] 법률 문서 검색")
async def search_documents(request: SearchRequest):
    """
    **[미구현]** app/rag/retriever/multi_retriever.py 기반으로 재구현 예정.
    """
    raise HTTPException(status_code=501, detail="미구현: app/rag/ 모듈 연동 후 활성화 예정")


@router.get("/stats", response_model=StatsResponse, summary="[미구현] RAG 통계 조회")
async def get_stats():
    """**[미구현]** 인덱싱된 문서 통계를 반환합니다."""
    raise HTTPException(status_code=501, detail="미구현: app/rag/ 모듈 연동 후 활성화 예정")


@router.delete("/collection", summary="[미구현] 컬렉션 삭제")
async def delete_collection():
    """**[미구현]** 모든 인덱싱된 문서가 삭제됩니다."""
    raise HTTPException(status_code=501, detail="미구현: app/rag/ 모듈 연동 후 활성화 예정")
