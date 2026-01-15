from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.api.routers import api_router


# 스레드 풀 (LangChain 동기 호출 처리)
executor = ThreadPoolExecutor(max_workers=4)


app = FastAPI(
    title="A-LAW Contract Analysis AI",
    description="법률 문서 분석 및 위험도 평가 시스템",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(api_router, prefix="/api")


# 헬스체크 엔드포인트
@app.get("/health", tags=["Health"])
async def health_check():
    """서버 헬스체크"""
    return {
        "status": "healthy",
        "service": "A-LAW FastAPI",
        "version": "1.0.0"
    }


@app.get("/", tags=["Root"])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "A-LAW Contract Analysis API",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)