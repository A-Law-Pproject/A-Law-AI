from dotenv import load_dotenv
load_dotenv()

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from prometheus_fastapi_instrumentator import Instrumentator
from app.api.routers import api_router
from app.services.rabbitmq_consumer import start_consumer, stop_consumer, consumer
from loguru import logger


class _MetricsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "GET /metrics" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(_MetricsFilter())


executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan - RabbitMQ Consumer 시작/종료"""
    logger.info("Starting RabbitMQ Consumer...")
    await start_consumer()
    logger.info("RabbitMQ Consumer started successfully")

    yield

    logger.info("Stopping RabbitMQ Consumer...")
    await stop_consumer()
    logger.info("RabbitMQ Consumer stopped")


app = FastAPI(
    title="A-LAW Contract Analysis AI",
    description="임대차 계약서 AI 분석 및 위험도 평가 시스템",
    version="1.0.0",
    lifespan=lifespan,
)

# Prometheus 메트릭 엔드포인트 (/metrics)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["https://www.a-law.site", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(api_router, prefix="/ai")


@app.get("/health", tags=["시스템"])
async def health_check():
    """서버 헬스체크"""
    if not consumer.is_healthy():
        raise HTTPException(status_code=503, detail="RabbitMQ is not connected")

    return {
        "status": "healthy",
        "service": "A-LAW FastAPI",
        "version": "1.0.0",
        "rabbitmq": "connected"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
