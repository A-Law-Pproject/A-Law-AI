from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from app.api.routers import api_router
from app.services.rabbitmq_consumer import start_consumer, stop_consumer
from loguru import logger


executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan - RabbitMQ Consumer 시작/종료"""
    logger.info("Starting RabbitMQ Consumer...")
    try:
        await start_consumer()
        logger.info("RabbitMQ Consumer started successfully")
    except Exception as e:
        logger.warning(f"RabbitMQ Consumer failed to start: {e}")
        logger.warning("The API will still work, but message queue processing is disabled")

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

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(api_router, prefix="/ai")


@app.get("/health", tags=["시스템"])
async def health_check():
    """서버 헬스체크"""
    return {
        "status": "healthy",
        "service": "A-LAW FastAPI",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
