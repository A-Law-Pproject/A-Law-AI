"""
API router registration.
"""
from fastapi import APIRouter

from app.api.endpoints import chat, contract, ocr, rag, voice_standalone

api_router = APIRouter()

api_router.include_router(
    contract.router,
    prefix="/contracts",
    tags=["계약서 분석"],
)

api_router.include_router(
    ocr.router,
    prefix="/contracts",
    tags=["계약서 OCR"],
)

api_router.include_router(
    rag.router,
    prefix="/contracts",
    tags=["계약서 분석"],
)

api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["챗봇"],
)

api_router.include_router(
    voice_standalone.router,
    prefix="/voice",
    tags=["음성 분석"],
)

api_router.include_router(
    voice_standalone.legacy_router,
    prefix="/voice",
    tags=["음성 분석"],
)
