"""
API router registration.
"""
from fastapi import APIRouter

from app.api.endpoints import chat, contract, ocr, rag, voice_standalone

api_router = APIRouter()

api_router.include_router(
    contract.router,
    prefix="/contracts",
    tags=["怨꾩빟??遺꾩꽍"],
)

api_router.include_router(
    ocr.router,
    prefix="/contracts",
    tags=["怨꾩빟??OCR"],
)

api_router.include_router(
    rag.router,
    prefix="/contracts",
    tags=["怨꾩빟??遺꾩꽍"],
)

api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["梨쀫큸"],
)

api_router.include_router(
    voice_standalone.router,
    prefix="/voice",
    tags=["?뚯꽦 遺꾩꽍"],
)

api_router.include_router(
    voice_standalone.legacy_router,
    prefix="/voice",
    tags=["?뚯꽦 遺꾩꽍"],
)
