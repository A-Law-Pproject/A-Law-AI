"""
RAG 공유 싱글톤 의존성
- VectorDB, KUREEmbeddings, ChatOpenAI 인스턴스를 앱 전체에서 재사용
- VECTOR_DB=pinecone → PineconeAdapter
"""
import certifi
from loguru import logger
from langchain_openai import ChatOpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone

from app.core.config import settings
from app.rag.embedding.kure import KUREEmbeddings
from app.rag.vector_store.base import VectorDB

_vector_db: VectorDB | None = None
_embeddings: KUREEmbeddings | None = None
_llm: ChatOpenAI | None = None
_mongo_client: AsyncIOMotorClient | None = None


def get_vector_db() -> VectorDB:
    global _vector_db
    if _vector_db is None:
        from app.rag.vector_store.pinecone_adapter import PineconeAdapter
        _vector_db = PineconeAdapter(
            api_key=settings.PINECONE_API_KEY,
            index_name=settings.PINECONE_INDEX,
        )
        logger.info(f"VectorDB: Pinecone (index={settings.PINECONE_INDEX})")
    return _vector_db


def get_embeddings() -> KUREEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = KUREEmbeddings()
        logger.info("KUREEmbeddings 싱글톤 초기화 완료")
    return _embeddings


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
            timeout=300,  # OpenAI API 호출 단위 타임아웃 (초)
        )
        logger.info(f"ChatOpenAI 싱글톤 초기화 완료 (model={settings.MODEL_NAME})")
    return _llm


def get_mongo_client() -> AsyncIOMotorClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = AsyncIOMotorClient(
            settings.MONGODB_URI,
            serverSelectionTimeoutMS=5000,   # 서버 선택 타임아웃 5초
            connectTimeoutMS=5000,           # 연결 타임아웃 5초
            socketTimeoutMS=5000,            # 소켓 타임아웃 5초
            tlsCAFile=certifi.where(),       # CA 인증서 명시 (Atlas TLS 검증)
        )
        logger.info("MongoDB 싱글톤 초기화 완료")
    return _mongo_client


async def fetch_ocr_text(s3_key: str) -> str:
    """
    MongoDB ocr_results 컬렉션에서 s3_key로 OCR 텍스트 조회.

    우선순위:
    1. rawText / raw_text
    2. fullText / full_text
    """
    client = get_mongo_client()
    collection = client[settings.MONGODB_DB][settings.MONGODB_OCR_COLLECTION]
    # Spring Boot는 camelCase / snake_case 혼용 가능 → 모두 시도
    doc = await collection.find_one(
        {"$or": [{"s3Key": s3_key}, {"s3_key": s3_key}]},
        {"rawText": 1, "raw_text": 1, "fullText": 1, "full_text": 1}
    )
    if doc is None:
        raise ValueError(f"OCR 결과 없음: s3_key={s3_key}")

    text = (
        doc.get("rawText")
        or doc.get("raw_text")
        or doc.get("fullText")
        or doc.get("full_text", "")
    )
    if not text:
        raise ValueError(f"OCR 텍스트가 비어 있음: s3_key={s3_key}")
    return text


async def save_ocr_result(s3_key: str, result) -> None:
    """S3 OCR 결과를 MongoDB ocr_results 컬렉션에 upsert.

    Spring Boot가 OCR 결과를 저장하는 흐름과 병행 가능하도록 동일 s3_key를
    기준으로 덮어쓴다. 분석 consumer는 fetch_ocr_text()로 이 컬렉션을 조회한다.
    """
    client = get_mongo_client()
    collection = client[settings.MONGODB_DB][settings.MONGODB_OCR_COLLECTION]

    data = result.model_dump(mode="json") if hasattr(result, "model_dump") else dict(result)
    text = data.get("full_text") or data.get("markdown") or ""

    await collection.update_one(
        {"$or": [{"s3Key": s3_key}, {"s3_key": s3_key}]},
        {
            "$set": {
                "s3Key": s3_key,
                "s3_key": s3_key,
                "rawText": text,
                "raw_text": text,
                "fullText": data.get("full_text", ""),
                "full_text": data.get("full_text", ""),
                "markdown": data.get("markdown", ""),
                "words": data.get("words") or [],
                "contractData": data.get("contract_data"),
                "validation": data.get("validation"),
                "ocrSuccess": data.get("success", True),
                "processingTime": data.get("processing_time", 0),
                "updatedAt": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )
