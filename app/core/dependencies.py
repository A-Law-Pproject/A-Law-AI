"""
RAG 공유 싱글톤 의존성
- QdrantClient, KUREEmbeddings, ChatOpenAI 인스턴스를 앱 전체에서 재사용
- rag.py / rabbitmq_consumer.py 등에서 중복 초기화하지 않도록 통합
"""
from loguru import logger
from langchain_openai import ChatOpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import QdrantClient

from app.core.config import settings
from app.rag.embedding.kure import KUREEmbeddings

_qdrant_client: QdrantClient | None = None
_embeddings: KUREEmbeddings | None = None
_llm: ChatOpenAI | None = None
_mongo_client: AsyncIOMotorClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        logger.info("QdrantClient 싱글톤 초기화 완료")
    return _qdrant_client


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
        )
        logger.info(f"ChatOpenAI 싱글톤 초기화 완료 (model={settings.MODEL_NAME})")
    return _llm


def get_mongo_client() -> AsyncIOMotorClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = AsyncIOMotorClient(settings.MONGODB_URI)
        logger.info("MongoDB 싱글톤 초기화 완료")
    return _mongo_client


async def fetch_ocr_text(contract_id: int) -> str:
    """
    MongoDB ocr_results 컬렉션에서 contract_id로 full_text 조회.
    Spring Boot OCR 처리 후 저장한 텍스트를 가져옴.
    """
    client = get_mongo_client()
    collection = client[settings.MONGODB_DB][settings.MONGODB_OCR_COLLECTION]
    doc = await collection.find_one({"contractId": contract_id}, {"full_text": 1})
    if doc is None:
        raise ValueError(f"OCR 결과 없음: contract_id={contract_id}")
    text = doc.get("full_text", "")
    if not text:
        raise ValueError(f"full_text가 비어 있음: contract_id={contract_id}")
    return text
