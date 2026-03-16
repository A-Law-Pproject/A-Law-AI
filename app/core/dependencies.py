"""
RAG 공유 싱글톤 의존성
- VectorDB, KUREEmbeddings, ChatOpenAI 인스턴스를 앱 전체에서 재사용
- VECTOR_DB=qdrant → QdrantAdapter (개발)
- VECTOR_DB=pinecone → PineconeAdapter (배포)
"""
from loguru import logger
from langchain_openai import ChatOpenAI
from motor.motor_asyncio import AsyncIOMotorClient

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
        if settings.VECTOR_DB == "pinecone":
            from app.rag.vector_store.pinecone_adapter import PineconeAdapter
            _vector_db = PineconeAdapter(
                api_key=settings.PINECONE_API_KEY,
                index_name=settings.PINECONE_INDEX,
            )
            logger.info(f"VectorDB: Pinecone (index={settings.PINECONE_INDEX})")
        else:
            from app.rag.vector_store.qdrant_adapter import QdrantAdapter
            _vector_db = QdrantAdapter(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
            logger.info(f"VectorDB: Qdrant (url={settings.QDRANT_URL})")
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
        )
        logger.info(f"ChatOpenAI 싱글톤 초기화 완료 (model={settings.MODEL_NAME})")
    return _llm


def get_mongo_client() -> AsyncIOMotorClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = AsyncIOMotorClient(settings.MONGODB_URI)
        logger.info("MongoDB 싱글톤 초기화 완료")
    return _mongo_client


async def fetch_ocr_text(s3_key: str) -> str:
    """
    MongoDB ocr_results 컬렉션에서 s3_key로 full_text 조회.
    Spring Boot OCR 처리 후 저장한 텍스트를 가져옴.
    """
    client = get_mongo_client()
    collection = client[settings.MONGODB_DB][settings.MONGODB_OCR_COLLECTION]
    # Spring Boot는 camelCase(s3Key, fullText)로 저장 가능 → 두 필드명 모두 시도
    doc = await collection.find_one(
        {"$or": [{"s3Key": s3_key}, {"s3_key": s3_key}]},
        {"fullText": 1, "full_text": 1}
    )
    if doc is None:
        raise ValueError(f"OCR 결과 없음: s3_key={s3_key}")
    text = doc.get("fullText") or doc.get("full_text", "")
    if not text:
        raise ValueError(f"fullText가 비어 있음: s3_key={s3_key}")
    return text
