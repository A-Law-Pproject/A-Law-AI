"""
Shared dependency helpers for vector DB, embeddings, LLM, and MongoDB.
"""
from datetime import datetime, timezone

import certifi
from langchain_openai import ChatOpenAI
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as aioredis

from app.core.config import settings
from app.rag.embedding.kure import KUREEmbeddings
from app.rag.vector_store.base import VectorDB

_vector_db: VectorDB | None = None
_embeddings: KUREEmbeddings | None = None
_llm: ChatOpenAI | None = None
_mongo_client: AsyncIOMotorClient | None = None
_redis_client: aioredis.Redis | None = None
_law_mcp_bridge = None


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
        _embeddings = KUREEmbeddings(model_name=settings.EMBEDDING_MODEL)
        logger.info("KUREEmbeddings singleton initialized")
    return _embeddings


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
            timeout=300,
        )
        logger.info(f"ChatOpenAI singleton initialized (model={settings.MODEL_NAME})")
    return _llm


def get_mongo_client() -> AsyncIOMotorClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = AsyncIOMotorClient(
            settings.MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000,
            tlsCAFile=certifi.where(),
        )
        logger.info("MongoDB singleton initialized")
    return _mongo_client


async def get_redis_client() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )
        logger.info("Redis singleton initialized")
    return _redis_client


def get_law_mcp_bridge():
    global _law_mcp_bridge
    if _law_mcp_bridge is None:
        from app.mcp.bridge import LawMCPBridge

        _law_mcp_bridge = LawMCPBridge()
        logger.info("Law MCP bridge singleton initialized")
    return _law_mcp_bridge


def _extract_ocr_text(doc: dict | None, *, lookup: str) -> str:
    if doc is None:
        raise ValueError(f"OCR result not found: {lookup}")

    text = (
        doc.get("rawText")
        or doc.get("raw_text")
        or doc.get("fullText")
        or doc.get("full_text", "")
    )
    if not text:
        raise ValueError(f"OCR text is empty: {lookup}")
    return text


async def fetch_ocr_text(s3_key: str) -> str:
    """Fetch OCR text by `s3_key` from MongoDB."""
    client = get_mongo_client()
    collection = client[settings.MONGODB_DB][settings.MONGODB_OCR_COLLECTION]
    doc = await collection.find_one(
        {"$or": [{"s3Key": s3_key}, {"s3_key": s3_key}]},
        {"rawText": 1, "raw_text": 1, "fullText": 1, "full_text": 1},
    )
    return _extract_ocr_text(doc, lookup=f"s3_key={s3_key}")


async def fetch_contract_text(contract_id: int | str, s3_key: str | None = None) -> str:
    """
    Fetch OCR text by `contract_id` first and `s3_key` second.

    This supports the voice fact-check pipeline without embedding the full
    contract text in the RabbitMQ payload.
    """
    client = get_mongo_client()
    collection = client[settings.MONGODB_DB][settings.MONGODB_OCR_COLLECTION]

    contract_id_str = str(contract_id)
    query_candidates = [
        {"contractId": contract_id_str},
        {"contract_id": contract_id_str},
    ]

    try:
        contract_id_int = int(contract_id_str)
    except (TypeError, ValueError):
        contract_id_int = None

    if contract_id_int is not None:
        query_candidates.extend(
            [
                {"contractId": contract_id_int},
                {"contract_id": contract_id_int},
            ]
        )

    doc = await collection.find_one(
        {"$or": query_candidates},
        {"rawText": 1, "raw_text": 1, "fullText": 1, "full_text": 1},
        sort=[("updatedAt", -1)],
    )
    if doc is not None:
        try:
            return _extract_ocr_text(doc, lookup=f"contract_id={contract_id_str}")
        except ValueError:
            pass

    if s3_key:
        return await fetch_ocr_text(s3_key)

    raise ValueError(
        f"OCR result not found: contract_id={contract_id_str}, s3_key={s3_key}"
    )


async def save_ocr_result(s3_key: str, result) -> None:
    """Upsert OCR results in MongoDB keyed by `s3_key`."""
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
