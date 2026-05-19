"""
Audio hashing and storage helpers.
"""
import asyncio
import hashlib
from datetime import datetime, timezone

import boto3
from loguru import logger

from app.core.config import settings
from app.core.dependencies import get_mongo_client
from app.schemas.voice_shared import VoiceAudioMeta


def compute_sha256(file_bytes: bytes) -> str:
    digest = hashlib.sha256(file_bytes).hexdigest()
    return f"sha256:{digest}"


def build_s3_key(source_id: str, timestamp: str, file_hash: str, ext: str) -> str:
    hash_short = file_hash.replace("sha256:", "")[:8]
    safe_ts = timestamp.replace(":", "").replace(".", "").replace("+", "").replace("-", "")[:15]
    return f"audio/analysis/{source_id}/{safe_ts}_{hash_short}.{ext}"


def _build_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
    )


def _upload_to_s3_sync(
    file_bytes: bytes,
    s3_key: str,
    content_type: str,
) -> None:
    s3_client = _build_s3_client()
    s3_client.put_object(
        Bucket=settings.AWS_S3_BUCKET,
        Key=s3_key,
        Body=file_bytes,
        ContentType=content_type,
    )


def _download_from_s3_sync(s3_key: str) -> bytes:
    s3_client = _build_s3_client()
    response = s3_client.get_object(
        Bucket=settings.AWS_S3_BUCKET,
        Key=s3_key,
    )
    return response["Body"].read()


async def upload_to_s3(
    file_bytes: bytes,
    s3_key: str,
    content_type: str = "audio/mpeg",
) -> bool:
    if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
        logger.warning(f"AWS credentials missing, skip S3 upload: {s3_key}")
        return False

    try:
        await asyncio.to_thread(_upload_to_s3_sync, file_bytes, s3_key, content_type)
        logger.info(f"S3 upload completed: s3://{settings.AWS_S3_BUCKET}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"S3 upload failed [{s3_key}]: {e}")
        return False


async def download_from_s3(s3_key: str) -> bytes:
    if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
        raise RuntimeError("AWS credentials are not configured for S3 download")

    try:
        file_bytes = await asyncio.to_thread(_download_from_s3_sync, s3_key)
        logger.info(
            f"S3 download completed: s3://{settings.AWS_S3_BUCKET}/{s3_key} ({len(file_bytes)} bytes)"
        )
        return file_bytes
    except Exception as e:
        logger.error(f"S3 download failed [{s3_key}]: {e}")
        raise RuntimeError(f"S3 download failed: {e}") from e


async def save_audio_meta(meta: VoiceAudioMeta) -> str:
    try:
        client = get_mongo_client()
        collection = client[settings.MONGODB_DB][settings.MONGODB_VOICE_EVIDENCE_COLLECTION]
        doc = meta.model_dump(mode="json")
        doc["savedAt"] = datetime.now(timezone.utc).isoformat()

        result = await collection.insert_one(doc)
        inserted_id = str(result.inserted_id)
        logger.info(
            f"Audio metadata saved: source_id={meta.source_id}, hash={meta.file_hash[:20]}..., _id={inserted_id}"
        )
        return inserted_id
    except Exception as e:
        logger.error(f"MongoDB audio metadata save failed: {e}")
        return ""


async def save_voice_analysis_result(doc: dict) -> str:
    """Standalone 분석 결과를 voice_analysis_results 컬렉션에 저장한다."""
    try:
        client = get_mongo_client()
        collection = client[settings.MONGODB_DB][settings.MONGODB_VOICE_ANALYSIS_COLLECTION]
        doc["savedAt"] = datetime.now(timezone.utc).isoformat()
        result = await collection.insert_one(doc)
        inserted_id = str(result.inserted_id)
        logger.info(f"Voice analysis result saved: _id={inserted_id}")
        return inserted_id
    except Exception as e:
        logger.error(f"MongoDB voice analysis result save failed: {e}")
        return ""


async def save_voice_fact_check_result(doc: dict) -> str:
    """팩트체크 결과를 voice_fact_check_results 컬렉션에 저장한다."""
    try:
        client = get_mongo_client()
        collection = client[settings.MONGODB_DB][settings.MONGODB_VOICE_FACT_CHECK_COLLECTION]
        doc["savedAt"] = datetime.now(timezone.utc).isoformat()
        result = await collection.insert_one(doc)
        inserted_id = str(result.inserted_id)
        logger.info(f"Voice fact-check result saved: _id={inserted_id}")
        return inserted_id
    except Exception as e:
        logger.error(f"MongoDB voice fact-check result save failed: {e}")
        return ""


async def process_and_store_audio(
    file_bytes: bytes,
    original_filename: str,
    source_id: str,
    content_type: str = "audio/mpeg",
) -> VoiceAudioMeta:
    file_hash = await asyncio.to_thread(compute_sha256, file_bytes)
    created_at = datetime.now(timezone.utc).isoformat()
    ext = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else "mp3"
    s3_key = build_s3_key(source_id, created_at, file_hash, ext)

    await upload_to_s3(file_bytes, s3_key, content_type)

    meta = VoiceAudioMeta(
        file_hash=file_hash,
        original_filename=original_filename,
        created_at=created_at,
        s3_key=s3_key,
        source_id=source_id,
        file_size_bytes=len(file_bytes),
        content_type=content_type,
    )
    await save_audio_meta(meta)
    return meta
