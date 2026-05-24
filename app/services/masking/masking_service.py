"""
Masking orchestration for OCR outputs and masked image storage.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

from loguru import logger

from app.core.config import settings
from app.schemas.masking import MaskingMetadata, MaskingStoreResult
from app.services.masking.image_masker import mask_image_with_words
from app.services.masking.text_masker import TextMasker
from app.util.s3_client import S3Client


def _build_masked_s3_key(original_s3_key: str) -> str:
    parts = original_s3_key.rsplit("/", 1)
    if len(parts) == 2:
        prefix, filename = parts
        if prefix.endswith("/original"):
            new_prefix = prefix[: -len("/original")] + "/masked"
        else:
            new_prefix = prefix + "/masked"
        return f"{new_prefix}/{filename}"
    return f"masked/{original_s3_key}"


async def _upload_masked_image_async(
    s3_client: S3Client,
    masked_image_bytes: bytes,
    masked_s3_key: str,
) -> None:
    await asyncio.to_thread(
        s3_client.upload_file,
        masked_image_bytes,
        masked_s3_key,
        "image/jpeg",
    )


async def _update_masking_metadata_in_mongo(
    s3_key: str,
    masked_text: str,
    metadata: MaskingMetadata,
) -> None:
    from app.core.dependencies import get_mongo_client

    client = get_mongo_client()
    collection = client[settings.MONGODB_DB][settings.MONGODB_OCR_COLLECTION]

    update_doc = {
        "s3Key": s3_key,
        "s3_key": s3_key,
        "maskedText": masked_text,
        "maskingMetadata": metadata.model_dump(by_alias=True, mode="json"),
    }

    result = await collection.update_one(
        {"$or": [{"s3Key": s3_key}, {"s3_key": s3_key}]},
        {"$set": update_doc},
        upsert=True,
    )

    if result.matched_count > 0:
        logger.info(
            "Masking metadata updated in MongoDB: "
            f"matched={result.matched_count}, modified={result.modified_count}"
        )
    elif result.upserted_id is not None:
        logger.warning(
            "OCR document was missing, so masking metadata was upserted separately: "
            f"upserted_id={result.upserted_id}"
        )
    else:
        logger.error("Masking metadata update returned neither a match nor an upserted id")


async def mask_and_store(
    original_text: str,
    original_image_bytes: bytes,
    s3_key: str,
    ocr_words: Optional[List[dict]] = None,
    img_width: int = 0,
    img_height: int = 0,
    image_bytes_for_storage: Optional[bytes] = None,
    pre_mask_count: int = 0,
    pre_mask_types: Optional[List[str]] = None,
) -> MaskingStoreResult:
    if not settings.ENABLE_MASKING:
        return MaskingStoreResult(
            success=True,
            masked_text=original_text,
            metadata=MaskingMetadata(masking_failed=False),
        )

    working_image_bytes = image_bytes_for_storage or original_image_bytes
    masked_s3_key: Optional[str] = None

    try:
        masker = TextMasker()
        text_result = masker.mask_all(original_text)
        logger.info(
            "Text masking completed for storage: "
            f"count={text_result.mask_count}, types={text_result.mask_types_found}"
        )

        image_mask_count = 0
        masked_image_bytes = working_image_bytes

        if working_image_bytes and ocr_words and img_width > 0 and img_height > 0:
            try:
                masked_image_bytes, image_mask_count = await asyncio.to_thread(
                    mask_image_with_words,
                    working_image_bytes,
                    ocr_words,
                    img_width,
                    img_height,
                )
            except Exception as exc:
                logger.warning(f"Post-OCR image masking failed; keeping pre-masked image: {exc}")
                masked_image_bytes = working_image_bytes

        masked_s3_key = _build_masked_s3_key(s3_key)
        logger.info(f"[MASKING] S3 업로드 시도: {masked_s3_key} ({len(masked_image_bytes)} bytes)")
        try:
            await _upload_masked_image_async(S3Client(), masked_image_bytes, masked_s3_key)
            logger.info(f"[MASKING] S3 업로드 성공: {masked_s3_key}")
        except Exception as exc:
            logger.error(f"[MASKING] S3 업로드 실패: {type(exc).__name__}: {exc}")
            masked_s3_key = None

        mask_types = set(text_result.mask_types_found)
        mask_types.update(pre_mask_types or [])
        if image_mask_count > 0:
            mask_types.add("seal_signature")

        metadata = MaskingMetadata(
            masked_at=datetime.now(timezone.utc),
            mask_count=text_result.mask_count + pre_mask_count + image_mask_count,
            mask_types=sorted(mask_types),
            masked_s3_key=masked_s3_key,
            masking_version=settings.MASKING_VERSION,
            masking_failed=False,
        )

        try:
            await _update_masking_metadata_in_mongo(
                s3_key=s3_key,
                masked_text=text_result.masked_text,
                metadata=metadata,
            )
        except Exception as exc:
            logger.warning(f"Failed to update masking metadata in MongoDB: {exc}")

        return MaskingStoreResult(
            success=True,
            masked_text=text_result.masked_text,
            masked_s3_key=masked_s3_key,
            metadata=metadata,
        )

    except Exception as exc:
        logger.error(f"Masking pipeline failed: {type(exc).__name__}: {exc}")
        failure_metadata = MaskingMetadata(
            masked_at=datetime.now(timezone.utc),
            mask_count=0,
            mask_types=[],
            masked_s3_key=None,
            masking_version=settings.MASKING_VERSION,
            masking_failed=True,
        )
        try:
            await _update_masking_metadata_in_mongo(
                s3_key=s3_key,
                masked_text=original_text,
                metadata=failure_metadata,
            )
        except Exception as record_error:
            logger.warning(f"Failed to record masking failure metadata: {record_error}")

        return MaskingStoreResult(
            success=False,
            masked_text=original_text,
            masked_s3_key=None,
            metadata=failure_metadata,
            error_message=f"{type(exc).__name__}: {exc}",
        )
