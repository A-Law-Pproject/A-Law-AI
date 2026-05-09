"""
PII 마스킹 서비스 오케스트레이터

텍스트 마스킹 → 이미지 마스킹 → S3 저장 → MongoDB 업데이트를 순서대로 처리한다.

설계 원칙:
- 마스킹 실패 시 서비스를 중단하지 않는다 (maskingFailed=True 기록 후 원본 유지)
- 로그에 PII 원본값을 절대 포함하지 않는다
- 모든 IO 작업은 async/await로 처리한다
"""
import asyncio
import os
from datetime import datetime, timezone
from typing import List, Optional

from loguru import logger

from app.core.config import settings
from app.core.dependencies import get_mongo_client
from app.schemas.masking import MaskingMetadata, MaskingStoreResult
from app.services.masking.image_masker import mask_image_with_words
from app.services.masking.text_masker import TextMasker
from app.util.s3_client import S3Client


def _build_masked_s3_key(original_s3_key: str) -> str:
    """
    원본 S3 키에서 마스킹본 S3 키를 생성한다.

    규칙: {prefix}/masked/{filename}
    예: contracts/1/10/original/image.jpg → contracts/1/10/masked/image.jpg
         contracts/1/10/image.jpg         → contracts/1/10/masked/image.jpg

    Args:
        original_s3_key: 원본 이미지 S3 키

    Returns:
        마스킹본 S3 키
    """
    parts = original_s3_key.rsplit("/", 1)
    if len(parts) == 2:
        prefix, filename = parts
        # "original" 세그먼트가 있으면 "masked"로 교체, 없으면 같은 레벨에 masked 디렉토리 삽입
        if prefix.endswith("/original"):
            new_prefix = prefix[: -len("/original")] + "/masked"
        else:
            new_prefix = prefix + "/masked"
        return f"{new_prefix}/{filename}"
    else:
        # 경로 구분자가 없는 경우 (단순 파일명)
        return f"masked/{original_s3_key}"


async def _upload_masked_image_async(
    s3_client: S3Client,
    masked_image_bytes: bytes,
    masked_s3_key: str,
) -> None:
    """
    마스킹된 이미지를 S3에 비동기로 업로드한다.
    boto3 동기 호출을 asyncio.to_thread로 감싼다.
    """
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
    """
    MongoDB OCR 컬렉션 도큐먼트에 마스킹 메타데이터와 마스킹된 텍스트를 추가한다.
    기존 필드는 건드리지 않는다 ($set으로 신규 필드만 추가).

    save_ocr_result가 선행 실패한 경우를 대비해 upsert=True로 안전망을 둔다.
    upsert 시 도큐먼트 식별을 위해 s3Key/s3_key 필드도 함께 기록한다.
    """
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
            f"MongoDB 마스킹 메타데이터 갱신 완료(기존 OCR 도큐먼트): "
            f"matched={result.matched_count}, modified={result.modified_count}"
        )
    elif result.upserted_id is not None:
        # save_ocr_result가 실패한 경우 — 마스킹 결과만이라도 보존
        logger.warning(
            f"기존 OCR 도큐먼트가 없어 마스킹 결과를 신규 도큐먼트로 upsert함: "
            f"upserted_id={result.upserted_id}. save_ocr_result 단계 점검 필요."
        )
    else:
        logger.error("MongoDB 마스킹 갱신: 매칭 0건 + upsert 실패 (예상치 못한 상태)")


async def mask_and_store(
    original_text: str,
    original_image_bytes: bytes,
    s3_key: str,
    ocr_words: Optional[List[dict]] = None,
    img_width: int = 0,
    img_height: int = 0,
) -> MaskingStoreResult:
    """
    계약서 OCR 결과에 대해 텍스트 + 이미지 마스킹을 수행하고
    마스킹본을 S3/MongoDB에 저장한다.

    Args:
        original_text: OCR로 추출된 원본 텍스트
        original_image_bytes: S3에서 가져온 원본 이미지 바이트
        s3_key: 원본 이미지 S3 키
        ocr_words: OCRWord dict 리스트 (인감/서명 영역 탐지용, None이면 이미지 마스킹 스킵)
        img_width: 이미지 너비 (px)
        img_height: 이미지 높이 (px)

    Returns:
        MaskingStoreResult
    """
    # ENABLE_MASKING 토글 확인
    if not settings.ENABLE_MASKING:
        logger.debug("ENABLE_MASKING=False — 마스킹 스킵")
        return MaskingStoreResult(
            success=True,
            masked_text=original_text,
            metadata=MaskingMetadata(masking_failed=False),
        )

    masked_s3_key: Optional[str] = None

    try:
        # ─────────────────────────────────────────
        # 1단계: 텍스트 마스킹
        # ─────────────────────────────────────────
        masker = TextMasker()
        text_result = masker.mask_all(original_text)
        logger.info(
            f"텍스트 마스킹: {text_result.mask_count}건, "
            f"유형={text_result.mask_types_found}"
        )

        # ─────────────────────────────────────────
        # 2단계: 이미지 마스킹 (인감/서명 bbox 블랙박스)
        # ─────────────────────────────────────────
        image_mask_count = 0
        masked_image_bytes = original_image_bytes

        if original_image_bytes:
            try:
                masked_image_bytes, image_mask_count = await asyncio.to_thread(
                    mask_image_with_words,
                    original_image_bytes,
                    ocr_words,
                    img_width,
                    img_height,
                )
                logger.info(f"이미지 마스킹: {image_mask_count}개 영역 처리")
            except Exception as img_err:
                # 이미지 마스킹 실패는 텍스트 마스킹 결과를 버리지 않는다
                logger.warning(f"이미지 마스킹 실패 (원본 유지): {img_err}")
                masked_image_bytes = original_image_bytes

        # ─────────────────────────────────────────
        # 3단계: 마스킹본 S3 업로드
        # ─────────────────────────────────────────
        masked_s3_key = _build_masked_s3_key(s3_key)
        try:
            s3_client = S3Client()
            await _upload_masked_image_async(s3_client, masked_image_bytes, masked_s3_key)
            logger.info(f"마스킹본 S3 업로드 완료: key_hint=[MASKED]")
        except Exception as s3_err:
            logger.warning(f"마스킹본 S3 업로드 실패 (MongoDB에는 텍스트만 저장): {s3_err}")
            masked_s3_key = None  # S3 실패 시 키 기록 안 함

        # ─────────────────────────────────────────
        # 4단계: MongoDB 마스킹 메타데이터 업데이트
        # ─────────────────────────────────────────
        total_mask_count = text_result.mask_count + image_mask_count
        mask_types = list(text_result.mask_types_found)
        if image_mask_count > 0:
            mask_types.append("seal_signature")

        metadata = MaskingMetadata(
            masked_at=datetime.now(timezone.utc),
            mask_count=total_mask_count,
            mask_types=mask_types,
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
        except Exception as mongo_err:
            logger.warning(f"MongoDB 마스킹 메타데이터 업데이트 실패: {mongo_err}")
            # MongoDB 실패는 치명적이지 않음 — 계속 진행

        return MaskingStoreResult(
            success=True,
            masked_text=text_result.masked_text,
            masked_s3_key=masked_s3_key,
            metadata=metadata,
        )

    except Exception as e:
        # 예상치 못한 전체 마스킹 실패 — 서비스는 중단하지 않는다
        logger.error(f"마스킹 처리 전체 실패: {type(e).__name__} — 원본 유지")

        # 실패 플래그를 MongoDB에 기록
        try:
            failure_metadata = MaskingMetadata(
                masked_at=datetime.now(timezone.utc),
                mask_count=0,
                mask_types=[],
                masked_s3_key=None,
                masking_version=settings.MASKING_VERSION,
                masking_failed=True,
            )
            await _update_masking_metadata_in_mongo(
                s3_key=s3_key,
                masked_text=original_text,  # 실패 시 원본 텍스트 유지
                metadata=failure_metadata,
            )
        except Exception as record_err:
            logger.warning(f"실패 플래그 MongoDB 기록도 실패: {record_err}")

        return MaskingStoreResult(
            success=False,
            masked_text=original_text,
            masked_s3_key=None,
            metadata=MaskingMetadata(masking_failed=True),
            error_message=f"{type(e).__name__}: {e}",
        )
