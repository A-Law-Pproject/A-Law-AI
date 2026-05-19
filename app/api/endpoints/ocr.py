"""
OCR endpoints with pre-OCR masking support.
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import settings
from app.schemas.ocr_response import ContractOCRResponse
from app.services.masking.image_masker import ImageMaskingResult, mask_image_for_ocr
from app.services.masking.masking_service import mask_and_store
from app.services.masking.text_masker import TextMasker
from app.services.ocr.ocr_service import OCRService
from app.util.s3_client import S3Client


router = APIRouter()


class OCRRequest(BaseModel):
    s3_key: str = Field(..., description="S3 object key")
    image_url: str | None = Field(None, description="S3 이미지 URL")


def get_s3_client() -> S3Client:
    return S3Client()


def get_ocr_service() -> OCRService:
    return OCRService()


async def save_ocr_result(
    s3_key: str, result: ContractOCRResponse, *, image_url: str | None = None
) -> None:
    from app.core.dependencies import save_ocr_result as _save_ocr_result

    await _save_ocr_result(s3_key, result, image_url=image_url)


def _mask_nested_strings(value: Any, masker: TextMasker) -> Any:
    if isinstance(value, str):
        return masker.mask_all(value).masked_text
    if isinstance(value, dict):
        return {key: _mask_nested_strings(item, masker) for key, item in value.items()}
    if isinstance(value, list):
        return [_mask_nested_strings(item, masker) for item in value]
    return value


def _apply_text_masking_to_result(result: ContractOCRResponse) -> ContractOCRResponse:
    if not settings.ENABLE_MASKING:
        return result

    masker = TextMasker()
    if result.full_text:
        result.full_text = masker.mask_all(result.full_text).masked_text
    if result.markdown:
        result.markdown = masker.mask_all(result.markdown).masked_text
    if result.contract_data:
        result.contract_data = _mask_nested_strings(result.contract_data, masker)
    return result


async def _prepare_image_for_ocr(image_bytes: bytes) -> ImageMaskingResult:
    if not settings.ENABLE_MASKING:
        return ImageMaskingResult(image_bytes=image_bytes)

    masking_result = await asyncio.to_thread(mask_image_for_ocr, image_bytes)
    if masking_result.masking_failed:
        logger.error(f"Blocking OCR because pre-OCR masking failed: {masking_result.error_message}")
        raise HTTPException(500, "사전 마스킹에 실패하여 OCR을 진행할 수 없습니다.")
    return masking_result


@router.post("/ocr", response_model=ContractOCRResponse, summary="[완료] S3 이미지 OCR")
async def run_ocr_from_s3(
    request: OCRRequest,
    include_overlay: bool = Query(True, description="Include OCR word boxes in response"),
    s3_client: S3Client = Depends(get_s3_client),
    service: OCRService = Depends(get_ocr_service),
):
    try:
        original_image_bytes = await asyncio.to_thread(s3_client.get_image, request.s3_key)
        pre_mask_result = await _prepare_image_for_ocr(original_image_bytes)

        ocr_include_overlay = include_overlay or settings.ENABLE_MASKING
        result = await asyncio.to_thread(
            service.process_and_map,
            image_bytes=pre_mask_result.image_bytes,
            structurize=False,
            include_overlay=ocr_include_overlay,
        )
        result = _apply_text_masking_to_result(result)

        # 마스킹 완료 후 최종 image_url 결정 (마스킹 URL이 있으면 그걸 사용)
        effective_image_url = request.image_url
        if settings.ENABLE_MASKING:
            try:
                words_dicts = [word.model_dump() for word in result.words] if result.words else None
                masking_result = await mask_and_store(
                    original_text=result.full_text or result.markdown or "",
                    original_image_bytes=original_image_bytes,
                    s3_key=request.s3_key,
                    ocr_words=words_dicts,
                    img_width=result.image_width,
                    img_height=result.image_height,
                    image_bytes_for_storage=pre_mask_result.image_bytes,
                    pre_mask_count=pre_mask_result.mask_count,
                    pre_mask_types=pre_mask_result.mask_types,
                )
                if masking_result.masked_s3_key:
                    result.masked_image_url = (
                        f"https://{settings.AWS_S3_BUCKET}.s3.amazonaws.com"
                        f"/{masking_result.masked_s3_key}"
                    )
                    effective_image_url = result.masked_image_url
                    # 마스킹된 버전으로 교체됐으므로 원본 삭제
                    await asyncio.to_thread(s3_client.delete_file, request.s3_key)
            except Exception as exc:
                logger.warning(f"Failed to persist masking artifacts after OCR: {type(exc).__name__}: {exc}")

        # MongoDB 저장: 마스킹 후 최종 URL로 저장해야 Spring findByImageUrl 조회가 일치함
        try:
            await save_ocr_result(request.s3_key, result, image_url=effective_image_url)
        except Exception as exc:
            logger.warning(f"Failed to save OCR result to MongoDB: {exc}")

        if not include_overlay:
            result.words = None

        return result

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(404, f"파일을 찾을 수 없습니다: {request.s3_key}")
    except Exception as exc:
        logger.error(f"OCR processing failed: {exc}")
        raise HTTPException(500, f"OCR 처리 실패: {exc}")


@router.post("/ocr/full", response_model=ContractOCRResponse, summary="이미지 직접 업로드 OCR")
async def run_ocr_full(
    file: UploadFile = File(...),
    structurize: bool = Query(True, description="Run contract structuring"),
    include_overlay: bool = Query(True, description="Include OCR word boxes in response"),
    service: OCRService = Depends(get_ocr_service),
):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(400, "이미지 파일만 가능합니다.")

        original_image_bytes = await file.read()
        pre_mask_result = await _prepare_image_for_ocr(original_image_bytes)

        ocr_include_overlay = include_overlay or settings.ENABLE_MASKING
        result = await asyncio.to_thread(
            service.process_and_map,
            image_bytes=pre_mask_result.image_bytes,
            structurize=structurize,
            include_overlay=ocr_include_overlay,
        )
        result = _apply_text_masking_to_result(result)

        if not include_overlay:
            result.words = None

        return result

    except HTTPException:
        raise
    except ValueError as exc:
        logger.error(f"Invalid image input: {exc}")
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.error(f"OCR processing failed: {exc}")
        raise HTTPException(500, f"OCR 처리 실패: {exc}")
