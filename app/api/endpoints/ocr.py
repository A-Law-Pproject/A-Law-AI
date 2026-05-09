"""
OCR API 엔드포인트
Spring Boot에서 S3 키를 받아 OCR 처리 후 텍스트 + 오버레이 반환
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from loguru import logger
from pydantic import BaseModel, Field

from app.util.s3_client import S3Client
from app.services.ocr.ocr_service import OCRService
from app.schemas.ocr_response import ContractOCRResponse
from app.core.dependencies import save_ocr_result
from app.core.config import settings
from app.services.masking.masking_service import mask_and_store


router = APIRouter()


# ===========================================
# Request 스키마
# ===========================================

class OCRRequest(BaseModel):
    s3_key: str = Field(..., description="S3 객체 키")


# ===========================================
# 의존성
# ===========================================

def get_s3_client() -> S3Client:
    return S3Client()


def get_ocr_service() -> OCRService:
    return OCRService()


# ===========================================
# API 엔드포인트
# ===========================================

@router.post("/ocr", response_model=ContractOCRResponse, summary="[완료] S3 이미지 OCR")
async def run_ocr_from_s3(
    request: OCRRequest,
    include_overlay: bool = Query(True, description="오버레이(단어 좌표) 포함 여부"),
    s3_client: S3Client = Depends(get_s3_client),
    service: OCRService = Depends(get_ocr_service),
):
    """S3에서 이미지를 가져와 OCR 처리. words 좌표 포함."""
    try:
        image_bytes = s3_client.get_image(request.s3_key)

        result = service.process_and_map(
            image_bytes=image_bytes,
            structurize=False,
            include_overlay=include_overlay,
        )

        # OCR 결과를 MongoDB에 먼저 저장 (원본 텍스트 + words)
        try:
            await save_ocr_result(request.s3_key, result)
        except Exception as e:
            logger.warning(f"MongoDB OCR 결과 저장 실패(s3_key 로그 생략): {e}")

        # OCR 저장 완료 후 PII 마스킹 처리 (ENABLE_MASKING 설정에 따라 실행)
        # 마스킹 실패 시 서비스를 중단하지 않는다 — 원본 결과를 그대로 반환
        if settings.ENABLE_MASKING:
            try:
                # words 좌표를 dict 형태로 변환 (이미지 인감/서명 영역 탐지용)
                words_dicts = None
                if result.words:
                    words_dicts = [w.model_dump() for w in result.words]

                await mask_and_store(
                    original_text=result.full_text or result.markdown or "",
                    original_image_bytes=image_bytes,
                    s3_key=request.s3_key,
                    ocr_words=words_dicts,
                    img_width=result.image_width,
                    img_height=result.image_height,
                )
            except Exception as e:
                # 마스킹 실패는 OCR 응답을 막지 않는다
                logger.warning(f"PII 마스킹 처리 실패 (원본 응답 반환): {type(e).__name__}")

        # 기존 응답 구조를 그대로 반환 (마스킹 여부와 무관)
        return result

    except FileNotFoundError as e:
        logger.error(f"S3 파일을 찾을 수 없음: {e}")
        raise HTTPException(404, f"파일을 찾을 수 없습니다: {request.s3_key}")
    except Exception as e:
        logger.error(f"OCR 처리 중 오류 발생: {e}")
        raise HTTPException(500, f"OCR 처리 실패: {str(e)}")


@router.post("/ocr/full", response_model=ContractOCRResponse, summary="이미지 직접 업로드 OCR")
async def run_ocr_full(
    file: UploadFile = File(...),
    structurize: bool = Query(True, description="구조화 여부"),
    include_overlay: bool = Query(True, description="오버레이(단어 좌표) 포함 여부"),
    service: OCRService = Depends(get_ocr_service),
):
    """이미지 파일을 직접 업로드하여 OCR 처리 (전체 결과)."""
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(400, "이미지 파일만 가능합니다")

        image_bytes = await file.read()

        result = service.process_and_map(
            image_bytes=image_bytes,
            structurize=structurize,
            include_overlay=include_overlay,
        )

        return result

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"이미지 처리 오류: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"OCR 처리 중 오류: {e}")
        raise HTTPException(500, f"OCR 처리 실패: {str(e)}")
