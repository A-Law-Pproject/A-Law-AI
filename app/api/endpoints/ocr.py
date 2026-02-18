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

@router.post("/ocr", response_model=ContractOCRResponse, summary="S3 이미지 OCR")
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
