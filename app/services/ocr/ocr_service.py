"""
OCR 서비스 - Upstage OCR 파이프라인 래퍼
"""
import time
import cv2
import numpy as np
from loguru import logger

from app.schemas.ocr_response import ContractOCRResponse


def bytes_to_cv2(image_bytes: bytes):
    """바이트를 OpenCV 이미지로 변환"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        logger.error(f"이미지 변환 실패: {e}")
        return None


# Upstage OCR 파이프라인 초기화
upstage_pipeline = None

try:
    from app.ocr.upstage_ocr import UpstageOCRPipeline
    upstage_pipeline = UpstageOCRPipeline()
    logger.info("Upstage OCR 파이프라인 초기화 완료")
except ImportError as e:
    logger.warning(f"Upstage OCR 파이프라인 로드 실패: {e}")
except Exception as e:
    logger.warning(f"Upstage 파이프라인 초기화 실패: {e}")


class OCRService:
    """OCR 처리 서비스 (Upstage 전용)"""

    def process_and_map(
        self,
        image_bytes: bytes,
        structurize: bool,
        include_overlay: bool
    ) -> ContractOCRResponse:
        """
        이미지를 OCR 처리하고 응답 DTO로 매핑

        Args:
            image_bytes: 이미지 바이트
            structurize: 구조화 여부 (GPT로 JSON 변환)
            include_overlay: 오버레이 포함 여부 (블록 좌표)

        Returns:
            ContractOCRResponse
        """
        if upstage_pipeline is None:
            raise RuntimeError("OCR 파이프라인이 초기화되지 않았습니다. UPSTAGE_API_KEY와 OPENAI_API_KEY를 확인하세요.")

        start_time = time.time()

        # 이미지 크기 확인
        image = bytes_to_cv2(image_bytes)
        if image is None:
            raise ValueError("이미지 디코딩 실패")

        h, w = image.shape[:2]

        # Upstage 파이프라인 실행
        result = upstage_pipeline.process(
            image_bytes=image_bytes,
            image_width=w,
            image_height=h,
            structurize=structurize
        )

        # DTO 매핑
        processing_time = round(time.time() - start_time, 2)
        return ContractOCRResponse.from_result(result, processing_time, include_overlay)

    def extract_text_only(self, image_bytes: bytes) -> str:
        """
        이미지에서 텍스트만 추출 (분석용)

        Args:
            image_bytes: 이미지 바이트

        Returns:
            추출된 텍스트
        """
        if upstage_pipeline is None:
            raise RuntimeError("OCR 파이프라인이 초기화되지 않았습니다. UPSTAGE_API_KEY와 OPENAI_API_KEY를 확인하세요.")

        image = bytes_to_cv2(image_bytes)
        if image is None:
            raise ValueError("이미지 디코딩 실패")

        h, w = image.shape[:2]

        # structurize=False로 텍스트만 추출 (빠름)
        result = upstage_pipeline.process(
            image_bytes=image_bytes,
            image_width=w,
            image_height=h,
            structurize=False
        )

        return result.full_text if hasattr(result, 'full_text') else str(result)
