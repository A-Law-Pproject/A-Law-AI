"""
OCR 서비스 - 엔진 선택형 래퍼

아래 'OCR 엔진 선택' 블록에서 사용할 엔진 하나만 활성화하고
나머지는 주석 처리하세요. 서버 재시작 후 즉시 적용됩니다.
"""
import time
import cv2
import numpy as np
from loguru import logger

from app.schemas.ocr_response import ContractOCRResponse
from app.core.config import settings


def bytes_to_cv2(image_bytes: bytes):
    """바이트를 OpenCV 이미지로 변환"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"이미지 변환 실패: {e}")
        return None


# ================================================================
# OCR 엔진 선택 — 하나만 활성화, 나머지는 주석 처리
# ================================================================

# ── ① Upstage Document Parse (기본값) ───────────────────────────
from app.ocr.upstage_ocr import UpstageOCRPipeline
_ocr_pipeline = UpstageOCRPipeline()

# ── ② Naver Clova OCR ───────────────────────────────────────────
# from app.ocr.clova_ocr import ClovaOCRPipeline
# _ocr_pipeline = ClovaOCRPipeline()

# ================================================================


logger.info(f"OCR 엔진 초기화 완료: {type(_ocr_pipeline).__name__}")


class OCRService:
    """OCR 처리 서비스 (엔진 독립적)"""

    def process_and_map(
        self,
        image_bytes: bytes,
        structurize: bool,
        include_overlay: bool,
    ) -> ContractOCRResponse:
        """
        이미지를 OCR 처리하고 응답 DTO로 매핑.

        Args:
            image_bytes    : 이미지 바이트
            structurize    : GPT 구조화 수행 여부
            include_overlay: 단어별 바운딩 박스 포함 여부

        Returns:
            ContractOCRResponse
        """
        start_time = time.time()

        image = bytes_to_cv2(image_bytes)
        if image is None:
            raise ValueError("이미지 디코딩 실패")
        h, w = image.shape[:2]

        result = _ocr_pipeline.process(
            image_bytes=image_bytes,
            image_width=w,
            image_height=h,
            structurize=structurize,
            enable_llm_table_fix=settings.ENABLE_LLM_TABLE_FIX,
        )

        return ContractOCRResponse.from_result(
            result, round(time.time() - start_time, 2), include_overlay
        )

    def extract_text_only(self, image_bytes: bytes) -> str:
        """이미지에서 텍스트만 추출 (분석용, 빠름)."""
        image = bytes_to_cv2(image_bytes)
        if image is None:
            raise ValueError("이미지 디코딩 실패")
        h, w = image.shape[:2]

        result = _ocr_pipeline.process(
            image_bytes=image_bytes,
            image_width=w,
            image_height=h,
            structurize=False,
            enable_llm_table_fix=False,
        )
        return result.full_text if hasattr(result, "full_text") else str(result)
