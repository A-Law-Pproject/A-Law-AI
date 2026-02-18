"""
OCR 응답 스키마
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# ============================================
# 기본 스키마
# ============================================

class OCRWord(BaseModel):
    """단어 단위 바운딩 박스 (프론트엔드 텍스트 오버레이용)"""
    text: str
    x: float = Field(..., description="좌상단 X (%, 0-100)")
    y: float = Field(..., description="좌상단 Y (%, 0-100)")
    width: float = Field(..., description="너비 (%)")
    height: float = Field(..., description="높이 (%)")
    confidence: float = Field(default=1.0, description="신뢰도 (0-1)")


class CellResult(BaseModel):
    """셀 OCR 결과"""
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    text: str
    confidence: float


class TableResult(BaseModel):
    """표 분석 결과"""
    table_x: int
    table_y: int
    table_width: int
    table_height: int
    rows: int
    cols: int
    cells: List[CellResult]


# ============================================
# 응답 스키마
# ============================================

class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    ocr_available: bool
    pipeline_available: bool
    upstage_v2_available: bool = False
    version: str


class ContractOCRResponse(BaseModel):
    """계약서 OCR 응답"""
    success: bool
    processing_time: float
    image_width: int = 0
    image_height: int = 0

    # 텍스트 결과
    full_text: Optional[str] = None
    markdown: Optional[str] = None

    # 구조화된 데이터
    contract_data: Optional[Dict[str, Any]] = None

    # 검증 결과
    validation: Optional[Dict[str, Any]] = None

    # 오버레이 데이터 - 단어별 정확한 좌표 (Upstage Document OCR)
    words: Optional[List[OCRWord]] = None

    # 경고/에러
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None

    model_config = {"from_attributes": True}

    @classmethod
    def from_result(
        cls,
        result: Any,
        processing_time: float,
        include_overlay: bool
    ) -> "ContractOCRResponse":
        """OCR 결과를 응답 DTO로 변환"""
        data = {
            "success": getattr(result, 'success', True),
            "processing_time": processing_time,
            "image_width": getattr(result, 'image_width', 0),
            "image_height": getattr(result, 'image_height', 0),
            "markdown": getattr(result, 'markdown', None),
            "full_text": getattr(result, 'full_text', None),
            "contract_data": getattr(result, 'contract_data', None),
            "validation": getattr(result, 'validation', None),
            "warnings": getattr(result, 'warnings', []),
            "error": getattr(result, 'error', None)
        }

        # 오버레이: 단어별 좌표 (Document OCR API에서 획득)
        if include_overlay and hasattr(result, 'words') and result.words:
            data["words"] = [
                OCRWord(
                    text=w.text,
                    x=w.x,
                    y=w.y,
                    width=w.width,
                    height=w.height,
                    confidence=getattr(w, 'confidence', 1.0),
                ) if hasattr(w, 'text') else w
                for w in result.words
            ]

        return cls(**data)


class AnalysisResponse(BaseModel):
    """분석 응답"""
    success: bool
    processing_time: float
    tables: List[TableResult] = Field(default_factory=list)
    full_text: str = ""
    raw_ocr: Optional[List[Dict]] = None
