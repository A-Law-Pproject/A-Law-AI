# ============================================================
# [LEGACY] 이 파일은 더 이상 사용되지 않습니다.
# PaddleOCR 기반 로컬 OCR 모델로, Upstage Document Parse API로 대체되었습니다.
# 새로운 구현: app/ocr/upstage_ocr.py
# TODO: 안정화 후 삭제 예정
# ============================================================
"""
계약서 OCR 통합 모델 (LEGACY - Upstage OCR로 대체됨)

5단계 파이프라인:
1. OpenCV 전처리 - 직선 검출, 셀 좌표 추출
2. Layout Annotation - 영역별 라벨링, 좌표+텍스트 JSON 생성
3. 영역별 OCR 최적화 - 도장 제거, 이진화, 구조화된 데이터 획득
4. Semantic 구조 파싱 - 계층 구조 데이터화 (제1조>1항>1호)
5. Post-processing & Validation - 오타 교정, 검증, Pydantic 구조화

사용법:
    from ocr_engine.contract_ocr_model import ContractOCRModel

    model = ContractOCRModel()
    result = model.process("계약서.png")

    # 프론트엔드 양식에 매핑
    print(result.to_frontend_format())
"""
import cv2
import numpy as np
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from loguru import logger

# Pydantic for schema validation
from pydantic import BaseModel, Field, field_validator, model_validator

# PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logger.warning("PaddleOCR 미설치. pip install paddleocr")

# LLM API (OpenAI 호환)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI 미설치. pip install openai")


# ===========================================
# 1. 데이터 클래스 정의
# ===========================================

@dataclass
class BoundingBox:
    """바운딩 박스"""
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def to_dict(self) -> Dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class TextRegion:
    """텍스트 영역 (Layout Annotation 결과)"""
    text: str
    bbox: BoundingBox
    confidence: float = 0.0
    field_type: Optional[str] = None  # 보증금, 월세, 임대인 등
    row: int = -1
    col: int = -1

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "field_type": self.field_type,
            "row": self.row,
            "col": self.col,
            **self.bbox.to_dict()
        }


@dataclass
class TableCell:
    """표 셀"""
    bbox: BoundingBox
    row: int
    col: int
    text: str = ""
    confidence: float = 0.0


@dataclass
class SemanticNode:
    """의미 구조 노드 (제1조>1항>1호)"""
    level: int  # 0: 조, 1: 항, 2: 호
    number: str  # "제1조", "1항", "1호"
    title: str
    content: str
    children: List["SemanticNode"] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "level": self.level,
            "number": self.number,
            "title": self.title,
            "content": self.content,
            "children": [c.to_dict() for c in self.children]
        }


# ===========================================
# 2. Pydantic 스키마 (구조 강제)
# ===========================================

class MoneyAmount(BaseModel):
    """금액"""
    value: int = Field(0, ge=0, description="금액 (원)")
    text: str = Field("", description="원본 텍스트")

    @classmethod
    def from_text(cls, text: str) -> "MoneyAmount":
        """텍스트에서 금액 파싱"""
        # 숫자 추출
        numbers = re.findall(r"[\d,]+", text)
        if numbers:
            value = int(numbers[0].replace(",", ""))
            return cls(value=value, text=text)
        return cls(value=0, text=text)


class PersonInfo(BaseModel):
    """당사자 정보"""
    name: str = ""
    resident_id: str = ""
    address: str = ""
    phone: str = ""

    @field_validator("phone")
    @classmethod
    def normalize_phone(cls, v: str) -> str:
        if not v:
            return v
        digits = re.sub(r"[^\d]", "", v)
        if len(digits) == 11:
            return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
        elif len(digits) == 10:
            return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        return v


class PropertyInfo(BaseModel):
    """부동산 정보"""
    address: str = ""
    property_type: str = ""  # 아파트, 빌라, 오피스텔 등
    area_m2: Optional[float] = None
    area_pyeong: Optional[float] = None
    floor: str = ""
    unit_number: str = ""


class ContractTerms(BaseModel):
    """계약 조건"""
    deposit: MoneyAmount = Field(default_factory=MoneyAmount)
    monthly_rent: MoneyAmount = Field(default_factory=MoneyAmount)
    contract_money: MoneyAmount = Field(default_factory=MoneyAmount)
    middle_payment: MoneyAmount = Field(default_factory=MoneyAmount)
    balance: MoneyAmount = Field(default_factory=MoneyAmount)
    start_date: str = ""
    end_date: str = ""
    contract_date: str = ""

    @model_validator(mode="after")
    def validate_money_sum(self):
        """금액 합계 검증: 보증금 = 계약금 + 중도금 + 잔금"""
        if self.deposit.value > 0:
            total = self.contract_money.value + self.middle_payment.value + self.balance.value
            if total > 0 and total != self.deposit.value:
                logger.warning(
                    f"금액 불일치: 보증금({self.deposit.value}) != "
                    f"계약금+중도금+잔금({total})"
                )
        return self


class ContractData(BaseModel):
    """계약서 전체 데이터 (Pydantic 강제 스키마)"""
    contract_type: str = "임대차"
    lessor: PersonInfo = Field(default_factory=PersonInfo)  # 임대인
    lessee: PersonInfo = Field(default_factory=PersonInfo)  # 임차인
    property_info: PropertyInfo = Field(default_factory=PropertyInfo)
    terms: ContractTerms = Field(default_factory=ContractTerms)
    special_terms: List[str] = Field(default_factory=list)
    semantic_structure: List[Dict] = Field(default_factory=list)

    # 메타 정보
    ocr_confidence: float = 0.0
    source_file: str = ""
    processed_at: str = ""

    def to_frontend_format(self) -> Dict[str, Any]:
        """프론트엔드 양식용 포맷"""
        return {
            "계약유형": self.contract_type,
            "임대인": {
                "성명": self.lessor.name,
                "주민등록번호": self.lessor.resident_id,
                "주소": self.lessor.address,
                "전화번호": self.lessor.phone,
            },
            "임차인": {
                "성명": self.lessee.name,
                "주민등록번호": self.lessee.resident_id,
                "주소": self.lessee.address,
                "전화번호": self.lessee.phone,
            },
            "부동산": {
                "소재지": self.property_info.address,
                "건물유형": self.property_info.property_type,
                "전용면적_m2": self.property_info.area_m2,
                "전용면적_평": self.property_info.area_pyeong,
                "층수": self.property_info.floor,
                "호수": self.property_info.unit_number,
            },
            "계약조건": {
                "보증금": self.terms.deposit.value,
                "보증금_텍스트": self.terms.deposit.text,
                "월세": self.terms.monthly_rent.value,
                "월세_텍스트": self.terms.monthly_rent.text,
                "계약금": self.terms.contract_money.value,
                "중도금": self.terms.middle_payment.value,
                "잔금": self.terms.balance.value,
                "계약시작일": self.terms.start_date,
                "계약종료일": self.terms.end_date,
                "계약체결일": self.terms.contract_date,
            },
            "특약사항": self.special_terms,
            "의미구조": self.semantic_structure,
            "메타정보": {
                "OCR신뢰도": round(self.ocr_confidence * 100, 1),
                "원본파일": self.source_file,
                "처리일시": self.processed_at,
            }
        }


# ===========================================
# 3. 계약서 용어 사전 (오타 교정용)
# ===========================================

CONTRACT_TERMS_DICT = {
    "임대인": ["임대인", "임디인", "임래인", "암대인", "입대인"],
    "임차인": ["임차인", "임치인", "임쟈인", "암차인", "입차인"],
    "보증금": ["보증금", "보즈금", "보증굼", "보중금", "보정금"],
    "월세": ["월세", "월쎄", "월새", "뭘세"],
    "계약금": ["계약금", "게약금", "계악금", "겨약금"],
    "중도금": ["중도금", "중도굼", "줄도금"],
    "잔금": ["잔금", "잔굼", "간금"],
    "전세": ["전세", "전쎄", "전새"],
    "특약사항": ["특약사항", "특약시항", "특약상항"],
    "원정": ["원정", "원쩡", "원졍"],
    "일금": ["일금", "일굼", "잃금"],
}

# 역매핑 생성
TYPO_TO_CORRECT = {}
for correct, typos in CONTRACT_TERMS_DICT.items():
    for typo in typos:
        if typo != correct:
            TYPO_TO_CORRECT[typo] = correct


# ===========================================
# 4. 정규식 패턴 (필드별 검증)
# ===========================================

FIELD_PATTERNS = {
    "money": re.compile(r"(\d{1,3}(,\d{3})*)\s*(원|만원|천원)"),
    "money_korean": re.compile(r"(일금\s*)?([일이삼사오육칠팔구십백천만억]+)\s*(원정?|원)"),
    "resident_id": re.compile(r"(\d{6})\s*[-－−]\s*(\d{7}|\*{6,7}|\d\*{6})"),
    "date": re.compile(r"(\d{4})\s*[년./-]\s*(\d{1,2})\s*[월./-]\s*(\d{1,2})\s*일?"),
    "phone": re.compile(r"(0\d{1,2})[-－−.\s]?(\d{3,4})[-－−.\s]?(\d{4})"),
    "area": re.compile(r"(\d+(?:\.\d+)?)\s*(㎡|평|제곱미터|m2)"),
}


# ===========================================
# 5. 메인 모델 클래스
# ===========================================

class ContractOCRModel:
    """
    계약서 OCR 통합 모델

    5단계 파이프라인을 하나의 클래스로 통합:
    1. preprocess() - OpenCV 전처리
    2. detect_layout() - Layout Annotation
    3. extract_text() - 영역별 OCR
    4. parse_semantic() - Semantic 구조 파싱
    5. validate_and_structure() - Post-processing & Validation (LLM 기반)
    """

    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = "korean",
        min_confidence: float = 0.3,  # 낮춤: 더 많은 텍스트 감지
        enable_correction: bool = True,
        use_multi_scale: bool = True,  # 다중 스케일 OCR
        use_cell_detection: bool = False,  # 셀 검출 대신 전체 OCR 우선
        use_llm: bool = True,  # LLM 기반 구조화 사용
        llm_api_key: Optional[str] = None,  # OpenAI API Key
        llm_model: str = "gpt-4o-mini",  # LLM 모델
    ):
        """
        Args:
            use_gpu: GPU 사용 여부
            lang: OCR 언어
            min_confidence: 최소 신뢰도 임계값 (낮을수록 더 많은 텍스트 감지)
            enable_correction: 오타 교정 활성화
            use_multi_scale: 다중 스케일 OCR 사용 (정확도 향상)
            use_cell_detection: 셀 검출 사용 여부 (False면 전체 이미지 OCR)
            use_llm: LLM 기반 구조화 사용 (권장)
            llm_api_key: OpenAI API Key (환경변수 OPENAI_API_KEY로도 설정 가능)
            llm_model: 사용할 LLM 모델 (기본: gpt-4o-mini)
        """
        self.use_gpu = use_gpu
        self.lang = lang
        self.min_confidence = min_confidence
        self.enable_correction = enable_correction
        self.use_multi_scale = use_multi_scale
        self.use_cell_detection = use_cell_detection
        self.use_llm = use_llm
        self.llm_model = llm_model

        # LLM 클라이언트 초기화
        self.llm_client = None
        if use_llm and OPENAI_AVAILABLE:
            import os
            api_key = llm_api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm_client = OpenAI(api_key=api_key)
                logger.info(f"LLM 구조화 활성화 (모델: {llm_model})")
            else:
                logger.warning("OPENAI_API_KEY 미설정. 규칙 기반 추출로 전환.")
                self.use_llm = False
        elif use_llm and not OPENAI_AVAILABLE:
            logger.warning("OpenAI 미설치. 규칙 기반 추출로 전환.")
            self.use_llm = False

        # OCR 엔진 초기화 (더 민감한 설정)
        if PADDLE_AVAILABLE:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False,
                # 텍스트 검출 임계값 낮춤 (더 많은 텍스트 감지)
                det_db_thresh=0.2,      # 기본 0.3 → 0.2
                det_db_box_thresh=0.3,  # 기본 0.5 → 0.3
                det_db_unclip_ratio=1.8,  # 박스 확장 비율
                rec_batch_num=16,       # 인식 배치 크기 증가
            )
            logger.info("PaddleOCR 초기화 완료 (고감도 모드)")
        else:
            self.ocr = None
            logger.error("PaddleOCR 미설치")

        # 처리 결과 저장
        self._current_image = None
        self._cells: List[TableCell] = []
        self._text_regions: List[TextRegion] = []
        self._semantic_nodes: List[SemanticNode] = []

    # =========================================
    # Stage 1: OpenCV 전처리
    # =========================================

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        1단계: OpenCV 전처리

        - 도장 제거 (붉은색 → 흰색)
        - 기울기 보정
        - 대비 향상
        - 이진화
        """
        result = image.copy()

        # 1-1. 도장 제거 (붉은색 계열)
        result = self._remove_stamp(result)

        # 1-2. 기울기 보정
        result = self._deskew(result)

        # 1-3. 대비 향상 (CLAHE)
        result = self._enhance_contrast(result)

        logger.info("전처리 완료")
        return result

    def _remove_stamp(self, image: np.ndarray) -> np.ndarray:
        """도장 제거 (붉은색 → 흰색)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 빨간색 범위 (HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        # 마스크 확장
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 흰색으로 대체
        result = image.copy()
        result[mask > 0] = [255, 255, 255]

        return result

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """기울기 보정"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)

        coords = np.column_stack(np.where(gray > 0))
        if len(coords) == 0:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > 10:
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """대비 향상 (CLAHE)"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # =========================================
    # Stage 2: Layout Annotation
    # =========================================

    def detect_layout(self, image: np.ndarray) -> List[TableCell]:
        """
        2단계: Layout Annotation

        - 가로/세로 직선 검출
        - 셀 좌표 추출
        - 각 셀을 개별 이미지로 분리

        Returns:
            List[TableCell]: 검출된 셀 목록
        """
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 이진화
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # 가로선 검출
        h_kernel_size = max(image.shape[1] // 30, 10)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)

        # 세로선 검출
        v_kernel_size = max(image.shape[0] // 30, 10)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)

        # 선 합치기
        table_mask = cv2.bitwise_or(horizontal, vertical)
        kernel = np.ones((3, 3), np.uint8)
        table_mask = cv2.dilate(table_mask, kernel, iterations=1)

        # 셀 영역 찾기
        table_mask_inv = cv2.bitwise_not(table_mask)
        contours, _ = cv2.findContours(table_mask_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        min_cell_size = (20, 15)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # 최소 크기 필터링
            if w >= min_cell_size[0] and h >= min_cell_size[1]:
                # 전체 이미지가 아닌 경우만
                if w < image.shape[1] * 0.95 and h < image.shape[0] * 0.95:
                    cells.append(TableCell(
                        bbox=BoundingBox(x=x, y=y, width=w, height=h),
                        row=-1, col=-1
                    ))

        # 행/열 번호 할당
        cells = self._assign_row_col(cells)

        self._cells = cells
        logger.info(f"레이아웃 검출: {len(cells)}개 셀")

        return cells

    def _assign_row_col(self, cells: List[TableCell]) -> List[TableCell]:
        """셀에 행/열 번호 할당"""
        if not cells:
            return cells

        threshold = 10

        # Y 좌표로 행 그룹화
        y_coords = sorted(set(c.bbox.y for c in cells))
        y_groups = []
        for y in y_coords:
            added = False
            for group in y_groups:
                if abs(y - group[0]) < threshold:
                    group.append(y)
                    added = True
                    break
            if not added:
                y_groups.append([y])

        y_to_row = {}
        for row_idx, group in enumerate(sorted(y_groups, key=lambda g: min(g))):
            for y in group:
                y_to_row[y] = row_idx

        # X 좌표로 열 그룹화
        x_coords = sorted(set(c.bbox.x for c in cells))
        x_groups = []
        for x in x_coords:
            added = False
            for group in x_groups:
                if abs(x - group[0]) < threshold:
                    group.append(x)
                    added = True
                    break
            if not added:
                x_groups.append([x])

        x_to_col = {}
        for col_idx, group in enumerate(sorted(x_groups, key=lambda g: min(g))):
            for x in group:
                x_to_col[x] = col_idx

        for cell in cells:
            cell.row = y_to_row.get(cell.bbox.y, -1)
            cell.col = x_to_col.get(cell.bbox.x, -1)

        return cells

    # =========================================
    # Stage 3: 영역별 OCR
    # =========================================

    def extract_text(self, image: np.ndarray, cells: Optional[List[TableCell]] = None) -> List[TextRegion]:
        """
        3단계: 영역별 OCR (100% 텍스트 감지 목표)

        - 전체 이미지 OCR 우선 (더 정확함)
        - 다중 스케일 OCR로 작은 텍스트도 감지
        - 저신뢰도 영역 재시도

        Returns:
            List[TextRegion]: {"text": "보증금", "x": 100, "y": 250, ...}
        """
        if self.ocr is None:
            raise RuntimeError("OCR 엔진이 초기화되지 않았습니다.")

        text_regions = []

        # 다중 스케일 OCR (작은 텍스트도 감지)
        if self.use_multi_scale:
            text_regions = self._multi_scale_ocr(image)
        else:
            text_regions = self._full_image_ocr(image)

        # 셀 검출 모드: 셀별로 추가 OCR (선택적)
        if self.use_cell_detection and cells:
            cell_regions = self._ocr_by_cells(image, cells)
            # 중복 제거하며 병합
            text_regions = self._merge_regions(text_regions, cell_regions)

        # 저신뢰도 영역 재시도
        text_regions = self._retry_low_confidence(image, text_regions)

        # 필드 타입 자동 감지
        text_regions = self._detect_field_types(text_regions)

        self._text_regions = text_regions
        logger.info(f"OCR 완료: {len(text_regions)}개 텍스트 영역")

        return text_regions

    def _full_image_ocr(self, image: np.ndarray) -> List[TextRegion]:
        """전체 이미지 OCR"""
        text_regions = []

        result = self.ocr.ocr(image, cls=True)
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    points = line[0]
                    text = line[1][0]
                    confidence = line[1][1]

                    # 바운딩 박스 변환
                    pts = np.array(points)
                    x_min, y_min = int(pts[:, 0].min()), int(pts[:, 1].min())
                    x_max, y_max = int(pts[:, 0].max()), int(pts[:, 1].max())

                    text_regions.append(TextRegion(
                        text=text,
                        bbox=BoundingBox(x=x_min, y=y_min, width=x_max-x_min, height=y_max-y_min),
                        confidence=confidence
                    ))

        return text_regions

    def _multi_scale_ocr(self, image: np.ndarray, scales: List[float] = [1.0, 1.5]) -> List[TextRegion]:
        """
        다중 스케일 OCR - 작은 텍스트도 감지

        Args:
            image: 원본 이미지
            scales: 스케일 목록 (1.0 = 원본, 1.5 = 1.5배 확대)
        """
        all_regions = []

        for scale in scales:
            if scale == 1.0:
                scaled_image = image
            else:
                scaled_image = cv2.resize(
                    image, None,
                    fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC
                )

            logger.info(f"OCR 스케일 {scale}x 처리 중...")
            result = self.ocr.ocr(scaled_image, cls=True)

            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        points = line[0]
                        text = line[1][0]
                        confidence = line[1][1]

                        # 좌표를 원본 스케일로 변환
                        pts = np.array(points) / scale
                        x_min, y_min = int(pts[:, 0].min()), int(pts[:, 1].min())
                        x_max, y_max = int(pts[:, 0].max()), int(pts[:, 1].max())

                        all_regions.append(TextRegion(
                            text=text,
                            bbox=BoundingBox(x=x_min, y=y_min, width=x_max-x_min, height=y_max-y_min),
                            confidence=confidence
                        ))

        # 중복 제거 (IoU 기반)
        merged = self._deduplicate_regions(all_regions)
        logger.info(f"다중 스케일 OCR: {len(all_regions)} → {len(merged)} (중복 제거)")

        return merged

    def _ocr_by_cells(self, image: np.ndarray, cells: List[TableCell]) -> List[TextRegion]:
        """셀별 OCR"""
        text_regions = []

        for cell in cells:
            roi = image[cell.bbox.y:cell.bbox.y2, cell.bbox.x:cell.bbox.x2]
            text, confidence = self._ocr_region(roi)

            cell.text = text
            cell.confidence = confidence

            if text.strip():
                text_regions.append(TextRegion(
                    text=text,
                    bbox=cell.bbox,
                    confidence=confidence,
                    row=cell.row,
                    col=cell.col
                ))

        return text_regions

    def _retry_low_confidence(
        self,
        image: np.ndarray,
        regions: List[TextRegion],
        threshold: float = 0.6
    ) -> List[TextRegion]:
        """저신뢰도 영역 고해상도 재시도"""
        updated = []

        for region in regions:
            if region.confidence >= threshold:
                updated.append(region)
                continue

            # 영역 크롭 (패딩 추가)
            bbox = region.bbox
            padding = 5
            x1 = max(0, bbox.x - padding)
            y1 = max(0, bbox.y - padding)
            x2 = min(image.shape[1], bbox.x2 + padding)
            y2 = min(image.shape[0], bbox.y2 + padding)

            roi = image[y1:y2, x1:x2]

            # 2배 확대 후 재시도
            enlarged = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            text, confidence = self._ocr_region(enlarged)

            if confidence > region.confidence and text.strip():
                region.text = text
                region.confidence = confidence
                logger.debug(f"재시도 성공: {region.confidence:.2f}")

            updated.append(region)

        return updated

    def _deduplicate_regions(self, regions: List[TextRegion], iou_threshold: float = 0.5) -> List[TextRegion]:
        """중복 영역 제거 (IoU 기반)"""
        if not regions:
            return []

        # 신뢰도 순 정렬
        regions = sorted(regions, key=lambda r: r.confidence, reverse=True)

        kept = []
        used = set()

        for i, r1 in enumerate(regions):
            if i in used:
                continue

            # 같은 영역 찾기
            for j, r2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue
                if self._calculate_iou(r1.bbox, r2.bbox) > iou_threshold:
                    used.add(j)

            kept.append(r1)
            used.add(i)

        return kept

    def _merge_regions(self, regions1: List[TextRegion], regions2: List[TextRegion]) -> List[TextRegion]:
        """두 영역 리스트 병합 (중복 제거)"""
        all_regions = regions1 + regions2
        return self._deduplicate_regions(all_regions)

    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """IoU (Intersection over Union) 계산"""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _ocr_region(self, roi: np.ndarray) -> Tuple[str, float]:
        """단일 영역 OCR"""
        if roi.size == 0:
            return "", 0.0

        try:
            result = self.ocr.ocr(roi, cls=True)
            if not result or not result[0]:
                return "", 0.0

            texts = []
            confidences = []

            for line in result[0]:
                if line and len(line) >= 2:
                    texts.append(line[1][0])
                    confidences.append(line[1][1])

            text = " ".join(texts)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return text, avg_conf

        except Exception as e:
            logger.error(f"OCR 오류: {e}")
            return "", 0.0

    def _detect_field_types(self, regions: List[TextRegion]) -> List[TextRegion]:
        """필드 타입 자동 감지"""
        field_keywords = {
            "임대인": ["임대인", "갑", "소유자"],
            "임차인": ["임차인", "을", "세입자"],
            "보증금": ["보증금", "전세금"],
            "월세": ["월세", "차임", "월임대료"],
            "계약금": ["계약금"],
            "중도금": ["중도금"],
            "잔금": ["잔금"],
            "주소": ["소재지", "주소", "목적물"],
            "면적": ["면적", "전용면적"],
            "특약": ["특약", "특약사항"],
        }

        for region in regions:
            for field_type, keywords in field_keywords.items():
                if any(kw in region.text for kw in keywords):
                    region.field_type = field_type
                    break

        return regions

    # =========================================
    # Stage 4: Semantic 구조 파싱
    # =========================================

    def parse_semantic(self, text_regions: List[TextRegion]) -> List[SemanticNode]:
        """
        4단계: Semantic 구조 파싱

        - 제1조 > 1항 > 1호 계층 구조 파싱
        - 조/항/호 패턴 인식

        Returns:
            List[SemanticNode]: 계층 구조 트리
        """
        # 전체 텍스트 조합
        full_text = "\n".join([r.text for r in text_regions])

        # 조/항/호 패턴
        jo_pattern = re.compile(r"제(\d+)조\s*[(\[]?([^)\]]*)[)\]]?\s*(.*)")  # 제1조
        hang_pattern = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩]|(\d+)\s*[.항]\s*(.*)")  # ① 또는 1항
        ho_pattern = re.compile(r"(\d+)\s*호\s*(.*)")  # 1호

        nodes = []
        current_jo = None
        current_hang = None

        for line in full_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # 조 매칭
            jo_match = jo_pattern.match(line)
            if jo_match:
                current_jo = SemanticNode(
                    level=0,
                    number=f"제{jo_match.group(1)}조",
                    title=jo_match.group(2) or "",
                    content=jo_match.group(3) or ""
                )
                nodes.append(current_jo)
                current_hang = None
                continue

            # 항 매칭
            hang_match = hang_pattern.match(line)
            if hang_match and current_jo:
                hang_num = hang_match.group(1) or line[0]
                current_hang = SemanticNode(
                    level=1,
                    number=f"{hang_num}항",
                    title="",
                    content=hang_match.group(2) if hang_match.group(2) else line[1:].strip()
                )
                current_jo.children.append(current_hang)
                continue

            # 호 매칭
            ho_match = ho_pattern.match(line)
            if ho_match and current_hang:
                ho_node = SemanticNode(
                    level=2,
                    number=f"{ho_match.group(1)}호",
                    title="",
                    content=ho_match.group(2)
                )
                current_hang.children.append(ho_node)

        self._semantic_nodes = nodes
        logger.info(f"의미 구조 파싱: {len(nodes)}개 조항")

        return nodes

    # =========================================
    # Stage 5: Post-processing & Validation
    # =========================================

    def validate_and_structure(
        self,
        text_regions: List[TextRegion],
        semantic_nodes: List[SemanticNode],
        source_file: str = ""
    ) -> ContractData:
        """
        5단계: Post-processing & Validation

        1. 오타 교정 (용어 사전 기반)
        2. 정규식 검증 (금액, 주민번호, 날짜 등)
        3. 금액 합계 검증
        4. Pydantic 스키마 강제 매핑

        Returns:
            ContractData: 구조화된 계약서 데이터
        """
        # 5-1. 오타 교정
        if self.enable_correction:
            text_regions = self._correct_typos(text_regions)

        # 5-2. 필드별 데이터 추출 (LLM 또는 규칙 기반)
        extracted = self._extract_fields(text_regions)

        # 5-3. Pydantic 스키마로 구조화
        # LLM이 추출한 contract_type 사용, 없으면 규칙 기반
        contract_type = extracted.get("contract_type") or self._detect_contract_type(text_regions)

        contract = ContractData(
            contract_type=contract_type,
            lessor=PersonInfo(**extracted.get("임대인", {})),
            lessee=PersonInfo(**extracted.get("임차인", {})),
            property_info=PropertyInfo(**extracted.get("부동산", {})),
            terms=ContractTerms(
                deposit=extracted.get("보증금", MoneyAmount()),
                monthly_rent=extracted.get("월세", MoneyAmount()),
                contract_money=extracted.get("계약금", MoneyAmount()),
                middle_payment=extracted.get("중도금", MoneyAmount()),
                balance=extracted.get("잔금", MoneyAmount()),
                start_date=extracted.get("계약시작일", ""),
                end_date=extracted.get("계약종료일", ""),
                contract_date=extracted.get("계약체결일", ""),
            ),
            special_terms=extracted.get("특약사항", []),
            semantic_structure=[n.to_dict() for n in semantic_nodes],
            ocr_confidence=self._calculate_avg_confidence(text_regions),
            source_file=source_file,
            processed_at=datetime.now().isoformat()
        )

        logger.info("데이터 구조화 완료")
        return contract

    def _correct_typos(self, regions: List[TextRegion]) -> List[TextRegion]:
        """오타 교정 (용어 사전 기반)"""
        for region in regions:
            original = region.text
            corrected = original

            for typo, correct in TYPO_TO_CORRECT.items():
                if typo in corrected:
                    corrected = corrected.replace(typo, correct)

            if corrected != original:
                logger.debug(f"오타 교정: '{original}' → '{corrected}'")
                region.text = corrected

        return regions

    def _extract_fields_with_llm(self, regions: List[TextRegion]) -> Dict[str, Any]:
        """
        LLM 기반 필드 추출 (권장)

        OCR로 추출한 모든 텍스트를 LLM에 전달하여
        계약서 필드에 정확하게 매핑합니다.
        """
        if not self.llm_client:
            logger.warning("LLM 클라이언트 없음. 규칙 기반 추출로 전환.")
            return self._extract_fields_rule_based(regions)

        # OCR 결과를 위치순으로 정렬 (위→아래, 왼쪽→오른쪽)
        sorted_regions = sorted(regions, key=lambda r: (r.bbox.y, r.bbox.x))

        # 텍스트와 위치 정보를 포함한 컨텍스트 생성
        ocr_lines = []
        for i, region in enumerate(sorted_regions):
            ocr_lines.append(f"[{i+1}] (y:{region.bbox.y}, x:{region.bbox.x}) {region.text}")

        ocr_text = "\n".join(ocr_lines)

        # LLM 프롬프트
        system_prompt = """당신은 한국 부동산 계약서 분석 전문가입니다.
OCR로 추출된 텍스트를 분석하여 계약서의 각 필드를 정확하게 추출해주세요.

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 순수 JSON만 출력하세요.
값이 없는 필드는 빈 문자열("")로 남겨두세요.

{
  "contract_type": "임대차/전세/매매 중 하나",
  "lessor": {
    "name": "임대인 성명",
    "resident_id": "주민등록번호 (123456-1234567 형식)",
    "address": "임대인 주소",
    "phone": "전화번호 (010-1234-5678 형식)"
  },
  "lessee": {
    "name": "임차인 성명",
    "resident_id": "주민등록번호",
    "address": "임차인 주소",
    "phone": "전화번호"
  },
  "property": {
    "address": "부동산 소재지",
    "type": "건물 유형 (아파트/빌라/오피스텔/상가 등)",
    "area_m2": 전용면적(숫자만, m2 기준),
    "floor": "층수",
    "unit": "호수"
  },
  "terms": {
    "deposit": 보증금(숫자만, 원 단위),
    "deposit_text": "보증금 원본 텍스트",
    "monthly_rent": 월세(숫자만),
    "monthly_rent_text": "월세 원본 텍스트",
    "contract_money": 계약금(숫자만),
    "middle_payment": 중도금(숫자만),
    "balance": 잔금(숫자만),
    "start_date": "계약 시작일 (YYYY년 MM월 DD일)",
    "end_date": "계약 종료일",
    "contract_date": "계약 체결일"
  },
  "special_terms": ["특약사항1", "특약사항2"]
}

주의사항:
1. 금액은 반드시 숫자만 (예: 10000000)
2. "일금 일천만원정"처럼 한글 금액은 숫자로 변환
3. 날짜는 "YYYY년 MM월 DD일" 형식
4. 임대인(갑)과 임차인(을)을 정확히 구분
5. 특약사항은 개별 항목으로 분리"""

        user_prompt = f"""다음은 계약서 이미지에서 OCR로 추출한 텍스트입니다.
위치 정보(y, x 좌표)를 참고하여 각 필드를 정확하게 매핑해주세요.

=== OCR 결과 ===
{ocr_text}
================

위 텍스트를 분석하여 계약서 정보를 JSON으로 추출해주세요."""

        try:
            logger.info("LLM 구조화 요청 중...")
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # 낮은 temperature로 일관된 출력
                max_tokens=2000,
            )

            result_text = response.choices[0].message.content.strip()

            # JSON 파싱 (코드 블록 제거)
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            llm_result = json.loads(result_text)
            logger.info("LLM 구조화 완료")

            # 결과 변환
            return self._convert_llm_result(llm_result)

        except json.JSONDecodeError as e:
            logger.error(f"LLM JSON 파싱 실패: {e}")
            logger.debug(f"LLM 응답: {result_text[:500]}")
            return self._extract_fields_rule_based(regions)
        except Exception as e:
            logger.error(f"LLM 요청 실패: {e}")
            return self._extract_fields_rule_based(regions)

    def _convert_llm_result(self, llm_result: Dict) -> Dict[str, Any]:
        """LLM 결과를 내부 형식으로 변환"""
        extracted = {
            "임대인": {},
            "임차인": {},
            "부동산": {},
            "특약사항": [],
        }

        # 임대인
        if "lessor" in llm_result:
            lessor = llm_result["lessor"]
            extracted["임대인"] = {
                "name": lessor.get("name", ""),
                "resident_id": lessor.get("resident_id", ""),
                "address": lessor.get("address", ""),
                "phone": lessor.get("phone", ""),
            }

        # 임차인
        if "lessee" in llm_result:
            lessee = llm_result["lessee"]
            extracted["임차인"] = {
                "name": lessee.get("name", ""),
                "resident_id": lessee.get("resident_id", ""),
                "address": lessee.get("address", ""),
                "phone": lessee.get("phone", ""),
            }

        # 부동산
        if "property" in llm_result:
            prop = llm_result["property"]
            extracted["부동산"] = {
                "address": prop.get("address", ""),
                "property_type": prop.get("type", ""),
                "area_m2": prop.get("area_m2"),
                "floor": prop.get("floor", ""),
                "unit_number": prop.get("unit", ""),
            }

        # 계약 조건
        if "terms" in llm_result:
            terms = llm_result["terms"]
            extracted["보증금"] = MoneyAmount(
                value=terms.get("deposit", 0) or 0,
                text=terms.get("deposit_text", "")
            )
            extracted["월세"] = MoneyAmount(
                value=terms.get("monthly_rent", 0) or 0,
                text=terms.get("monthly_rent_text", "")
            )
            extracted["계약금"] = MoneyAmount(
                value=terms.get("contract_money", 0) or 0,
                text=""
            )
            extracted["중도금"] = MoneyAmount(
                value=terms.get("middle_payment", 0) or 0,
                text=""
            )
            extracted["잔금"] = MoneyAmount(
                value=terms.get("balance", 0) or 0,
                text=""
            )
            extracted["계약시작일"] = terms.get("start_date", "")
            extracted["계약종료일"] = terms.get("end_date", "")
            extracted["계약체결일"] = terms.get("contract_date", "")

        # 특약사항
        if "special_terms" in llm_result:
            extracted["특약사항"] = llm_result["special_terms"] or []

        # 계약 유형
        extracted["contract_type"] = llm_result.get("contract_type", "임대차")

        return extracted

    def _extract_fields_rule_based(self, regions: List[TextRegion]) -> Dict[str, Any]:
        """규칙 기반 필드 추출 (fallback)"""
        extracted = {
            "임대인": {},
            "임차인": {},
            "부동산": {},
            "특약사항": [],
        }

        current_party = None

        for region in regions:
            text = region.text

            # 당사자 컨텍스트 감지
            if "임대인" in text or region.field_type == "임대인":
                current_party = "임대인"
            elif "임차인" in text or region.field_type == "임차인":
                current_party = "임차인"

            # 금액 추출
            if region.field_type == "보증금" or "보증금" in text:
                extracted["보증금"] = self._parse_money(text)
            elif region.field_type == "월세" or "월세" in text:
                extracted["월세"] = self._parse_money(text)
            elif region.field_type == "계약금" or "계약금" in text:
                extracted["계약금"] = self._parse_money(text)
            elif region.field_type == "잔금" or "잔금" in text:
                extracted["잔금"] = self._parse_money(text)

            # 날짜 추출
            date_match = FIELD_PATTERNS["date"].search(text)
            if date_match:
                date_str = f"{date_match.group(1)}년 {date_match.group(2)}월 {date_match.group(3)}일"
                if "시작" in text or "부터" in text:
                    extracted["계약시작일"] = date_str
                elif "종료" in text or "까지" in text:
                    extracted["계약종료일"] = date_str
                else:
                    extracted["계약체결일"] = date_str

            # 전화번호 추출
            phone_match = FIELD_PATTERNS["phone"].search(text)
            if phone_match and current_party:
                phone = f"{phone_match.group(1)}-{phone_match.group(2)}-{phone_match.group(3)}"
                extracted[current_party]["phone"] = phone

            # 주민번호 추출
            resident_match = FIELD_PATTERNS["resident_id"].search(text)
            if resident_match and current_party:
                extracted[current_party]["resident_id"] = f"{resident_match.group(1)}-{resident_match.group(2)}"

            # 주소/소재지
            if region.field_type == "주소" or "소재지" in text:
                extracted["부동산"]["address"] = text

            # 면적
            area_match = FIELD_PATTERNS["area"].search(text)
            if area_match:
                value = float(area_match.group(1))
                unit = area_match.group(2)
                if unit in ["㎡", "m2", "제곱미터"]:
                    extracted["부동산"]["area_m2"] = value
                else:
                    extracted["부동산"]["area_pyeong"] = value

            # 특약사항
            if region.field_type == "특약" or "특약" in text:
                extracted["특약사항"].append(text)

        return extracted

    def _extract_fields(self, regions: List[TextRegion]) -> Dict[str, Any]:
        """필드별 데이터 추출 (LLM 또는 규칙 기반)"""
        if self.use_llm and self.llm_client:
            return self._extract_fields_with_llm(regions)
        return self._extract_fields_rule_based(regions)

    def _parse_money(self, text: str) -> MoneyAmount:
        """금액 파싱"""
        # 숫자 형식
        match = FIELD_PATTERNS["money"].search(text)
        if match:
            value = int(match.group(1).replace(",", ""))
            return MoneyAmount(value=value, text=text)

        # 한글 형식
        match = FIELD_PATTERNS["money_korean"].search(text)
        if match:
            # 간단한 한글→숫자 변환
            korean_text = match.group(2)
            value = self._korean_to_number(korean_text)
            return MoneyAmount(value=value, text=text)

        return MoneyAmount(value=0, text=text)

    def _korean_to_number(self, korean: str) -> int:
        """한글 금액 → 숫자"""
        units = {
            "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5,
            "육": 6, "칠": 7, "팔": 8, "구": 9,
            "십": 10, "백": 100, "천": 1000, "만": 10000, "억": 100000000
        }

        result = 0
        current = 0

        for char in korean:
            if char in units:
                num = units[char]
                if num >= 10:
                    if current == 0:
                        current = 1
                    current *= num
                    if num >= 10000:
                        result += current
                        current = 0
                else:
                    current = num

        return result + current

    def _detect_contract_type(self, regions: List[TextRegion]) -> str:
        """계약서 유형 감지"""
        full_text = " ".join([r.text for r in regions])

        if "전세" in full_text:
            return "전세"
        elif "매매" in full_text:
            return "매매"
        elif "상가" in full_text:
            return "상가임대차"
        else:
            return "임대차"

    def _calculate_avg_confidence(self, regions: List[TextRegion]) -> float:
        """평균 신뢰도 계산"""
        if not regions:
            return 0.0
        return sum(r.confidence for r in regions) / len(regions)

    # =========================================
    # 메인 처리 함수
    # =========================================

    def process(self, image_input: Union[str, np.ndarray]) -> ContractData:
        """
        전체 파이프라인 실행

        Args:
            image_input: 이미지 파일 경로 또는 numpy 배열

        Returns:
            ContractData: 구조화된 계약서 데이터

        Usage:
            model = ContractOCRModel()
            result = model.process("계약서.png")
            print(result.to_frontend_format())
        """
        # 이미지 로드 (한글 경로 지원)
        if isinstance(image_input, str):
            # 한글 경로 지원을 위해 numpy로 읽기
            image = cv2.imdecode(
                np.fromfile(image_input, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if image is None:
                raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_input}")
            source_file = Path(image_input).name
        else:
            image = image_input
            source_file = ""

        self._current_image = image
        logger.info(f"이미지 로드: {image.shape}")

        # Stage 1: 전처리
        logger.info("Stage 1: 전처리")
        processed = self.preprocess(image)

        # Stage 2: 레이아웃 검출 (선택적)
        cells = None
        if self.use_cell_detection:
            logger.info("Stage 2: 레이아웃 검출")
            cells = self.detect_layout(processed)
        else:
            logger.info("Stage 2: 레이아웃 검출 (스킵 - 전체 OCR 모드)")

        # Stage 3: OCR (전체 이미지 + 다중 스케일)
        logger.info("Stage 3: OCR (다중 스케일)")
        text_regions = self.extract_text(processed, cells)

        # Stage 4: 의미 구조 파싱
        logger.info("Stage 4: 의미 구조 파싱")
        semantic_nodes = self.parse_semantic(text_regions)

        # Stage 5: 검증 및 구조화
        logger.info("Stage 5: 검증 및 구조화")
        contract = self.validate_and_structure(text_regions, semantic_nodes, source_file)

        return contract

    def process_to_json(self, image_input: Union[str, np.ndarray]) -> str:
        """JSON 문자열로 반환"""
        result = self.process(image_input)
        return json.dumps(result.to_frontend_format(), ensure_ascii=False, indent=2)

    def get_raw_ocr_results(self) -> List[Dict]:
        """원시 OCR 결과 반환 (디버깅용)"""
        return [r.to_dict() for r in self._text_regions]

    def get_cells(self) -> List[Dict]:
        """셀 정보 반환 (디버깅용)"""
        return [
            {
                "row": c.row,
                "col": c.col,
                "text": c.text,
                **c.bbox.to_dict()
            }
            for c in self._cells
        ]


# ===========================================
# 편의 함수
# ===========================================

def process_contract(image_path: str) -> Dict[str, Any]:
    """
    계약서 이미지 처리 편의 함수

    Args:
        image_path: 이미지 파일 경로

    Returns:
        프론트엔드용 데이터 딕셔너리
    """
    model = ContractOCRModel()
    result = model.process(image_path)
    return result.to_frontend_format()


# ===========================================
# 메인 실행
# ===========================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]

        model = ContractOCRModel()
        result = model.process(image_path)

        print("\n" + "=" * 50)
        print("계약서 OCR 결과")
        print("=" * 50)
        print(json.dumps(result.to_frontend_format(), ensure_ascii=False, indent=2))

    else:
        print("사용법: python contract_ocr_model.py <이미지경로>")
        print("\n예시:")
        print("  from ocr_engine.contract_ocr_model import ContractOCRModel")
        print("")
        print("  model = ContractOCRModel()")
        print("  result = model.process('계약서.png')")
        print("  print(result.to_frontend_format())")
"""
