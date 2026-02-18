"""
Upstage Document Parse 기반 OCR 파이프라인

3단계 처리:
1. Upstage Document Parse - 레이아웃 분석 + 텍스트 추출 (Markdown + 좌표)
2. GPT-4o-mini - Markdown → 구조화된 JSON (DTO)
3. Pydantic - 비즈니스 로직 검증

프론트엔드 요구사항:
- 이미지 위 텍스트 드래그 (OCR Overlay)
- 좌표를 백분율(%)로 정규화
- 단어(Word)별 좌표 정보
"""
import os
import base64
import httpx
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, field_validator
from loguru import logger

# YAML 설정 로드
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False

# OpenAI 호환 API
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI 미설치. pip install openai")


def load_api_keys() -> Dict[str, str]:
    """
    API 키 로드 (환경변수 우선, 없으면 pydantic Settings 또는 config 파일에서 읽기)

    Returns:
        {"upstage": "...", "openai": "..."}
    """
    keys = {
        "upstage": os.getenv("UPSTAGE_API_KEY", ""),
        "openai": os.getenv("OPENAI_API_KEY", "")
    }

    # 1) Pydantic Settings에서 읽기 (로컬 .env를 Settings가 로드했을 때 유용)
    try:
        from app.core.config import settings
        if not keys["upstage"]:
            keys["upstage"] = getattr(settings, "UPSTAGE_API_KEY", "") or keys["upstage"]
        if not keys["openai"]:
            keys["openai"] = getattr(settings, "OPENAI_API_KEY", "") or keys["openai"]
    except Exception:
        # 안전하게 무시 (settings가 아직 초기화되지 않았거나 import 순환이 있는 경우)
        pass

    # 2) 환경변수에 없으면 config 파일에서 읽기
    if not keys["upstage"] or not keys["openai"]:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"

        if config_path.exists() and YAML_AVAILABLE:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                api_keys = config.get("api_keys", {})

                if not keys["upstage"]:
                    keys["upstage"] = api_keys.get("upstage", "")
                if not keys["openai"]:
                    keys["openai"] = api_keys.get("openai", "")

                if keys["upstage"] or keys["openai"]:
                    logger.info("Config 파일에서 API 키 로드 완료")

            except Exception as e:
                logger.warning(f"Config 파일 로드 실패: {e}")

    return keys


# ============================================================
# 1. 응답 스키마 정의 (프론트엔드용)
# ============================================================

class WordBox(BaseModel):
    """단어 단위 바운딩 박스 (프론트엔드 드래그용)"""
    text: str
    # 백분율 좌표 (0-100)
    x: float = Field(..., ge=0, le=100, description="좌상단 X (%, 0-100)")
    y: float = Field(..., ge=0, le=100, description="좌상단 Y (%, 0-100)")
    width: float = Field(..., ge=0, le=100, description="너비 (%)")
    height: float = Field(..., ge=0, le=100, description="높이 (%)")
    confidence: float = Field(default=1.0, ge=0, le=1, description="신뢰도 (0-1)")

    # 원본 픽셀 좌표 (디버깅용)
    px_x: Optional[int] = None
    px_y: Optional[int] = None
    px_width: Optional[int] = None
    px_height: Optional[int] = None


class TextBlock(BaseModel):
    """텍스트 블록 (문단/셀 단위)"""
    text: str
    block_type: str = "paragraph"  # paragraph, table_cell, title, etc.
    words: List[WordBox] = Field(default_factory=list)
    # 블록 전체 좌표 (백분율)
    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0


class TableCell(BaseModel):
    """표 셀"""
    row: int
    col: int
    text: str
    rowspan: int = 1
    colspan: int = 1
    # 좌표 (백분율)
    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0


class TableData(BaseModel):
    """표 데이터"""
    rows: int
    cols: int
    cells: List[TableCell] = Field(default_factory=list)
    # 표 전체 좌표 (백분율)
    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0


class OCROverlayResponse(BaseModel):
    """
    프론트엔드용 OCR 응답

    이미지 위에 투명 텍스트 레이어를 씌워
    드래그 & 복사가 가능하도록 구성
    """
    success: bool = True

    # 이미지 정보
    image_width: int = Field(..., description="원본 이미지 너비 (px)")
    image_height: int = Field(..., description="원본 이미지 높이 (px)")

    # 텍스트 오버레이용 단어 목록 (백분율 좌표)
    words: List[WordBox] = Field(default_factory=list, description="단어별 좌표 (드래그용)")

    # 구조화된 블록 (문단, 표 등)
    blocks: List[TextBlock] = Field(default_factory=list)
    tables: List[TableData] = Field(default_factory=list)

    # 전체 텍스트 (Markdown)
    markdown: str = ""
    full_text: str = ""

    # 구조화된 계약서 데이터 (DTO)
    contract_data: Optional[Dict[str, Any]] = None

    # 검증 결과
    validation: Optional[Dict[str, Any]] = None

    # 메타 정보
    processing_time: float = 0.0
    confidence: float = 0.0
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


# ============================================================
# 2. 계약서 DTO 스키마 (GPT가 채울 구조)
# ============================================================

class PersonDTO(BaseModel):
    """당사자 정보"""
    name: str = ""
    resident_id: str = ""  # 주민등록번호
    address: str = ""
    phone: str = ""


class PropertyDTO(BaseModel):
    """부동산 정보"""
    address: str = ""  # 소재지
    land_category: str = ""  # 지목
    land_area_m2: Optional[float] = None  # 토지 면적
    building_structure: str = ""  # 건물 구조/용도
    building_area_m2: Optional[float] = None  # 건물 면적
    lease_area_m2: Optional[float] = None  # 임대 면적
    floor: str = ""  # 층수
    ho: str = ""  # 호수


class ContractTermsDTO(BaseModel):
    """계약 조건"""
    deposit: int = 0  # 보증금 (원)
    deposit_text: str = ""  # 보증금 한글
    monthly_rent: int = 0  # 월세 (원)
    monthly_rent_text: str = ""
    down_payment: int = 0  # 계약금
    middle_payment: int = 0  # 중도금
    balance: int = 0  # 잔금

    contract_start_date: str = ""  # 계약 시작일
    contract_end_date: str = ""  # 계약 종료일
    contract_date: str = ""  # 계약 체결일

    payment_day: int = 0  # 월세 지불일

    @field_validator('deposit', 'monthly_rent', 'down_payment', 'middle_payment', 'balance', mode='before')
    @classmethod
    def parse_money(cls, v):
        if isinstance(v, str):
            # "100,000,000" → 100000000
            return int(v.replace(',', '').replace('원', '').strip() or 0)
        return v or 0


class SpecialTermDTO(BaseModel):
    """특약사항"""
    index: int
    content: str
    is_toxic: bool = False  # 독소조항 여부
    toxic_category: str = ""  # 독소조항 카테고리


class BrokerDTO(BaseModel):
    """중개업자 정보"""
    company_name: str = ""
    representative: str = ""
    address: str = ""
    registration_number: str = ""
    phone: str = ""


class LeaseContractDTO(BaseModel):
    """임대차 계약서 전체 DTO"""
    contract_type: str = "임대차"

    lessor: PersonDTO = Field(default_factory=PersonDTO, description="임대인")
    lessee: PersonDTO = Field(default_factory=PersonDTO, description="임차인")

    property: PropertyDTO = Field(default_factory=PropertyDTO, description="부동산 정보")
    terms: ContractTermsDTO = Field(default_factory=ContractTermsDTO, description="계약 조건")

    special_terms: List[SpecialTermDTO] = Field(default_factory=list, description="특약사항")

    broker1: Optional[BrokerDTO] = None
    broker2: Optional[BrokerDTO] = None

    # 메타 정보
    source_file: str = ""
    ocr_confidence: float = 0.0


# ============================================================
# 3. Upstage Document Parse 클라이언트
# ============================================================

class UpstageClient:
    """Upstage Document Parse API 클라이언트"""

    API_URL = "https://api.upstage.ai/v1/document-ai/document-parse"

    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            self.api_key = api_key
        else:
            keys = load_api_keys()
            self.api_key = keys.get("upstage", "")

        if not self.api_key:
            raise ValueError("UPSTAGE_API_KEY 필요 (환경변수 또는 configs/default.yaml)")

    def parse(
        self,
        image_path: str = None,
        image_bytes: bytes = None,
        ocr: bool = True,
        coordinates: bool = True,
        output_formats: List[str] = None
    ) -> Dict[str, Any]:
        """
        문서 파싱 수행

        Args:
            image_path: 이미지 파일 경로
            image_bytes: 이미지 바이트
            ocr: OCR 수행 여부
            coordinates: 좌표 정보 포함 여부
            output_formats: ["text", "html", "markdown"]

        Returns:
            파싱 결과 딕셔너리
        """
        output_formats = output_formats or ["markdown", "text"]

        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        # 파일 준비
        if image_path:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            filename = Path(image_path).name
        else:
            filename = "image.png"

        files = {
            "document": (filename, image_bytes, "image/png")
        }

        data = {
            "ocr": str(ocr).lower(),
            "coordinates": str(coordinates).lower(),
            "output_formats": json.dumps(output_formats)
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    self.API_URL,
                    headers=headers,
                    files=files,
                    data=data
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Upstage API 오류: {e}")
            raise

    # ----------------------------------------------------------
    # Document OCR API - 단어별 bounding box 반환
    # ----------------------------------------------------------
    OCR_API_URL = "https://api.upstage.ai/v1/document-digitization"

    def ocr(
        self,
        image_path: str = None,
        image_bytes: bytes = None,
    ) -> Dict[str, Any]:
        """
        Document OCR 수행 (단어별 좌표 반환)

        Returns:
            {
              "pages": [{
                "width": int, "height": int,
                "words": [{"text": str, "boundingBox": {"vertices": [{"x","y"}, ...]}}]
              }],
              "text": str
            }
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}

        if image_path:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            filename = Path(image_path).name
        else:
            filename = "image.png"

        files = {"document": (filename, image_bytes, "image/png")}
        data = {"model": "ocr"}

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    self.OCR_API_URL,
                    headers=headers,
                    files=files,
                    data=data,
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Upstage OCR API 오류: {e}")
            raise


# ============================================================
# 4. GPT 구조화 클라이언트
# ============================================================

class GPTStructurizer:
    """GPT-4o-mini로 Markdown → JSON 변환"""

    SYSTEM_PROMPT = """당신은 한국 부동산 임대차 계약서를 분석하는 전문가입니다.
주어진 계약서 Markdown 텍스트를 분석하여 JSON 형태로 구조화하세요.

반드시 아래 JSON 스키마를 따르세요:
{
  "contract_type": "임대차",
  "lessor": {
    "name": "임대인 이름",
    "resident_id": "주민등록번호 (앞6자리-뒷7자리)",
    "address": "주소",
    "phone": "전화번호"
  },
  "lessee": {
    "name": "임차인 이름",
    "resident_id": "주민등록번호",
    "address": "주소",
    "phone": "전화번호"
  },
  "property": {
    "address": "부동산 소재지",
    "land_category": "지목 (대지, 전, 답 등)",
    "land_area_m2": 면적(숫자),
    "building_structure": "건물 구조/용도",
    "building_area_m2": 면적(숫자),
    "lease_area_m2": 임대면적(숫자),
    "floor": "층수",
    "ho": "호수"
  },
  "terms": {
    "deposit": 보증금(숫자, 원 단위),
    "deposit_text": "보증금 한글 표기",
    "monthly_rent": 월세(숫자),
    "monthly_rent_text": "월세 한글 표기",
    "down_payment": 계약금(숫자),
    "middle_payment": 중도금(숫자),
    "balance": 잔금(숫자),
    "contract_start_date": "YYYY-MM-DD",
    "contract_end_date": "YYYY-MM-DD",
    "contract_date": "YYYY-MM-DD",
    "payment_day": 매월 지불일(숫자)
  },
  "special_terms": [
    {
      "index": 1,
      "content": "특약사항 내용",
      "is_toxic": false,
      "toxic_category": ""
    }
  ],
  "broker1": {
    "company_name": "중개업소명",
    "representative": "대표자",
    "address": "소재지",
    "registration_number": "등록번호",
    "phone": "전화번호"
  }
}

중요:
- 값이 없으면 빈 문자열("") 또는 0을 사용
- 금액은 반드시 숫자(원 단위)로 변환
- 날짜는 YYYY-MM-DD 형식으로
- 특약사항 중 임차인에게 불리한 조항은 is_toxic: true로 표시
- JSON만 출력하고 다른 텍스트는 포함하지 마세요
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI 패키지 필요: pip install openai")

        if api_key:
            self.api_key = api_key
        else:
            keys = load_api_keys()
            self.api_key = keys.get("openai", "")

        self.model = model

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )

    def structurize(self, markdown: str) -> LeaseContractDTO:
        """
        Markdown 텍스트를 구조화된 DTO로 변환

        Args:
            markdown: 계약서 Markdown 텍스트

        Returns:
            LeaseContractDTO
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"다음 계약서를 분석하세요:\n\n{markdown}"}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            data = json.loads(content)

            return LeaseContractDTO(**data)

        except Exception as e:
            logger.error(f"GPT 구조화 오류: {e}")
            return LeaseContractDTO()


# ============================================================
# 5. 좌표 변환 유틸리티
# ============================================================

def normalize_coordinates(
    boxes: List[Dict],
    image_width: int,
    image_height: int
) -> List[WordBox]:
    """
    픽셀 좌표를 백분율로 정규화

    Args:
        boxes: [{"text": str, "x": int, "y": int, "width": int, "height": int}, ...]
        image_width: 이미지 너비
        image_height: 이미지 높이

    Returns:
        List[WordBox] - 백분율 좌표
    """
    normalized = []

    for box in boxes:
        px_x = box.get("x", 0)
        px_y = box.get("y", 0)
        px_w = box.get("width", 0)
        px_h = box.get("height", 0)

        word = WordBox(
            text=box.get("text", ""),
            x=(px_x / image_width) * 100,
            y=(px_y / image_height) * 100,
            width=(px_w / image_width) * 100,
            height=(px_h / image_height) * 100,
            confidence=box.get("confidence", 1.0),
            px_x=px_x,
            px_y=px_y,
            px_width=px_w,
            px_height=px_h
        )
        normalized.append(word)

    return normalized


def parse_upstage_coordinates(coords: Any) -> Tuple[float, float, float, float]:
    """
    Upstage 좌표 형식 파싱

    Upstage API 좌표 형식 (실제):
    - 4개의 딕셔너리: [{"x": 0.07, "y": 0.01}, {"x": 0.96, "y": 0.01}, ...]
    - 값은 0~1 사이의 상대 좌표 (이미 정규화됨)

    Returns:
        (x, y, width, height) - 백분율 (0-100)
    """
    if not coords:
        return 0, 0, 0, 0

    try:
        # 좌표가 리스트인 경우 (Upstage 형식)
        if isinstance(coords, list) and len(coords) >= 4:
            # 첫 번째 요소로 형식 판단
            first = coords[0]

            if isinstance(first, dict):
                # [{"x": 0.07, "y": 0.01}, ...] 형식 (Upstage 실제 형식)
                xs = [p.get("x", 0) for p in coords]
                ys = [p.get("y", 0) for p in coords]
            elif isinstance(first, (list, tuple)):
                # [[x1, y1], [x2, y2], ...] 형식
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
            else:
                return 0, 0, 0, 0

            min_x = min(xs)
            min_y = min(ys)
            max_x = max(xs)
            max_y = max(ys)

            # 0-1 값을 0-100 백분율로 변환
            return (
                min_x * 100,
                min_y * 100,
                (max_x - min_x) * 100,
                (max_y - min_y) * 100
            )

        # 딕셔너리 형식 (폴백 - x, y, width, height)
        elif isinstance(coords, dict):
            x = coords.get("x", 0)
            y = coords.get("y", 0)
            w = coords.get("width", 0)
            h = coords.get("height", 0)
            return x * 100, y * 100, w * 100, h * 100

    except Exception as e:
        logger.warning(f"좌표 파싱 오류: {e}, coords={coords}")

    return 0, 0, 0, 0


def parse_ocr_words(
    ocr_result: Dict[str, Any],
    image_width: int,
    image_height: int
) -> List[WordBox]:
    """
    Upstage Document OCR API 응답에서 단어별 좌표를 추출

    OCR API 응답:
      pages[].words[]: { text, confidence, boundingBox: { vertices: [{x, y}, ...] } }
      vertices의 x, y는 픽셀 좌표

    Returns:
        WordBox 리스트 (백분율 좌표)
    """
    words = []

    pages = ocr_result.get("pages", [])
    for page in pages:
        page_w = page.get("width", image_width) or image_width
        page_h = page.get("height", image_height) or image_height

        for w_data in page.get("words", []):
            text = w_data.get("text", "").strip()
            if not text:
                continue

            confidence = w_data.get("confidence", 1.0)
            bbox = w_data.get("boundingBox", {})
            vertices = bbox.get("vertices", [])

            if len(vertices) < 4:
                continue

            # vertices: 4개의 꼭짓점 {x, y} (픽셀 좌표)
            xs = [v.get("x", 0) for v in vertices]
            ys = [v.get("y", 0) for v in vertices]

            min_x = min(xs)
            min_y = min(ys)
            max_x = max(xs)
            max_y = max(ys)

            # 픽셀 → 백분율 변환
            pct_x = round(min_x / page_w * 100, 2) if page_w else 0
            pct_y = round(min_y / page_h * 100, 2) if page_h else 0
            pct_w = round((max_x - min_x) / page_w * 100, 2) if page_w else 0
            pct_h = round((max_y - min_y) / page_h * 100, 2) if page_h else 0

            words.append(WordBox(
                text=text,
                x=pct_x,
                y=pct_y,
                width=pct_w,
                height=pct_h,
                confidence=confidence,
                px_x=int(min_x),
                px_y=int(min_y),
                px_width=int(max_x - min_x),
                px_height=int(max_y - min_y),
            ))

    return words


def parse_upstage_elements(
    elements: List[Dict],
    image_width: int,
    image_height: int
) -> Tuple[List[WordBox], List[TextBlock], List[TableData]]:
    """
    Upstage 응답의 elements를 파싱

    Upstage Document Parse API 좌표 형식:
    - coordinates: 4개의 (x, y) 쌍 (바운딩 박스 꼭짓점)
    - 값은 0~1 사이의 상대 좌표 (이미 정규화됨)

    Returns:
        (words, blocks, tables)
    """
    words = []
    blocks = []
    tables = []

    for elem in elements:
        elem_type = elem.get("category", "paragraph")
        content = elem.get("content", {})

        # 텍스트 추출 (text → markdown → html 순으로 fallback)
        if isinstance(content, str):
            text = content
        elif isinstance(content, dict):
            text = content.get("text", "")
            if not text:
                # table 등은 text가 없고 markdown/html만 있음
                md = content.get("markdown", "")
                if md:
                    # markdown 테이블 태그/기호 제거하여 plain text 추출
                    import re as _re
                    text = _re.sub(r'<[^>]+>', ' ', md)  # HTML 태그 제거
                    text = _re.sub(r'[|]', ' ', text)     # 테이블 구분자 제거
                    text = _re.sub(r'-{2,}', '', text)    # 구분선 제거
                    text = _re.sub(r'\s+', ' ', text).strip()
        else:
            text = ""

        coords = elem.get("coordinates", [])

        # Upstage 좌표 파싱 (0-1 상대좌표 → 0-100 백분율)
        pct_x, pct_y, pct_w, pct_h = parse_upstage_coordinates(coords)

        # 픽셀 좌표 계산 (디버깅용)
        px_x = int(pct_x / 100 * image_width) if image_width else 0
        px_y = int(pct_y / 100 * image_height) if image_height else 0
        px_w = int(pct_w / 100 * image_width) if image_width else 0
        px_h = int(pct_h / 100 * image_height) if image_height else 0

        # 단어 분리 (텍스트를 공백으로 분리하여 각 단어에 대략적 위치 부여)
        if text and pct_w > 0:
            word_list = text.split()
            if word_list:
                # 각 단어의 대략적 너비 계산
                total_chars = sum(len(w) for w in word_list)

                current_x = pct_x
                for word_text in word_list:
                    # 글자 수 기반 너비 배분
                    word_ratio = len(word_text) / total_chars if total_chars > 0 else 1 / len(word_list)
                    word_width = pct_w * word_ratio

                    # 단어 간 간격 추가 (총 너비의 5% 정도)
                    spacing = pct_w * 0.02 if len(word_list) > 1 else 0

                    words.append(WordBox(
                        text=word_text,
                        x=round(current_x, 2),
                        y=round(pct_y, 2),
                        width=round(max(word_width - spacing, 0.1), 2),
                        height=round(pct_h, 2),
                        px_x=int(current_x / 100 * image_width) if image_width else 0,
                        px_y=px_y,
                        px_width=int((word_width - spacing) / 100 * image_width) if image_width else 0,
                        px_height=px_h
                    ))

                    current_x += word_width

        # 모든 element를 블록으로 생성 (table 포함)
        if text:
            blocks.append(TextBlock(
                text=text,
                block_type=elem_type,
                x=round(pct_x, 2),
                y=round(pct_y, 2),
                width=round(pct_w, 2),
                height=round(pct_h, 2)
            ))

    return words, blocks, tables


# ============================================================
# 6. 통합 파이프라인
# ============================================================

class UpstageOCRPipeline:
    """
    Upstage + GPT 기반 OCR 파이프라인

    1. Upstage Document Parse - 텍스트 추출 + 좌표
    2. GPT-4o-mini - 구조화
    3. Pydantic - 검증
    """

    def __init__(
        self,
        upstage_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        gpt_model: str = "gpt-4o-mini"
    ):
        self.upstage = UpstageClient(upstage_api_key)
        self.gpt = GPTStructurizer(openai_api_key, gpt_model)

    def process(
        self,
        image_path: str = None,
        image_bytes: bytes = None,
        image_width: int = None,
        image_height: int = None,
        structurize: bool = True
    ) -> OCROverlayResponse:
        """
        전체 OCR 파이프라인 실행

        Args:
            image_path: 이미지 경로
            image_bytes: 이미지 바이트
            image_width: 이미지 너비 (미리 알고 있는 경우)
            image_height: 이미지 높이
            structurize: GPT 구조화 수행 여부

        Returns:
            OCROverlayResponse
        """
        import time
        start_time = time.time()

        response = OCROverlayResponse(
            image_width=image_width or 0,
            image_height=image_height or 0
        )

        try:
            # 1. Upstage 파싱
            logger.info("1. Upstage Document Parse 시작")
            parse_result = self.upstage.parse(
                image_path=image_path,
                image_bytes=image_bytes,
                ocr=True,
                coordinates=True,
                output_formats=["markdown", "text"]
            )

            # 디버깅: Upstage 응답 구조 확인
            logger.info(f"Upstage 응답 키: {list(parse_result.keys())}")

            # 이미지 크기 추출
            if not image_width or not image_height:
                meta = parse_result.get("metadata", {})
                response.image_width = meta.get("width", 0)
                response.image_height = meta.get("height", 0)

            # Markdown & 텍스트 추출
            content = parse_result.get("content", {})
            if isinstance(content, dict):
                response.markdown = content.get("markdown", "")
                response.full_text = content.get("text", "")
            else:
                # content가 문자열인 경우
                response.markdown = str(content) if content else ""
                response.full_text = str(content) if content else ""

            # Document OCR API 호출 - 단어별 정확한 좌표 획득
            logger.info("2. Upstage Document OCR 시작 (단어별 좌표)")
            try:
                ocr_result = self.upstage.ocr(
                    image_path=image_path,
                    image_bytes=image_bytes,
                )
                ocr_words = parse_ocr_words(
                    ocr_result,
                    response.image_width,
                    response.image_height
                )
                response.words = ocr_words
                logger.info(f"Document OCR 완료: {len(ocr_words)} words")
            except Exception as ocr_err:
                logger.warning(f"Document OCR 실패 (Document Parse로 fallback): {ocr_err}")
                # fallback: Document Parse elements에서 블록 추출
                elements = parse_result.get("elements", [])
                _, blocks, _ = parse_upstage_elements(
                    elements, response.image_width, response.image_height
                )
                response.blocks = blocks

            # 2. GPT 구조화
            if structurize and response.markdown:
                logger.info("2. GPT 구조화 시작")
                contract = self.gpt.structurize(response.markdown)
                contract_dict = contract.model_dump() if hasattr(contract, 'model_dump') else contract
                
                # contract_dict가 dict인지 확인
                if isinstance(contract_dict, dict):
                    response.contract_data = contract_dict
                else:
                    logger.error(f"GPT 응답이 dict가 아닙니다: {type(contract_dict)}")
                    response.contract_data = {}
                
                logger.info("GPT 구조화 완료")

            # 3. 검증
            if response.contract_data:
                logger.info("3. 검증 시작")
                response.validation = self._validate(response.contract_data)

            response.processing_time = time.time() - start_time
            logger.info(f"처리 완료: {response.processing_time:.2f}초")

        except Exception as e:
            logger.error(f"파이프라인 오류: {e}")
            response.success = False
            response.error = str(e)

        return response

    def _validate(self, contract_data: Dict) -> Dict:
        """비즈니스 로직 검증"""
        warnings = []
        errors = []
        
        # contract_data가 dict가 아닌 경우 처리
        if not isinstance(contract_data, dict):
            logger.warning(f"contract_data가 dict가 아닙니다: {type(contract_data)}")
            return {"warnings": warnings, "errors": errors}

        terms = contract_data.get("terms", {})

        # 1. 금액 합계 검증
        deposit = terms.get("deposit", 0)
        down = terms.get("down_payment", 0)
        middle = terms.get("middle_payment", 0)
        balance = terms.get("balance", 0)

        if deposit > 0 and (down + middle + balance) > 0:
            total = down + middle + balance
            if total != deposit:
                warnings.append({
                    "field": "deposit",
                    "message": f"보증금({deposit:,}원)과 계약금+중도금+잔금 합계({total:,}원)가 일치하지 않습니다."
                })

        # 2. 날짜 유효성
        start_date = terms.get("contract_start_date", "")
        end_date = terms.get("contract_end_date", "")

        if start_date and end_date:
            if start_date > end_date:
                errors.append({
                    "field": "contract_date",
                    "message": "계약 시작일이 종료일보다 늦습니다."
                })

        # 3. 필수 필드 체크
        required_fields = [
            ("lessor.name", "임대인 이름"),
            ("lessee.name", "임차인 이름"),
            ("property.address", "부동산 소재지"),
            ("terms.deposit", "보증금"),
        ]

        for path, label in required_fields:
            parts = path.split(".")
            value = contract_data
            for part in parts:
                # dict가 아니면 None으로 설정하고 루프 탈출
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(part, None)

            if not value:
                warnings.append({
                    "field": path,
                    "message": f"{label}이(가) 누락되었습니다."
                })

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    # 환경변수 필요:
    # - UPSTAGE_API_KEY
    # - OPENAI_API_KEY

    pipeline = UpstageOCRPipeline()

    # 테스트 이미지
    test_image = "data/실제계약서이미지/계약서.png"

    if Path(test_image).exists():
        result = pipeline.process(image_path=test_image)

        print(f"\n=== OCR 결과 ===")
        print(f"처리 시간: {result.processing_time:.2f}초")
        print(f"이미지 크기: {result.image_width}x{result.image_height}")
        print(f"단어 수: {len(result.words)}")
        print(f"표 수: {len(result.tables)}")

        if result.contract_data:
            print(f"\n=== 구조화된 데이터 ===")
            print(json.dumps(result.contract_data, ensure_ascii=False, indent=2))

        if result.validation:
            print(f"\n=== 검증 결과 ===")
            print(json.dumps(result.validation, ensure_ascii=False, indent=2))
    else:
        print(f"테스트 이미지 없음: {test_image}")
