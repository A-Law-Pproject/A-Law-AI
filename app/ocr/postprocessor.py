# ============================================================
# [LEGACY] 이 파일은 더 이상 사용되지 않습니다.
# PaddleOCR용 후처리로, Upstage OCR로 대체되었습니다.
# TODO: 안정화 후 삭제 예정
# ============================================================
"""
OCR 후처리 모듈 (LEGACY - Upstage OCR로 대체됨)
- 오타 교정 (계약서 용어 사전 기반)
- 정규식 검증 (금액, 주민번호, 날짜 등)
- 문맥 기반 교정
"""
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from loguru import logger


# ===========================================
# 계약서 용어 사전
# ===========================================

CONTRACT_TERMS = {
    # 기본 용어
    "임대인": ["임대인", "임디인", "임래인", "암대인"],
    "임차인": ["임차인", "임치인", "임쟈인", "암차인"],
    "보증금": ["보증금", "보즈금", "보증굼", "보중금"],
    "월세": ["월세", "월쎄", "월새", "뭘세"],
    "전세": ["전세", "전쎄", "전새"],
    "계약금": ["계약금", "게약금", "계악금", "겨약금"],
    "중도금": ["중도금", "중도굼", "줄도금"],
    "잔금": ["잔금", "잔굼", "간금"],
    "계약": ["계약", "게약", "계악", "겨약"],
    "임대차": ["임대차", "임대쟈", "입대차"],
    "주택": ["주택", "주댁", "쥬택"],
    "아파트": ["아파트", "아팥트", "아파투"],
    "오피스텔": ["오피스텔", "오피스탤", "오피스텔"],
    "상가": ["상가", "상까"],
    "건물": ["건물", "건믈", "건볼"],
    "면적": ["면적", "면젹", "면적"],
    "평방미터": ["평방미터", "㎡", "m2"],

    # 법적 용어
    "특약사항": ["특약사항", "특약시항", "특약상항"],
    "계약기간": ["계약기간", "게약기간", "계약기깐"],
    "갱신": ["갱신", "갠신", "겡신"],
    "해지": ["해지", "해자", "헤지"],
    "위약금": ["위약금", "위약굼", "휘약금"],
    "소유자": ["소유자", "소유져", "소유쟈"],
    "등기부등본": ["등기부등본", "등기부동본"],
    "확정일자": ["확정일자", "화정일자", "확정일져"],

    # 금액 단위
    "원정": ["원정", "원쩡", "원졍"],
    "일금": ["일금", "일굼", "잃금"],
    "만원": ["만원", "만원"],
    "천원": ["천원", "천원"],
    "백원": ["백원", "백원"],

    # 한글 숫자
    "일천만": ["일천만", "일쳔만"],
    "이천만": ["이천만", "이쳔만"],
    "삼천만": ["삼천만", "삼쳔만"],
    "사천만": ["사천만", "사쳔만"],
    "오천만": ["오천만", "오쳔만"],
    "육천만": ["육천만", "육쳔만"],
    "칠천만": ["칠천만", "칠쳔만"],
    "팔천만": ["팔천만", "팔쳔만"],
    "구천만": ["구천만", "구쳔만"],
}

# 오타 → 정답 매핑 생성
TYPO_CORRECTIONS = {}
for correct, variants in CONTRACT_TERMS.items():
    for variant in variants:
        if variant != correct:
            TYPO_CORRECTIONS[variant] = correct


# ===========================================
# 정규식 패턴
# ===========================================

PATTERNS = {
    # 금액 패턴 (숫자 + 원/만원/천원)
    "money_numeric": re.compile(r"(\d{1,3}(,\d{3})*)\s*(원|만원|천원)"),

    # 금액 패턴 (한글)
    "money_korean": re.compile(
        r"(일금\s*)?([일이삼사오육칠팔구십백천만억조]+)\s*(원정?|원)"
    ),

    # 주민등록번호 (뒷자리 마스킹 포함)
    "resident_id": re.compile(
        r"(\d{6})\s*[-－−]\s*(\d{7}|\*{6,7}|\d\*{6})"
    ),

    # 날짜 패턴
    "date_ymd": re.compile(
        r"(\d{4})\s*[년./-]\s*(\d{1,2})\s*[월./-]\s*(\d{1,2})\s*일?"
    ),
    "date_korean": re.compile(
        r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"
    ),

    # 전화번호
    "phone": re.compile(
        r"(0\d{1,2})[-－−.\s]?(\d{3,4})[-－−.\s]?(\d{4})"
    ),

    # 주소
    "address": re.compile(
        r"([가-힣]+(?:시|도))\s*([가-힣]+(?:구|군|시))\s*([가-힣]+(?:동|읍|면|로|길))"
    ),

    # 면적 (평/㎡)
    "area": re.compile(
        r"(\d+(?:\.\d+)?)\s*(㎡|평|제곱미터|m2)"
    ),

    # 층수
    "floor": re.compile(
        r"(\d+|지하\s*\d+)\s*층"
    ),

    # 호수
    "unit": re.compile(
        r"(\d+)\s*호"
    ),
}


@dataclass
class CorrectionResult:
    """교정 결과"""
    original: str
    corrected: str
    corrections: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ValidationResult:
    """검증 결과"""
    field_name: str
    value: str
    is_valid: bool
    expected_pattern: str
    message: str = ""


class OCRPostProcessor:
    """OCR 후처리기"""

    def __init__(
        self,
        enable_typo_correction: bool = True,
        enable_pattern_validation: bool = True,
        custom_terms: Optional[Dict[str, List[str]]] = None
    ):
        self.enable_typo_correction = enable_typo_correction
        self.enable_pattern_validation = enable_pattern_validation

        # 사용자 정의 용어 추가
        if custom_terms:
            for correct, variants in custom_terms.items():
                for variant in variants:
                    if variant != correct:
                        TYPO_CORRECTIONS[variant] = correct

    def correct_typos(self, text: str) -> CorrectionResult:
        """
        오타 교정

        계약서 용어 사전 기반으로 오타를 교정합니다.
        """
        if not self.enable_typo_correction:
            return CorrectionResult(original=text, corrected=text)

        corrected = text
        corrections = []

        # 용어 사전 기반 교정
        for typo, correct in TYPO_CORRECTIONS.items():
            if typo in corrected:
                corrected = corrected.replace(typo, correct)
                corrections.append({"from": typo, "to": correct})

        # 일반적인 OCR 오류 패턴 교정
        ocr_fixes = [
            (r"[oO0]", "0"),  # O/o → 0 (숫자 문맥에서)
            (r"[lI1]", "1"),  # l/I → 1 (숫자 문맥에서)
            (r"[\u3000\xa0]", " "),  # 전각 공백 → 반각 공백
        ]

        # 숫자 필드에서만 적용
        for pattern, replacement in ocr_fixes:
            # 금액 패턴 내에서만 적용
            if PATTERNS["money_numeric"].search(corrected):
                corrected = re.sub(pattern, replacement, corrected)

        return CorrectionResult(
            original=text,
            corrected=corrected,
            corrections=corrections,
            confidence=1.0 if not corrections else 0.9
        )

    def validate_field(
        self,
        field_name: str,
        value: str
    ) -> ValidationResult:
        """
        필드별 패턴 검증

        Args:
            field_name: 필드 이름 (보증금, 월세, 주민번호, 날짜 등)
            value: 필드 값

        Returns:
            ValidationResult
        """
        value = value.strip()

        # 필드 타입별 검증
        if field_name in ["보증금", "월세", "계약금", "잔금", "중도금", "금액"]:
            return self._validate_money(field_name, value)

        elif field_name in ["주민등록번호", "주민번호"]:
            return self._validate_resident_id(field_name, value)

        elif field_name in ["계약일", "시작일", "종료일", "날짜"]:
            return self._validate_date(field_name, value)

        elif field_name in ["전화번호", "연락처", "휴대폰"]:
            return self._validate_phone(field_name, value)

        elif field_name in ["면적", "전용면적", "공급면적"]:
            return self._validate_area(field_name, value)

        elif field_name in ["주소", "소재지"]:
            return self._validate_address(field_name, value)

        # 기본: 빈 값 체크만
        return ValidationResult(
            field_name=field_name,
            value=value,
            is_valid=bool(value),
            expected_pattern="any",
            message="" if value else "빈 값입니다."
        )

    def _validate_money(self, field_name: str, value: str) -> ValidationResult:
        """금액 필드 검증"""
        # 숫자 + 원 패턴
        if PATTERNS["money_numeric"].search(value):
            return ValidationResult(
                field_name=field_name,
                value=value,
                is_valid=True,
                expected_pattern="숫자+원"
            )

        # 한글 금액 패턴
        if PATTERNS["money_korean"].search(value):
            return ValidationResult(
                field_name=field_name,
                value=value,
                is_valid=True,
                expected_pattern="한글금액"
            )

        return ValidationResult(
            field_name=field_name,
            value=value,
            is_valid=False,
            expected_pattern="금액 형식 (예: 1,000,000원 또는 일금일백만원정)",
            message="금액 형식이 올바르지 않습니다."
        )

    def _validate_resident_id(self, field_name: str, value: str) -> ValidationResult:
        """주민등록번호 검증"""
        match = PATTERNS["resident_id"].search(value)
        if match:
            return ValidationResult(
                field_name=field_name,
                value=value,
                is_valid=True,
                expected_pattern="주민등록번호"
            )

        return ValidationResult(
            field_name=field_name,
            value=value,
            is_valid=False,
            expected_pattern="주민등록번호 (예: 000000-0000000)",
            message="주민등록번호 형식이 올바르지 않습니다."
        )

    def _validate_date(self, field_name: str, value: str) -> ValidationResult:
        """날짜 검증"""
        if PATTERNS["date_ymd"].search(value) or PATTERNS["date_korean"].search(value):
            return ValidationResult(
                field_name=field_name,
                value=value,
                is_valid=True,
                expected_pattern="날짜"
            )

        return ValidationResult(
            field_name=field_name,
            value=value,
            is_valid=False,
            expected_pattern="날짜 (예: 2024년 1월 1일)",
            message="날짜 형식이 올바르지 않습니다."
        )

    def _validate_phone(self, field_name: str, value: str) -> ValidationResult:
        """전화번호 검증"""
        if PATTERNS["phone"].search(value):
            return ValidationResult(
                field_name=field_name,
                value=value,
                is_valid=True,
                expected_pattern="전화번호"
            )

        return ValidationResult(
            field_name=field_name,
            value=value,
            is_valid=False,
            expected_pattern="전화번호 (예: 010-1234-5678)",
            message="전화번호 형식이 올바르지 않습니다."
        )

    def _validate_area(self, field_name: str, value: str) -> ValidationResult:
        """면적 검증"""
        if PATTERNS["area"].search(value):
            return ValidationResult(
                field_name=field_name,
                value=value,
                is_valid=True,
                expected_pattern="면적"
            )

        return ValidationResult(
            field_name=field_name,
            value=value,
            is_valid=False,
            expected_pattern="면적 (예: 84.5㎡ 또는 25.6평)",
            message="면적 형식이 올바르지 않습니다."
        )

    def _validate_address(self, field_name: str, value: str) -> ValidationResult:
        """주소 검증"""
        # 주소는 최소 길이만 체크 (너무 엄격하면 실패)
        if len(value) >= 10 and any(
            keyword in value for keyword in ["시", "도", "구", "동", "로", "길"]
        ):
            return ValidationResult(
                field_name=field_name,
                value=value,
                is_valid=True,
                expected_pattern="주소"
            )

        return ValidationResult(
            field_name=field_name,
            value=value,
            is_valid=False,
            expected_pattern="주소",
            message="주소가 너무 짧거나 형식이 올바르지 않습니다."
        )

    def process(
        self,
        text: str,
        field_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        전체 후처리 수행

        Args:
            text: OCR 결과 텍스트
            field_name: 필드 이름 (검증용)

        Returns:
            처리 결과 딕셔너리
        """
        # 1. 오타 교정
        correction = self.correct_typos(text)

        # 2. 패턴 검증 (필드명이 있는 경우)
        validation = None
        if field_name and self.enable_pattern_validation:
            validation = self.validate_field(field_name, correction.corrected)

        return {
            "original": text,
            "corrected": correction.corrected,
            "corrections": correction.corrections,
            "correction_confidence": correction.confidence,
            "validation": validation.is_valid if validation else None,
            "validation_message": validation.message if validation else None,
            "field_name": field_name
        }

    def process_batch(
        self,
        data: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        여러 필드 일괄 처리

        Args:
            data: {"필드명": "OCR 값", ...}

        Returns:
            {"필드명": {처리결과}, ...}
        """
        results = {}
        for field_name, value in data.items():
            results[field_name] = self.process(value, field_name)
        return results


class ContractFieldExtractor:
    """
    계약서 필드 추출기

    정규식을 사용하여 특정 필드 값을 추출합니다.
    """

    @staticmethod
    def extract_money(text: str) -> Optional[str]:
        """금액 추출"""
        # 숫자 형식
        match = PATTERNS["money_numeric"].search(text)
        if match:
            return match.group(0)

        # 한글 형식
        match = PATTERNS["money_korean"].search(text)
        if match:
            return match.group(0)

        return None

    @staticmethod
    def extract_date(text: str) -> Optional[str]:
        """날짜 추출"""
        match = PATTERNS["date_korean"].search(text)
        if match:
            return f"{match.group(1)}년 {match.group(2)}월 {match.group(3)}일"

        match = PATTERNS["date_ymd"].search(text)
        if match:
            return f"{match.group(1)}년 {match.group(2)}월 {match.group(3)}일"

        return None

    @staticmethod
    def extract_phone(text: str) -> Optional[str]:
        """전화번호 추출"""
        match = PATTERNS["phone"].search(text)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        return None

    @staticmethod
    def extract_resident_id(text: str) -> Optional[str]:
        """주민등록번호 추출"""
        match = PATTERNS["resident_id"].search(text)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        return None

    @staticmethod
    def extract_area(text: str) -> Optional[Tuple[float, str]]:
        """면적 추출 (값, 단위)"""
        match = PATTERNS["area"].search(text)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            return (value, unit)
        return None

    @staticmethod
    def korean_to_number(korean_text: str) -> Optional[int]:
        """한글 금액을 숫자로 변환"""
        korean_nums = {
            "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5,
            "육": 6, "칠": 7, "팔": 8, "구": 9, "십": 10,
            "백": 100, "천": 1000, "만": 10000, "억": 100000000
        }

        # 간단한 변환 (복잡한 경우 별도 구현 필요)
        result = 0
        current = 0

        for char in korean_text:
            if char in korean_nums:
                num = korean_nums[char]
                if num >= 10:  # 단위
                    if current == 0:
                        current = 1
                    current *= num
                    if num >= 10000:  # 만 이상은 바로 합산
                        result += current
                        current = 0
                else:  # 숫자
                    current = num

        result += current
        return result if result > 0 else None


if __name__ == "__main__":
    # 테스트
    processor = OCRPostProcessor()

    test_cases = [
        ("보즈금 일금 일천만원쩡", "보증금"),
        ("게약일: 2024년 1월 15일", "계약일"),
        ("010-1234-5678", "전화번호"),
        ("서울시 강남구 테헤란로 123", "주소"),
    ]

    for text, field in test_cases:
        result = processor.process(text, field)
        print(f"원본: {result['original']}")
        print(f"교정: {result['corrected']}")
        print(f"검증: {result['validation']}")
        print("-" * 50)
"""
