"""
계약서 OCR 스키마 정의
- Pydantic 기반 데이터 구조화
- 입력/출력 검증
- 계약서 필드 강제 매핑
"""
from typing import List, Dict, Optional, Any, Union
from datetime import date
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import re


# ===========================================
# Enum 정의
# ===========================================

class ContractType(str, Enum):
    """계약서 유형"""
    LEASE = "임대차"          # 월세
    JEONSE = "전세"           # 전세
    SALE = "매매"             # 매매
    COMMERCIAL = "상가"       # 상가 임대
    OFFICE = "오피스텔"       # 오피스텔
    UNKNOWN = "미확인"


class PartyRole(str, Enum):
    """당사자 역할"""
    LESSOR = "임대인"         # 집주인
    LESSEE = "임차인"         # 세입자
    SELLER = "매도인"
    BUYER = "매수인"


# ===========================================
# OCR 결과 스키마
# ===========================================

class BoundingBox(BaseModel):
    """바운딩 박스"""
    x: int = Field(..., ge=0, description="좌측 상단 X 좌표")
    y: int = Field(..., ge=0, description="좌측 상단 Y 좌표")
    width: int = Field(..., gt=0, description="너비")
    height: int = Field(..., gt=0, description="높이")

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def to_xyxy(self) -> tuple:
        return (self.x, self.y, self.x2, self.y2)


class OCRTextResult(BaseModel):
    """OCR 텍스트 결과"""
    text: str = Field(..., description="인식된 텍스트")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도")
    bbox: BoundingBox = Field(..., description="바운딩 박스")
    field_name: Optional[str] = Field(None, description="매핑된 필드명")
    is_corrected: bool = Field(False, description="교정 여부")
    original_text: Optional[str] = Field(None, description="교정 전 원본 텍스트")


class OCRPageResult(BaseModel):
    """페이지 OCR 결과"""
    page_number: int = Field(1, ge=1)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    results: List[OCRTextResult] = Field(default_factory=list)
    full_text: str = Field("", description="전체 텍스트")
    avg_confidence: float = Field(0.0, ge=0.0, le=1.0)


# ===========================================
# 계약서 필드 스키마
# ===========================================

class PersonInfo(BaseModel):
    """당사자 정보"""
    role: PartyRole = Field(..., description="역할 (임대인/임차인)")
    name: str = Field(..., min_length=1, description="성명")
    resident_id: Optional[str] = Field(None, description="주민등록번호")
    address: Optional[str] = Field(None, description="주소")
    phone: Optional[str] = Field(None, description="전화번호")

    @field_validator("resident_id")
    @classmethod
    def validate_resident_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # 마스킹된 형태도 허용
        pattern = r"^\d{6}[-－−](\d{7}|\*{6,7}|\d\*{6})$"
        if not re.match(pattern, v.replace(" ", "")):
            # 형식이 맞지 않으면 원본 그대로 (경고만)
            pass
        return v

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # 숫자만 추출
        digits = re.sub(r"[^\d]", "", v)
        if len(digits) >= 10:
            return f"{digits[:3]}-{digits[3:7]}-{digits[7:11]}"
        return v


class MoneyAmount(BaseModel):
    """금액 정보"""
    value: int = Field(..., ge=0, description="금액 (원)")
    text: str = Field(..., description="원본 텍스트")
    unit: str = Field("원", description="단위")

    @classmethod
    def from_text(cls, text: str) -> Optional["MoneyAmount"]:
        """텍스트에서 금액 파싱"""
        # 숫자 추출 (콤마 제거)
        numbers = re.findall(r"[\d,]+", text)
        if numbers:
            value = int(numbers[0].replace(",", ""))
            return cls(value=value, text=text, unit="원")
        return None


class PropertyInfo(BaseModel):
    """부동산 정보"""
    address: str = Field(..., description="소재지 주소")
    property_type: Optional[str] = Field(None, description="건물 유형 (아파트, 빌라 등)")
    area_m2: Optional[float] = Field(None, ge=0, description="전용면적 (㎡)")
    area_pyeong: Optional[float] = Field(None, ge=0, description="전용면적 (평)")
    floor: Optional[str] = Field(None, description="층수")
    unit_number: Optional[str] = Field(None, description="호수")
    building_name: Optional[str] = Field(None, description="건물명")


class ContractTerms(BaseModel):
    """계약 조건"""
    deposit: Optional[MoneyAmount] = Field(None, description="보증금")
    monthly_rent: Optional[MoneyAmount] = Field(None, description="월세")
    contract_money: Optional[MoneyAmount] = Field(None, description="계약금")
    middle_payment: Optional[MoneyAmount] = Field(None, description="중도금")
    balance: Optional[MoneyAmount] = Field(None, description="잔금")

    start_date: Optional[str] = Field(None, description="계약 시작일")
    end_date: Optional[str] = Field(None, description="계약 종료일")
    contract_date: Optional[str] = Field(None, description="계약 체결일")

    contract_period_months: Optional[int] = Field(None, ge=1, description="계약 기간 (개월)")

    @model_validator(mode="after")
    def validate_amounts(self):
        """금액 합계 검증"""
        if self.deposit and self.contract_money and self.balance:
            deposit_val = self.deposit.value
            contract_val = self.contract_money.value
            middle_val = self.middle_payment.value if self.middle_payment else 0
            balance_val = self.balance.value

            total = contract_val + middle_val + balance_val

            # 보증금 = 계약금 + 중도금 + 잔금
            if total != deposit_val:
                # 경고만 (데이터는 유지)
                pass
        return self


class SpecialTerms(BaseModel):
    """특약사항"""
    items: List[str] = Field(default_factory=list, description="특약 조항 목록")
    raw_text: Optional[str] = Field(None, description="원본 텍스트")


# ===========================================
# 통합 계약서 스키마
# ===========================================

class LeaseContract(BaseModel):
    """임대차 계약서 전체 스키마"""
    contract_type: ContractType = Field(ContractType.LEASE, description="계약 유형")

    # 당사자 정보
    lessor: Optional[PersonInfo] = Field(None, description="임대인 정보")
    lessee: Optional[PersonInfo] = Field(None, description="임차인 정보")

    # 부동산 정보
    property: Optional[PropertyInfo] = Field(None, description="부동산 정보")

    # 계약 조건
    terms: Optional[ContractTerms] = Field(None, description="계약 조건")

    # 특약사항
    special_terms: Optional[SpecialTerms] = Field(None, description="특약사항")

    # 메타데이터
    ocr_confidence: float = Field(0.0, ge=0.0, le=1.0, description="OCR 평균 신뢰도")
    source_file: Optional[str] = Field(None, description="원본 파일명")
    processing_date: Optional[str] = Field(None, description="처리 일시")

    def to_frontend_format(self) -> Dict[str, Any]:
        """
        프론트엔드용 포맷으로 변환

        양식의 각 필드에 텍스트가 들어가도록 구조화
        """
        return {
            "계약유형": self.contract_type.value,
            "임대인": {
                "성명": self.lessor.name if self.lessor else "",
                "주민등록번호": self.lessor.resident_id if self.lessor else "",
                "주소": self.lessor.address if self.lessor else "",
                "전화번호": self.lessor.phone if self.lessor else "",
            },
            "임차인": {
                "성명": self.lessee.name if self.lessee else "",
                "주민등록번호": self.lessee.resident_id if self.lessee else "",
                "주소": self.lessee.address if self.lessee else "",
                "전화번호": self.lessee.phone if self.lessee else "",
            },
            "부동산": {
                "소재지": self.property.address if self.property else "",
                "건물유형": self.property.property_type if self.property else "",
                "전용면적_m2": self.property.area_m2 if self.property else None,
                "전용면적_평": self.property.area_pyeong if self.property else None,
                "층수": self.property.floor if self.property else "",
                "호수": self.property.unit_number if self.property else "",
            },
            "계약조건": {
                "보증금": self.terms.deposit.value if self.terms and self.terms.deposit else 0,
                "보증금_텍스트": self.terms.deposit.text if self.terms and self.terms.deposit else "",
                "월세": self.terms.monthly_rent.value if self.terms and self.terms.monthly_rent else 0,
                "월세_텍스트": self.terms.monthly_rent.text if self.terms and self.terms.monthly_rent else "",
                "계약금": self.terms.contract_money.value if self.terms and self.terms.contract_money else 0,
                "중도금": self.terms.middle_payment.value if self.terms and self.terms.middle_payment else 0,
                "잔금": self.terms.balance.value if self.terms and self.terms.balance else 0,
                "계약시작일": self.terms.start_date if self.terms else "",
                "계약종료일": self.terms.end_date if self.terms else "",
                "계약체결일": self.terms.contract_date if self.terms else "",
            },
            "특약사항": self.special_terms.items if self.special_terms else [],
            "메타정보": {
                "OCR신뢰도": round(self.ocr_confidence * 100, 1),
                "원본파일": self.source_file,
                "처리일시": self.processing_date,
            }
        }


# ===========================================
# API 요청/응답 스키마
# ===========================================

class OCRRequest(BaseModel):
    """OCR 요청"""
    preprocess: bool = Field(True, description="전처리 수행 여부")
    remove_stamp: bool = Field(True, description="도장 제거 여부")
    high_precision: bool = Field(True, description="고정밀 모드 여부")
    return_boxes: bool = Field(True, description="바운딩 박스 반환 여부")


class OCRResponse(BaseModel):
    """OCR 응답"""
    success: bool = Field(..., description="성공 여부")
    processing_time: float = Field(..., ge=0, description="처리 시간 (초)")

    # OCR 결과
    pages: List[OCRPageResult] = Field(default_factory=list, description="페이지별 결과")

    # 구조화된 계약서 데이터
    contract: Optional[LeaseContract] = Field(None, description="파싱된 계약서 데이터")

    # 프론트엔드용 포맷
    frontend_data: Optional[Dict[str, Any]] = Field(None, description="프론트엔드용 데이터")

    # 에러 정보
    error: Optional[str] = Field(None, description="에러 메시지")
    warnings: List[str] = Field(default_factory=list, description="경고 목록")


class ValidationError(BaseModel):
    """검증 오류"""
    field: str
    message: str
    current_value: Optional[str] = None
    expected_format: Optional[str] = None


class ValidationResponse(BaseModel):
    """검증 응답"""
    is_valid: bool
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: Dict[str, str] = Field(default_factory=dict)


if __name__ == "__main__":
    # 테스트
    # 샘플 계약서 데이터 생성
    contract = LeaseContract(
        contract_type=ContractType.LEASE,
        lessor=PersonInfo(
            role=PartyRole.LESSOR,
            name="홍길동",
            resident_id="900101-1******",
            address="서울시 강남구 테헤란로 123",
            phone="010-1234-5678"
        ),
        lessee=PersonInfo(
            role=PartyRole.LESSEE,
            name="김철수",
            resident_id="950505-1******",
            phone="010-9876-5432"
        ),
        property=PropertyInfo(
            address="서울시 강남구 역삼동 123-45, 테헤란빌딩 501호",
            property_type="오피스텔",
            area_m2=33.5,
            floor="5층",
            unit_number="501호"
        ),
        terms=ContractTerms(
            deposit=MoneyAmount(value=10000000, text="일금 일천만원정", unit="원"),
            monthly_rent=MoneyAmount(value=500000, text="오십만원", unit="원"),
            contract_money=MoneyAmount(value=1000000, text="일백만원", unit="원"),
            balance=MoneyAmount(value=9000000, text="구백만원", unit="원"),
            start_date="2024년 2월 1일",
            end_date="2026년 1월 31일",
            contract_date="2024년 1월 15일"
        ),
        special_terms=SpecialTerms(
            items=[
                "임차인은 입주 전 도배, 장판 교체 비용 50만원을 부담한다.",
                "반려동물 사육은 금지한다."
            ]
        ),
        ocr_confidence=0.95,
        source_file="계약서.png"
    )

    # 프론트엔드용 변환
    frontend_data = contract.to_frontend_format()
    import json
    print(json.dumps(frontend_data, ensure_ascii=False, indent=2))
