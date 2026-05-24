"""
Shared masking patterns for text and OCR-word image masking.
"""
from __future__ import annotations

import re


def _spaced_term(term: str) -> str:
    return r"\s*".join(re.escape(char) for char in term if not char.isspace())


def _label_union(*groups: list[str]) -> str:
    values: list[str] = []
    for group in groups:
        values.extend(group)
    return "(?:" + "|".join(_spaced_term(value) for value in values) + ")"


NAME_FIELD_LABELS = [
    "성명",
    "이름",
    "대표자",
    "예금주",
    "임대인",
    "임차인",
    "소유자",
    "대리인",
    "위임인",
    "수임인",
    "신청인",
    "계약자",
    "채권자",
    "채무자",
]
ADDRESS_FIELD_LABELS = [
    "주소",
    "소재지",
    "거주지",
    "현주소",
    "본점소재지",
]
PHONE_FIELD_LABELS = [
    "휴대전화",
    "휴대폰",
    "핸드폰",
    "연락처",
    "전화번호",
    "휴대전화번호",
    "팩스",
    "FAX",
]
EMAIL_FIELD_LABELS = [
    "이메일",
    "전자우편",
    "메일주소",
    "email",
    "e-mail",
]
RESIDENT_FIELD_LABELS = [
    "주민등록번호",
    "주민번호",
    "외국인등록번호",
]
ACCOUNT_FIELD_LABELS = [
    "계좌번호",
    "입금계좌",
    "반환계좌",
    "예금계좌",
    "은행계좌",
    "계좌정보",
    "계좌",
]
BUSINESS_NO_FIELD_LABELS = [
    "사업자등록번호",
]
CORPORATE_NO_FIELD_LABELS = [
    "법인등록번호",
]
BIRTH_DATE_FIELD_LABELS = [
    "생년월일",
    "출생일자",
    "생년월일자",
]
PASSPORT_FIELD_LABELS = [
    "여권번호",
]
DRIVER_LICENSE_FIELD_LABELS = [
    "운전면허번호",
    "면허번호",
]

ALL_FIELD_LABEL_PATTERN = _label_union(
    NAME_FIELD_LABELS,
    ADDRESS_FIELD_LABELS,
    PHONE_FIELD_LABELS,
    EMAIL_FIELD_LABELS,
    RESIDENT_FIELD_LABELS,
    ACCOUNT_FIELD_LABELS,
    BUSINESS_NO_FIELD_LABELS,
    CORPORATE_NO_FIELD_LABELS,
    BIRTH_DATE_FIELD_LABELS,
    PASSPORT_FIELD_LABELS,
    DRIVER_LICENSE_FIELD_LABELS,
)


def _compile_label_value_pattern(labels: list[str]) -> re.Pattern[str]:
    label_pattern = _label_union(labels)
    return re.compile(
        rf"(?P<label>{label_pattern})\s*[:：]?\s*(?P<value>.+?)"
        rf"(?=(?:\s+(?:{ALL_FIELD_LABEL_PATTERN}))|$)",
        re.IGNORECASE | re.MULTILINE,
    )


NAME_FIELD_PATTERN = _compile_label_value_pattern(NAME_FIELD_LABELS)
ADDRESS_FIELD_PATTERN = _compile_label_value_pattern(ADDRESS_FIELD_LABELS)
PHONE_FIELD_PATTERN = _compile_label_value_pattern(PHONE_FIELD_LABELS)
EMAIL_FIELD_PATTERN = _compile_label_value_pattern(EMAIL_FIELD_LABELS)
RESIDENT_FIELD_PATTERN = _compile_label_value_pattern(RESIDENT_FIELD_LABELS)
ACCOUNT_FIELD_PATTERN = _compile_label_value_pattern(ACCOUNT_FIELD_LABELS)
BUSINESS_NO_FIELD_PATTERN = _compile_label_value_pattern(BUSINESS_NO_FIELD_LABELS)
CORPORATE_NO_FIELD_PATTERN = _compile_label_value_pattern(CORPORATE_NO_FIELD_LABELS)
BIRTH_DATE_FIELD_PATTERN = _compile_label_value_pattern(BIRTH_DATE_FIELD_LABELS)
PASSPORT_FIELD_PATTERN = _compile_label_value_pattern(PASSPORT_FIELD_LABELS)
DRIVER_LICENSE_FIELD_PATTERN = _compile_label_value_pattern(DRIVER_LICENSE_FIELD_LABELS)

RESIDENT_ID_PATTERN = re.compile(r"\b(\d{6})\s*[- ]?\s*([1-8])\s*(\d{6})\b")
PHONE_PATTERN = re.compile(
    r"\b(0(?:1[016789]|2|[3-9][0-9]?))\s*[- ]?\s*(\d{3,4})\s*[- ]?\s*(\d{4})\b"
)
EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b", re.IGNORECASE)
BUSINESS_NO_PATTERN = re.compile(r"\b(\d{3})\s*[- ]?\s*(\d{2})\s*[- ]?\s*(\d{5})\b")
CORPORATE_NO_PATTERN = re.compile(r"\b(\d{6})\s*[- ]?\s*(\d{7})\b")
BIRTH_DATE_PATTERN = re.compile(
    r"\b(?:19|20)?\d{2}\s*[./-]\s*(?:0?[1-9]|1[0-2])\s*[./-]\s*(?:0?[1-9]|[12]\d|3[01])\b|\b\d{8}\b"
)
ACCOUNT_VALUE_PATTERN = re.compile(r"\b\d{2,6}(?:[- ]?\d{2,6}){2,4}\b")
BANK_ACCOUNT_PATTERN = re.compile(
    r"\b(?:국민|신한|하나|우리|농협|기업|카카오뱅크|토스뱅크|새마을금고|수협|SC제일|우체국|대구|부산|광주|전북|경남|제주)"
    r"(?:은행|뱅크)?\s*\d{2,6}(?:[- ]?\d{2,6}){2,4}\b"
)
PASSPORT_VALUE_PATTERN = re.compile(r"\b[A-Z]{1,2}\d{7,8}\b", re.IGNORECASE)
DRIVER_LICENSE_VALUE_PATTERN = re.compile(r"\b\d{2}-\d{2}-\d{6}-\d{2}\b|\b\d{12}\b")

STANDALONE_NAME_PATTERN = re.compile(
    r"^\s*(?:[가-힣]{2,4}|[A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){0,2})\s*$"
)
ADDRESS_FRAGMENT_PATTERN = re.compile(
    r"(?:(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)"
    r"(?:특별시|광역시|특별자치시|특별자치도|도|시)?)\s+"
    r"[^\n]{2,80}?(?:시|군|구)\s+[^\n]{1,40}?(?:읍|면|동|리|로|길)[^\n]{0,60}\d+(?:-\d+)?(?:\([^)]+\))?"
)

SEAL_KEYWORDS = ("도장", "서명", "인장", "날인", "사인", "인")
