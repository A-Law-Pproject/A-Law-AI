"""
데이터 증강 모듈
- 텍스트 생성 (Faker 기반)
- 텍스트 렌더링 및 합성
- 계약서 합성 데이터 생성
- PDF 템플릿 관리
"""

# 핵심 클래스
from .text_generator import TextGenerator
from .data_synthesizer import DataSynthesizer
from .contract_synthesizer import ContractSynthesizer, ToxicClauseGenerator
from .pdf_template import PDFTemplateManager

# 텍스트 렌더링
from .text_renderer import FontManager, TextRenderer, ImageEffects

# 데이터 모델
from .models import (
    GeneratedText,
    FieldConfig,
    FieldAnnotation,
    TextAnnotation,
    SynthesizedImage,
    TemplateDefinition,
    FieldDefinition,
)

# 필드 정의
from .field_definitions import (
    COURT_LEASE_FIELDS,
    COURT_LEASE_CONTRACT_FIELDS,
    SPECIAL_TERMS_AREA,
)

# 상수
from .constants import (
    SYSTEM_FONTS,
    TEXT_COLORS,
    DEFAULT_FONT_SIZE,
    DEFAULT_DPI,
    DEFAULT_TOXIC_RATIO,
)

__all__ = [
    # 핵심 클래스
    "TextGenerator",
    "DataSynthesizer",
    "ContractSynthesizer",
    "ToxicClauseGenerator",
    "PDFTemplateManager",
    # 텍스트 렌더링
    "FontManager",
    "TextRenderer",
    "ImageEffects",
    # 데이터 모델
    "GeneratedText",
    "FieldConfig",
    "FieldAnnotation",
    "TextAnnotation",
    "SynthesizedImage",
    "TemplateDefinition",
    "FieldDefinition",
    # 필드 정의
    "COURT_LEASE_FIELDS",
    "COURT_LEASE_CONTRACT_FIELDS",
    "SPECIAL_TERMS_AREA",
    # 상수
    "SYSTEM_FONTS",
    "TEXT_COLORS",
    "DEFAULT_FONT_SIZE",
    "DEFAULT_DPI",
    "DEFAULT_TOXIC_RATIO",
]
