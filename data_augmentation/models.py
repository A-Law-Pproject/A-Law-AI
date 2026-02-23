"""
데이터 증강 모듈 데이터 모델 정의
- 텍스트 어노테이션
- 필드 설정
- 합성 결과
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from PIL import Image


@dataclass
class GeneratedText:
    """Faker로 생성된 텍스트 정보"""
    text: str
    field_type: str  # name, amount, date, address, phone, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FieldConfig:
    """필드 설정 (렌더링 위치 및 스타일)"""
    name: str
    field_type: str
    x: int
    y: int
    width: int
    height: int
    font_size: int = 14
    align: str = "left"  # left, center, right
    color: Tuple[int, int, int] = (0, 0, 0)


@dataclass
class FieldAnnotation:
    """필드 어노테이션 (학습 라벨용)"""
    field_name: str
    field_type: str
    text: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    is_toxic: bool = False  # 독소조항 여부


@dataclass
class TextAnnotation:
    """텍스트 어노테이션 (레거시 호환용)"""
    text: str
    x: int
    y: int
    width: int
    height: int
    field_type: str
    font_name: str
    font_size: int
    rotation: float


@dataclass
class SynthesizedImage:
    """합성된 이미지 정보"""
    image: Image.Image
    annotations: List[FieldAnnotation]
    source_template: str
    has_toxic: bool = False


@dataclass
class TemplateDefinition:
    """PDF 템플릿 정의"""
    name: str
    pdf_path: str
    image_path: str
    width: int
    height: int
    fields: List[FieldConfig]


@dataclass
class FieldDefinition:
    """필드 정의 (PDF 템플릿용, 레거시 호환)"""
    name: str
    field_type: str
    x: int
    y: int
    width: int
    height: int
    font_size: int = 16
    align: str = "left"
