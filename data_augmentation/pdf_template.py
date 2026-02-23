"""
PDF 양식 템플릿 처리 모듈
- PDF를 이미지로 변환
- 필드 위치 정의 및 관리
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import asdict
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from loguru import logger

from .constants import DEFAULT_DPI
from .models import FieldDefinition, TemplateDefinition
from .field_definitions import COURT_LEASE_CONTRACT_FIELDS


class PDFTemplateManager:
    """PDF 템플릿 관리자"""

    def __init__(self, templates_dir: str = "data/계약서양식"):
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, TemplateDefinition] = {}

    def pdf_to_image(
        self,
        pdf_path: str,
        output_path: str = None,
        dpi: int = DEFAULT_DPI,
        page_num: int = 0
    ) -> Image.Image:
        """
        PDF를 이미지로 변환

        Args:
            pdf_path: PDF 파일 경로
            output_path: 저장할 이미지 경로 (None이면 저장 안함)
            dpi: 해상도
            page_num: 변환할 페이지 번호

        Returns:
            PIL Image
        """
        if fitz is None:
            raise ImportError("PyMuPDF가 설치되지 않았습니다. pip install pymupdf")

        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)

        # 해상도 설정
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # PIL Image로 변환
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if output_path:
            img.save(output_path)
            logger.info(f"PDF 변환 완료: {output_path} ({img.width}x{img.height})")

        doc.close()
        return img

    def register_template(
        self,
        name: str,
        pdf_path: str,
        fields: List[FieldDefinition],
        dpi: int = DEFAULT_DPI
    ) -> TemplateDefinition:
        """
        템플릿 등록

        Args:
            name: 템플릿 이름
            pdf_path: PDF 파일 경로
            fields: 필드 정의 리스트
            dpi: 변환 해상도
        """
        pdf_path = Path(pdf_path)

        # 이미지 변환
        image_dir = self.templates_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{name}.png"

        img = self.pdf_to_image(str(pdf_path), str(image_path), dpi)

        template = TemplateDefinition(
            name=name,
            pdf_path=str(pdf_path),
            image_path=str(image_path),
            width=img.width,
            height=img.height,
            fields=fields
        )

        self.templates[name] = template
        logger.info(f"템플릿 등록: {name} (필드 {len(fields)}개)")

        return template

    def save_template_config(self, name: str, output_path: str = None):
        """템플릿 설정 저장"""
        if name not in self.templates:
            raise ValueError(f"템플릿 '{name}'이 등록되지 않았습니다.")

        template = self.templates[name]
        output_path = output_path or str(self.templates_dir / f"{name}_config.json")

        config = {
            "name": template.name,
            "pdf_path": template.pdf_path,
            "image_path": template.image_path,
            "width": template.width,
            "height": template.height,
            "fields": [asdict(f) for f in template.fields]
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        logger.info(f"템플릿 설정 저장: {output_path}")

    def load_template_config(self, config_path: str) -> TemplateDefinition:
        """템플릿 설정 로드"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        fields = [FieldDefinition(**f) for f in config["fields"]]

        template = TemplateDefinition(
            name=config["name"],
            pdf_path=config["pdf_path"],
            image_path=config["image_path"],
            width=config["width"],
            height=config["height"],
            fields=fields
        )

        self.templates[template.name] = template
        logger.info(f"템플릿 로드: {template.name}")

        return template

    def get_template_image(self, name: str) -> Image.Image:
        """템플릿 이미지 반환"""
        if name not in self.templates:
            raise ValueError(f"템플릿 '{name}'이 등록되지 않았습니다.")

        return Image.open(self.templates[name].image_path).convert("RGB")

    def get_field_positions(self, name: str) -> Dict[str, Tuple[int, int]]:
        """필드 위치 딕셔너리 반환 (기존 코드 호환용)"""
        if name not in self.templates:
            raise ValueError(f"템플릿 '{name}'이 등록되지 않았습니다.")

        return {
            f.name: (f.x, f.y)
            for f in self.templates[name].fields
        }



if __name__ == "__main__":
    # 테스트
    manager = PDFTemplateManager()

    pdf_path = "data/계약서양식/법원/부동산_임대차_계약서.pdf"

    if Path(pdf_path).exists():
        template = manager.register_template(
            name="법원_부동산_임대차_계약서",
            pdf_path=pdf_path,
            fields=COURT_LEASE_CONTRACT_FIELDS
        )

        manager.save_template_config("법원_부동산_임대차_계약서")
        print(f"템플릿 등록 완료: {template.name}")
        print(f"이미지 크기: {template.width}x{template.height}")
        print(f"필드 수: {len(template.fields)}")
    else:
        print(f"PDF 파일이 없습니다: {pdf_path}")
