"""
계약서 합성 데이터 생성 모듈
- PDF 양식 기반 합성 이미지 생성
- 실제 계약서와 유사한 데이터 생성
- 독소조항/정상조항 특약사항 생성
"""
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from loguru import logger

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from PIL import Image, ImageDraw

from .constants import DEFAULT_DPI, DEFAULT_TOXIC_RATIO
from .models import FieldConfig, FieldAnnotation
from .text_generator import TextGenerator
from .text_renderer import FontManager, TextRenderer, ImageEffects
from .field_definitions import COURT_LEASE_FIELDS, SPECIAL_TERMS_AREA


class ToxicClauseGenerator:
    """독소조항/정상조항 생성기"""

    def __init__(self, toxic_file: str = "data/독소조항/illegal.md"):
        self.toxic_clauses = self._load_toxic_clauses(toxic_file)
        self.normal_clauses = self._get_normal_clauses()

    def _load_toxic_clauses(self, filepath: str) -> List[Dict]:
        """마크다운 파일에서 독소조항 로드"""
        clauses = []
        path = Path(filepath)

        if not path.exists():
            logger.warning(f"독소조항 파일 없음: {filepath}")
            return clauses

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # 마크다운 테이블 파싱
        lines = content.strip().split("\n")
        for line in lines[2:]:  # 헤더와 구분선 제외
            if line.startswith("|") and not line.startswith("| ---"):
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) >= 3:
                    try:
                        clauses.append({
                            "id": int(parts[0]),
                            "category": parts[1],
                            "content": parts[2]
                        })
                    except (ValueError, IndexError):
                        continue

        logger.info(f"독소조항 {len(clauses)}개 로드됨")
        return clauses

    def _get_normal_clauses(self) -> List[str]:
        """정상적인 특약사항 예시"""
        return [
            "임차인은 계약 기간 중 주택의 시설물을 선량한 관리자의 주의로 사용하여야 한다.",
            "계약 기간 만료 시 임차인은 주택을 원상회복하여 반환한다. 단, 통상의 사용으로 인한 마모는 제외한다.",
            "임대인은 임대차 목적물의 하자 발생 시 수리할 의무가 있다.",
            "보증금은 계약 종료 후 명도 완료 시 즉시 반환한다.",
            "임차인의 동의 없이 임대인은 임대차 목적물에 출입할 수 없다.",
            "전기, 가스, 수도 요금은 임차인이 부담한다.",
            "관리비는 월 10만원으로 하며, 실비 정산한다.",
            "계약 갱신 시 임대료 인상은 5% 이내로 한다.",
            "임대인은 임차인의 전입신고 및 확정일자 취득에 협조한다.",
            "화재보험은 임대인이 가입하며, 보험료는 임대인이 부담한다.",
            "애완동물 사육은 소형견에 한하여 허용한다.",
            "주차장 1대 사용을 허용한다.",
            "계약 기간 중 임대인이 변경되어도 본 계약은 유효하다.",
            "임차인은 월세를 매월 말일까지 납부한다.",
            "보증금 반환 지연 시 연 5%의 지연이자를 지급한다.",
            "임대차 계약 종료 시 현 시설 상태로 인수인계한다.",
            "입주 시 시설물 점검표를 작성하여 쌍방이 보관한다.",
            "계약 체결 시 중개대상물 확인설명서를 교부받았음을 확인한다.",
        ]

    def generate_toxic_clause(self) -> Tuple[str, str]:
        """독소조항 생성 (내용, 카테고리)"""
        if not self.toxic_clauses:
            return "", ""
        clause = random.choice(self.toxic_clauses)
        return clause["content"], clause["category"]

    def generate_normal_clause(self) -> str:
        """정상 특약조항 생성"""
        return random.choice(self.normal_clauses)

    def generate_special_terms(
        self,
        toxic_ratio: float = 0.3,
        num_clauses: int = 3
    ) -> Tuple[List[str], List[bool], List[str]]:
        """
        특약사항 생성

        Args:
            toxic_ratio: 독소조항 비율 (0.0 ~ 1.0)
            num_clauses: 생성할 조항 수

        Returns:
            (조항 리스트, 독소여부 리스트, 카테고리 리스트)
        """
        clauses = []
        is_toxic_list = []
        categories = []

        for i in range(num_clauses):
            if random.random() < toxic_ratio and self.toxic_clauses:
                content, category = self.generate_toxic_clause()
                clauses.append(f"{i+1}. {content}")
                is_toxic_list.append(True)
                categories.append(category)
            else:
                content = self.generate_normal_clause()
                clauses.append(f"{i+1}. {content}")
                is_toxic_list.append(False)
                categories.append("정상")

        return clauses, is_toxic_list, categories


class ContractSynthesizer:
    """계약서 합성 데이터 생성기"""

    def __init__(self, font_dir: str = None, toxic_file: str = "data/독소조항/illegal.md"):
        self.text_generator = TextGenerator()
        self.toxic_generator = ToxicClauseGenerator(toxic_file)
        self.font_manager = FontManager(font_dir)
        self.text_renderer = TextRenderer(self.font_manager)
        self.image_effects = ImageEffects()

    def pdf_to_image(self, pdf_path: str, dpi: int = DEFAULT_DPI, page: int = 0) -> Image.Image:
        """PDF를 이미지로 변환"""
        if fitz is None:
            raise ImportError("PyMuPDF 필요: pip install pymupdf")

        doc = fitz.open(pdf_path)
        pg = doc.load_page(page)
        zoom = dpi / 300
        mat = fitz.Matrix(zoom, zoom)
        pix = pg.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img

    def generate_field_value(self, field_type: str) -> str:
        """필드 타입에 따른 값 생성"""
        gen = self.text_generator

        generators = {
            "name": lambda: gen.generate_name().text,
            "address": lambda: gen.generate_address().text,
            "phone": lambda: gen.generate_phone().text,
            "resident_id": lambda: gen.generate_resident_id().text,
            "amount": lambda: gen.generate_amount(10000000, 500000000, "korean").text,
            "amount_numeric": lambda: gen.generate_amount(10000000, 500000000, "numeric").text,
            "amount_small": lambda: gen.generate_amount(100000, 5000000, "korean").text,
            "date": lambda: gen.generate_date("%Y년 %m월 %d일").text,
            "date_short": lambda: gen.generate_date("%Y. %m. %d").text,
            "year": lambda: str(random.randint(2023, 2026)),
            "month": lambda: str(random.randint(1, 12)),
            "day": lambda: str(random.randint(1, 28)),
            "area": lambda: f"{random.randint(10, 200)}.{random.randint(0, 99):02d}",
            "percentage": lambda: f"{random.uniform(0.1, 0.9):.1f}",
            "text": lambda: random.choice(["대지", "전", "답", "임야", "잡종지"]),
            "structure": lambda: random.choice([
                "철근콘크리트조", "벽돌조", "목조", "철골조",
                "연립주택", "다세대주택", "아파트", "단독주택"
            ]),
            "company": lambda: random.choice([
                "행복부동산", "미래공인중개사", "신뢰부동산",
                "희망공인중개사사무소", "성공부동산중개"
            ]),
            "business_number": lambda: f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(10000, 99999)}",
        }

        return generators.get(field_type, lambda: "")()


    def synthesize(
        self,
        template_image: Image.Image,
        fields: List[FieldConfig],
        toxic_ratio: float = DEFAULT_TOXIC_RATIO,
        add_effects: bool = True
    ) -> Tuple[Image.Image, List[FieldAnnotation], bool]:
        """
        템플릿에 데이터 합성

        Args:
            template_image: 템플릿 이미지
            fields: 필드 설정 리스트
            toxic_ratio: 독소조항 비율
            add_effects: 효과 추가 여부

        Returns:
            (합성 이미지, 어노테이션 리스트, 독소조항 포함 여부)
        """
        image = template_image.copy()
        draw = ImageDraw.Draw(image)
        annotations = []
        has_toxic = False

        # 일반 필드 렌더링
        for field in fields:
            # 값 생성
            text = self.generate_field_value(field.field_type)
            if not text:
                continue

            # 렌더링
            bbox = self.text_renderer.render_text(draw, text, field)

            # 어노테이션
            annotations.append(FieldAnnotation(
                field_name=field.name,
                field_type=field.field_type,
                text=text,
                bbox=bbox,
                is_toxic=False
            ))

        # 특약사항 생성 및 렌더링
        clauses, is_toxic_list, categories = self.toxic_generator.generate_special_terms(
            toxic_ratio=toxic_ratio,
            num_clauses=random.randint(2, 4)
        )

        # 특약사항 영역
        area = SPECIAL_TERMS_AREA
        font = self.font_manager.get_font(area["font_size"])

        current_y = area["y"]
        for i, (clause, is_toxic, category) in enumerate(zip(clauses, is_toxic_list, categories)):
            bboxes, current_y = self.text_renderer.render_multiline_text(
                draw, clause,
                area["x"], current_y,
                area["width"], font,
                line_spacing=area["line_spacing"]
            )

            if is_toxic:
                has_toxic = True

            # 전체 조항의 bbox (첫 줄부터 마지막 줄까지)
            if bboxes:
                full_bbox = (
                    bboxes[0][0],
                    bboxes[0][1],
                    max(b[2] for b in bboxes),
                    bboxes[-1][3]
                )

                annotations.append(FieldAnnotation(
                    field_name=f"특약사항_{i+1}",
                    field_type="special_term",
                    text=clause,
                    bbox=full_bbox,
                    is_toxic=is_toxic
                ))

            current_y += area["clause_spacing"]

        # 현실적인 효과 추가
        if add_effects:
            image = self.image_effects.add_realistic_effects(image)

        return image, annotations, has_toxic

    def save_result(
        self,
        image: Image.Image,
        annotations: List[FieldAnnotation],
        output_dir: str,
        filename: str,
        has_toxic: bool = False
    ) -> Tuple[str, str]:
        """결과 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 이미지 저장
        img_path = output_path / f"{filename}.png"
        image.save(str(img_path))

        # 어노테이션 저장
        json_path = output_path / f"{filename}.json"

        # 특약사항 분리
        special_terms = [a for a in annotations if a.field_type == "special_term"]
        other_fields = [a for a in annotations if a.field_type != "special_term"]

        data = {
            "image": str(img_path.name),
            "width": image.width,
            "height": image.height,
            "has_toxic_clause": has_toxic,
            "annotations": [
                {
                    "field_name": a.field_name,
                    "field_type": a.field_type,
                    "text": a.text,
                    "bbox": list(a.bbox),
                    "is_toxic": a.is_toxic
                }
                for a in other_fields
            ],
            "special_terms": [
                {
                    "field_name": a.field_name,
                    "text": a.text,
                    "bbox": list(a.bbox),
                    "is_toxic": a.is_toxic
                }
                for a in special_terms
            ]
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return str(img_path), str(json_path)


def generate_batch(
    pdf_path: str,
    output_dir: str,
    count: int = 100,
    toxic_ratio: float = DEFAULT_TOXIC_RATIO,
    dpi: int = DEFAULT_DPI
) -> List[Tuple[str, str]]:
    """
    배치 합성 데이터 생성

    Args:
        pdf_path: 양식 PDF 경로
        output_dir: 출력 디렉토리
        count: 생성 개수
        toxic_ratio: 독소조항 포함 비율 (0.0 ~ 1.0)
        dpi: 해상도

    Returns:
        [(이미지 경로, JSON 경로), ...] 리스트
    """
    synthesizer = ContractSynthesizer()

    # PDF를 이미지로 변환
    logger.info(f"PDF 로드: {pdf_path}")
    template = synthesizer.pdf_to_image(pdf_path, dpi=dpi, page=0)  # 1페이지
    logger.info(f"템플릿 크기: {template.width}x{template.height}")

    results = []
    toxic_count = 0
    normal_count = 0

    # 독소조항 포함 계약서 수 계산 (30%가 독소조항 포함)
    target_toxic_contracts = int(count * toxic_ratio)

    for i in range(count):
        # 독소조항 포함 여부 결정
        if toxic_count < target_toxic_contracts:
            # 남은 것 중에서 확률적으로 결정
            remaining = count - i
            remaining_toxic = target_toxic_contracts - toxic_count
            use_toxic = random.random() < (remaining_toxic / remaining)
        else:
            use_toxic = False

        # 합성 (독소조항 포함 계약서는 100% 확률, 아니면 0%)
        current_toxic_ratio = 1.0 if use_toxic else 0.0
        image, annotations, has_toxic = synthesizer.synthesize(
            template,
            COURT_LEASE_FIELDS,
            toxic_ratio=current_toxic_ratio
        )

        if has_toxic:
            toxic_count += 1
        else:
            normal_count += 1

        # 저장
        filename = f"contract_{i:04d}"
        paths = synthesizer.save_result(image, annotations, output_dir, filename, has_toxic)
        results.append(paths)

        if (i + 1) % 10 == 0:
            logger.info(f"생성 진행: {i + 1}/{count} (독소: {toxic_count}, 정상: {normal_count})")

    logger.info(f"배치 생성 완료: {len(results)}개 -> {output_dir}")
    logger.info(f"독소조항 포함: {toxic_count}개 ({toxic_count/count*100:.1f}%)")
    logger.info(f"정상 계약서: {normal_count}개 ({normal_count/count*100:.1f}%)")

    return results


if __name__ == "__main__":
    # 테스트 실행
    pdf_path = "data/계약서양식/법원/부동산_임대차_계약서.pdf"
    output_dir = "data/augmented/test"

    if Path(pdf_path).exists():
        results = generate_batch(pdf_path, output_dir, count=5, toxic_ratio=0.3)
        print(f"\n생성 완료: {len(results)}개")
        for img_path, json_path in results:
            print(f"  - {img_path}")
    else:
        print(f"PDF 파일이 없습니다: {pdf_path}")
