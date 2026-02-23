"""
데이터 합성 모듈
- 빈 양식 이미지에 텍스트 렌더링
- 다양한 폰트, 각도, 색상 변형
- 학습용 라벨 데이터 생성
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import asdict
from PIL import Image
from loguru import logger

from .constants import DEFAULT_FONT_SIZE, ROTATION_RANGE, POSITION_JITTER
from .models import TextAnnotation, SynthesizedImage
from .text_generator import TextGenerator
from .text_renderer import FontManager, TextRenderer


class DataSynthesizer:
    """데이터 합성기"""

    def __init__(
        self,
        font_dir: str = "fonts",
        default_font_size: int = DEFAULT_FONT_SIZE
    ):
        """
        Args:
            font_dir: 폰트 파일 디렉토리
            default_font_size: 기본 폰트 크기
        """
        self.default_font_size = default_font_size
        self.text_generator = TextGenerator()
        self.font_manager = FontManager(font_dir)
        self.text_renderer = TextRenderer(self.font_manager)


    def synthesize(
        self,
        template_image: Image.Image,
        field_positions: Dict[str, Tuple[int, int]],
        template_name: str = "unknown"
    ) -> SynthesizedImage:
        """
        양식 이미지에 데이터 합성

        Args:
            template_image: 빈 양식 이미지
            field_positions: {필드명: (x, y)} 위치 정보
            template_name: 템플릿 이름

        Returns:
            SynthesizedImage
        """
        image = template_image.copy()
        annotations = []

        # 필드 타입별 생성기 매핑
        field_generators = {
            "name": self.text_generator.generate_name,
            "임대인": self.text_generator.generate_name,
            "임차인": self.text_generator.generate_name,
            "phone": self.text_generator.generate_phone,
            "전화번호": self.text_generator.generate_phone,
            "address": self.text_generator.generate_address,
            "주소": self.text_generator.generate_address,
            "date": self.text_generator.generate_date,
            "계약일": self.text_generator.generate_date,
            "amount": self.text_generator.generate_amount,
            "보증금": lambda: self.text_generator.generate_amount(10000000, 500000000),
            "월세": lambda: self.text_generator.generate_amount(100000, 5000000),
            "계약금": lambda: self.text_generator.generate_amount(1000000, 50000000),
            "resident_id": self.text_generator.generate_resident_id,
            "주민등록번호": self.text_generator.generate_resident_id,
        }

        for field_name, position in field_positions.items():
            # 생성기 찾기
            generator = field_generators.get(
                field_name,
                self.text_generator.generate_name  # 기본값
            )

            # 텍스트 생성
            generated = generator()

            # 랜덤 변형
            font_size = self.default_font_size + random.randint(-2, 2)
            font = self.font_manager.get_font(font_size)

            # 위치 미세 조정
            x = position[0] + random.randint(*POSITION_JITTER)
            y = position[1] + random.randint(*POSITION_JITTER)

            # 간단한 텍스트 렌더링 (회전 없음)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(image)
            bbox_coords = draw.textbbox((0, 0), generated.text, font=font)
            text_width = bbox_coords[2] - bbox_coords[0]
            text_height = bbox_coords[3] - bbox_coords[1]

            color = random.choice(self.text_renderer.text_colors)
            draw.text((x, y), generated.text, font=font, fill=color)

            # 어노테이션 생성
            annotation = TextAnnotation(
                text=generated.text,
                x=x,
                y=y,
                width=text_width,
                height=text_height,
                field_type=generated.field_type,
                font_name="random",
                font_size=font_size,
                rotation=0.0
            )
            annotations.append(annotation)

        return SynthesizedImage(
            image=image,
            annotations=annotations,
            source_template=template_name
        )

    def save_with_annotations(
        self,
        synthesized: SynthesizedImage,
        output_dir: str,
        filename: str
    ) -> Tuple[str, str]:
        """
        이미지와 어노테이션 저장

        Returns:
            (이미지 경로, JSON 경로)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 이미지 저장
        img_path = output_path / f"{filename}.png"
        synthesized.image.save(str(img_path))

        # 어노테이션 JSON 저장
        json_path = output_path / f"{filename}.json"
        annotations_dict = {
            "image": str(img_path),
            "template": synthesized.source_template,
            "annotations": [asdict(a) for a in synthesized.annotations]
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(annotations_dict, f, ensure_ascii=False, indent=2)

        return str(img_path), str(json_path)

    def generate_batch(
        self,
        template_path: str,
        field_positions: Dict[str, Tuple[int, int]],
        output_dir: str,
        count: int = 100
    ) -> List[Tuple[str, str]]:
        """
        배치 데이터 생성

        Args:
            template_path: 빈 양식 이미지 경로
            field_positions: 필드 위치 정보
            output_dir: 출력 디렉토리
            count: 생성 개수

        Returns:
            [(이미지 경로, JSON 경로), ...] 리스트
        """
        template = Image.open(template_path).convert("RGB")
        template_name = Path(template_path).stem

        results = []
        for i in range(count):
            synthesized = self.synthesize(template, field_positions, template_name)
            paths = self.save_with_annotations(
                synthesized,
                output_dir,
                f"{template_name}_{i:04d}"
            )
            results.append(paths)

            if (i + 1) % 10 == 0:
                logger.info(f"생성 진행: {i + 1}/{count}")

        logger.info(f"배치 생성 완료: {len(results)}개")
        return results


if __name__ == "__main__":
    synthesizer = DataSynthesizer()
    print("DataSynthesizer 모듈 로드 완료")

    # 테스트
    gen = synthesizer.text_generator
    print(f"\n이름: {gen.generate_name().text}")
    print(f"금액: {gen.generate_amount().text}")
