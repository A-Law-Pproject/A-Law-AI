"""
텍스트 렌더링 모듈
- 이미지에 텍스트 렌더링
- 폰트 관리
- 노이즈 및 효과 추가
"""
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from loguru import logger

from .constants import (
    SYSTEM_FONTS,
    TEXT_COLORS,
    DEFAULT_FONT_SIZE,
    NOISE_STD,
    BRIGHTNESS_RANGE,
    BLUR_PROBABILITY,
    BLUR_RADIUS,
    POSITION_JITTER,
    FONT_SIZE_JITTER,
)
from .models import FieldConfig


class FontManager:
    """폰트 관리자"""

    def __init__(self, font_dir: Optional[str] = None):
        self.font_dir = Path(font_dir) if font_dir else None
        self.fonts = self._load_fonts()

    def _load_fonts(self) -> List[str]:
        """사용 가능한 폰트 로드"""
        fonts = []

        # 사용자 지정 폰트 디렉토리
        if self.font_dir and self.font_dir.exists():
            for ext in ["*.ttf", "*.otf", "*.TTF", "*.OTF"]:
                fonts.extend([str(p) for p in self.font_dir.glob(ext)])

        # 시스템 폰트
        for font_path in SYSTEM_FONTS:
            if Path(font_path).exists():
                fonts.append(font_path)

        if fonts:
            logger.info(f"폰트 {len(fonts)}개 로드됨")
        else:
            logger.warning("한글 폰트를 찾을 수 없습니다")

        return fonts

    def get_font(self, size: int = DEFAULT_FONT_SIZE, font_path: Optional[str] = None) -> ImageFont.FreeTypeFont:
        """폰트 객체 반환"""
        if font_path:
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                pass

        if self.fonts:
            font_path = random.choice(self.fonts)
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                pass

        return ImageFont.load_default()

    def get_random_font(self, size: int = DEFAULT_FONT_SIZE) -> ImageFont.FreeTypeFont:
        """랜덤 폰트 반환"""
        return self.get_font(size)


class TextRenderer:
    """텍스트 렌더링 클래스"""

    def __init__(self, font_manager: Optional[FontManager] = None):
        self.font_manager = font_manager or FontManager()
        self.text_colors = TEXT_COLORS

    def render_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        field: FieldConfig,
        font: Optional[ImageFont.FreeTypeFont] = None,
        add_jitter: bool = True
    ) -> Tuple[int, int, int, int]:
        """
        텍스트 렌더링 및 bbox 반환

        Args:
            draw: ImageDraw 객체
            text: 렌더링할 텍스트
            field: 필드 설정
            font: 폰트 (None이면 자동 선택)
            add_jitter: 위치 변동 추가 여부

        Returns:
            (x1, y1, x2, y2) bbox
        """
        if font is None:
            font_size = field.font_size + random.randint(*FONT_SIZE_JITTER)
            font = self.font_manager.get_random_font(font_size)

        # 위치 계산 (jitter 적용)
        x = field.x
        y = field.y
        if add_jitter:
            x += random.randint(*POSITION_JITTER)
            y += random.randint(*POSITION_JITTER)

        # 텍스트 크기 계산
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 정렬
        if field.align == "center":
            x = field.x + (field.width - text_width) // 2
        elif field.align == "right":
            x = field.x + field.width - text_width

        # 색상 선택
        color = random.choice(self.text_colors)

        # 렌더링
        draw.text((x, y), text, font=font, fill=color)

        return (x, y, x + text_width, y + text_height)

    def render_multiline_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int,
        y: int,
        max_width: int,
        font: Optional[ImageFont.FreeTypeFont] = None,
        line_spacing: int = 5
    ) -> Tuple[List[Tuple[int, int, int, int]], int]:
        """
        여러 줄 텍스트 렌더링

        Args:
            draw: ImageDraw 객체
            text: 렌더링할 텍스트
            x: 시작 x 좌표
            y: 시작 y 좌표
            max_width: 최대 너비
            font: 폰트
            line_spacing: 줄 간격

        Returns:
            (bbox 리스트, 마지막 y 좌표)
        """
        if font is None:
            font = self.font_manager.get_random_font()

        color = random.choice(self.text_colors)
        bboxes = []
        current_y = y

        # 텍스트를 단어 단위로 분리
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        # 각 줄 렌더링
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_height = bbox[3] - bbox[1]

            draw.text((x, current_y), line, font=font, fill=color)
            bboxes.append((x, current_y, x + bbox[2] - bbox[0], current_y + line_height))

            current_y += line_height + line_spacing

        return bboxes, current_y


class ImageEffects:
    """이미지 효과 추가 클래스"""

    @staticmethod
    def add_gaussian_noise(image: Image.Image, std: float = NOISE_STD) -> Image.Image:
        """가우시안 노이즈 추가"""
        img_array = np.array(image)
        noise = np.random.normal(0, std, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    @staticmethod
    def add_brightness_variation(image: Image.Image) -> Image.Image:
        """밝기 변화 추가"""
        img_array = np.array(image)
        brightness = random.uniform(*BRIGHTNESS_RANGE)
        img_array = np.clip(img_array * brightness, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    @staticmethod
    def add_blur(image: Image.Image, probability: float = BLUR_PROBABILITY) -> Image.Image:
        """확률적으로 블러 추가"""
        if random.random() < probability:
            return image.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
        return image

    @classmethod
    def add_realistic_effects(cls, image: Image.Image) -> Image.Image:
        """실제 스캔 이미지처럼 효과 추가"""
        image = cls.add_gaussian_noise(image)
        image = cls.add_brightness_variation(image)
        image = cls.add_blur(image)
        return image
