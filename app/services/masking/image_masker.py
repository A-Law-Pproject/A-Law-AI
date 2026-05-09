"""
이미지 PII 마스킹 모듈
OCR 결과의 bbox 좌표를 기반으로 계약서 이미지에서 개인정보 영역을 검정 박스로 마스킹한다.

- 인감/서명/날인 키워드 근처 영역 자동 탐지
- Pillow (PIL.ImageDraw) 기반으로 처리
"""
import io
from typing import List, Optional, Tuple

from loguru import logger

# 인감/서명 영역 탐지 키워드 (단독 "인"은 오탐 방지를 위해 제외)
_SEAL_KEYWORDS = {"인감", "서명", "날인", "印", "서명란", "인감란"}

# 주민번호/전화번호 관련 OCR 텍스트 패턴 힌트 (bbox 마스킹 보조)
_PII_KEYWORDS = {"주민등록번호", "주민번호", "생년월일", "연락처", "전화번호", "핸드폰"}

# 마스킹 박스 확장 픽셀 (bbox 주변 여백)
_BOX_PADDING_PX = 4


def _pct_to_px(value_pct: float, dimension_px: int) -> int:
    """백분율 좌표(0-100)를 픽셀 좌표로 변환"""
    return int(value_pct / 100.0 * dimension_px)


def _compute_pixel_box(
    x_pct: float, y_pct: float,
    w_pct: float, h_pct: float,
    img_width: int, img_height: int,
    padding: int = _BOX_PADDING_PX,
) -> Tuple[int, int, int, int]:
    """
    OCRWord의 % 좌표를 픽셀 좌표 (left, top, right, bottom)로 변환

    Args:
        x_pct: 좌상단 X (%)
        y_pct: 좌상단 Y (%)
        w_pct: 너비 (%)
        h_pct: 높이 (%)
        img_width: 이미지 너비 (px)
        img_height: 이미지 높이 (px)
        padding: 박스 확장 픽셀

    Returns:
        (left, top, right, bottom) in pixels
    """
    left = max(0, _pct_to_px(x_pct, img_width) - padding)
    top = max(0, _pct_to_px(y_pct, img_height) - padding)
    right = min(img_width, _pct_to_px(x_pct + w_pct, img_width) + padding)
    bottom = min(img_height, _pct_to_px(y_pct + h_pct, img_height) + padding)
    return left, top, right, bottom


def find_seal_regions(
    words: List[dict],
    img_width: int,
    img_height: int,
) -> List[Tuple[int, int, int, int]]:
    """
    OCR words에서 인감/서명 관련 키워드를 포함하는 bbox를 탐지하여
    픽셀 좌표 리스트로 반환한다.

    Args:
        words: OCRWord dict 리스트 (text, x, y, width, height 포함)
        img_width: 이미지 너비
        img_height: 이미지 높이

    Returns:
        마스킹할 픽셀 박스 목록 [(left, top, right, bottom), ...]
    """
    regions = []
    for word in words:
        text = word.get("text", "")
        # 인감/서명 키워드 확인
        if any(keyword in text for keyword in _SEAL_KEYWORDS):
            box = _compute_pixel_box(
                x_pct=word.get("x", 0),
                y_pct=word.get("y", 0),
                w_pct=word.get("width", 0),
                h_pct=word.get("height", 0),
                img_width=img_width,
                img_height=img_height,
                padding=_BOX_PADDING_PX * 3,  # 인감 영역은 더 넓게 확장
            )
            regions.append(box)
            logger.debug(f"인감/서명 영역 탐지: text_hint=[MASKED], box={box}")

    return regions


def mask_image_regions(
    image_bytes: bytes,
    mask_regions: List[Tuple[int, int, int, int]],
) -> bytes:
    """
    이미지 바이트에서 지정된 픽셀 영역을 검정 박스로 마스킹한다.

    Args:
        image_bytes: 원본 이미지 바이트
        mask_regions: 마스킹할 픽셀 박스 목록 [(left, top, right, bottom), ...]

    Returns:
        마스킹된 이미지 바이트 (JPEG)

    Raises:
        ValueError: 이미지 디코딩 실패
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise RuntimeError(
            "Pillow 라이브러리가 필요합니다. `pip install Pillow`를 실행하세요."
        )

    if not mask_regions:
        return image_bytes

    # 이미지 디코딩
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")  # JPEG 저장을 위해 RGB 변환
    except Exception as e:
        raise ValueError(f"이미지 디코딩 실패: {e}")

    draw = ImageDraw.Draw(img)

    # 각 영역에 검정 박스 오버레이
    for region in mask_regions:
        left, top, right, bottom = region
        if right > left and bottom > top:  # 유효한 영역만 처리
            draw.rectangle([left, top, right, bottom], fill=(0, 0, 0))

    # JPEG로 인코딩 후 바이트 반환
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=95)
    output.seek(0)
    return output.read()


def mask_image_with_words(
    image_bytes: bytes,
    words: Optional[List[dict]],
    img_width: int,
    img_height: int,
) -> Tuple[bytes, int]:
    """
    OCR words 목록을 기반으로 이미지에서 인감/서명 영역을 마스킹한다.

    Args:
        image_bytes: 원본 이미지 바이트
        words: OCRWord dict 리스트 (None이면 마스킹 스킵)
        img_width: OCR이 처리한 이미지 너비
        img_height: OCR이 처리한 이미지 높이

    Returns:
        (masked_image_bytes, mask_count): 마스킹된 이미지 바이트, 마스킹된 영역 수
    """
    if not words or img_width <= 0 or img_height <= 0:
        logger.debug("마스킹할 OCR words 없음 또는 이미지 크기 미확인 — 이미지 마스킹 스킵")
        return image_bytes, 0

    # 인감/서명 영역 탐지
    seal_regions = find_seal_regions(words, img_width, img_height)

    if not seal_regions:
        logger.debug("인감/서명 영역이 탐지되지 않았음")
        return image_bytes, 0

    logger.info(f"이미지 마스킹 영역 {len(seal_regions)}건 탐지")

    # 검정 박스 오버레이 적용
    masked_bytes = mask_image_regions(image_bytes, seal_regions)
    return masked_bytes, len(seal_regions)
