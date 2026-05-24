"""
Image masking helpers used before and after OCR.

The module supports two flows:
1. Pre-OCR masking using local Tesseract word boxes plus image heuristics.
2. Post-OCR masking using word boxes returned by the upstream OCR engine.
"""
from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger

from app.services.masking.patterns import (
    ACCOUNT_FIELD_PATTERN,
    ADDRESS_FIELD_PATTERN,
    ADDRESS_FRAGMENT_PATTERN,
    ALL_FIELD_LABEL_PATTERN,
    BANK_ACCOUNT_PATTERN,
    BIRTH_DATE_FIELD_PATTERN,
    BUSINESS_NO_FIELD_PATTERN,
    BUSINESS_NO_PATTERN,
    CORPORATE_NO_FIELD_PATTERN,
    CORPORATE_NO_PATTERN,
    DRIVER_LICENSE_FIELD_PATTERN,
    DRIVER_LICENSE_VALUE_PATTERN,
    EMAIL_FIELD_PATTERN,
    EMAIL_PATTERN,
    NAME_FIELD_PATTERN,
    PASSPORT_FIELD_PATTERN,
    PASSPORT_VALUE_PATTERN,
    PHONE_FIELD_PATTERN,
    PHONE_PATTERN,
    RESIDENT_FIELD_PATTERN,
    RESIDENT_ID_PATTERN,
    SEAL_KEYWORDS,
)


Box = Tuple[int, int, int, int]

_BOX_PADDING_PX = 6
_MERGE_GAP_PX = 12
_IDENTITY_EXTRA_RIGHT_RATIO = 0.12

_IDENTITY_FIELD_LABELS = [
    "성명",
    "임대인",
    "임차인",
    "대리인",
    "대표",
    "서명",
    "서명인",
    "인감",
    "도장",
    "날인",
    "사인",
]


def _spaced_term(term: str) -> str:
    return r"\s*".join(re.escape(char) for char in term if not char.isspace())


_IDENTITY_FIELD_LABEL_PATTERN = "(?:" + "|".join(_spaced_term(label) for label in _IDENTITY_FIELD_LABELS) + ")"
_IDENTITY_STOP_LABEL_PATTERN = rf"(?:{ALL_FIELD_LABEL_PATTERN}|{_IDENTITY_FIELD_LABEL_PATTERN})"
_IDENTITY_FIELD_PATTERN = re.compile(
    rf"(?P<label>{_IDENTITY_FIELD_LABEL_PATTERN})\s*[:：]?\s*(?P<value>.+?)"
    rf"(?=(?:\s+(?:{_IDENTITY_STOP_LABEL_PATTERN}))|$)",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class ImageMaskingResult:
    image_bytes: bytes
    mask_count: int = 0
    mask_types: List[str] = field(default_factory=list)
    regions: List[Box] = field(default_factory=list)
    masking_failed: bool = False
    error_message: Optional[str] = None


@dataclass
class _LineWord:
    text: str
    box: Box
    start: int
    end: int


@dataclass
class _Line:
    words: List[dict]
    top: int
    bottom: int


def _decode_image(image_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image bytes")
    return image


def _image_size(image_bytes: bytes) -> Tuple[int, int]:
    image = _decode_image(image_bytes)
    height, width = image.shape[:2]
    return width, height


def _pct_to_px(value_pct: float, dimension_px: int) -> int:
    return int(value_pct / 100.0 * dimension_px)


def _expand_box(
    box: Box,
    img_width: int,
    img_height: int,
    *,
    left: int = 0,
    top: int = 0,
    right: int = 0,
    bottom: int = 0,
) -> Box:
    box_left, box_top, box_right, box_bottom = box
    return (
        max(0, box_left - left),
        max(0, box_top - top),
        min(img_width, box_right + right),
        min(img_height, box_bottom + bottom),
    )


def _combine_boxes(boxes: Sequence[Box]) -> Optional[Box]:
    if not boxes:
        return None
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    return left, top, right, bottom


def _boxes_touch(box1: Box, box2: Box, gap: int = _MERGE_GAP_PX) -> bool:
    return not (
        box1[2] + gap < box2[0]
        or box2[2] + gap < box1[0]
        or box1[3] + gap < box2[1]
        or box2[3] + gap < box1[1]
    )


def _merge_regions(regions: Sequence[Tuple[Box, str]]) -> Tuple[List[Box], List[str]]:
    if not regions:
        return [], []

    merged = [
        {"box": box, "types": {mask_type}}
        for box, mask_type in sorted(regions, key=lambda item: (item[0][1], item[0][0]))
    ]

    changed = True
    while changed:
        changed = False
        next_regions = []
        while merged:
            current = merged.pop(0)
            current_box = current["box"]
            current_types = set(current["types"])
            keep_merging = True
            while keep_merging:
                keep_merging = False
                survivors = []
                for other in merged:
                    if _boxes_touch(current_box, other["box"]):
                        current_box = _combine_boxes([current_box, other["box"]]) or current_box
                        current_types.update(other["types"])
                        keep_merging = True
                        changed = True
                    else:
                        survivors.append(other)
                merged = survivors
            next_regions.append({"box": current_box, "types": current_types})
        merged = next_regions

    boxes = [item["box"] for item in merged]
    mask_types = sorted({mask_type for item in merged for mask_type in item["types"]})
    return boxes, mask_types


def _word_to_box(word: dict, img_width: int, img_height: int) -> Box:
    if all(key in word for key in ("px_x", "px_y", "px_width", "px_height")):
        left = int(word["px_x"])
        top = int(word["px_y"])
        right = left + int(word["px_width"])
        bottom = top + int(word["px_height"])
        return _expand_box(
            (left, top, right, bottom),
            img_width,
            img_height,
            left=_BOX_PADDING_PX,
            top=_BOX_PADDING_PX,
            right=_BOX_PADDING_PX,
            bottom=_BOX_PADDING_PX,
        )

    x_pct = float(word.get("x", 0))
    y_pct = float(word.get("y", 0))
    width_pct = float(word.get("width", 0))
    height_pct = float(word.get("height", 0))
    return _expand_box(
        (
            _pct_to_px(x_pct, img_width),
            _pct_to_px(y_pct, img_height),
            _pct_to_px(x_pct + width_pct, img_width),
            _pct_to_px(y_pct + height_pct, img_height),
        ),
        img_width,
        img_height,
        left=_BOX_PADDING_PX,
        top=_BOX_PADDING_PX,
        right=_BOX_PADDING_PX,
        bottom=_BOX_PADDING_PX,
    )


def _group_words_into_lines(words: Sequence[dict], img_width: int, img_height: int) -> List[_Line]:
    candidates = []
    for word in words:
        text = str(word.get("text", "")).strip()
        if not text:
            continue
        box = _word_to_box(word, img_width, img_height)
        candidates.append({"text": text, "box": box, "raw": word})

    candidates.sort(key=lambda item: (item["box"][1], item["box"][0]))
    lines: List[_Line] = []

    for item in candidates:
        left, top, right, bottom = item["box"]
        center_y = (top + bottom) / 2
        assigned = False

        for line in lines:
            line_center = (line.top + line.bottom) / 2
            line_height = max(1, line.bottom - line.top)
            word_height = max(1, bottom - top)
            threshold = max(line_height, word_height) * 0.7
            if abs(center_y - line_center) <= threshold:
                line.words.append({"text": item["text"], "box": item["box"], "raw": item["raw"]})
                line.top = min(line.top, top)
                line.bottom = max(line.bottom, bottom)
                assigned = True
                break

        if not assigned:
            lines.append(
                _Line(
                    words=[{"text": item["text"], "box": item["box"], "raw": item["raw"]}],
                    top=top,
                    bottom=bottom,
                )
            )

    for line in lines:
        line.words.sort(key=lambda item: item["box"][0])

    return lines


def _build_line_tokens(line: _Line) -> Tuple[str, List[_LineWord]]:
    parts: List[str] = []
    tokens: List[_LineWord] = []
    cursor = 0

    for index, word in enumerate(line.words):
        text = word["text"].strip()
        if not text:
            continue
        if index > 0:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(text)
        cursor += len(text)
        tokens.append(_LineWord(text=text, box=word["box"], start=start, end=cursor))

    return "".join(parts), tokens


def _boxes_for_span(tokens: Sequence[_LineWord], start: int, end: int) -> List[Box]:
    return [token.box for token in tokens if not (token.end <= start or token.start >= end)]


def _collect_pattern_regions(
    lines: Sequence[_Line],
    pattern: re.Pattern[str],
    *,
    region_type: str,
    group_name: Optional[str],
    img_width: int,
    img_height: int,
    extra_right_px: int = 0,
    extra_vertical_px: int = 0,
) -> List[Tuple[Box, str]]:
    regions: List[Tuple[Box, str]] = []

    for line in lines:
        line_text, tokens = _build_line_tokens(line)
        if not line_text:
            continue

        for match in pattern.finditer(line_text):
            span_start, span_end = match.span(group_name) if group_name else match.span()
            boxes = _boxes_for_span(tokens, span_start, span_end)
            combined = _combine_boxes(boxes)
            if combined is None:
                continue

            if extra_right_px or extra_vertical_px:
                combined = _expand_box(
                    combined,
                    img_width,
                    img_height,
                    top=extra_vertical_px,
                    bottom=extra_vertical_px,
                    right=extra_right_px,
                )

            regions.append((combined, region_type))

    return regions


def _find_red_seal_regions(image_bytes: bytes) -> List[Box]:
    image = _decode_image(image_bytes)
    img_height, img_width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array([0, 45, 40]), np.array([20, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 45, 40]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(120, int(img_width * img_height * 0.00035))
    regions: List[Box] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        if width < 12 or height < 12:
            continue

        region = _expand_box(
            (x, y, x + width, y + height),
            img_width,
            img_height,
            left=10,
            top=10,
            right=10,
            bottom=10,
        )
        regions.append(region)

    return regions


_TESSERACT_MIN_WIDTH_PX = 2000  # 작은 이미지 업스케일 임계값 (한글 가독성)
_TESSERACT_UPSCALE_FACTOR = 2.0
_TESSERACT_MIN_CONFIDENCE = 30.0  # conf < 30 단어는 노이즈로 폐기
_TESSERACT_SPARSE_FALLBACK_THRESHOLD = 20  # 1차 단어 수가 이보다 적으면 PSM 11로 보강


def _prepare_tesseract_input(image_bytes: bytes) -> Tuple[bytes, float]:
    """Tesseract 입력 전처리. 업스케일 비율도 반환해서 좌표 역변환에 사용."""
    image = _decode_image(image_bytes)
    height, width = image.shape[:2]

    scale = 1.0
    if width and width < _TESSERACT_MIN_WIDTH_PX:
        scale = _TESSERACT_UPSCALE_FACTOR
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Otsu가 LSTM 한글 인식에 더 안정적 (adaptive는 작은 한글 획을 잘라먹는 경향)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ok, encoded = cv2.imencode(".png", binary)
    if not ok:
        raise ValueError("Failed to encode preprocessed image for Tesseract")
    return encoded.tobytes(), scale


def _run_tesseract_tsv(
    tesseract_cmd: str,
    input_path: str,
    psm: str,
) -> str:
    command = [
        tesseract_cmd,
        input_path,
        "stdout",
        "-l", "kor+eng",
        "--oem", "1",
        "--psm", psm,
        "-c", "preserve_interword_spaces=1",
        "-c", "user_defined_dpi=300",
        "tsv",
    ]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        timeout=30,
        check=False,
    )
    if process.returncode != 0:
        stderr = process.stderr.strip() or "unknown tesseract error"
        raise RuntimeError(f"Tesseract word detection failed (psm={psm}): {stderr}")
    return process.stdout


def _parse_tesseract_tsv(
    tsv_output: str,
    img_width: int,
    img_height: int,
    scale: float,
) -> List[dict]:
    words: List[dict] = []
    reader = csv.DictReader(io.StringIO(tsv_output), delimiter="\t")
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue

        try:
            conf = float(row.get("conf") or -1)
            left = int(row.get("left") or 0)
            top = int(row.get("top") or 0)
            width = int(row.get("width") or 0)
            height = int(row.get("height") or 0)
        except ValueError:
            continue

        if conf < _TESSERACT_MIN_CONFIDENCE or width <= 0 or height <= 0:
            continue

        # 업스케일된 좌표를 원본 픽셀 좌표로 환산
        if scale and scale != 1.0:
            left = int(left / scale)
            top = int(top / scale)
            width = int(width / scale)
            height = int(height / scale)

        words.append(
            {
                "text": text,
                "x": round(left / img_width * 100, 2) if img_width else 0.0,
                "y": round(top / img_height * 100, 2) if img_height else 0.0,
                "width": round(width / img_width * 100, 2) if img_width else 0.0,
                "height": round(height / img_height * 100, 2) if img_height else 0.0,
                "confidence": max(0.0, min(conf / 100.0, 1.0)),
                "px_x": left,
                "px_y": top,
                "px_width": width,
                "px_height": height,
            }
        )
    return words


def detect_words_with_tesseract(image_bytes: bytes) -> List[dict]:
    tesseract_cmd = shutil.which("tesseract")
    if not tesseract_cmd:
        raise RuntimeError("Tesseract is not installed")

    img_width, img_height = _image_size(image_bytes)
    prepared_bytes, scale = _prepare_tesseract_input(image_bytes)

    with tempfile.TemporaryDirectory(prefix="masking-ocr-") as temp_dir:
        input_path = f"{temp_dir}/input.png"
        with open(input_path, "wb") as file:
            file.write(prepared_bytes)

        # 1차: PSM 3 (auto layout) — 계약서처럼 블록 구성이 다양한 문서에 적합
        primary_tsv = _run_tesseract_tsv(tesseract_cmd, input_path, psm="3")
        words = _parse_tesseract_tsv(primary_tsv, img_width, img_height, scale)

        # 보강: 1차에서 단어가 적게 잡히면 PSM 11(sparse text)로 한 번 더 시도
        if len(words) < _TESSERACT_SPARSE_FALLBACK_THRESHOLD:
            try:
                fallback_tsv = _run_tesseract_tsv(tesseract_cmd, input_path, psm="11")
                fallback_words = _parse_tesseract_tsv(
                    fallback_tsv, img_width, img_height, scale
                )
                # 중복 제거: 동일 위치(±5px) + 동일 텍스트 단어는 1차 결과 우선
                existing_keys = {
                    (w["text"], w["px_x"] // 10, w["px_y"] // 10) for w in words
                }
                for w in fallback_words:
                    key = (w["text"], w["px_x"] // 10, w["px_y"] // 10)
                    if key not in existing_keys:
                        words.append(w)
                        existing_keys.add(key)
                logger.info(
                    f"Tesseract PSM11 fallback added {len(fallback_words)} words "
                    f"(total now {len(words)})"
                )
            except Exception as exc:
                logger.warning(f"Tesseract PSM11 fallback skipped: {exc}")

    logger.info(
        f"Local Tesseract detected {len(words)} words for pre-OCR masking "
        f"(scale={scale:.2f}, min_conf={_TESSERACT_MIN_CONFIDENCE})"
    )
    return words


def _contains_seal_keyword(text: str) -> bool:
    stripped = text.strip()
    if stripped == "인":
        return True
    return any(keyword in stripped for keyword in SEAL_KEYWORDS if len(keyword) > 1)


def find_seal_regions(
    words: List[dict],
    img_width: int,
    img_height: int,
) -> List[Box]:
    lines = _group_words_into_lines(words, img_width, img_height)
    regions = _collect_pattern_regions(
        lines,
        _IDENTITY_FIELD_PATTERN,
        region_type="seal_signature",
        group_name="value",
        img_width=img_width,
        img_height=img_height,
        extra_right_px=max(int(img_width * _IDENTITY_EXTRA_RIGHT_RATIO), 36),
        extra_vertical_px=10,
    )

    if not regions:
        return []

    boxes, _ = _merge_regions(regions)
    return boxes


def find_sensitive_regions_from_words(
    words: List[dict],
    img_width: int,
    img_height: int,
    image_bytes: Optional[bytes] = None,
) -> Tuple[List[Box], List[str]]:
    if not words or img_width <= 0 or img_height <= 0:
        base_regions = _find_red_seal_regions(image_bytes) if image_bytes else []
        return base_regions, (["seal_signature"] if base_regions else [])

    lines = _group_words_into_lines(words, img_width, img_height)
    regions: List[Tuple[Box, str]] = []

    labelled_specs = [
        (RESIDENT_FIELD_PATTERN, "resident_id"),
        (PHONE_FIELD_PATTERN, "phone"),
        (ADDRESS_FIELD_PATTERN, "address"),
        (NAME_FIELD_PATTERN, "name"),
        (ACCOUNT_FIELD_PATTERN, "account"),
        (EMAIL_FIELD_PATTERN, "email"),
        (BUSINESS_NO_FIELD_PATTERN, "business_no"),
        (CORPORATE_NO_FIELD_PATTERN, "corporation_no"),
        (BIRTH_DATE_FIELD_PATTERN, "birth_date"),
        (PASSPORT_FIELD_PATTERN, "passport_no"),
        (DRIVER_LICENSE_FIELD_PATTERN, "driver_license_no"),
    ]
    for pattern, region_type in labelled_specs:
        regions.extend(
            _collect_pattern_regions(
                lines,
                pattern,
                region_type=region_type,
                group_name="value",
                img_width=img_width,
                img_height=img_height,
            )
        )

    direct_specs = [
        (RESIDENT_ID_PATTERN, "resident_id"),
        (PHONE_PATTERN, "phone"),
        (EMAIL_PATTERN, "email"),
        (BUSINESS_NO_PATTERN, "business_no"),
        (CORPORATE_NO_PATTERN, "corporation_no"),
        (BANK_ACCOUNT_PATTERN, "account"),
        (PASSPORT_VALUE_PATTERN, "passport_no"),
        (DRIVER_LICENSE_VALUE_PATTERN, "driver_license_no"),
        (ADDRESS_FRAGMENT_PATTERN, "address"),
    ]
    for pattern, region_type in direct_specs:
        regions.extend(
            _collect_pattern_regions(
                lines,
                pattern,
                region_type=region_type,
                group_name=None,
                img_width=img_width,
                img_height=img_height,
            )
        )

    regions.extend(
        _collect_pattern_regions(
            lines,
            _IDENTITY_FIELD_PATTERN,
            region_type="seal_signature",
            group_name="value",
            img_width=img_width,
            img_height=img_height,
            extra_right_px=max(int(img_width * _IDENTITY_EXTRA_RIGHT_RATIO), 36),
            extra_vertical_px=10,
        )
    )

    for word in words:
        text = str(word.get("text", "")).strip()
        if text and _contains_seal_keyword(text):
            regions.append((_word_to_box(word, img_width, img_height), "seal_signature"))

    if image_bytes:
        regions.extend((region, "seal_signature") for region in _find_red_seal_regions(image_bytes))

    return _merge_regions(regions)


def mask_image_regions(image_bytes: bytes, mask_regions: List[Box]) -> bytes:
    if not mask_regions:
        return image_bytes

    image = _decode_image(image_bytes)

    for left, top, right, bottom in mask_regions:
        if right <= left or bottom <= top:
            continue
        roi = image[top:bottom, left:right]
        roi_h, roi_w = roi.shape[:2]
        if roi_h < 1 or roi_w < 1:
            continue
        # Scale the blur kernel with the masked region so small boxes are still obscured.
        kernel_size = max(51, (min(roi_w, roi_h) // 2) | 1)
        blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        image[top:bottom, left:right] = blurred

    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("Failed to encode blurred image")
    return encoded.tobytes()


def mask_image_with_words(
    image_bytes: bytes,
    words: Optional[List[dict]],
    img_width: int,
    img_height: int,
) -> Tuple[bytes, int]:
    if not words or img_width <= 0 or img_height <= 0:
        return image_bytes, 0

    regions, mask_types = find_sensitive_regions_from_words(
        words,
        img_width,
        img_height,
        image_bytes=image_bytes,
    )
    if not regions:
        logger.debug("No sensitive image regions were detected from OCR words")
        return image_bytes, 0

    logger.info(f"Masking {len(regions)} image regions from OCR words: {mask_types}")
    return mask_image_regions(image_bytes, regions), len(regions)


def mask_image_for_ocr(image_bytes: bytes) -> ImageMaskingResult:
    try:
        img_width, img_height = _image_size(image_bytes)
        words = detect_words_with_tesseract(image_bytes)
        regions, mask_types = find_sensitive_regions_from_words(
            words,
            img_width,
            img_height,
            image_bytes=image_bytes,
        )
        if not regions:
            return ImageMaskingResult(
                image_bytes=image_bytes,
                mask_count=0,
                mask_types=[],
                regions=[],
            )

        masked_bytes = mask_image_regions(image_bytes, regions)
        logger.info(f"Pre-OCR masking applied to {len(regions)} regions: {mask_types}")
        return ImageMaskingResult(
            image_bytes=masked_bytes,
            mask_count=len(regions),
            mask_types=mask_types,
            regions=regions,
        )
    except Exception as exc:
        logger.error(f"Pre-OCR masking failed: {type(exc).__name__}: {exc}")
        return ImageMaskingResult(
            image_bytes=image_bytes,
            mask_count=0,
            mask_types=[],
            regions=[],
            masking_failed=True,
            error_message=f"{type(exc).__name__}: {exc}",
        )
