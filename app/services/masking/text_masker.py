"""
Text-based PII masking for OCR output.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from loguru import logger

from app.schemas.masking import MaskPosition, TextMaskingResult


_ADDRESS_MARKER = "[주소 마스킹]"
_ACCOUNT_MARKER = "[계좌번호 마스킹]"

_RESIDENT_ID_PATTERN = re.compile(r"(\d{6})\s*[- ]\s*([1-8])\s*(\d{6})")
_PHONE_PATTERN = re.compile(r"(01[016789])\s*[- ]\s*(\d{3,4})\s*[- ]\s*(\d{4})")

_ADDRESS_STOP = (
    r"(?:주\s*민\s*(?:등\s*록\s*)?번\s*호|휴\s*대\s*전\s*화|전\s*화|연\s*락\s*처|"
    r"성\s*명|대\s*표|등\s*록\s*번\s*호|상\s*호|소\s*속\s*공\s*인\s*중\s*개\s*사)"
)
_ADDRESS_FIELD_PATTERN = re.compile(
    rf"(?P<label>주\s*소|소\s*재\s*지|재\s*지)\s*[:：]?\s*(?P<value>.+?)"
    rf"(?=(?:\s+(?:{_ADDRESS_STOP}))|$)",
    re.MULTILINE,
)
_ADDRESS_UNIT_PATTERN = re.compile(
    r"(?:(?:\d{1,4}\s*동)\s*)?(?:\d{1,4}\s*층\s*)?(?:\d{1,4}\s*호)(?:\s*,?\s*\d{1,4}\s*호)?"
)

_ACCOUNT_PATTERN = re.compile(
    r"(?:국민|신한|하나|우리|농협|기업|카카오뱅크|케이뱅크|토스|SC제일|우체국|수협|새마을|부산|대구|광주|전북|경남)?\s*"
    r"(?:은행)?\s*"
    r"(\d{2,6}(?:-\d{2,6}){2,4})"
)


class TextMasker:
    def mask_all(self, text: str) -> TextMaskingResult:
        positions: List[MaskPosition] = []
        current_text = text
        total_offset = 0

        current_text, resident_positions, resident_offset = self._mask_resident_id(current_text, total_offset)
        positions.extend(resident_positions)
        total_offset += resident_offset

        current_text, phone_positions, phone_offset = self._mask_phone(current_text, total_offset)
        positions.extend(phone_positions)
        total_offset += phone_offset

        current_text, address_positions, address_offset = self._mask_address(current_text, total_offset)
        positions.extend(address_positions)
        total_offset += address_offset

        current_text, account_positions, account_offset = self._mask_account(current_text, total_offset)
        positions.extend(account_positions)
        total_offset += account_offset

        mask_types_found = sorted({position.mask_type for position in positions})
        logger.info(f"Text masking completed: {len(positions)} matches, types={mask_types_found}")

        return TextMaskingResult(
            masked_text=current_text,
            positions=positions,
            mask_count=len(positions),
            mask_types_found=mask_types_found,
        )

    def mask_resident_id(self, text: str) -> Tuple[str, List[MaskPosition]]:
        result, positions, _ = self._mask_resident_id(text, 0)
        return result, positions

    def mask_phone(self, text: str) -> Tuple[str, List[MaskPosition]]:
        result, positions, _ = self._mask_phone(text, 0)
        return result, positions

    def mask_address(self, text: str) -> Tuple[str, List[MaskPosition]]:
        result, positions, _ = self._mask_address(text, 0)
        return result, positions

    def mask_account(self, text: str) -> Tuple[str, List[MaskPosition]]:
        result, positions, _ = self._mask_account(text, 0)
        return result, positions

    def _mask_resident_id(self, text: str, base_offset: int) -> Tuple[str, List[MaskPosition], int]:
        positions: List[MaskPosition] = []
        offset = 0

        def replace_fn(match: re.Match[str]) -> str:
            nonlocal offset
            original = match.group(0)
            masked = f"{match.group(1)}-{match.group(2)}******"
            start = match.start() + base_offset + offset
            end = start + len(original)
            positions.append(
                MaskPosition(
                    start=start,
                    end=end,
                    mask_type="resident_id",
                    original_length=len(original),
                )
            )
            offset += len(masked) - len(original)
            return masked

        return _RESIDENT_ID_PATTERN.sub(replace_fn, text), positions, offset

    def _mask_phone(self, text: str, base_offset: int) -> Tuple[str, List[MaskPosition], int]:
        positions: List[MaskPosition] = []
        offset = 0

        def replace_fn(match: re.Match[str]) -> str:
            nonlocal offset
            original = match.group(0)
            masked = f"{match.group(1)}-****-****"
            start = match.start() + base_offset + offset
            end = start + len(original)
            positions.append(
                MaskPosition(
                    start=start,
                    end=end,
                    mask_type="phone",
                    original_length=len(original),
                )
            )
            offset += len(masked) - len(original)
            return masked

        return _PHONE_PATTERN.sub(replace_fn, text), positions, offset

    def _mask_address(self, text: str, base_offset: int) -> Tuple[str, List[MaskPosition], int]:
        positions: List[MaskPosition] = []
        offset = 0

        def replace_labelled(match: re.Match[str]) -> str:
            nonlocal offset
            original = match.group(0)
            label = re.sub(r"\s+", "", match.group("label"))
            value = match.group("value").strip()
            if not value:
                return original

            replacement = f"{label} {_ADDRESS_MARKER}"
            value_start = match.start("value") + base_offset + offset
            value_end = value_start + len(match.group("value"))
            positions.append(
                MaskPosition(
                    start=value_start,
                    end=value_end,
                    mask_type="address",
                    original_length=len(match.group("value")),
                )
            )
            offset += len(replacement) - len(original)
            return replacement

        current_text = _ADDRESS_FIELD_PATTERN.sub(replace_labelled, text)

        def replace_unit(match: re.Match[str]) -> str:
            nonlocal offset
            original = match.group(0)
            start = match.start() + base_offset + offset
            end = start + len(original)
            positions.append(
                MaskPosition(
                    start=start,
                    end=end,
                    mask_type="address",
                    original_length=len(original),
                )
            )
            offset += len(_ADDRESS_MARKER) - len(original)
            return _ADDRESS_MARKER

        current_text = _ADDRESS_UNIT_PATTERN.sub(replace_unit, current_text)
        return current_text, positions, offset

    def _mask_account(self, text: str, base_offset: int) -> Tuple[str, List[MaskPosition], int]:
        positions: List[MaskPosition] = []
        offset = 0

        def replace_fn(match: re.Match[str]) -> str:
            nonlocal offset
            original = match.group(0)
            start = match.start() + base_offset + offset
            end = start + len(original)
            positions.append(
                MaskPosition(
                    start=start,
                    end=end,
                    mask_type="account",
                    original_length=len(original),
                )
            )
            offset += len(_ACCOUNT_MARKER) - len(original)
            return _ACCOUNT_MARKER

        return _ACCOUNT_PATTERN.sub(replace_fn, text), positions, offset
