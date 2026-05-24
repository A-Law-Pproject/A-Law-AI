"""
Text-based PII masking for OCR output.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from loguru import logger

from app.schemas.masking import MaskPosition, TextMaskingResult
from app.services.masking.patterns import (
    ACCOUNT_FIELD_PATTERN,
    ADDRESS_FIELD_PATTERN,
    ADDRESS_FRAGMENT_PATTERN,
    BANK_ACCOUNT_PATTERN,
    BIRTH_DATE_FIELD_PATTERN,
    BIRTH_DATE_PATTERN,
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
    STANDALONE_NAME_PATTERN,
)


_ADDRESS_MARKER = "[주소 마스킹]"
_ACCOUNT_MARKER = "[계좌번호 마스킹]"
_EMAIL_MARKER = "[이메일 마스킹]"
_BUSINESS_NO_MARKER = "[사업자번호 마스킹]"
_CORPORATE_NO_MARKER = "[법인번호 마스킹]"
_NAME_MARKER = "[성명 마스킹]"
_PHONE_MARKER = "[전화번호 마스킹]"
_RESIDENT_MARKER = "[신분번호 마스킹]"
_BIRTH_DATE_MARKER = "[생년월일 마스킹]"
_PASSPORT_MARKER = "[여권번호 마스킹]"
_DRIVER_LICENSE_MARKER = "[면허번호 마스킹]"


class TextMasker:
    def mask_all(self, text: str) -> TextMaskingResult:
        positions: List[MaskPosition] = []
        current_text = text
        total_offset = 0

        pre_labelled_maskers = [
            self._mask_address,
            self._mask_labelled_account,
            self._mask_labelled_email,
            self._mask_labelled_business_no,
            self._mask_labelled_corporate_no,
            self._mask_labelled_birth_date,
            self._mask_labelled_passport_no,
            self._mask_labelled_driver_license_no,
            self._mask_name,
        ]
        direct_maskers = [
            self._mask_resident_id,
            self._mask_phone,
            self._mask_email,
            self._mask_business_no,
            self._mask_corporate_no,
            self._mask_account,
            self._mask_passport_no,
            self._mask_driver_license_no,
        ]
        post_labelled_maskers = [
            self._mask_labelled_phone,
            self._mask_labelled_resident_id,
        ]

        for masker in [*pre_labelled_maskers, *direct_maskers, *post_labelled_maskers]:
            current_text, new_positions, delta = masker(current_text, total_offset)
            positions.extend(new_positions)
            total_offset += delta

        current_text, standalone_positions, standalone_delta = self._mask_standalone_name(
            current_text,
            total_offset,
        )
        positions.extend(standalone_positions)
        total_offset += standalone_delta

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
        result, positions, offset = self._mask_account(text, 0)
        result, labelled_positions, _ = self._mask_labelled_account(result, offset)
        positions.extend(labelled_positions)
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
                MaskPosition(start=start, end=end, mask_type="resident_id", original_length=len(original))
            )
            offset += len(masked) - len(original)
            return masked

        return RESIDENT_ID_PATTERN.sub(replace_fn, text), positions, offset

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
                MaskPosition(start=start, end=end, mask_type="phone", original_length=len(original))
            )
            offset += len(masked) - len(original)
            return masked

        return PHONE_PATTERN.sub(replace_fn, text), positions, offset

    def _mask_email(self, text: str, base_offset: int) -> Tuple[str, List[MaskPosition], int]:
        return self._mask_with_marker(
            text,
            base_offset,
            EMAIL_PATTERN,
            marker=_EMAIL_MARKER,
            mask_type="email",
        )

    def _mask_business_no(self, text: str, base_offset: int) -> Tuple[str, List[MaskPosition], int]:
        return self._mask_with_marker(
            text,
            base_offset,
            BUSINESS_NO_PATTERN,
            marker=_BUSINESS_NO_MARKER,
            mask_type="business_no",
        )

    def _mask_corporate_no(self, text: str, base_offset: int) -> Tuple[str, List[MaskPosition], int]:
        return self._mask_with_marker(
            text,
            base_offset,
            CORPORATE_NO_PATTERN,
            marker=_CORPORATE_NO_MARKER,
            mask_type="corporation_no",
        )

    def _mask_birth_date(self, text: str, base_offset: int) -> Tuple[str, List[MaskPosition], int]:
        return self._mask_with_marker(
            text,
            base_offset,
            BIRTH_DATE_PATTERN,
            marker=_BIRTH_DATE_MARKER,
            mask_type="birth_date",
        )

    def _mask_passport_no(self, text: str, base_offset: int) -> Tuple[str, List[MaskPosition], int]:
        return self._mask_with_marker(
            text,
            base_offset,
            PASSPORT_VALUE_PATTERN,
            marker=_PASSPORT_MARKER,
            mask_type="passport_no",
        )

    def _mask_driver_license_no(
        self,
        text: str,
        base_offset: int,
    ) -> Tuple[str, List[MaskPosition], int]:
        return self._mask_with_marker(
            text,
            base_offset,
            DRIVER_LICENSE_VALUE_PATTERN,
            marker=_DRIVER_LICENSE_MARKER,
            mask_type="driver_license_no",
        )

    def _mask_account(self, text: str, base_offset: int) -> Tuple[str, List[MaskPosition], int]:
        return self._mask_with_marker(
            text,
            base_offset,
            BANK_ACCOUNT_PATTERN,
            marker=_ACCOUNT_MARKER,
            mask_type="account",
        )

    def _mask_address(self, text: str, base_offset: int) -> Tuple[str, List[MaskPosition], int]:
        current_text, positions, offset = self._mask_labelled_field(
            text,
            base_offset,
            ADDRESS_FIELD_PATTERN,
            marker=_ADDRESS_MARKER,
            mask_type="address",
        )
        current_text, direct_positions, direct_offset = self._mask_with_marker(
            current_text,
            base_offset + offset,
            ADDRESS_FRAGMENT_PATTERN,
            marker=_ADDRESS_MARKER,
            mask_type="address",
        )
        positions.extend(direct_positions)
        offset += direct_offset
        return current_text, positions, offset

    def _mask_labelled_phone(self, text: str, base_offset: int) -> Tuple[str, List[MaskPosition], int]:
        return self._mask_labelled_field(
            text,
            base_offset,
            PHONE_FIELD_PATTERN,
            marker=_PHONE_MARKER,
            mask_type="phone",
        )

    def _mask_labelled_resident_id(
        self,
        text: str,
        base_offset: int,
    ) -> Tuple[str, List[MaskPosition], int]:
        return self._mask_labelled_field(
            text,
            base_offset,
            RESIDENT_FIELD_PATTERN,
            marker=_RESIDENT_MARKER,
            mask_type="resident_id",
        )

    def _mask_labelled_account(self, text: str, base_offset: int) -> Tuple[str, List[MaskPosition], int]:
        return self._mask_labelled_field(
            text,
            base_offset,
            ACCOUNT_FIELD_PATTERN,
            marker=_ACCOUNT_MARKER,
            mask_type="account",
        )

    def _mask_labelled_email(self, text: str, base_offset: int) -> Tuple[str, List[MaskPosition], int]:
        return self._mask_labelled_field(
            text,
            base_offset,
            EMAIL_FIELD_PATTERN,
            marker=_EMAIL_MARKER,
            mask_type="email",
        )

    def _mask_labelled_business_no(
        self,
        text: str,
        base_offset: int,
    ) -> Tuple[str, List[MaskPosition], int]:
        return self._mask_labelled_field(
            text,
            base_offset,
            BUSINESS_NO_FIELD_PATTERN,
            marker=_BUSINESS_NO_MARKER,
            mask_type="business_no",
        )

    def _mask_labelled_corporate_no(
        self,
        text: str,
        base_offset: int,
    ) -> Tuple[str, List[MaskPosition], int]:
        return self._mask_labelled_field(
            text,
            base_offset,
            CORPORATE_NO_FIELD_PATTERN,
            marker=_CORPORATE_NO_MARKER,
            mask_type="corporation_no",
        )

    def _mask_labelled_birth_date(
        self,
        text: str,
        base_offset: int,
    ) -> Tuple[str, List[MaskPosition], int]:
        return self._mask_labelled_field(
            text,
            base_offset,
            BIRTH_DATE_FIELD_PATTERN,
            marker=_BIRTH_DATE_MARKER,
            mask_type="birth_date",
        )

    def _mask_labelled_passport_no(
        self,
        text: str,
        base_offset: int,
    ) -> Tuple[str, List[MaskPosition], int]:
        return self._mask_labelled_field(
            text,
            base_offset,
            PASSPORT_FIELD_PATTERN,
            marker=_PASSPORT_MARKER,
            mask_type="passport_no",
        )

    def _mask_labelled_driver_license_no(
        self,
        text: str,
        base_offset: int,
    ) -> Tuple[str, List[MaskPosition], int]:
        return self._mask_labelled_field(
            text,
            base_offset,
            DRIVER_LICENSE_FIELD_PATTERN,
            marker=_DRIVER_LICENSE_MARKER,
            mask_type="driver_license_no",
        )

    def _mask_name(self, text: str, base_offset: int) -> Tuple[str, List[MaskPosition], int]:
        return self._mask_labelled_field(
            text,
            base_offset,
            NAME_FIELD_PATTERN,
            marker=_NAME_MARKER,
            mask_type="name",
        )

    def _mask_standalone_name(self, text: str, base_offset: int) -> Tuple[str, List[MaskPosition], int]:
        stripped = text.strip()
        if not stripped or not STANDALONE_NAME_PATTERN.fullmatch(stripped):
            return text, [], 0

        start = text.index(stripped) + base_offset
        end = start + len(stripped)
        positions = [
            MaskPosition(start=start, end=end, mask_type="name", original_length=len(stripped))
        ]
        return text.replace(stripped, _NAME_MARKER, 1), positions, len(_NAME_MARKER) - len(stripped)

    def _mask_with_marker(
        self,
        text: str,
        base_offset: int,
        pattern: re.Pattern[str],
        *,
        marker: str,
        mask_type: str,
    ) -> Tuple[str, List[MaskPosition], int]:
        positions: List[MaskPosition] = []
        offset = 0

        def replace_fn(match: re.Match[str]) -> str:
            nonlocal offset
            original = match.group(0)
            start = match.start() + base_offset + offset
            end = start + len(original)
            positions.append(
                MaskPosition(start=start, end=end, mask_type=mask_type, original_length=len(original))
            )
            offset += len(marker) - len(original)
            return marker

        return pattern.sub(replace_fn, text), positions, offset

    def _mask_labelled_field(
        self,
        text: str,
        base_offset: int,
        pattern: re.Pattern[str],
        *,
        marker: str,
        mask_type: str,
    ) -> Tuple[str, List[MaskPosition], int]:
        positions: List[MaskPosition] = []
        offset = 0

        def replace_fn(match: re.Match[str]) -> str:
            nonlocal offset
            original = match.group(0)
            label = self._normalize_label(match.group("label"))
            value_text = match.group("value")
            value = value_text.strip()
            if not value:
                return original
            if self._is_already_masked(value):
                return original

            replacement = f"{label} {marker}"
            value_start = match.start("value") + base_offset + offset
            value_end = value_start + len(value_text)
            positions.append(
                MaskPosition(
                    start=value_start,
                    end=value_end,
                    mask_type=mask_type,
                    original_length=len(value_text),
                )
            )
            offset += len(replacement) - len(original)
            return replacement

        return pattern.sub(replace_fn, text), positions, offset

    def _normalize_label(self, label: str) -> str:
        return re.sub(r"\s+", "", label).strip(":：")

    def _is_already_masked(self, value: str) -> bool:
        return "마스킹" in value or "*" in value
