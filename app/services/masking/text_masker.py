"""
텍스트 PII 마스킹 모듈
OCR로 추출된 계약서 텍스트에서 개인정보를 정규식으로 탐지하고 마스킹한다.

주의: 마스킹 처리 중 PII 원본값을 로그에 절대 출력하지 않는다.
"""
import re
from typing import List, Tuple

from loguru import logger

from app.schemas.masking import MaskPosition, TextMaskingResult


# ================================================
# 마스킹 패턴 상수
# ================================================

# 주민번호 앞자리 6자리 + 하이픈 + 뒷자리 7자리 (내국인: 1~4, 외국인: 5~8)
_RESIDENT_ID_PATTERN = re.compile(
    r"(\d{6})-([1-8]\d{6})"
)

# 전화번호: 010/011/016/017/018/019 + 3~4자리 + 4자리
_PHONE_PATTERN = re.compile(
    r"(01[016789])-(\d{3,4})-(\d{4})"
)

# 상세주소: "숫자동 숫자호" 또는 "숫자-숫자호" 패턴
_ADDRESS_DETAIL_PATTERN = re.compile(
    r"(\d+)\s*동\s*(\d+)\s*호"
)

# 아파트명 + 동호: "○○아파트 숫자동 숫자호" 패턴
_APARTMENT_PATTERN = re.compile(
    r"([가-힣a-zA-Z0-9]+\s*아파트)\s+\d+\s*동\s*\d+\s*호"
)

# 계좌번호: 은행명(선택) + 숫자-숫자-숫자 패턴 (10~14자리 이상)
_ACCOUNT_PATTERN = re.compile(
    r"(?:국민|신한|하나|우리|농협|기업|카카오|케이|토스|SC제일|씨티|우체국|수협|새마을|제주|전북|광주|대구|부산|경남)?\s*"
    r"(?:은행)?\s*"
    r"(\d{3,6}-\d{2,6}-\d{4,6}(?:-\d{2,4})?)"
)


class TextMasker:
    """
    계약서 텍스트 PII 마스킹 처리기

    각 마스킹 유형은 독립 메서드로 분리되어 단위 테스트 가능하다.
    마스킹은 우선순위 순으로 적용된다:
    1. 주민번호/외국인등록번호
    2. 전화번호
    3. 상세주소
    4. 계좌번호
    """

    def mask_all(self, text: str) -> TextMaskingResult:
        """
        모든 PII 패턴을 순서대로 마스킹하고 결과를 반환한다.

        Args:
            text: 원본 OCR 텍스트

        Returns:
            TextMaskingResult (마스킹된 텍스트 + 위치 목록)
        """
        positions: List[MaskPosition] = []
        current_text = text
        total_offset = 0  # 누적 오프셋 (마스킹 후 길이 변화 반영)

        # 1. 주민번호 마스킹
        current_text, res_positions, res_offset = self._mask_resident_id(current_text, total_offset)
        positions.extend(res_positions)
        total_offset += res_offset

        # 2. 전화번호 마스킹
        current_text, phone_positions, phone_offset = self._mask_phone(current_text, total_offset)
        positions.extend(phone_positions)
        total_offset += phone_offset

        # 3. 상세주소 마스킹 (아파트 패턴 우선)
        current_text, addr_positions, addr_offset = self._mask_address(current_text, total_offset)
        positions.extend(addr_positions)
        total_offset += addr_offset

        # 4. 계좌번호 마스킹
        current_text, acc_positions, acc_offset = self._mask_account(current_text, total_offset)
        positions.extend(acc_positions)
        total_offset += acc_offset

        mask_types_found = list({p.mask_type for p in positions})

        logger.info(
            f"텍스트 마스킹 완료 - 총 {len(positions)}건, 유형: {mask_types_found}"
        )

        return TextMaskingResult(
            masked_text=current_text,
            positions=positions,
            mask_count=len(positions),
            mask_types_found=mask_types_found,
        )

    def mask_resident_id(self, text: str) -> Tuple[str, List[MaskPosition]]:
        """
        주민번호/외국인등록번호 마스킹 (공개 인터페이스 - 단위 테스트용)

        패턴: XXXXXX-YYYYYYY → XXXXXX-Y******
        앞자리 6자리는 그대로, 뒷자리 첫 자리만 공개하고 나머지 6자리를 마스킹
        """
        result, positions, _ = self._mask_resident_id(text, 0)
        return result, positions

    def mask_phone(self, text: str) -> Tuple[str, List[MaskPosition]]:
        """전화번호 마스킹 (공개 인터페이스 - 단위 테스트용)"""
        result, positions, _ = self._mask_phone(text, 0)
        return result, positions

    def mask_address(self, text: str) -> Tuple[str, List[MaskPosition]]:
        """상세주소 마스킹 (공개 인터페이스 - 단위 테스트용)"""
        result, positions, _ = self._mask_address(text, 0)
        return result, positions

    def mask_account(self, text: str) -> Tuple[str, List[MaskPosition]]:
        """계좌번호 마스킹 (공개 인터페이스 - 단위 테스트용)"""
        result, positions, _ = self._mask_account(text, 0)
        return result, positions

    # ------------------------------------------------
    # 내부 마스킹 메서드 (offset 반환 포함)
    # ------------------------------------------------

    def _mask_resident_id(
        self, text: str, base_offset: int
    ) -> Tuple[str, List[MaskPosition], int]:
        """
        주민번호 뒷자리 마스킹
        예: 901225-1234567 → 901225-1******
        """
        positions: List[MaskPosition] = []
        offset = 0  # 현재 누적 길이 변화

        def replace_fn(match: re.Match) -> str:
            nonlocal offset
            original = match.group(0)
            front = match.group(1)   # 앞자리 6자리
            back = match.group(2)    # 뒷자리 7자리
            masked = f"{front}-{back[0]}******"

            # 위치 기록 (원본 기준 + base_offset)
            start = match.start() + base_offset + offset
            end = start + len(original)
            positions.append(MaskPosition(
                start=start,
                end=end,
                mask_type="resident_id",
                original_length=len(original),
            ))

            # 길이 변화 추적: 원본 13자리 → 마스킹 13자리 (동일 길이)
            offset += len(masked) - len(original)
            return masked

        result = _RESIDENT_ID_PATTERN.sub(replace_fn, text)
        return result, positions, offset

    def _mask_phone(
        self, text: str, base_offset: int
    ) -> Tuple[str, List[MaskPosition], int]:
        """
        전화번호 마스킹
        예: 010-1234-5678 → 010-****-****
        """
        positions: List[MaskPosition] = []
        offset = 0

        def replace_fn(match: re.Match) -> str:
            nonlocal offset
            original = match.group(0)
            prefix = match.group(1)  # 010/011/...
            masked = f"{prefix}-****-****"

            start = match.start() + base_offset + offset
            end = start + len(original)
            positions.append(MaskPosition(
                start=start,
                end=end,
                mask_type="phone",
                original_length=len(original),
            ))

            offset += len(masked) - len(original)
            return masked

        result = _PHONE_PATTERN.sub(replace_fn, text)
        return result, positions, offset

    def _mask_address(
        self, text: str, base_offset: int
    ) -> Tuple[str, List[MaskPosition], int]:
        """
        상세주소 마스킹 (동호 단위)
        예: "101동 502호" → "[상세주소 마스킹]"
        예: "래미안아파트 101동 502호" → "[상세주소 마스킹]"
        """
        positions: List[MaskPosition] = []
        offset = 0
        masked_marker = "[상세주소 마스킹]"

        def replace_fn(match: re.Match) -> str:
            nonlocal offset
            original = match.group(0)

            start = match.start() + base_offset + offset
            end = start + len(original)
            positions.append(MaskPosition(
                start=start,
                end=end,
                mask_type="address",
                original_length=len(original),
            ))

            offset += len(masked_marker) - len(original)
            return masked_marker

        # 아파트 패턴 먼저 적용 (더 넓은 범위)
        text = _APARTMENT_PATTERN.sub(replace_fn, text)

        # 일반 동호 패턴 적용
        result = _ADDRESS_DETAIL_PATTERN.sub(replace_fn, text)
        return result, positions, offset

    def _mask_account(
        self, text: str, base_offset: int
    ) -> Tuple[str, List[MaskPosition], int]:
        """
        계좌번호 마스킹
        예: "국민은행 123-456-789012" → "[계좌번호 마스킹]"
        """
        positions: List[MaskPosition] = []
        offset = 0
        masked_marker = "[계좌번호 마스킹]"

        def replace_fn(match: re.Match) -> str:
            nonlocal offset
            original = match.group(0)

            start = match.start() + base_offset + offset
            end = start + len(original)
            positions.append(MaskPosition(
                start=start,
                end=end,
                mask_type="account",
                original_length=len(original),
            ))

            offset += len(masked_marker) - len(original)
            return masked_marker

        result = _ACCOUNT_PATTERN.sub(replace_fn, text)
        return result, positions, offset
