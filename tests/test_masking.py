"""
PII 마스킹 단위 테스트

텍스트 마스킹 및 이미지 마스킹 기능을 검증한다.
실제 S3/MongoDB 연결 없이 순수 로직만 테스트한다.
"""
import io
import pytest

from app.services.masking.text_masker import TextMasker
from app.services.masking.image_masker import (
    find_seal_regions,
    mask_image_regions,
    mask_image_with_words,
)
from app.services.masking.masking_service import _build_masked_s3_key


# ================================================
# TextMasker 테스트
# ================================================

class TestResidentIdMasking:
    """주민번호 마스킹 테스트"""

    def test_basic_resident_id(self):
        """기본 주민번호 마스킹"""
        masker = TextMasker()
        text = "홍길동 (주민번호: 901225-1234567)"
        result, positions = masker.mask_resident_id(text)

        assert "901225-1******" in result
        assert "1234567" not in result  # 뒷자리 원본이 남아있으면 안 됨
        assert len(positions) == 1
        assert positions[0].mask_type == "resident_id"

    def test_foreigner_registration_number(self):
        """외국인등록번호 마스킹 (뒷자리 5~8로 시작)"""
        masker = TextMasker()
        text = "외국인: 901225-5123456"
        result, positions = masker.mask_resident_id(text)

        assert "901225-5******" in result
        assert "5123456" not in result
        assert len(positions) == 1

    def test_multiple_resident_ids(self):
        """여러 주민번호 동시 마스킹"""
        masker = TextMasker()
        text = "임대인: 850315-1234567, 임차인: 920811-2345678"
        result, positions = masker.mask_resident_id(text)

        assert "1234567" not in result
        assert "2345678" not in result
        assert len(positions) == 2

    def test_no_resident_id(self):
        """주민번호 없는 텍스트 — 변경 없어야 함"""
        masker = TextMasker()
        original = "제1조 본 계약의 목적물은 서울시 강남구에 소재한다."
        result, positions = masker.mask_resident_id(original)

        assert result == original
        assert len(positions) == 0


class TestPhoneMasking:
    """전화번호 마스킹 테스트"""

    def test_basic_phone(self):
        """기본 010 전화번호 마스킹"""
        masker = TextMasker()
        text = "연락처: 010-1234-5678"
        result, positions = masker.mask_phone(text)

        assert "010-****-****" in result
        assert "1234" not in result
        assert "5678" not in result
        assert len(positions) == 1
        assert positions[0].mask_type == "phone"

    def test_phone_011(self):
        """011 번호 마스킹"""
        masker = TextMasker()
        text = "전화: 011-123-4567"
        result, positions = masker.mask_phone(text)

        assert "011-****-****" in result
        assert len(positions) == 1

    def test_phone_016(self):
        """016 번호 마스킹"""
        masker = TextMasker()
        text = "연락: 016-9876-5432"
        result, positions = masker.mask_phone(text)

        assert "016-****-****" in result
        assert len(positions) == 1

    def test_no_phone(self):
        """전화번호 없는 텍스트 — 변경 없어야 함"""
        masker = TextMasker()
        original = "보증금 1억원, 월세 50만원"
        result, positions = masker.mask_phone(original)

        assert result == original
        assert len(positions) == 0


class TestAddressMasking:
    """상세주소 마스킹 테스트"""

    def test_dong_ho_pattern(self):
        """동호 패턴 마스킹"""
        masker = TextMasker()
        text = "서울시 강남구 테헤란로 123, 101동 502호"
        result, positions = masker.mask_address(text)

        assert "[상세주소 마스킹]" in result
        # 주소 대분류(강남구, 테헤란로)는 남아있어야 함
        assert "강남구" in result
        assert len(positions) >= 1
        assert any(p.mask_type == "address" for p in positions)

    def test_apartment_pattern(self):
        """아파트명 + 동호 패턴 마스킹"""
        masker = TextMasker()
        text = "래미안아파트 101동 502호에 거주"
        result, positions = masker.mask_address(text)

        assert "[상세주소 마스킹]" in result
        assert "101동" not in result
        assert "502호" not in result

    def test_no_address_detail(self):
        """상세주소 없는 텍스트 — 변경 없어야 함"""
        masker = TextMasker()
        original = "서울특별시 강남구 역삼동 소재 부동산"
        result, positions = masker.mask_address(original)

        # 동호 패턴이 없으므로 변경 없어야 함
        assert "101동" not in original
        assert len(positions) == 0


class TestAccountMasking:
    """계좌번호 마스킹 테스트"""

    def test_basic_account(self):
        """기본 계좌번호 마스킹"""
        masker = TextMasker()
        text = "보증금 반환 계좌: 국민은행 123-456-789012"
        result, positions = masker.mask_account(text)

        assert "[계좌번호 마스킹]" in result
        assert "789012" not in result
        assert len(positions) == 1
        assert positions[0].mask_type == "account"

    def test_no_account(self):
        """계좌번호 없는 텍스트 — 변경 없어야 함"""
        masker = TextMasker()
        original = "제2조 임대 기간은 2년으로 한다."
        result, positions = masker.mask_account(original)

        assert result == original
        assert len(positions) == 0


class TestMaskAll:
    """전체 마스킹 (mask_all) 테스트"""

    def test_mask_all_combined(self):
        """여러 PII가 혼재된 계약서 텍스트 전체 마스킹"""
        masker = TextMasker()
        text = (
            "임대인 홍길동 (주민번호: 850315-1234567, 연락처: 010-1234-5678)\n"
            "주소: 서울시 강남구 테헤란로 123, 101동 502호\n"
            "보증금 반환: 국민은행 123-456-789012\n"
            "제1조 본 계약은 선량한 관리자의 주의의무를 다한다."
        )
        result = masker.mask_all(text)

        # 주민번호 뒷자리 마스킹 확인
        assert "1234567" not in result.masked_text
        assert "850315-1******" in result.masked_text

        # 전화번호 마스킹 확인
        assert "010-****-****" in result.masked_text

        # 상세주소 마스킹 확인
        assert "[상세주소 마스킹]" in result.masked_text

        # 계좌번호 마스킹 확인
        assert "[계좌번호 마스킹]" in result.masked_text

        # PII 없는 계약 조항은 그대로 남아있어야 함
        assert "선량한 관리자의 주의의무" in result.masked_text

        # 마스킹 카운트 확인
        assert result.mask_count >= 4

    def test_no_pii_text_unchanged(self):
        """PII가 없는 계약 조항 텍스트는 변경되면 안 됨"""
        masker = TextMasker()
        original = (
            "제1조 (목적물) 임대인은 다음 부동산을 임차인에게 임대한다.\n"
            "제2조 (임대차 기간) 임대차 기간은 2024년 1월 1일부터 2025년 12월 31일까지로 한다.\n"
            "제3조 (차임) 월 차임은 금 오십만원정 (\\500,000)으로 한다."
        )
        result = masker.mask_all(original)

        assert result.masked_text == original
        assert result.mask_count == 0
        assert result.mask_types_found == []


# ================================================
# 이미지 마스킹 테스트
# ================================================

class TestImageMasking:
    """이미지 마스킹 테스트"""

    def _create_dummy_jpeg(self, width: int = 100, height: int = 100) -> bytes:
        """단위 테스트용 더미 JPEG 이미지 생성"""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow 미설치 — 이미지 마스킹 테스트 스킵")

        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        output = io.BytesIO()
        img.save(output, format="JPEG")
        output.seek(0)
        return output.read()

    def test_mask_image_regions_basic(self):
        """픽셀 좌표 기반 이미지 마스킹 기본 동작"""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow 미설치")

        image_bytes = self._create_dummy_jpeg(200, 200)
        regions = [(10, 10, 50, 50)]  # 검정 박스 영역

        masked = mask_image_regions(image_bytes, regions)
        assert masked is not None
        assert len(masked) > 0

        # 마스킹된 이미지에서 해당 영역이 검정인지 확인
        img = Image.open(io.BytesIO(masked))
        pixel = img.getpixel((30, 30))  # 마스킹 영역 내부
        assert pixel == (0, 0, 0), f"마스킹 영역이 검정이 아님: {pixel}"

    def test_mask_image_regions_no_regions(self):
        """마스킹 영역이 없으면 원본 반환"""
        image_bytes = self._create_dummy_jpeg(100, 100)
        result = mask_image_regions(image_bytes, [])

        # 빈 regions면 원본 그대로 반환
        assert result == image_bytes

    def test_find_seal_regions_with_keywords(self):
        """인감/서명 키워드가 포함된 words에서 영역 탐지"""
        words = [
            {"text": "인감", "x": 10.0, "y": 20.0, "width": 5.0, "height": 3.0},
            {"text": "서명란", "x": 50.0, "y": 80.0, "width": 8.0, "height": 4.0},
            {"text": "계약내용", "x": 30.0, "y": 50.0, "width": 10.0, "height": 3.0},
        ]
        regions = find_seal_regions(words, img_width=1000, img_height=1000)

        assert len(regions) == 2  # "인감"과 "서명란"만 탐지
        # "계약내용"은 탐지되지 않아야 함

    def test_find_seal_regions_no_keywords(self):
        """인감/서명 키워드가 없으면 빈 리스트 반환"""
        words = [
            {"text": "제1조", "x": 10.0, "y": 10.0, "width": 5.0, "height": 3.0},
            {"text": "임대인", "x": 20.0, "y": 20.0, "width": 6.0, "height": 3.0},
        ]
        regions = find_seal_regions(words, img_width=1000, img_height=1000)

        assert len(regions) == 0

    def test_mask_image_with_words_no_words(self):
        """words가 None이면 원본 이미지 반환"""
        image_bytes = self._create_dummy_jpeg(100, 100)
        masked, count = mask_image_with_words(
            image_bytes=image_bytes,
            words=None,
            img_width=100,
            img_height=100,
        )

        assert masked == image_bytes
        assert count == 0

    def test_mask_image_with_words_seal_detected(self):
        """인감 키워드 포함 words로 이미지 마스킹 수행"""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow 미설치")

        image_bytes = self._create_dummy_jpeg(500, 500)
        words = [
            {"text": "인감", "x": 10.0, "y": 10.0, "width": 10.0, "height": 5.0},
        ]
        masked, count = mask_image_with_words(
            image_bytes=image_bytes,
            words=words,
            img_width=500,
            img_height=500,
        )

        assert count == 1
        assert masked != image_bytes  # 마스킹 후 이미지가 변경되어야 함


# ================================================
# S3 키 변환 테스트
# ================================================

class TestS3KeyConversion:
    """마스킹본 S3 키 생성 테스트"""

    def test_original_path_conversion(self):
        """original 경로가 masked로 교체되는지 확인"""
        original_key = "contracts/1/10/original/image.jpg"
        masked_key = _build_masked_s3_key(original_key)

        assert masked_key == "contracts/1/10/masked/image.jpg"

    def test_flat_path_conversion(self):
        """/original 없는 경로에 /masked 삽입"""
        original_key = "contracts/1/10/image.jpg"
        masked_key = _build_masked_s3_key(original_key)

        assert masked_key == "contracts/1/10/masked/image.jpg"

    def test_simple_filename(self):
        """경로 구분자 없는 단순 파일명"""
        original_key = "image.jpg"
        masked_key = _build_masked_s3_key(original_key)

        assert masked_key == "masked/image.jpg"
