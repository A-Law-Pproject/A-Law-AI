from __future__ import annotations

import asyncio
import io

import pytest

from app.api.endpoints import ocr as ocr_endpoint
from app.schemas.ocr_response import ContractOCRResponse, OCRWord
from app.services.masking.image_masker import (
    ImageMaskingResult,
    find_sensitive_regions_from_words,
    mask_image_regions,
    mask_image_with_words,
)
from app.services.masking.masking_service import _build_masked_s3_key
from app.services.masking.text_masker import TextMasker


def _create_dummy_jpeg(width: int = 800, height: int = 600, *, striped: bool = False) -> bytes:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        pytest.skip("Pillow is required for image masking tests")

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    if striped:
        draw = ImageDraw.Draw(image)
        for x in range(width):
            color = (0, 0, 0) if (x // 4) % 2 == 0 else (255, 255, 255)
            draw.line((x, 0, x, height), fill=color)

    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def _contract_words() -> list[dict]:
    return [
        {"text": "주소", "x": 5.0, "y": 10.0, "width": 5.0, "height": 4.0},
        {"text": "경기도", "x": 12.0, "y": 10.0, "width": 8.0, "height": 4.0},
        {"text": "수원시", "x": 22.0, "y": 10.0, "width": 8.0, "height": 4.0},
        {"text": "장안구", "x": 32.0, "y": 10.0, "width": 8.0, "height": 4.0},
        {"text": "서부로", "x": 42.0, "y": 10.0, "width": 8.0, "height": 4.0},
        {"text": "2126번길", "x": 52.0, "y": 10.0, "width": 12.0, "height": 4.0},
        {"text": "137-12", "x": 66.0, "y": 10.0, "width": 10.0, "height": 4.0},
        {"text": "주민번호", "x": 5.0, "y": 20.0, "width": 10.0, "height": 4.0},
        {"text": "850315-1234567", "x": 18.0, "y": 20.0, "width": 22.0, "height": 4.0},
        {"text": "휴대전화", "x": 45.0, "y": 20.0, "width": 10.0, "height": 4.0},
        {"text": "010-8771-5169", "x": 58.0, "y": 20.0, "width": 18.0, "height": 4.0},
        {"text": "성명", "x": 78.0, "y": 20.0, "width": 6.0, "height": 4.0},
        {"text": "홍길동", "x": 86.0, "y": 20.0, "width": 8.0, "height": 4.0},
    ]


def _extended_pii_words() -> list[dict]:
    return [
        {"text": "이메일", "x": 5.0, "y": 10.0, "width": 8.0, "height": 4.0},
        {"text": "alice@example.com", "x": 15.0, "y": 10.0, "width": 24.0, "height": 4.0},
        {"text": "계좌번호", "x": 43.0, "y": 10.0, "width": 10.0, "height": 4.0},
        {"text": "국민은행", "x": 55.0, "y": 10.0, "width": 8.0, "height": 4.0},
        {"text": "123-456-789012", "x": 65.0, "y": 10.0, "width": 20.0, "height": 4.0},
        {"text": "사업자등록번호", "x": 5.0, "y": 20.0, "width": 16.0, "height": 4.0},
        {"text": "123-45-67890", "x": 23.0, "y": 20.0, "width": 18.0, "height": 4.0},
        {"text": "법인등록번호", "x": 45.0, "y": 20.0, "width": 14.0, "height": 4.0},
        {"text": "110111-1234567", "x": 61.0, "y": 20.0, "width": 20.0, "height": 4.0},
        {"text": "생년월일", "x": 5.0, "y": 30.0, "width": 10.0, "height": 4.0},
        {"text": "1990-01-02", "x": 17.0, "y": 30.0, "width": 14.0, "height": 4.0},
        {"text": "여권번호", "x": 35.0, "y": 30.0, "width": 10.0, "height": 4.0},
        {"text": "M12345678", "x": 47.0, "y": 30.0, "width": 12.0, "height": 4.0},
        {"text": "운전면허번호", "x": 63.0, "y": 30.0, "width": 14.0, "height": 4.0},
        {"text": "12-34-567890-12", "x": 79.0, "y": 30.0, "width": 18.0, "height": 4.0},
    ]


class TestTextMasker:
    def test_resident_id_masks_rear_digits(self):
        masker = TextMasker()
        result, positions = masker.mask_resident_id("주민번호 850315-1234567")

        assert "850315-1******" in result
        assert "1234567" not in result
        assert positions[0].mask_type == "resident_id"

    def test_phone_masks_mobile_number(self):
        masker = TextMasker()
        result, positions = masker.mask_phone("휴대전화 010-8771-5169")

        assert "010-****-****" in result
        assert "8771" not in result
        assert positions[0].mask_type == "phone"

    def test_address_masks_full_labelled_address(self):
        masker = TextMasker()
        result, positions = masker.mask_address("주소 경기도 수원시 장안구 서부로 2126번길 137-12")

        assert result == "주소 [주소 마스킹]"
        assert positions
        assert positions[0].mask_type == "address"

    def test_mask_all_combines_common_pii_categories(self):
        masker = TextMasker()
        text = (
            "주민번호 850315-1234567\n"
            "휴대전화 010-8771-5169\n"
            "주소 경기도 수원시 장안구 서부로 2126번길 137-12\n"
            "반환계좌 국민은행 123-456-789012"
        )

        result = masker.mask_all(text)

        assert "850315-1******" in result.masked_text
        assert "010-****-****" in result.masked_text
        assert "주소 [주소 마스킹]" in result.masked_text
        assert "[계좌번호 마스킹]" in result.masked_text
        assert {"account", "address", "phone", "resident_id"} <= set(result.mask_types_found)

    def test_mask_all_masks_broader_labelled_personal_fields(self):
        masker = TextMasker()
        text = (
            "성명 홍길동\n"
            "이메일 alice@example.com\n"
            "사업자등록번호 123-45-67890\n"
            "법인등록번호 110111-1234567\n"
            "생년월일 1990-01-02\n"
            "여권번호 M12345678\n"
            "운전면허번호 12-34-567890-12"
        )

        result = masker.mask_all(text)

        assert "성명 [성명 마스킹]" in result.masked_text
        assert "이메일 [이메일 마스킹]" in result.masked_text
        assert "사업자등록번호 [사업자번호 마스킹]" in result.masked_text
        assert "법인등록번호 [법인번호 마스킹]" in result.masked_text
        assert "생년월일 [생년월일 마스킹]" in result.masked_text
        assert "여권번호 [여권번호 마스킹]" in result.masked_text
        assert "운전면허번호 [면허번호 마스킹]" in result.masked_text
        assert {
            "birth_date",
            "business_no",
            "corporation_no",
            "driver_license_no",
            "email",
            "name",
            "passport_no",
        } <= set(result.mask_types_found)

    def test_mask_all_keeps_general_contract_date_without_birth_label(self):
        masker = TextMasker()
        result = masker.mask_all("계약일 2024-05-24")

        assert "2024-05-24" in result.masked_text
        assert "birth_date" not in result.mask_types_found


class TestImageMasking:
    def test_mask_image_regions_blurs_region(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow is required for image masking tests")

        image_bytes = _create_dummy_jpeg(200, 200, striped=True)
        original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        masked = mask_image_regions(image_bytes, [(20, 20, 180, 180)])
        masked_image = Image.open(io.BytesIO(masked)).convert("RGB")

        assert masked_image.getpixel((100, 100)) != original.getpixel((100, 100))

    def test_find_sensitive_regions_detects_address_phone_resident_name_and_signature(self):
        regions, mask_types = find_sensitive_regions_from_words(
            _contract_words(),
            img_width=800,
            img_height=600,
        )

        assert len(regions) >= 4
        assert {"address", "name", "phone", "resident_id", "seal_signature"} <= set(mask_types)

    def test_find_sensitive_regions_detects_broader_personal_fields(self):
        regions, mask_types = find_sensitive_regions_from_words(
            _extended_pii_words(),
            img_width=800,
            img_height=600,
        )

        assert len(regions) >= 5
        assert {
            "account",
            "birth_date",
            "business_no",
            "corporation_no",
            "driver_license_no",
            "email",
            "passport_no",
        } <= set(mask_types)

    def test_mask_image_with_words_masks_sensitive_value_boxes(self):
        image_bytes = _create_dummy_jpeg(800, 600, striped=True)
        masked_bytes, count = mask_image_with_words(
            image_bytes=image_bytes,
            words=_contract_words(),
            img_width=800,
            img_height=600,
        )

        assert count >= 4
        assert masked_bytes != image_bytes


class TestS3KeyConversion:
    def test_original_path_conversion(self):
        assert _build_masked_s3_key("contracts/1/10/original/image.jpg") == "contracts/1/10/masked/image.jpg"

    def test_flat_path_conversion(self):
        assert _build_masked_s3_key("contracts/1/10/image.jpg") == "contracts/1/10/masked/image.jpg"

    def test_simple_filename(self):
        assert _build_masked_s3_key("image.jpg") == "masked/image.jpg"


class _DummyUploadFile:
    def __init__(self, content: bytes):
        self._content = content
        self.content_type = "image/png"

    async def read(self) -> bytes:
        return self._content


class _DummyService:
    def __init__(self, response: ContractOCRResponse):
        self.response = response
        self.received_bytes = None
        self.received_include_overlay = None

    def process_and_map(self, image_bytes: bytes, structurize: bool, include_overlay: bool) -> ContractOCRResponse:
        self.received_bytes = image_bytes
        self.received_include_overlay = include_overlay
        return self.response.model_copy(deep=True)


class _DummyS3Client:
    def __init__(self, image_bytes: bytes):
        self.image_bytes = image_bytes

    def get_image(self, _: str) -> bytes:
        return self.image_bytes


def test_run_ocr_full_uses_pre_masked_image(monkeypatch):
    monkeypatch.setattr(ocr_endpoint.settings, "ENABLE_MASKING", True)
    monkeypatch.setattr(
        ocr_endpoint,
        "mask_image_for_ocr",
        lambda image_bytes: ImageMaskingResult(
            image_bytes=b"masked-image",
            mask_count=2,
            mask_types=["phone", "resident_id"],
        ),
    )

    service = _DummyService(
        ContractOCRResponse(
            success=True,
            processing_time=0.1,
            image_width=800,
            image_height=600,
            full_text="휴대전화 010-8771-5169",
            markdown="주소 경기도 수원시 장안구 서부로 2126번길 137-12",
            contract_data={"tenantPhone": "010-8771-5169"},
        )
    )

    result = asyncio.run(
        ocr_endpoint.run_ocr_full(
            file=_DummyUploadFile(b"original-image"),
            structurize=False,
            include_overlay=False,
            service=service,
        )
    )

    assert service.received_bytes == b"masked-image"
    assert service.received_include_overlay is True
    assert result.words is None
    assert result.full_text == "휴대전화 010-****-****"
    assert result.markdown == "주소 [주소 마스킹]"
    assert result.contract_data["tenantPhone"] == "010-****-****"


def test_run_ocr_from_s3_passes_pre_masking_metadata_to_storage(monkeypatch):
    monkeypatch.setattr(ocr_endpoint.settings, "ENABLE_MASKING", True)
    monkeypatch.setattr(
        ocr_endpoint,
        "mask_image_for_ocr",
        lambda image_bytes: ImageMaskingResult(
            image_bytes=b"masked-image",
            mask_count=3,
            mask_types=["address", "phone", "resident_id"],
        ),
    )

    saved = {}

    async def fake_save_ocr_result(s3_key, result, *, image_url=None):
        saved["s3_key"] = s3_key
        saved["result"] = result
        saved["image_url"] = image_url

    async def fake_mask_and_store(**kwargs):
        saved["mask_kwargs"] = kwargs
        return type(
            "_MaskStoreResult",
            (),
            {
                "success": True,
                "masked_s3_key": None,
                "metadata": None,
                "error_message": None,
            },
        )()

    monkeypatch.setattr(ocr_endpoint, "save_ocr_result", fake_save_ocr_result)
    monkeypatch.setattr(ocr_endpoint, "mask_and_store", fake_mask_and_store)

    response = ContractOCRResponse(
        success=True,
        processing_time=0.2,
        image_width=800,
        image_height=600,
        full_text="주민번호 850315-1234567 휴대전화 010-8771-5169",
        words=[
            OCRWord(text="휴대전화", x=10, y=10, width=10, height=5),
            OCRWord(text="010-8771-5169", x=25, y=10, width=15, height=5),
        ],
    )
    service = _DummyService(response)

    result = asyncio.run(
        ocr_endpoint.run_ocr_from_s3(
            request=ocr_endpoint.OCRRequest(s3_key="contracts/1/10/original/image.jpg"),
            include_overlay=False,
            s3_client=_DummyS3Client(b"original-image"),
            service=service,
        )
    )

    assert service.received_bytes == b"masked-image"
    assert service.received_include_overlay is True
    assert result.words is None
    assert saved["s3_key"] == "contracts/1/10/original/image.jpg"
    assert saved["mask_kwargs"]["image_bytes_for_storage"] == b"masked-image"
    assert saved["mask_kwargs"]["pre_mask_count"] == 3
    assert saved["mask_kwargs"]["pre_mask_types"] == ["address", "phone", "resident_id"]
    assert saved["mask_kwargs"]["original_text"] == "주민번호 850315-1****** 휴대전화 010-****-****"


def test_pre_ocr_masking_failure_blocks_ocr(monkeypatch):
    monkeypatch.setattr(ocr_endpoint.settings, "ENABLE_MASKING", True)
    monkeypatch.setattr(
        ocr_endpoint,
        "mask_image_for_ocr",
        lambda image_bytes: ImageMaskingResult(
            image_bytes=image_bytes,
            masking_failed=True,
            error_message="boom",
        ),
    )

    service = _DummyService(
        ContractOCRResponse(success=True, processing_time=0.1, full_text="ok")
    )

    with pytest.raises(Exception) as exc_info:
        asyncio.run(
            ocr_endpoint.run_ocr_full(
                file=_DummyUploadFile(b"original-image"),
                structurize=False,
                include_overlay=False,
                service=service,
            )
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "사전 마스킹에 실패하여 OCR을 진행할 수 없습니다."
