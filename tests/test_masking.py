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


def _create_dummy_jpeg(width: int = 800, height: int = 600) -> bytes:
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow is required for image masking tests")

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def _contract_words() -> list[dict]:
    return [
        {"text": "주소", "x": 5.0, "y": 10.0, "width": 5.0, "height": 4.0},
        {"text": "경기도", "x": 14.0, "y": 10.0, "width": 8.0, "height": 4.0},
        {"text": "수원시", "x": 24.0, "y": 10.0, "width": 8.0, "height": 4.0},
        {"text": "장안구", "x": 34.0, "y": 10.0, "width": 8.0, "height": 4.0},
        {"text": "서부로", "x": 44.0, "y": 10.0, "width": 8.0, "height": 4.0},
        {"text": "2126번길", "x": 54.0, "y": 10.0, "width": 10.0, "height": 4.0},
        {"text": "137-12(천천동)", "x": 66.0, "y": 10.0, "width": 18.0, "height": 4.0},
        {"text": "주민번호", "x": 5.0, "y": 20.0, "width": 10.0, "height": 4.0},
        {"text": "850315-1234567", "x": 18.0, "y": 20.0, "width": 22.0, "height": 4.0},
        {"text": "휴대전화", "x": 45.0, "y": 20.0, "width": 10.0, "height": 4.0},
        {"text": "010-8771-5169", "x": 58.0, "y": 20.0, "width": 18.0, "height": 4.0},
        {"text": "성명", "x": 74.0, "y": 20.0, "width": 6.0, "height": 4.0},
        {"text": "홍길동", "x": 82.0, "y": 20.0, "width": 10.0, "height": 4.0},
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
        result, positions = masker.mask_address("주소 경기도 수원시 장안구 서부로 2126번길 137-12(천천동)")

        assert result == "주소 [주소 마스킹]"
        assert positions
        assert positions[0].mask_type == "address"

    def test_mask_all_combines_all_categories(self):
        masker = TextMasker()
        text = (
            "주민번호 850315-1234567\n"
            "휴대전화 010-8771-5169\n"
            "주소 경기도 수원시 장안구 서부로 2126번길 137-12(천천동)\n"
            "반환계좌 국민은행 123-456-789012"
        )

        result = masker.mask_all(text)

        assert "850315-1******" in result.masked_text
        assert "010-****-****" in result.masked_text
        assert "주소 [주소 마스킹]" in result.masked_text
        assert "[계좌번호 마스킹]" in result.masked_text
        assert set(result.mask_types_found) == {"account", "address", "phone", "resident_id"}


class TestImageMasking:
    def test_mask_image_regions_fills_box_in_black(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow is required for image masking tests")

        image_bytes = _create_dummy_jpeg(200, 200)
        masked = mask_image_regions(image_bytes, [(10, 10, 60, 60)])
        image = Image.open(io.BytesIO(masked))

        assert image.getpixel((30, 30)) == (0, 0, 0)

    def test_find_sensitive_regions_detects_address_phone_resident_and_signature(self):
        regions, mask_types = find_sensitive_regions_from_words(
            _contract_words(),
            img_width=800,
            img_height=600,
        )

        assert len(regions) >= 3
        assert {"address", "phone", "resident_id", "seal_signature"}.issubset(set(mask_types))

    def test_mask_image_with_words_masks_sensitive_value_boxes(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow is required for image masking tests")

        image_bytes = _create_dummy_jpeg(800, 600)
        masked_bytes, count = mask_image_with_words(
            image_bytes=image_bytes,
            words=_contract_words(),
            img_width=800,
            img_height=600,
        )
        image = Image.open(io.BytesIO(masked_bytes))

        assert count >= 3
        assert masked_bytes != image_bytes
        assert image.getpixel((560, 135)) == (0, 0, 0)


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
            markdown="주소 경기도 수원시 장안구 서부로 2126번길 137-12(천천동)",
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

    async def fake_save_ocr_result(s3_key, result):
        saved["s3_key"] = s3_key
        saved["result"] = result

    async def fake_mask_and_store(**kwargs):
        saved["mask_kwargs"] = kwargs
        return None

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

    assert "사전 마스킹" in str(exc_info.value)
