import pytest

from app.services.voice import hash_service


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


@pytest.mark.asyncio
async def test_upload_to_s3_uses_boto3_client(monkeypatch):
    calls = {}

    class _FakeS3Client:
        def put_object(self, **kwargs):
            calls["put_object"] = kwargs

    def _fake_boto3_client(service_name, **kwargs):
        calls["service_name"] = service_name
        calls["client_kwargs"] = kwargs
        return _FakeS3Client()

    monkeypatch.setattr(hash_service.boto3, "client", _fake_boto3_client)
    monkeypatch.setattr(hash_service.settings, "AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setattr(hash_service.settings, "AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setattr(hash_service.settings, "AWS_REGION", "ap-northeast-2")
    monkeypatch.setattr(hash_service.settings, "AWS_S3_BUCKET", "test-bucket")

    result = await hash_service.upload_to_s3(b"voice-bytes", "audio/test.mp3", "audio/mpeg")

    assert result is True
    assert calls["service_name"] == "s3"
    assert calls["client_kwargs"] == {
        "aws_access_key_id": "test-access-key",
        "aws_secret_access_key": "test-secret-key",
        "region_name": "ap-northeast-2",
    }
    assert calls["put_object"] == {
        "Bucket": "test-bucket",
        "Key": "audio/test.mp3",
        "Body": b"voice-bytes",
        "ContentType": "audio/mpeg",
    }


@pytest.mark.asyncio
async def test_download_from_s3_uses_boto3_client(monkeypatch):
    calls = {}

    class _FakeS3Client:
        def get_object(self, **kwargs):
            calls["get_object"] = kwargs
            return {"Body": _FakeBody(b"downloaded-audio")}

    def _fake_boto3_client(service_name, **kwargs):
        calls["service_name"] = service_name
        calls["client_kwargs"] = kwargs
        return _FakeS3Client()

    monkeypatch.setattr(hash_service.boto3, "client", _fake_boto3_client)
    monkeypatch.setattr(hash_service.settings, "AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setattr(hash_service.settings, "AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setattr(hash_service.settings, "AWS_REGION", "ap-northeast-2")
    monkeypatch.setattr(hash_service.settings, "AWS_S3_BUCKET", "test-bucket")

    result = await hash_service.download_from_s3("audio/test.mp3")

    assert result == b"downloaded-audio"
    assert calls["service_name"] == "s3"
    assert calls["client_kwargs"] == {
        "aws_access_key_id": "test-access-key",
        "aws_secret_access_key": "test-secret-key",
        "region_name": "ap-northeast-2",
    }
    assert calls["get_object"] == {
        "Bucket": "test-bucket",
        "Key": "audio/test.mp3",
    }
