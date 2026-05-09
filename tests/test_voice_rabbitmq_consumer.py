import pytest

from app.schemas.voice_contract_fact_check import VoiceContractFactCheckRequest
from app.services import voice_contract_fact_check_consumer, voice_rabbitmq_consumer
from app.services.voice import contract_fact_check_service


@pytest.mark.asyncio
async def test_resolve_contract_text_uses_contract_id_only(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_fetch_contract_text(contract_id, s3_key=None):
        captured["contract_id"] = contract_id
        captured["s3_key"] = s3_key
        return " contract text "

    monkeypatch.setattr(
        contract_fact_check_service,
        "fetch_contract_text",
        fake_fetch_contract_text,
    )

    request = VoiceContractFactCheckRequest(
        voiceRecordId=2,
        contractId=10,
        userId=1,
        jobId="9bd9b27a-f756-4386-ad91-4f6c97e77558",
        s3Key="contracts/example-audio.mp3",
    )

    text = await contract_fact_check_service.resolve_contract_text(request)

    assert text == "contract text"
    assert captured == {"contract_id": 10, "s3_key": None}


def test_legacy_consumer_module_reexports_new_consumer_aliases():
    assert (
        voice_rabbitmq_consumer.voice_consumer
        is voice_contract_fact_check_consumer.voice_contract_fact_check_consumer
    )
    assert (
        voice_rabbitmq_consumer.start_voice_consumer
        is voice_contract_fact_check_consumer.start_voice_contract_fact_check_consumer
    )
