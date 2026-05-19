from langchain_core.documents import Document

from app.rag.chain.chain import _legal_reference_from_doc
from app.schemas.contract_analysis_dto import ClauseRiskResult
from app.services.rabbitmq_consumer import RabbitMQConsumer


def test_clause_risk_result_serializes_related_work_alias():
    result = ClauseRiskResult(
        clause_title="보증금",
        clause_content="보증금은 계약 체결일에 지급한다.",
        risk_level="주의",
        category="보증금",
        score=52,
        legal_reference="주택임대차보호법 제3조의2",
        related_work="보증금 지급 시기와 반환 조건을 같이 확인해야 합니다.",
        reasoning_summary="보증금 지급 시기와 반환 조건을 같이 확인해야 합니다.",
    )

    payload = result.model_dump(by_alias=True)

    assert payload["legalReference"] == "주택임대차보호법 제3조의2"
    assert payload["relatedWork"] == "보증금 지급 시기와 반환 조건을 같이 확인해야 합니다."


def test_rabbitmq_consumer_recovers_clause_fields_from_analysis():
    clause = {
        "analysis": (
            "보증금 반환 시점이 불명확합니다. "
            "확인된 법령 근거: 주택임대차보호법 제3조의2; 민법 제623조."
        )
    }

    assert RabbitMQConsumer._resolve_clause_legal_reference(clause) == "주택임대차보호법 제3조의2"
    assert (
        RabbitMQConsumer._resolve_clause_related_work(clause)
        == "보증금 반환 시점이 불명확합니다. 확인된 법령 근거: 주택임대차보호법 제3조의2; 민법 제623조."
    )


def test_legal_reference_from_doc_falls_back_to_law_name():
    doc = Document(
        page_content="보증금 반환 관련 설명",
        metadata={
            "collection": "law_statutes",
            "law_name": "주택임대차보호법",
        },
    )

    assert _legal_reference_from_doc(doc) == "주택임대차보호법"
