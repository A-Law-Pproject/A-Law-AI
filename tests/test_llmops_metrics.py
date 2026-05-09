from types import SimpleNamespace

from app.monitoring.llmops_metrics import (
    has_legal_citation,
    is_empty_context,
    is_refusal_answer,
    observe_rag_interaction,
)


def test_has_legal_citation_detects_article_pattern():
    assert has_legal_citation("주택임대차보호법 제3조에 따라 보호됩니다.")
    assert not has_legal_citation("계약서 작성 전에 등기부등본을 확인하세요.")


def test_is_refusal_answer_detects_uncertain_answer():
    assert is_refusal_answer("정확한 조문 확인 필요")
    assert not is_refusal_answer("계약 갱신 요구권은 1회 행사할 수 있습니다.")


def test_is_empty_context_detects_fallback_phrase():
    assert is_empty_context("관련 법률 문서를 찾을 수 없습니다.")
    assert not is_empty_context("[문서 1 - law_statutes] 주택임대차보호법 제3조")


def test_observe_rag_interaction_accepts_fake_documents():
    docs = [SimpleNamespace(metadata={"rerank_score": 1.2})]
    observe_rag_interaction(
        endpoint="test",
        answer="주택임대차보호법 제3조에 따라 보호됩니다.",
        documents=docs,
        context="[문서 1 - law_statutes] 주택임대차보호법 제3조",
    )
