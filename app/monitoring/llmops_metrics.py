from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Any

from prometheus_client import Counter, Histogram

from app.core.config import settings

_LEGAL_CITATION_PATTERN = re.compile(
    r"(제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?|참조조문)"
)

_REFUSAL_PATTERNS = (
    "알 수 없",
    "정확한 조문 확인 필요",
    "관련 법률 문서를 찾을 수 없",
    "관련 법률 없음",
    "추가 확인이 필요",
    "법률 전문가와 상담",
)

RAG_LEGAL_CITATION_TOTAL = Counter(
    "rag_legal_citation_total",
    "법령 인용이 포함된 응답 수",
    ["endpoint", "has_citation"],
)

RAG_REFUSAL_RESPONSE_TOTAL = Counter(
    "rag_refusal_response_total",
    "회피성 또는 불확실성 응답 수",
    ["endpoint", "is_refusal"],
)

RAG_EMPTY_CONTEXT_TOTAL = Counter(
    "rag_empty_context_total",
    "실질 컨텍스트가 비어 있는 응답 수",
    ["endpoint", "is_empty_context"],
)

RAG_RETRIEVAL_HIT_TOTAL = Counter(
    "rag_retrieval_hit_total",
    "검색 결과 문서가 존재한 응답 수",
    ["endpoint", "has_documents"],
)

RAG_DOCUMENT_COUNT = Histogram(
    "rag_document_count",
    "응답 생성에 사용된 문서 수",
    ["endpoint"],
    buckets=[1, 2, 3, 5, 7, 10, 15, 20],
)

RAG_CONTEXT_LENGTH = Histogram(
    "rag_context_length_chars",
    "응답 생성에 사용된 컨텍스트 길이(문자 수)",
    ["endpoint"],
    buckets=[50, 100, 300, 600, 1000, 2000, 3000, 5000],
)

RAG_RESPONSE_LENGTH = Histogram(
    "rag_response_length_chars",
    "최종 응답 길이(문자 수)",
    ["endpoint"],
    buckets=[50, 100, 300, 600, 1000, 1500, 2500, 4000],
)

RAG_RERANKER_SCORE = Histogram(
    "rag_reranker_score",
    "재정렬된 문서의 reranker score",
    ["endpoint"],
    buckets=[-10, -5, -4, -3, -2, -1, 0, 1, 2, 4, 8],
)


def has_legal_citation(text: str) -> bool:
    return bool(_LEGAL_CITATION_PATTERN.search(text or ""))


def is_refusal_answer(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    return any(pattern in normalized for pattern in _REFUSAL_PATTERNS)


def is_empty_context(context: str) -> bool:
    normalized = re.sub(r"\s+", " ", context or "").strip()
    if not normalized:
        return True
    return any(
        phrase in normalized
        for phrase in ("관련 법률 없음", "관련 법률 문서를 찾을 수 없습니다.")
    )


def _bool_label(value: bool) -> str:
    return "true" if value else "false"


def _get_metadata(document: Any) -> dict[str, Any]:
    if hasattr(document, "metadata") and isinstance(document.metadata, dict):
        return document.metadata
    if isinstance(document, dict):
        metadata = document.get("metadata")
        if isinstance(metadata, dict):
            return metadata
    return {}


def _sample_enabled() -> bool:
    return (
        settings.ENABLE_LLMOPS_METRICS
        and settings.LLMOPS_METRIC_SAMPLE_RATE > 0
        and random.random() <= settings.LLMOPS_METRIC_SAMPLE_RATE
    )


def observe_rag_interaction(
    endpoint: str,
    answer: str,
    documents: Sequence[Any] | None,
    context: str,
) -> None:
    if not _sample_enabled():
        return

    docs = list(documents or [])
    citation = has_legal_citation(answer)
    refusal = is_refusal_answer(answer)
    empty = is_empty_context(context)
    has_docs = bool(docs)

    RAG_LEGAL_CITATION_TOTAL.labels(
        endpoint=endpoint,
        has_citation=_bool_label(citation),
    ).inc()
    RAG_REFUSAL_RESPONSE_TOTAL.labels(
        endpoint=endpoint,
        is_refusal=_bool_label(refusal),
    ).inc()
    RAG_EMPTY_CONTEXT_TOTAL.labels(
        endpoint=endpoint,
        is_empty_context=_bool_label(empty),
    ).inc()
    RAG_RETRIEVAL_HIT_TOTAL.labels(
        endpoint=endpoint,
        has_documents=_bool_label(has_docs),
    ).inc()
    RAG_DOCUMENT_COUNT.labels(endpoint=endpoint).observe(len(docs))
    RAG_CONTEXT_LENGTH.labels(endpoint=endpoint).observe(len(context or ""))
    RAG_RESPONSE_LENGTH.labels(endpoint=endpoint).observe(len(answer or ""))

    for document in docs[:3]:
        metadata = _get_metadata(document)
        rerank_score = metadata.get("rerank_score")
        if rerank_score is None:
            continue
        try:
            RAG_RERANKER_SCORE.labels(endpoint=endpoint).observe(float(rerank_score))
        except (TypeError, ValueError):
            continue
