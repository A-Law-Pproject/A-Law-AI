import time

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.rag.chain.langgraph_tool_pipelines import (
    _classify_query,
    _format_sources,
    _repair_in_scope_rejection,
)


class _FakeLLM:
    async def ainvoke(self, messages):
        return AIMessage(
            content=(
                "보증금은 계약서에 정한 지급일에 내는 것이 원칙입니다. "
                "지급일이 불명확하면 계약금·잔금·입주일 약정을 함께 확인하세요."
            )
        )


def test_classify_deposit_question_as_legal_qa():
    assert _classify_query("보증금을 언제 내야하나요") == "legal_qa"


def test_format_sources_hides_internal_collection_names():
    docs = [
        Document(
            page_content="주택의 인도와 주민등록을 마친 때에는 제삼자에 대하여 효력이 생긴다.",
            metadata={
                "collection": "law_statutes",
                "law_name": "주택임대차보호법",
                "article": "제3조",
            },
        )
    ]

    sources = _format_sources(docs)

    assert sources == [
        "[법령] 주택임대차보호법 제3조 — 주택의 인도와 주민등록을 마친 때에는 제삼자에 대하여 효력이 생긴다."
    ]
    assert all("law_statutes" not in source for source in sources)


@pytest.mark.asyncio
async def test_repair_in_scope_rejection_for_lease_question():
    docs = [
        Document(
            page_content="보증금은 약정한 날에 지급하는 것이 원칙이다.",
            metadata={
                "collection": "law_database",
                "law_name": "임대차 실무 해설",
                "title": "보증금 지급 시기",
            },
        )
    ]

    repaired = await _repair_in_scope_rejection(
        question="보증금을 언제 내야하나요",
        answer=(
            "죄송합니다. 저는 임대차 계약 관련 질문만 답변할 수 있습니다.\n"
            "전세·월세 계약, 보증금, 계약갱신, 독소조항 분석 등에 대해 질문해 주세요."
        ),
        history=[],
        context="[법령]\n보증금은 약정한 날에 지급하는 것이 원칙이다.",
        docs=docs,
        llm=_FakeLLM(),
        deadline=time.perf_counter() + 5,
    )

    assert "임대차 계약 관련 질문만 답변할 수 있습니다" not in repaired
    assert "보증금" in repaired
