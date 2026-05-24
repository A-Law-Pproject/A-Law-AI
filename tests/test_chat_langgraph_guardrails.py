import time

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.rag.chain.langgraph_tool_pipelines import (
    _classify_query,
    _ensure_readable_markdown_answer,
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


class _FakeInlineListLLM:
    async def ainvoke(self, messages):
        return AIMessage(
            content=(
                "집주인이 실거주를 이유로 계약 갱신을 거절할 경우, 몇 가지 중요한 사항이 있습니다: "
                "1. **증명 책임**: 집주인은 실제로 거주할 의사를 입증해야 합니다. "
                "2. **정당한 사유**: 본인 또는 직계존비속의 실거주 계획이 있어야 합니다. "
                "3. **갱신 요구 기간**: 임차인은 만료 전 법정 기간 안에 갱신을 요구해야 합니다. "
                "4. **손해배상**: 허위 실거주 후 제3자에게 임대하면 손해배상을 청구할 수 있습니다. "
                "따라서 집주인의 실제 거주 계획과 이후 사용 형태를 확인하는 것이 중요합니다."
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


def test_ensure_readable_markdown_answer_splits_inline_numbered_items():
    raw = (
        "집주인이 실거주를 이유로 계약 갱신을 거절할 경우, 몇 가지 중요한 사항이 있습니다: "
        "1. **증명 책임**: 집주인은 실제로 거주할 의사를 입증해야 합니다. "
        "2. **정당한 사유**: 본인 또는 직계존비속의 실거주 계획이 있어야 합니다. "
        "3. **갱신 요구 기간**: 임차인은 만료 전 법정 기간 안에 갱신을 요구해야 합니다. "
        "4. **손해배상**: 허위 실거주 후 제3자에게 임대하면 손해배상을 청구할 수 있습니다. "
        "따라서 집주인의 실제 거주 계획과 이후 사용 형태를 확인하는 것이 중요합니다."
    )

    formatted = _ensure_readable_markdown_answer(raw)

    assert "사항이 있습니다:\n\n1. **증명 책임**" in formatted
    assert "\n2. **정당한 사유**" in formatted
    assert "\n3. **갱신 요구 기간**" in formatted
    assert "\n4. **손해배상**" in formatted
    assert "\n\n따라서 집주인의 실제 거주 계획" in formatted


def test_ensure_readable_markdown_answer_strips_heading_markers():
    raw = "### ?붿빟\n蹂댁쬆湲덉? 怨꾩빟 醫낅즺 ??14???대궡 諛섑솚?대뒗 寃껋씠 ?먯튃?낅땲??"

    formatted = _ensure_readable_markdown_answer(raw)

    assert "###" not in formatted
    assert formatted.startswith("?붿빟\n")
    assert "蹂댁쬆湲덉?" in formatted


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


@pytest.mark.asyncio
async def test_repair_in_scope_rejection_formats_inline_numbered_list():
    docs = [
        Document(
            page_content="주택임대차보호법은 계약갱신요구권과 손해배상 규정을 둔다.",
            metadata={
                "collection": "law_database",
                "law_name": "주택임대차보호법 해설",
                "title": "계약갱신과 실거주 거절",
            },
        )
    ]

    repaired = await _repair_in_scope_rejection(
        question="집주인이 실거주를 이유로 갱신을 거절하면 어떻게 되나요",
        answer=(
            "죄송합니다. 저는 임대차 계약 관련 질문만 답변할 수 있습니다.\n"
            "전세·월세 계약, 보증금, 계약갱신, 독소조항 분석 등에 대해 질문해 주세요."
        ),
        history=[],
        context="[법령]\n주택임대차보호법은 계약갱신요구권과 손해배상 규정을 둔다.",
        docs=docs,
        llm=_FakeInlineListLLM(),
        deadline=time.perf_counter() + 5,
    )

    assert "사항이 있습니다:\n\n1. **증명 책임**" in repaired
    assert "\n2. **정당한 사유**" in repaired
    assert "\n\n따라서 집주인의 실제 거주 계획" in repaired
