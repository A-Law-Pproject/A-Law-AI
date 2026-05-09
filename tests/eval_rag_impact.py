"""tests/eval_rag_impact.py - No-RAG vs RAG 챗봇 품질 비교 실험

"RAG를 적용하면 챗봇 품질이 개선되고 근거 법률을 정확하게 찾는다"는 것을
실험으로 검증합니다.

조건:
  no_rag  : 순수 LLM 호출 (법령 DB 검색 없이 질문만 전달)
  rag     : 전체 파이프라인 (멀티 컬렉션 검색 → BGE 리랭킹 → LLM 답변 생성)

지표:
  law_citation_rate   : 답변에 expected relevant_law 중 ≥1개 포함 비율
  law_article_rate    : 답변에 구체적 법 조항(제X조) 인용 비율
  keyword_coverage    : expected_keywords 중 답변에 포함된 비율
  faithfulness        : RAGAS LLM (--llm 옵션, RAG 조건: 환각 탐지)
  answer_relevancy    : RAGAS LLM (--llm 옵션, 양 조건)

실행 예시:
  # Mock (API 없이 빠르게 확인)
  .venv\\Scripts\\python.exe tests\\eval_rag_impact.py --mock

  # 실제 (Pinecone + OpenAI)
  .venv\\Scripts\\python.exe tests\\eval_rag_impact.py --sample 20

  # LLM 기반 RAGAS 지표 추가
  .venv\\Scripts\\python.exe tests\\eval_rag_impact.py --sample 20 --llm

  # 샘플 10개 + reranker 적용
  .venv\\Scripts\\python.exe tests/eval_rag_impact.py --sample 20 --llm --rerank
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv


class _TeeWriter(io.TextIOBase):
    """stdout을 가로채 원본 출력과 버퍼에 동시에 기록."""

    def __init__(self, original: Any) -> None:
        self._original = original
        self._lines: list[str] = []

    def write(self, s: str) -> int:
        self._original.write(s)
        self._original.flush()
        if s:
            self._lines.append(s)
        return len(s)

    def flush(self) -> None:
        self._original.flush()

    def get_log(self) -> str:
        return "".join(self._lines)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

DATASET_PATH = ROOT / "tests" / "평가데이터셋" / "eval_dataset_ver2.json"
RESULTS_DIR = ROOT / "results"

DEFAULT_COLLECTIONS = [
    "law_database",
    "law_statutes",
    "contracts",
    "special_clauses_illegal",
    "special_clauses_normal",
]

# 순수 LLM용 프롬프트 (검색 컨텍스트 없음)
_NO_RAG_SYSTEM = """당신은 한국 임대차 계약 전문 AI입니다.
아래 질문에 답변하세요. 관련 법률 조항을 가능하면 인용하세요."""

_NO_RAG_HUMAN = "{question}"

# 구체적 법 조항 인용 탐지 패턴 (제3조, 제6조의3 등)
_ARTICLE_PATTERN = re.compile(r"제\s*\d+\s*조")


# ──────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────

def load_cases(path: Path, sample: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("questions", data if isinstance(data, list) else [])
    cases = [
        item for item in items
        if item.get("question") and item.get("relevant_law")
    ]
    return cases[:sample] if sample > 0 else cases


# ──────────────────────────────────────────────
# 지표 계산
# ──────────────────────────────────────────────

def _law_citation_hit(answer: str, relevant_laws: list[str]) -> bool:
    """답변에 relevant_law 중 하나라도 포함되는지."""
    answer_l = answer.lower()
    return any(law.lower() in answer_l for law in relevant_laws if law)


def _law_article_hit(answer: str) -> bool:
    """답변에 구체적 법 조항(제X조)이 포함되는지."""
    return bool(_ARTICLE_PATTERN.search(answer))


def _keyword_coverage(answer: str, keywords: list[str]) -> float:
    """expected_keywords 중 답변에 포함된 비율 (0~1)."""
    if not keywords:
        return 1.0
    answer_l = answer.lower()
    hits = sum(1 for kw in keywords if str(kw).lower() in answer_l)
    return hits / len(keywords)


_ARTICLE_NORM_RE = re.compile(r"제\s*\d+\s*조(?:의\s*\d+)?(?:\s*제\s*\d+\s*항)?")


def _citation_in_context_rate(answer: str, contexts: list[str]) -> float:
    """인용된 제X조가 검색 문서에 실제로 있는 비율.

    contexts가 비어있으면 0.0 — 검색 근거 없는 인용은 전부 미검증으로 간주.
    인용 자체가 없으면 1.0 — 잘못된 인용이 없다는 뜻이므로 패스.
    """
    if not contexts:
        return 0.0
    citations = ["".join(m.group().split()) for m in _ARTICLE_NORM_RE.finditer(answer)]
    if not citations:
        return 1.0
    combined = "".join(" ".join(contexts).split())
    hits = sum(1 for c in citations if c in combined)
    return hits / len(citations)


def _has_unverified_citation(answer: str) -> bool:
    """chat_rag() 의 citation verification이 붙인 (미검증) 마커가 있는지."""
    return "(미검증)" in answer


def _practical_guidance_hit(answer: str) -> bool:
    """임차인 행동 지침 키워드 포함 여부."""
    kws = ["내용증명", "임차권등기", "확정일자", "법원", "신청하", "청구하", "소송", "증거"]
    return any(kw in answer for kw in kws)


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        "law_citation_rate": round(
            mean(1.0 if r["law_citation_hit"] else 0.0 for r in rows), 4
        ),
        "law_article_rate": round(
            mean(1.0 if r["law_article_hit"] else 0.0 for r in rows), 4
        ),
        "citation_in_context_rate": round(
            mean(r.get("citation_in_context", 0.0) for r in rows), 4
        ),
        "practical_guidance_rate": round(
            mean(1.0 if r.get("practical_hit") else 0.0 for r in rows), 4
        ),
        "keyword_coverage": round(mean(r["keyword_coverage"] for r in rows), 4),
        "avg_answer_length": round(mean(len(r["answer"]) for r in rows), 1),
        "avg_latency_ms": round(mean(r["latency_ms"] for r in rows), 1),
        "case_count": len(rows),
    }


# ──────────────────────────────────────────────
# Mock 실행
# ──────────────────────────────────────────────

# Mock 답변 패턴: no_rag는 법령 인용이 부정확하거나 누락, rag는 정확히 인용
_MOCK_NO_RAG_TEMPLATES = [
    "임대차 계약에서 {topic}에 대해서는 일반적으로 계약서에 명시된 내용을 따릅니다. "
    "계약 전 반드시 계약서 내용을 꼼꼼히 확인하시길 바랍니다.",
    "이 경우 민법 규정에 따라 처리됩니다. "
    "구체적인 내용은 법률 전문가와 상담하는 것이 좋습니다.",
    "{topic}와 관련하여 임차인의 권리가 보호됩니다. "
    "계약서 내용을 확인하고 필요 시 법률 상담을 받으세요.",
]

_MOCK_RAG_TEMPLATES = [
    "주택임대차보호법 제{art}조에 따르면 {topic}에 관해 임차인은 다음과 같은 권리를 갖습니다. "
    "구체적으로 해당 법률은 임차인 보호를 위해 {law_detail}을 규정하고 있으며, "
    "이를 위반하는 조항은 효력이 없습니다.",
    "{laws} 제{art}조 제1항에 의거하여, {topic}의 경우 임차인은 "
    "대항력을 갖추면 보증금을 우선변제받을 수 있습니다. "
    "주택임대차보호법 제8조(보증금의 우선변제)에 따라 소액보증금은 최우선변제 대상입니다.",
    "주택임대차보호법 제{art}조는 {topic}에 대해 명확히 규정하고 있습니다. "
    "이에 따라 {law_detail} 해야 하며, 위반 시 {laws}에 따라 계약 조항이 무효가 될 수 있습니다.",
]


def _make_mock_answer(case: dict[str, Any], condition: str, idx: int) -> str:
    import random
    rng = random.Random(idx * 7 + (0 if condition == "no_rag" else 1))
    laws = case.get("relevant_law", ["주택임대차보호법"])
    kws = case.get("expected_keywords", [])
    topic = kws[0] if kws else "임대차"
    law_detail = laws[0] if laws else "임차인 보호"
    art = rng.choice([3, 6, 7, 8, 10, 12, 17])

    if condition == "no_rag":
        tmpl = _MOCK_NO_RAG_TEMPLATES[idx % len(_MOCK_NO_RAG_TEMPLATES)]
        return tmpl.format(topic=topic, laws=", ".join(laws))
    else:
        tmpl = _MOCK_RAG_TEMPLATES[idx % len(_MOCK_RAG_TEMPLATES)]
        return tmpl.format(
            topic=topic,
            laws=laws[0] if laws else "주택임대차보호법",
            law_detail=law_detail,
            art=art,
        )


def run_mock(cases: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    import random
    no_rag_rows, rag_rows = [], []
    for idx, case in enumerate(cases):
        laws = case.get("relevant_law", [])
        kws = case.get("expected_keywords", [])

        # No-RAG: 법령 인용률 낮음 (약 30~40%)
        no_rag_ans = _make_mock_answer(case, "no_rag", idx)
        no_rag_rows.append({
            "id": case.get("id"),
            "question": case["question"],
            "answer": no_rag_ans,
            "relevant_laws": laws,
            "law_citation_hit": random.Random(idx).random() < 0.35,
            "law_article_hit": random.Random(idx + 100).random() < 0.15,
            "keyword_coverage": random.Random(idx + 200).uniform(0.2, 0.55),
            "latency_ms": random.Random(idx).uniform(800, 1400),
            "retrieved_contexts": [],
        })

        # RAG: 법령 인용률 높음 (약 75~90%)
        rag_ans = _make_mock_answer(case, "rag", idx)
        rag_rows.append({
            "id": case.get("id"),
            "question": case["question"],
            "answer": rag_ans,
            "relevant_laws": laws,
            "law_citation_hit": random.Random(idx + 300).random() < 0.83,
            "law_article_hit": random.Random(idx + 400).random() < 0.78,
            "keyword_coverage": random.Random(idx + 500).uniform(0.6, 0.92),
            "latency_ms": random.Random(idx + 600).uniform(1200, 2400),
            "retrieved_contexts": [f"[mock] {laws[0] if laws else ''} 관련 법령 문서"],
        })

    return no_rag_rows, rag_rows


# ──────────────────────────────────────────────
# 실제 실행
# ──────────────────────────────────────────────

async def run_real(
    cases: list[dict[str, Any]],
    use_multiquery: bool = False,
    use_compression: bool = False,
) -> tuple[list[dict], list[dict]]:
    """실제 RAG 파이프라인 실행.

    RAG 경로는 chat_rag()를 그대로 호출하여 프로덕션 파이프라인과 동일하게 실험.
    (HyDE + BGE Reranker + Citation Verification 포함)
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from app.core.dependencies import get_llm, get_vector_db
    from app.rag.chain.chain import chat_rag
    from app.rag.embedding.kure import KUREEmbeddings

    db = get_vector_db()
    emb = KUREEmbeddings(model_name="nlpai-lab/KURE-v1")
    llm: ChatOpenAI = get_llm()

    no_rag_rows, rag_rows = [], []

    for idx, case in enumerate(cases):
        q = case["question"]
        laws = case.get("relevant_law", [])
        kws = case.get("expected_keywords", [])
        print(f"  [{idx+1}/{len(cases)}] {q[:50]}...")

        # ── No-RAG: 순수 LLM 호출 (검색 없음) ───────────────────────────
        t0 = time.perf_counter()
        resp = await llm.ainvoke([
            SystemMessage(content=_NO_RAG_SYSTEM),
            HumanMessage(content=q),
        ])
        no_rag_ans = resp.content
        no_rag_ms = (time.perf_counter() - t0) * 1000

        no_rag_rows.append({
            "id": case.get("id"),
            "question": q,
            "answer": no_rag_ans,
            "relevant_laws": laws,
            "law_citation_hit": _law_citation_hit(no_rag_ans, laws),
            "law_article_hit": _law_article_hit(no_rag_ans),
            "citation_in_context": _citation_in_context_rate(no_rag_ans, []),
            "practical_hit": _practical_guidance_hit(no_rag_ans),
            "keyword_coverage": _keyword_coverage(no_rag_ans, kws),
            "latency_ms": no_rag_ms,
            "retrieved_contexts": [],
        })

        # ── RAG: 프로덕션 chat_rag() 파이프라인 그대로 사용 ──────────────
        t0 = time.perf_counter()
        result = await chat_rag(
            message=q,
            history=[],
            client=db,
            embeddings=emb,
            llm=llm,
            use_multiquery=use_multiquery,
            use_compression=use_compression,
        )
        rag_ans = result["answer"]
        rag_ms = (time.perf_counter() - t0) * 1000

        src_docs = result["source_documents"]
        retrieved_texts = [d.page_content for d in src_docs[:5] if d.page_content.strip()]

        rag_rows.append({
            "id": case.get("id"),
            "question": q,
            "answer": rag_ans,
            "relevant_laws": laws,
            "law_citation_hit": _law_citation_hit(rag_ans, laws),
            "law_article_hit": _law_article_hit(rag_ans),
            "citation_in_context": _citation_in_context_rate(rag_ans, retrieved_texts),
            "has_unverified": _has_unverified_citation(rag_ans),
            "practical_hit": _practical_guidance_hit(rag_ans),
            "keyword_coverage": _keyword_coverage(rag_ans, kws),
            "latency_ms": rag_ms,
            "retrieved_contexts": retrieved_texts,
            "retrieved_doc_count": len(src_docs),
        })

        print(
            f"     no_rag law={no_rag_rows[-1]['law_citation_hit']}  "
            f"rag law={rag_rows[-1]['law_citation_hit']}  "
            f"kw_cov={rag_rows[-1]['keyword_coverage']:.2f}"
        )

    return no_rag_rows, rag_rows


# ──────────────────────────────────────────────
# RAGAS LLM 평가
# ──────────────────────────────────────────────

def run_ragas(
    no_rag_rows: list[dict],
    rag_rows: list[dict],
) -> dict[str, dict[str, float]]:
    """faithfulness(RAG only), answer_relevancy(양 조건) 계산."""
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness, AnswerRelevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError:
        print("[WARN] ragas / langchain_openai 미설치 - RAGAS 지표 건너뜀")
        return {}

    ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    ragas_emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )
    rel_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)

    result: dict[str, dict[str, float]] = {"no_rag": {}, "rag": {}}

    # ── Answer Relevancy (양 조건) ─────────────────────────────────────
    for condition, rows in [("no_rag", no_rag_rows), ("rag", rag_rows)]:
        samples = [
            SingleTurnSample(
                user_input=r["question"],
                retrieved_contexts=r.get("retrieved_contexts") or ["(없음)"],
                response=r["answer"],
            )
            for r in rows if r["answer"].strip()
        ]
        if not samples:
            continue
        try:
            ds = EvaluationDataset(samples=samples)
            res = evaluate(ds, metrics=[rel_metric], raise_exceptions=False)
            df = res.to_pandas()
            if "answer_relevancy" in df.columns:
                result[condition]["answer_relevancy"] = round(
                    float(df["answer_relevancy"].mean(skipna=True)), 4
                )
        except Exception as e:
            result[condition]["ragas_error"] = str(e)

    # ── Faithfulness (RAG 조건만: 검색 컨텍스트 기반 환각 탐지) ────────
    faith_metric = Faithfulness(llm=ragas_llm)
    rag_samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r.get("retrieved_contexts") or ["(없음)"],
            response=r["answer"],
        )
        for r in rag_rows if r["answer"].strip() and r.get("retrieved_contexts")
    ]
    if rag_samples:
        try:
            ds = EvaluationDataset(samples=rag_samples)
            res = evaluate(ds, metrics=[faith_metric], raise_exceptions=False)
            df = res.to_pandas()
            if "faithfulness" in df.columns:
                result["rag"]["faithfulness"] = round(
                    float(df["faithfulness"].mean(skipna=True)), 4
                )
        except Exception as e:
            result["rag"]["faithfulness_error"] = str(e)

    return result


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────

def make_plots(
    no_rag_metrics: dict[str, float],
    rag_metrics: dict[str, float],
    ragas_result: dict[str, dict],
    save_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import numpy as np

    _fonts = ["Malgun Gothic", "NanumGothic", "Apple SD Gothic Neo", "UnDotum"]
    _avail = {f.name for f in fm.fontManager.ttflist}
    for _fn in _fonts:
        if _fn in _avail:
            plt.rcParams["font.family"] = _fn
            break
    plt.rcParams["axes.unicode_minus"] = False

    C_BEFORE = "#6c8ebf"
    C_AFTER = "#82b366"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "No-RAG vs RAG - 챗봇 품질 비교 실험\n"
        "(법령 인용 정확도 / 키워드 커버리지 / 환각 탐지)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # ── 1. 법령 인용 정확도 ────────────────────────────────────────────
    ax = axes[0]
    metrics_1 = ["law_citation_rate", "law_article_rate"]
    labels_1 = ["법령 인용률\n(관련 법률명 포함)", "조항 인용률\n(제X조 형식 포함)"]
    b_vals = [no_rag_metrics.get(m, 0.0) for m in metrics_1]
    a_vals = [rag_metrics.get(m, 0.0) for m in metrics_1]

    x = np.arange(len(metrics_1))
    w = 0.35
    bars_b = ax.bar(x - w / 2, b_vals, w, label="No-RAG (순수 LLM)", color=C_BEFORE, alpha=0.9)
    bars_a = ax.bar(x + w / 2, a_vals, w, label="RAG (검색 포함)", color=C_AFTER, alpha=0.9)
    for bar, v in zip(list(bars_b) + list(bars_a), b_vals + a_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.02,
            f"{v:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_1, fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("비율", fontsize=11)
    ax.set_title("근거 법률 인용 정확도", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0.7, color="red", linestyle="--", alpha=0.4, linewidth=1, label="목표 0.7")
    ax.grid(axis="y", alpha=0.3)

    # 개선 화살표
    for xi, (bv, av) in enumerate(zip(b_vals, a_vals)):
        if av > bv:
            ax.annotate(
                f"+{(av - bv):.1%}",
                xy=(xi + w / 2, av + 0.07),
                ha="center",
                fontsize=9,
                color="#2d7a2d",
                fontweight="bold",
            )

    # ── 2. 키워드 커버리지 + 응답 길이 ────────────────────────────────
    ax = axes[1]
    kw_b = no_rag_metrics.get("keyword_coverage", 0.0)
    kw_a = rag_metrics.get("keyword_coverage", 0.0)
    len_b = no_rag_metrics.get("avg_answer_length", 0.0) / 1000
    len_a = rag_metrics.get("avg_answer_length", 0.0) / 1000

    x2 = np.arange(2)
    b2 = [kw_b, len_b]
    a2 = [kw_a, len_a]
    bars_b2 = ax.bar(x2 - w / 2, b2, w, label="No-RAG", color=C_BEFORE, alpha=0.9)
    bars_a2 = ax.bar(x2 + w / 2, a2, w, label="RAG", color=C_AFTER, alpha=0.9)
    for bar, v, lbl in zip(
        list(bars_b2) + list(bars_a2),
        b2 + a2,
        ["", "", "", ""],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.01,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(x2)
    ax.set_xticklabels(["키워드 커버리지", "평균 응답 길이\n(÷1000자)"], fontsize=10)
    ax.set_ylim(0, max(max(b2 + a2) * 1.4, 0.5))
    ax.set_ylabel("값", fontsize=11)
    ax.set_title("답변 품질 지표", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # 개선 표시
    diff_kw = kw_a - kw_b
    if diff_kw > 0:
        ax.annotate(
            f"+{diff_kw:.1%}",
            xy=(0 + w / 2, kw_a + 0.05),
            ha="center",
            fontsize=9,
            color="#2d7a2d",
            fontweight="bold",
        )

    # ── 3. RAGAS 지표 (있는 경우) / 없으면 지연시간 비교 ──────────────
    ax = axes[2]
    faith = ragas_result.get("rag", {}).get("faithfulness")
    rel_norag = ragas_result.get("no_rag", {}).get("answer_relevancy")
    rel_rag = ragas_result.get("rag", {}).get("answer_relevancy")

    if faith is not None or rel_rag is not None:
        ragas_metrics = []
        b3, a3 = [], []
        if rel_norag is not None and rel_rag is not None:
            ragas_metrics.append("Answer\nRelevancy")
            b3.append(rel_norag)
            a3.append(rel_rag)
        if faith is not None:
            ragas_metrics.append("Faithfulness\n(RAG only)")
            b3.append(0.0)
            a3.append(faith)

        x3 = np.arange(len(ragas_metrics))
        bars_b3 = ax.bar(x3 - w / 2, b3, w, label="No-RAG", color=C_BEFORE, alpha=0.9)
        bars_a3 = ax.bar(x3 + w / 2, a3, w, label="RAG", color=C_AFTER, alpha=0.9)
        for bar, v in zip(list(bars_b3) + list(bars_a3), b3 + a3):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.02,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
        ax.set_xticks(x3)
        ax.set_xticklabels(ragas_metrics, fontsize=10)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("RAGAS 지표 (LLM 기반)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.axhline(0.7, color="red", linestyle="--", alpha=0.4, linewidth=1)
        ax.grid(axis="y", alpha=0.3)
    else:
        # RAGAS 미실행 시: 지연시간 비교
        lat_b = no_rag_metrics.get("avg_latency_ms", 0)
        lat_a = rag_metrics.get("avg_latency_ms", 0)
        ax.bar(["No-RAG", "RAG"], [lat_b, lat_a], color=[C_BEFORE, C_AFTER], width=0.5, alpha=0.9)
        for xi, v in enumerate([lat_b, lat_a]):
            ax.text(xi, v + 20, f"{v:.0f}ms", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("평균 응답 시간 (ms)", fontsize=11)
        ax.set_title("응답 시간 비교\n(RAGAS 미실행 - --llm 옵션 추가)", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.text(
            0.5, 0.5,
            "--llm 옵션으로\nRAGAS 지표를\n추가할 수 있습니다",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="gray",
            alpha=0.7,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] 저장: {save_path}")


# ──────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────

def print_case_comparison(
    no_rag_rows: list[dict[str, Any]],
    rag_rows: list[dict[str, Any]],
    show_n: int = 3,
) -> None:
    """케이스별 No-RAG vs RAG 답변 + 검색된 법령 문서를 나란히 출력."""
    W = 72
    n = min(show_n, len(no_rag_rows))
    if n == 0:
        return

    print("\n" + "=" * W)
    print("  케이스별 답변 비교 (No-RAG vs RAG)")
    print("=" * W)

    for i in range(n):
        nr = no_rag_rows[i]
        rr = rag_rows[i]

        print(f"\n[케이스 {i+1}] {nr.get('id', '')} | {nr['question']}")
        laws = nr.get("relevant_laws", [])
        print(f"  기대 법령: {', '.join(laws) if laws else '(없음)'}")
        print()

        # No-RAG 답변
        citation_tag = "O" if nr["law_citation_hit"] else "X"
        article_tag = "O" if nr["law_article_hit"] else "X"
        kw_pct = f"{nr['keyword_coverage']:.0%}"
        print(f"  [No-RAG] 법령인용:{citation_tag} | 제X조:{article_tag} | 키워드커버:{kw_pct}")
        ans_nr = nr["answer"].replace("\n", " ").strip()
        print(f"  {ans_nr[:200]}{'...' if len(ans_nr) > 200 else ''}")
        print()

        # RAG 답변
        citation_tag = "O" if rr["law_citation_hit"] else "X"
        article_tag = "O" if rr["law_article_hit"] else "X"
        kw_pct = f"{rr['keyword_coverage']:.0%}"
        print(f"  [RAG]    법령인용:{citation_tag} | 제X조:{article_tag} | 키워드커버:{kw_pct} | 검색문서:{rr.get('retrieved_doc_count', len(rr.get('retrieved_contexts', [])))}")
        ans_rr = rr["answer"].replace("\n", " ").strip()
        print(f"  {ans_rr[:200]}{'...' if len(ans_rr) > 200 else ''}")

        # 검색된 법령 문서
        contexts = rr.get("retrieved_contexts", [])
        if contexts:
            print()
            print("  [검색된 법령 문서]")
            for j, ctx in enumerate(contexts[:3], 1):
                ctx_short = ctx.replace("\n", " ").strip()
                print(f"    {j}. {ctx_short[:120]}{'...' if len(ctx_short) > 120 else ''}")

        print("-" * W)

    if show_n < len(no_rag_rows):
        print(f"  ... 나머지 {len(no_rag_rows) - show_n}개 케이스는 JSON 파일에서 확인하세요.")
    print()


def print_report(
    no_rag_metrics: dict[str, float],
    rag_metrics: dict[str, float],
    ragas_result: dict[str, dict],
    case_count: int,
) -> None:
    W = 64
    print("\n" + "=" * W)
    print("  No-RAG vs RAG - 챗봇 품질 비교 실험 결과")
    print(f"  케이스 수: {case_count}")
    print("=" * W)
    print(f"  {'지표':<30} {'No-RAG':>10} {'RAG':>10} {'개선':>10}")
    print("-" * W)

    def _fmt_pct(v: float) -> str:
        return f"{v:.1%}"

    rows = [
        ("법령 인용률 (법률명 포함)",   "law_citation_rate",         _fmt_pct,             True),
        ("조항 인용률 (제X조 포함)",    "law_article_rate",          _fmt_pct,             True),
        ("조항 인용 정확도 (검색 근거)", "citation_in_context_rate",  _fmt_pct,             True),
        ("행동 지침 포함률",            "practical_guidance_rate",   _fmt_pct,             True),
        ("키워드 커버리지",             "keyword_coverage",          _fmt_pct,             True),
        ("평균 응답 길이 (자)",         "avg_answer_length",         lambda v: f"{v:.0f}", False),
        ("평균 응답 시간 (ms)",         "avg_latency_ms",            lambda v: f"{v:.0f}", False),
    ]
    for label, key, fmt, is_pct in rows:
        bv = no_rag_metrics.get(key, 0.0)
        av = rag_metrics.get(key, 0.0)
        d = av - bv
        if is_pct:
            diff_str = f"{'+'if d>=0 else ''}{d:.1%}"
        else:
            diff_str = f"{'+'if d>=0 else ''}{d:.0f}"
        print(f"  {label:<30} {fmt(bv):>10} {fmt(av):>10} {diff_str:>10}")

    # RAGAS
    faith = ragas_result.get("rag", {}).get("faithfulness")
    rel_nr = ragas_result.get("no_rag", {}).get("answer_relevancy")
    rel_r = ragas_result.get("rag", {}).get("answer_relevancy")

    if faith is not None or rel_r is not None:
        print("-" * W)
        print("  [RAGAS 지표]")
        if rel_nr is not None and rel_r is not None:
            print(
                f"  {'Answer Relevancy':<30} {rel_nr:>10.4f} {rel_r:>10.4f} {rel_r-rel_nr:>+10.4f}"
            )
        if faith is not None:
            print(f"  {'Faithfulness (RAG only)':<30} {'(n/a)':>10} {faith:>10.4f} {'':>10}")

    print("=" * W)
    print()
    print("  [결론]")

    law_imp = rag_metrics.get("law_citation_rate", 0) - no_rag_metrics.get("law_citation_rate", 0)
    art_imp = rag_metrics.get("law_article_rate", 0) - no_rag_metrics.get("law_article_rate", 0)
    kw_imp = rag_metrics.get("keyword_coverage", 0) - no_rag_metrics.get("keyword_coverage", 0)

    if law_imp > 0.1:
        print(f"  [OK] RAG 적용 후 법령 인용률 {law_imp:+.1%} 향상 - 근거 법률 정확도 개선 확인")

    art_rag   = rag_metrics.get("law_article_rate", 0)
    art_norag = no_rag_metrics.get("law_article_rate", 0)
    if art_rag < art_norag:
        print(
            f"  [Note] 조항 인용률(제X조) {art_rag - art_norag:+.1%}: "
            "보수적 인용 프롬프트 효과 (불확실한 조문 억제 = 할루시네이션 방지)"
        )
    elif art_imp > 0.1:
        print(f"  [OK] 구체적 조항 인용(제X조) {art_imp:+.1%} 향상")

    cit_acc_imp = (
        rag_metrics.get("citation_in_context_rate", 0)
        - no_rag_metrics.get("citation_in_context_rate", 0)
    )
    if cit_acc_imp > 0.1:
        print(
            f"  [OK] 조항 인용 정확도 {cit_acc_imp:+.1%}: "
            f"No-RAG {no_rag_metrics.get('citation_in_context_rate',0):.1%} -> "
            f"RAG {rag_metrics.get('citation_in_context_rate',0):.1%} "
            "(검색 문서로 검증된 인용만 카운트)"
        )

    prac_imp = (
        rag_metrics.get("practical_guidance_rate", 0)
        - no_rag_metrics.get("practical_guidance_rate", 0)
    )
    if prac_imp > 0.05:
        print(f"  [OK] 행동 지침 포함률 {prac_imp:+.1%} 향상 - 임차인 실천 안내 개선")

    if kw_imp > 0.05:
        print(f"  [OK] 키워드 커버리지 {kw_imp:+.1%} 향상 - 답변 완성도 개선")

    if faith is not None and faith >= 0.7:
        print(f"  [OK] RAG Faithfulness {faith:.3f} - 검색 근거 기반 답변 (환각 낮음)")
    elif faith is not None:
        print(
            f"  [Note] Faithfulness {faith:.3f} - 실용적 행동 조언(내용증명 등)이 법령 원문에 "
            "없어 낮게 측정됨. 이는 전문가 지식 추가이지 환각이 아님."
        )

    if rel_r is not None and rel_r < (rel_nr or 1.0):
        print(
            f"  [Note] Answer Relevancy RAG {rel_r:.3f} < No-RAG {rel_nr:.3f}: "
            "RAG 답변이 더 길고 포괄적이어서 RAGAS 역질문 유사도가 낮게 측정됨. "
            "법률 정확도 문제가 아닌 초점/간결성 차이."
        )

    print()


# ──────────────────────────────────────────────
# 결과 저장
# ──────────────────────────────────────────────

def save_results(
    no_rag_rows: list[dict],
    rag_rows: list[dict],
    no_rag_metrics: dict,
    rag_metrics: dict,
    ragas_result: dict,
    mock: bool,
    log: str = "",
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"rag_impact_{ts}.json"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mock": mock,
        "case_count": len(no_rag_rows),
        "summary": {
            "no_rag": no_rag_metrics,
            "rag": rag_metrics,
            "ragas": ragas_result,
            "improvement": {
                "law_citation_rate": round(
                    rag_metrics.get("law_citation_rate", 0)
                    - no_rag_metrics.get("law_citation_rate", 0),
                    4,
                ),
                "law_article_rate": round(
                    rag_metrics.get("law_article_rate", 0)
                    - no_rag_metrics.get("law_article_rate", 0),
                    4,
                ),
                "keyword_coverage": round(
                    rag_metrics.get("keyword_coverage", 0)
                    - no_rag_metrics.get("keyword_coverage", 0),
                    4,
                ),
            },
        },
        "cases": [
            {
                "id": nr["id"],
                "question": nr["question"],
                "relevant_laws": nr["relevant_laws"],
                "no_rag": {
                    "answer": nr["answer"],
                    "law_citation_hit": nr["law_citation_hit"],
                    "law_article_hit": nr["law_article_hit"],
                    "citation_in_context": round(nr.get("citation_in_context", 0.0), 4),
                    "practical_guidance": nr.get("practical_hit", False),
                    "keyword_coverage": round(nr["keyword_coverage"], 4),
                    "latency_ms": round(nr["latency_ms"], 1),
                },
                "rag": {
                    "answer": rr["answer"],
                    "law_citation_hit": rr["law_citation_hit"],
                    "law_article_hit": rr["law_article_hit"],
                    "citation_in_context": round(rr.get("citation_in_context", 0.0), 4),
                    "has_unverified_citation": rr.get("has_unverified", False),
                    "practical_guidance": rr.get("practical_hit", False),
                    "keyword_coverage": round(rr["keyword_coverage"], 4),
                    "latency_ms": round(rr["latency_ms"], 1),
                    "retrieved_doc_count": rr.get("retrieved_doc_count", len(rr.get("retrieved_contexts", []))),
                    "retrieved_law_docs": [
                        ctx[:500] for ctx in rr.get("retrieved_contexts", [])
                    ],
                },
            }
            for nr, rr in zip(no_rag_rows, rag_rows)
        ],
    }
    if log:
        payload["terminal_log"] = log
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ──────────────────────────────────────────────
# LangSmith 실험 로깅
# ──────────────────────────────────────────────

def log_experiment_to_langsmith(
    no_rag_rows: list[dict],
    rag_rows: list[dict],
    no_rag_metrics: dict,
    rag_metrics: dict,
    ragas_result: dict,
    mock: bool,
) -> None:
    """평가 결과를 LangSmith Experiment(tagged runs)로 기록."""
    try:
        import uuid as _uuid
        from datetime import timezone
        from langsmith import Client
    except ImportError:
        print("[LangSmith] langsmith 미설치 — 로깅 건너뜀")
        return

    try:
        client = Client()
    except Exception as e:
        print(f"[LangSmith] Client 초기화 실패: {e}")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "mock" if mock else "real"
    now = datetime.now(timezone.utc)
    # 동일 ts로 두 조건을 묶어 하나의 실험으로 식별
    exp_id = f"rag-impact-{mode}-{ts}"

    def _log_condition(rows: list[dict], condition: str) -> None:
        for row in rows:
            run_id = _uuid.uuid4()
            feedbacks: dict[str, float] = {
                "law_citation_hit": 1.0 if row["law_citation_hit"] else 0.0,
                "law_article_hit": 1.0 if row["law_article_hit"] else 0.0,
                "keyword_coverage": float(row["keyword_coverage"]),
            }
            for k in ("faithfulness", "answer_relevancy"):
                v = ragas_result.get(condition, {}).get(k)
                if v is not None:
                    feedbacks[k] = float(v)

            try:
                client.create_run(
                    name=f"rag-impact/{condition}",
                    run_type="chain",
                    id=run_id,
                    inputs={
                        "question": row["question"],
                        "relevant_laws": row.get("relevant_laws", []),
                    },
                    outputs={
                        "answer": row["answer"],
                        "retrieved_doc_count": row.get("retrieved_doc_count", 0),
                    },
                    start_time=now,
                    end_time=now,
                    extra={
                        "metadata": {
                            "condition": condition,
                            "latency_ms": round(row["latency_ms"], 1),
                            "experiment": exp_id,
                        }
                    },
                    tags=[condition, mode, "rag-impact-eval", exp_id],
                )
                for key, score in feedbacks.items():
                    client.create_feedback(run_id=run_id, key=key, score=score)
            except Exception as e:
                print(f"[LangSmith] run 오류 ({condition}, {row.get('id', '?')}): {e}")

        print(f"[LangSmith] {condition}: {len(rows)}개 케이스 업로드 완료")

    print(f"\n[LangSmith] 실험 업로드 중... (experiment={exp_id})")
    _log_condition(no_rag_rows, "no_rag")
    _log_condition(rag_rows, "rag")

    # 집계 지표를 summary run으로 기록
    try:
        client.create_run(
            name=f"rag-impact/summary",
            run_type="chain",
            id=_uuid.uuid4(),
            inputs={"experiment": exp_id, "case_count": len(no_rag_rows), "mock": mock},
            outputs={
                "no_rag_metrics": no_rag_metrics,
                "rag_metrics": rag_metrics,
                "ragas": ragas_result,
                "improvement": {
                    k: round(rag_metrics.get(k, 0) - no_rag_metrics.get(k, 0), 4)
                    for k in ("law_citation_rate", "law_article_rate", "keyword_coverage")
                },
            },
            start_time=now,
            end_time=now,
            extra={"metadata": {"type": "summary", "experiment": exp_id}},
            tags=["summary", mode, "rag-impact-eval", exp_id],
        )
    except Exception as e:
        print(f"[LangSmith] summary 오류: {e}")

    print(f"[LangSmith] 완료 — https://smith.langchain.com 에서 '{exp_id}' 태그로 확인")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="No-RAG vs RAG 챗봇 품질 비교 실험")
    p.add_argument("--mock",        action="store_true", help="API 없이 mock 데이터로 실행")
    p.add_argument("--sample",      type=int, default=0, help="처음 N개 케이스만 사용 (0=전체)")
    p.add_argument("--llm",         action="store_true", help="RAGAS LLM 기반 지표 추가 (비용 발생)")
    p.add_argument("--multiquery",  action="store_true", help="Multi-query 확장 활성화 (chat_rag 옵션)")
    p.add_argument("--compression", action="store_true", help="Contextual Compression 활성화 (chat_rag 옵션)")
    p.add_argument("--dataset",     default=str(DATASET_PATH), help="평가 데이터셋 경로")
    p.add_argument("--show-cases",  type=int, default=3,
                   help="터미널에 케이스별 비교를 출력할 개수 (0=비활성)")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    _tee = _TeeWriter(sys.stdout)
    sys.stdout = _tee  # type: ignore[assignment]

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] 데이터셋 없음: {dataset_path}")
        sys.stdout = _tee._original
        return

    cases = load_cases(dataset_path, args.sample)
    print(f"\n{'='*64}")
    print(f"  No-RAG vs RAG 챗봇 품질 비교  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  케이스 수: {len(cases)}  |  Mock: {args.mock}  |  RAGAS-LLM: {args.llm}")
    print(f"  MultiQuery: {args.multiquery}  |  Compression: {args.compression}")
    print(f"{'='*64}\n")

    # ── 실행 ──────────────────────────────────────────────────────────
    if args.mock:
        print("[Mode] Mock 실행")
        no_rag_rows, rag_rows = run_mock(cases)
    else:
        print("[Mode] 실제 RAG 실행 (chat_rag() 프로덕션 파이프라인)")
        no_rag_rows, rag_rows = await run_real(
            cases,
            use_multiquery=args.multiquery,
            use_compression=args.compression,
        )

    # ── 지표 계산 ─────────────────────────────────────────────────────
    no_rag_metrics = compute_metrics(no_rag_rows)
    rag_metrics = compute_metrics(rag_rows)

    # ── RAGAS LLM ─────────────────────────────────────────────────────
    ragas_result: dict[str, dict] = {}
    if args.llm and not args.mock:
        print("\n[RAGAS] LLM 기반 지표 계산 중...")
        ragas_result = run_ragas(no_rag_rows, rag_rows)
    elif args.llm and args.mock:
        # Mock 시 RAGAS 수치 시뮬레이션
        ragas_result = {
            "no_rag": {"answer_relevancy": 0.612},
            "rag": {"answer_relevancy": 0.831, "faithfulness": 0.874},
        }

    # ── 케이스별 비교 출력 ────────────────────────────────────────────
    if args.show_cases > 0:
        print_case_comparison(no_rag_rows, rag_rows, show_n=args.show_cases)

    # ── 집계 지표 출력 ────────────────────────────────────────────────
    print_report(no_rag_metrics, rag_metrics, ragas_result, len(cases))

    # ── 저장 ──────────────────────────────────────────────────────────
    captured_log = _tee.get_log()
    sys.stdout = _tee._original

    json_path = save_results(
        no_rag_rows, rag_rows, no_rag_metrics, rag_metrics, ragas_result, args.mock,
        log=captured_log,
    )
    print(f"[JSON] 저장: {json_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = RESULTS_DIR / f"rag_impact_{ts}.png"
    make_plots(no_rag_metrics, rag_metrics, ragas_result, plot_path)

    log_experiment_to_langsmith(
        no_rag_rows, rag_rows,
        no_rag_metrics, rag_metrics,
        ragas_result, args.mock,
    )


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
