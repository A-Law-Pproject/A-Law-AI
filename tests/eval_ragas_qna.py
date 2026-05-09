"""RAGAS 기반 검색 품질 + 답변 정확도 평가

데이터셋:
  챗봇_평가용_최종자료.xlsx (기본) — 검증된 QA 275개 (주유형/세부유형/질문/상세 답변)
  lease_faq.jsonl            — 검증된 FAQ 23개 (질문/관련법령) → HR@K / MRR 전용
  국가_제공_QnA_자료.xlsx    (--source qna) — 기존 QnA 자료

지표:
  [RAGAS 검색 품질]
    non_llm_context_precision_with_reference  키워드 기반 검색 정밀도
    non_llm_context_recall                    키워드 기반 검색 재현율
    llm_context_precision_with_reference      LLM 기반 정밀도  (--llm)
    context_recall                            LLM 기반 재현율  (--llm)

  [HR@K / MRR — lease_faq ground truth]
    hr@1 / hr@3 / hr@5    top-K 결과에 ground-truth 법령명 포함률
    mrr                   평균 역순위

  [답변 정확도 — --answer 옵션 필요]
    answer_correctness    LLM 생성 답변 vs 검증 정답 일치도  (--llm 필요)
    answer_similarity     의미적 유사도                       (--llm 필요)

실행 예시:
    # Mock (API 없이 빠른 확인)
    .venv\\Scripts\\python.exe tests\\eval_ragas_qna.py --mock

    # Mock + LLM 기반 RAGAS 지표
    .venv\\Scripts\\python.exe tests\\eval_ragas_qna.py --mock --llm

    # 실제 검색 (Pinecone)
    .venv\\Scripts\\python.exe tests\\eval_ragas_qna.py --sample 20

    # 실제 검색 + 답변 생성 + 전체 지표
    .venv\\Scripts\\python.exe tests\\eval_ragas_qna.py --sample 20 --llm --answer

    # 기존 국가_QnA 자료 사용
    .venv\\Scripts\\python.exe tests\\eval_ragas_qna.py --mock --source qna
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")
os.environ.setdefault("LANGCHAIN_PROJECT", "A-LAW-eval")

CHATBOT_XLSX  = ROOT / "tests" / "평가데이터셋" / "챗봇_평가용_최종자료.xlsx"
LEASE_FAQ_JSONL = ROOT / "tests" / "평가데이터셋" / "lease_faq.jsonl"
QNA_XLSX      = ROOT / "tests" / "평가데이터셋" / "국가_제공_QnA_자료.xlsx"
RESULTS_DIR   = ROOT / "results"

DEFAULT_COLLECTIONS = [
    "law_database",
    "law_statutes",
    "contracts",
    "special_clauses_illegal",
    "special_clauses_normal",
]

METRIC_LABELS = {
    "non_llm_context_precision_with_reference": "Context Precision\n(Non-LLM)",
    "non_llm_context_recall":                  "Context Recall\n(Non-LLM)",
    "llm_context_precision_with_reference":    "Context Precision\n(LLM)",
    "context_recall":                          "Context Recall\n(LLM)",
    "answer_correctness":                      "Answer\nCorrectness",
    "answer_similarity":                       "Answer\nSimilarity",
    "legal_fit":                               "Legal Fit\n(Citation)",
}

METRIC_COLORS = {
    "non_llm_context_precision_with_reference": "#4C72B0",
    "non_llm_context_recall":                  "#55A868",
    "llm_context_precision_with_reference":    "#C44E52",
    "context_recall":                          "#8172B2",
    "answer_correctness":                      "#CCB974",
    "answer_similarity":                       "#64B5CD",
    "legal_fit":                               "#DD8452",
}


# ──────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────

def load_chatbot_final(path: Path, sample: int) -> list[dict[str, Any]]:
    """챗봇_평가용_최종자료.xlsx 로드 (주유형/세부유형/질문/상세 답변)."""
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = ["주유형", "세부유형", "question", "answer"]
    df = df.dropna(subset=["question"])
    df["주유형"]  = df["주유형"].fillna("기타")
    df["세부유형"] = df["세부유형"].fillna("-")
    df["answer"]  = df["answer"].fillna("").astype(str)
    df = df.rename(columns={"주유형": "분류"})
    cases = df.to_dict("records")
    return cases[:sample] if sample > 0 else cases


def load_qna_legacy(path: Path, sample: int) -> list[dict[str, Any]]:
    """국가_제공_QnA_자료.xlsx 로드 (레거시)."""
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = ["분류", "question", "answer", "notes"]
    df = df[~df["분류"].isin(["분류", "상세 분류"])].reset_index(drop=True)
    df = df.dropna(subset=["question", "answer"])
    cases = df.to_dict("records")
    return cases[:sample] if sample > 0 else cases


def load_lease_faq(path: Path, sample: int) -> list[dict[str, Any]]:
    """lease_faq.jsonl 로드 — HR@K / MRR 평가용.

    반환 형식: {"question": str, "gt_laws": set[str], "gt_clauses": set[str]}
    """
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    cases: list[dict] = []
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not obj.get("question"):
            continue
        laws = obj.get("related_laws", [])
        gt_laws    = {str(law.get("법령명", "")).strip() for law in laws}
        gt_clauses = {str(law.get("조문명", "")).strip() for law in laws}
        gt_laws.discard("")
        gt_clauses.discard("")
        cases.append({
            "question":   obj["question"],
            "gt_laws":    gt_laws,
            "gt_clauses": gt_clauses,
        })
    return cases[:sample] if sample > 0 else cases


# ──────────────────────────────────────────────
# Mock
# ──────────────────────────────────────────────

def _mock_docs(case: dict, idx: int) -> list[str]:
    answer    = str(case.get("answer", ""))
    good_ctx  = answer if answer else "임대차 관련 법률 조항 참고 자료"
    noise_ctx = "임대차 관련 일반 안내 사항입니다. 계약서 작성 시 주의하세요."
    if idx % 5 == 0:
        return [noise_ctx, noise_ctx]
    if idx % 3 == 0:
        return [noise_ctx, good_ctx]
    return [good_ctx, noise_ctx]


def _mock_answer(case: dict, idx: int) -> str:
    """Mock LLM 답변: 정답을 부분 참고한 답변 시뮬레이션."""
    ref = str(case.get("answer", "")).strip()
    if not ref:
        return "해당 질문에 대한 답변을 생성하지 못했습니다."
    if idx % 2 == 0:
        return f"법률 검토 결과, {ref[:min(len(ref), 100)]}"
    words = ref.split()
    half  = words[: max(1, len(words) // 2)]
    return "법률에 따르면 " + " ".join(half) + " 등의 내용이 적용됩니다."


def _mock_hr_docs(case: dict, idx: int) -> list[str]:
    """lease_faq HR@K 용 mock."""
    laws  = list(case["gt_laws"])
    noise = "주택임대차보호법 시행령 관련 일반 안내"
    if not laws:
        return [noise, noise, noise, noise, noise]
    if idx % 4 == 0:
        return [noise, noise, noise, noise, noise]
    if idx % 3 == 0:
        return [noise, f"{laws[0]} 제6조 임차권등기명령 규정", noise, noise, noise]
    return [f"{laws[0]} 보증금 반환 관련 규정", noise, noise, noise, noise]


# ──────────────────────────────────────────────
# 실제 검색
# ──────────────────────────────────────────────

def _real_docs(
    question: str,
    db: Any,
    embeddings: Any,
    collections: list[str],
    k_per_collection: int,
    reranker: Any | None,
    rerank_top_n: int,
    llm: Any | None = None,
    use_multiquery: bool = False,
) -> tuple[list[str], list[dict]]:
    """Pinecone 검색 후 (텍스트 목록, 메타데이터 목록) 반환.

    use_multiquery=True + llm 전달 시: 원본 질문을 3개 변형으로 확장하여
    각각 검색 → 합산 중복 제거 → reranker 적용.
    """
    from app.rag.retriever.multi_retriever import _deduplicate, search_multi_index
    from langchain_core.documents import Document

    if use_multiquery and llm:
        from app.rag.retriever.query_expansion import expand_query_multi
        queries = expand_query_multi(question, llm, n=3)
        all_docs: list[Document] = []
        for q in queries:
            docs = search_multi_index(
                db, embeddings, q,
                collections=collections,
                k_per_collection=k_per_collection,
            )
            all_docs.extend(docs)
        all_docs = _deduplicate(all_docs)
        all_docs.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)
        if reranker is not None:
            all_docs = reranker.rerank(question, all_docs, top_n=rerank_top_n)
        docs_final = all_docs
    else:
        docs_final = search_multi_index(
            db, embeddings, question,
            collections=collections,
            k_per_collection=k_per_collection,
            reranker=reranker,
            rerank_top_n=rerank_top_n,
        )

    texts: list[str] = []
    metas: list[dict] = []
    for doc in docs_final:
        if hasattr(doc, "page_content"):
            text = str(doc.page_content)
        elif isinstance(doc, dict):
            text = str(doc.get("content") or doc.get("page_content") or "")
        else:
            text = str(getattr(doc, "content", ""))
        if text.strip():
            texts.append(text)
            metas.append(dict(doc.metadata) if hasattr(doc, "metadata") and isinstance(doc.metadata, dict) else {})
    return texts, metas


def _extract_cited_laws(metas: list[dict]) -> list[dict]:
    """검색 문서 메타데이터에서 법령 인용 정보를 중복 없이 추출.

    반환 형식: [{"law_name": str, "article": str, "collection": str, "score": float}, ...]
    """
    seen: set[tuple[str, str]] = set()
    laws: list[dict] = []
    for meta in metas:
        law_name   = str(meta.get("law_name") or "").strip()
        article    = str(meta.get("조문명") or meta.get("article") or "").strip()
        collection = str(meta.get("collection") or "").strip()
        score      = round(float(meta.get("rerank_score") or meta.get("score") or 0.0), 4)

        key = (law_name, article)
        if law_name and key not in seen:
            seen.add(key)
            laws.append({
                "law_name":   law_name,
                "article":    article,
                "collection": collection,
                "score":      score,
            })
    return laws


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text.strip())
    return text.strip()


def _mock_legal_fit(idx: int, cited_laws: list[dict]) -> dict[str, Any]:
    """Mock 환경에서 법리 적합성 점수를 결정적으로 생성."""
    if not cited_laws:
        return {"score": 0, "reason": "no_cited_laws"}
    return {
        "score": 0 if idx % 5 == 0 else 1,
        "reason": "mock_inappropriate" if idx % 5 == 0 else "mock_appropriate",
    }


async def _judge_legal_fit(
    question: str,
    response: str | None,
    cited_laws: list[dict],
    contexts: list[str],
    llm: Any,
) -> dict[str, Any]:
    """질문에 대해 인용된 조문이 직접적으로 맞는지 1/0으로 판정."""
    if not cited_laws:
        return {"score": 0, "reason": "no_cited_laws"}

    from langchain_core.messages import HumanMessage, SystemMessage

    citation_summary = json.dumps(cited_laws[:5], ensure_ascii=False, indent=2)
    context_summary = "\n\n".join(contexts[:3])[:2500] if contexts else "(검색 문맥 없음)"
    answer_text = str(response or "(답변 없음)")[:1200]

    system = (
        "당신은 한국 임대차 법률 QA 평가자입니다. "
        "질문과 답변, 인용 조문 정보를 보고 인용 조문이 질문에 직접적으로 적합한지 엄격하게 판정하세요."
    )
    human = f"""다음 케이스를 평가하세요.

질문:
{question}

답변:
{answer_text}

인용 조문 메타데이터:
{citation_summary}

후보 조문 본문:
{context_summary}

판정 기준:
1. 질문의 핵심 쟁점과 직접 관련된 조문이면 1
2. 카테고리가 다르거나 적용 방향이 반대면 0
3. 검색은 되었지만 질문 상황에 맞지 않는 조문 오용이면 0
4. 인용 조문이 없거나, 본문만으로 적합성을 확인할 수 없으면 0

반드시 아래 JSON만 출력하세요.
{{
  "legal_fit": 0,
  "reason": "한 줄 설명"
}}"""

    resp = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=human),
    ])
    parsed = json.loads(_strip_code_fence(resp.content))
    score = 1 if int(parsed.get("legal_fit", 0)) == 1 else 0
    reason = str(parsed.get("reason") or "").strip() or "no_reason"
    return {"score": score, "reason": reason}


async def _real_answer(question: str, contexts: list[str], llm: Any) -> str:
    """RAG 컨텍스트 기반 LLM 답변 생성."""
    from langchain_core.messages import HumanMessage, SystemMessage

    ctx_text = "\n\n".join(contexts[:5]) if contexts else "(검색 결과 없음)"
    system = (
        "당신은 한국 임대차 계약 전문 AI입니다. "
        "아래 참고 문서를 바탕으로 질문에 간결하고 정확하게 답변하세요.\n\n"
        f"[참고 문서]\n{ctx_text}"
    )
    resp = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=question),
    ])
    return resp.content


# ──────────────────────────────────────────────
# HR@K / MRR (lease_faq 전용)
# ──────────────────────────────────────────────

def compute_hr_k_mrr(
    cases: list[dict],
    docs_per_case: list[list[str]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """lease_faq ground-truth 법령명 기반 HR@K 및 MRR 계산."""
    if k_values is None:
        k_values = [1, 3, 5]
    if not cases:
        return {}

    hits: dict[int, int]    = {k: 0 for k in k_values}
    reciprocal_ranks: list[float] = []

    for case, docs in zip(cases, docs_per_case):
        gt_lower = {g.lower() for g in (case["gt_laws"] | case["gt_clauses"]) if g}

        rr = 0.0
        for rank, doc in enumerate(docs, 1):
            doc_lower = doc.lower()
            if any(law in doc_lower for law in gt_lower):
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        for k in k_values:
            combined = " ".join(docs[:k]).lower()
            if any(law in combined for law in gt_lower):
                hits[k] += 1

    n = len(cases)
    result: dict[str, float] = {f"hr@{k}": round(hits[k] / n, 4) for k in k_values}
    result["mrr"] = round(sum(reciprocal_ranks) / n, 4)
    return result


# ──────────────────────────────────────────────
# RAGAS
# ──────────────────────────────────────────────

def _build_ragas_dataset(rows: list[dict[str, Any]]) -> Any:
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    samples = []
    for row in rows:
        reference = str(row.get("answer") or "").strip()
        if not reference:
            continue
        contexts = [c for c in row.get("retrieved_contexts", []) if str(c).strip()]
        if not contexts:
            contexts = ["(검색 결과 없음)"]
        response = str(row.get("response") or "").strip() or None
        samples.append(SingleTurnSample(
            user_input=row["question"],
            retrieved_contexts=contexts,
            reference_contexts=[reference],
            reference=reference,
            response=response,
        ))
    return EvaluationDataset(samples=samples)


def _run_ragas(dataset: Any, use_llm: bool, has_response: bool) -> pd.DataFrame:
    from ragas import evaluate
    from ragas.metrics import (
        _NonLLMContextPrecisionWithReference,
        _NonLLMContextRecall,
    )

    metrics: list = [
        _NonLLMContextPrecisionWithReference(),
        _NonLLMContextRecall(),
    ]

    if use_llm:
        from ragas.metrics import _LLMContextPrecisionWithReference, _LLMContextRecall
        metrics += [_LLMContextPrecisionWithReference(), _LLMContextRecall()]

    if use_llm and has_response:
        try:
            try:
                from ragas.metrics.collections import AnswerCorrectness, AnswerSimilarity
            except ImportError:
                from ragas.metrics import AnswerCorrectness, AnswerSimilarity  # type: ignore[no-redef]
            try:
                from openai import OpenAI
                from ragas.llms import llm_factory
                from ragas.embeddings import embedding_factory
                _llm = llm_factory("gpt-4o-mini", client=OpenAI())
                _emb = embedding_factory("openai", model="text-embedding-3-small", client=OpenAI())
            except Exception:
                from ragas.llms import LangchainLLMWrapper
                from ragas.embeddings import LangchainEmbeddingsWrapper
                from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                _llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
                _emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
            metrics += [
                AnswerCorrectness(llm=_llm, embeddings=_emb),
                AnswerSimilarity(embeddings=_emb),
            ]
        except Exception as e:
            print(f"[WARN] 답변 품질 지표 초기화 실패: {e}")

    result = evaluate(dataset, metrics=metrics, raise_exceptions=False)
    return result.to_pandas()


def _metric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c in METRIC_LABELS and not df[c].isna().all()]


# ──────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────

def _make_plots(
    rows: list[dict[str, Any]],
    scores_df: pd.DataFrame,
    hr_k_results: dict[str, float],
    save_path: Path,
    source_label: str = "챗봇 평가용 최종자료",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import numpy as np

    _korean_fonts = ["Malgun Gothic", "NanumGothic", "Apple SD Gothic Neo", "UnDotum"]
    _available = {f.name for f in fm.fontManager.ttflist}
    for _fn in _korean_fonts:
        if _fn in _available:
            plt.rcParams["font.family"] = _fn
            break
    plt.rcParams["axes.unicode_minus"] = False

    metric_cols = _metric_cols(scores_df)
    has_hr_k    = bool(hr_k_results)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        f"RAG 검색 품질 + 답변 정확도 + 법리 적합성 + HR@K 평가\n({source_label} + lease_faq)",
        fontsize=14, fontweight="bold", y=0.99,
    )

    # ── 1. RAGAS 전체 평균 바 차트 ────────────────────────────────
    ax1 = axes[0, 0]
    if metric_cols:
        overall = {col: float(scores_df[col].mean(skipna=True)) for col in metric_cols}
        labels  = [METRIC_LABELS.get(k, k) for k in overall]
        values  = list(overall.values())
        colors  = [METRIC_COLORS.get(k, "#999999") for k in overall]

        bars = ax1.barh(labels, values, color=colors, edgecolor="white", height=0.55)
        for bar, v in zip(bars, values):
            ax1.text(
                min(v + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=10, fontweight="bold",
            )
        ax1.set_xlim(0, 1.15)
        ax1.set_xlabel("Score", fontsize=10)
        ax1.axvline(0.7, color="red", linestyle="--", alpha=0.5, linewidth=1, label="목표 0.7")
        ax1.legend(fontsize=8)
        ax1.grid(axis="x", alpha=0.3)
    ax1.set_title("RAGAS 전체 평균 지표", fontsize=12, fontweight="bold")

    # ── 2. HR@K / MRR (lease_faq 기반) ────────────────────────────
    ax2 = axes[0, 1]
    if has_hr_k:
        hr_labels = list(hr_k_results.keys())
        hr_values = list(hr_k_results.values())
        hr_colors = ["#5B9BD5", "#70AD47", "#ED7D31", "#9E480E"][:len(hr_labels)]
        bars2 = ax2.bar(hr_labels, hr_values, color=hr_colors, edgecolor="white", width=0.55)
        for bar, v in zip(bars2, hr_values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.3f}", ha="center", fontsize=11, fontweight="bold",
            )
        ax2.set_ylim(0, 1.2)
        ax2.set_ylabel("Score", fontsize=10)
        ax2.axhline(0.7, color="red", linestyle="--", alpha=0.5, linewidth=1, label="목표 0.7")
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "HR@K 데이터 없음\n(lease_faq.jsonl 로딩 실패)",
                 ha="center", va="center", fontsize=11, color="gray", transform=ax2.transAxes)
    ax2.set_title("HR@K / MRR (lease_faq ground truth)", fontsize=12, fontweight="bold")

    # ── 3. 주유형별 context_recall 비교 ────────────────────────────
    ax3 = axes[1, 0]
    recall_col = next(
        (c for c in ["non_llm_context_recall", "context_recall"] if c in scores_df.columns),
        None,
    )
    if recall_col and rows:
        cats = [r.get("분류", "기타") for r in rows[: len(scores_df)]]
        tmp  = scores_df[[recall_col]].copy()
        tmp["분류"] = cats
        cat_mean = (
            tmp.groupby("분류")[recall_col]
            .mean()
            .sort_values(ascending=False)
        )
        cat_mean = cat_mean[cat_mean.index.notna()]
        colors3 = plt.cm.tab20.colors[: len(cat_mean)]
        bars3 = ax3.barh(
            cat_mean.index.tolist(), cat_mean.values,
            color=colors3, edgecolor="white", height=0.6,
        )
        for bar, v in zip(bars3, cat_mean.values):
            ax3.text(
                min(v + 0.01, 0.98), bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=9,
            )
        ax3.set_xlim(0, 1.15)
        ax3.set_xlabel("Context Recall", fontsize=10)
        ax3.axvline(0.7, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax3.grid(axis="x", alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "주유형 분류 데이터 없음",
                 ha="center", va="center", fontsize=11, color="gray", transform=ax3.transAxes)
    ax3.set_title("주유형별 Context Recall", fontsize=12, fontweight="bold")

    # ── 4. 샘플별 히트맵 (상위 20개) ─────────────────────────────
    ax4 = axes[1, 1]
    if metric_cols and len(scores_df) > 0:
        heat_df = scores_df[metric_cols].copy().iloc[:20]
        q_labels = [
            f"Q{i+1}: {rows[i]['question'][:22]}..."
            if len(rows[i]["question"]) > 22 else f"Q{i+1}: {rows[i]['question']}"
            for i in range(min(len(rows), len(heat_df)))
        ]
        heat_df.index = q_labels

        import numpy as np
        im = ax4.imshow(heat_df.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax4.set_xticks(range(len(metric_cols)))
        ax4.set_xticklabels(
            [METRIC_LABELS.get(c, c) for c in metric_cols],
            fontsize=8, rotation=20, ha="right",
        )
        ax4.set_yticks(range(len(heat_df)))
        ax4.set_yticklabels(heat_df.index, fontsize=7)
        ax4.set_title(f"질문별 점수 히트맵 (상위 {len(heat_df)}개)", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        for ri in range(len(heat_df)):
            for ci in range(len(metric_cols)):
                val = heat_df.values[ri, ci]
                if not (val != val):  # NaN 방어
                    ax4.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                             fontsize=6, color="black" if 0.3 < val < 0.8 else "white")
    else:
        ax4.text(0.5, 0.5, "히트맵 데이터 없음",
                 ha="center", va="center", fontsize=11, color="gray", transform=ax4.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] 저장: {save_path}")


# ──────────────────────────────────────────────
# LangSmith 로깅
# ──────────────────────────────────────────────

def log_eval_to_langsmith(
    *,
    rows: list[dict[str, Any]],
    faq_cases: list[dict[str, Any]],
    faq_docs_per_case: list[list[str]],
    faq_metas_per_case: list[list[dict]],
    metric_cols: list[str],
    scores_df: pd.DataFrame,
    overall: dict[str, float],
    hr_k_results: dict[str, float],
    failure_counts: dict[str, int],
    source_label: str,
    mode: str,
    llm_judge: bool,
    answer_eval: bool,
    json_path: Path,
    plot_path: Path,
) -> None:
    """평가 결과를 LangSmith run/feedback으로 업로드한다."""
    try:
        import uuid as _uuid
        from datetime import timezone
        from langsmith import Client
        from urllib3.util.retry import Retry
    except ImportError:
        print("[LangSmith] langsmith 미설치 - 로깅 건너뜀")
        return

    try:
        client = Client(
            timeout_ms=(2000, 3000),
            retry_config=Retry(total=0, connect=0, read=0, redirect=0, status=0),
            info={},
        )
    except Exception as e:
        print(f"[LangSmith] Client 초기화 실패: {e}")
        return

    endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    parsed = urlparse(endpoint)
    host = parsed.hostname or "api.smith.langchain.com"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=2):
            pass
    except OSError as e:
        print(f"[LangSmith] 연결 불가로 로깅 건너뜀: {host}:{port} ({e.__class__.__name__})")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project = os.getenv("LANGCHAIN_PROJECT", "A-LAW-eval")
    exp_id = f"ragas-qna-{mode}-{ts}"
    now = datetime.now(timezone.utc)
    tags = [
        "ragas-qna-eval",
        mode,
        source_label.replace(" ", "_"),
        exp_id,
    ]
    if llm_judge:
        tags.append("llm-judge")
    if answer_eval:
        tags.append("answer-eval")

    def _safe_feedback(run_id: Any, key: str, score: Any) -> None:
        if score is None:
            return
        try:
            if pd.isna(score):
                return
        except Exception:
            pass
        try:
            client.create_feedback(run_id=run_id, key=key, score=float(score))
        except Exception as e:
            print(f"[LangSmith] feedback 오류 ({key}): {e}")

    def _is_connection_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        name = exc.__class__.__name__.lower()
        return (
            "connection" in msg
            or "proxy" in msg
            or "timeout" in msg
            or "connection" in name
            or "timeout" in name
        )

    print(f"\n[LangSmith] 업로드 중... (project={project}, experiment={exp_id})")

    for i, row in enumerate(rows):
        run_id = None
        try:
            run_id = _uuid.uuid4()
            case_tags = tags + ["case"]
            if row.get("error"):
                case_tags.append("case-error")

            outputs = {
                "response": row.get("response"),
                "retrieved_context_count": len(row.get("retrieved_contexts", [])),
                "cited_laws": row.get("cited_laws", []),
                "legal_fit_reason": row.get("legal_fit_reason"),
                "error": row.get("error"),
            }
            client.create_run(
                name="ragas-qna/case",
                run_type="chain",
                id=run_id,
                inputs={
                    "question": row["question"],
                    "reference_answer": row.get("answer"),
                    "source": source_label,
                },
                outputs=outputs,
                start_time=now,
                end_time=now,
                extra={
                    "metadata": {
                        "experiment": exp_id,
                        "case_index": i,
                        "mode": mode,
                        "source": source_label,
                        "category": row.get("분류"),
                        "sub_category": row.get("세부유형"),
                        "latency_ms": row.get("latency_ms"),
                    }
                },
                tags=case_tags,
            )

            for col in metric_cols:
                value = scores_df.iloc[i][col] if i < len(scores_df) and col in scores_df.columns else None
                _safe_feedback(run_id, col, value)
            _safe_feedback(run_id, "latency_ms", row.get("latency_ms"))
        except Exception as e:
            if _is_connection_error(e):
                print(f"[LangSmith] case 업로드 오류 ({i}): 연결성 문제")
                print("[LangSmith] 연결성 오류로 나머지 업로드를 중단합니다.")
                return
            print(f"[LangSmith] case 업로드 오류 ({i}): {e}")

    for i, faq in enumerate(faq_cases):
        try:
            run_id = _uuid.uuid4()
            client.create_run(
                name="ragas-qna/faq-retrieval",
                run_type="chain",
                id=run_id,
                inputs={
                    "question": faq["question"],
                    "ground_truth_laws": list(faq.get("gt_laws", [])),
                    "ground_truth_clauses": list(faq.get("gt_clauses", [])),
                },
                outputs={
                    "retrieved": faq_docs_per_case[i][:3],
                    "cited_laws": _extract_cited_laws(faq_metas_per_case[i]),
                },
                start_time=now,
                end_time=now,
                extra={
                    "metadata": {
                        "experiment": exp_id,
                        "faq_index": i,
                        "mode": mode,
                        "source": "lease_faq",
                    }
                },
                tags=tags + ["faq-retrieval"],
            )
        except Exception as e:
            if _is_connection_error(e):
                print(f"[LangSmith] faq 업로드 오류 ({i}): 연결성 문제")
                print("[LangSmith] 연결성 오류로 나머지 업로드를 중단합니다.")
                return
            print(f"[LangSmith] faq 업로드 오류 ({i}): {e}")

    try:
        summary_run_id = _uuid.uuid4()
        client.create_run(
            name="ragas-qna/summary",
            run_type="chain",
            id=summary_run_id,
            inputs={
                "experiment": exp_id,
                "source": source_label,
                "mode": mode,
                "case_count": len(rows),
                "faq_count": len(faq_cases),
                "llm_judge": llm_judge,
                "answer_eval": answer_eval,
            },
            outputs={
                "overall_metrics": overall,
                "hr_k_mrr": hr_k_results,
                "failure_counts": failure_counts,
                "json_path": str(json_path),
                "plot_path": str(plot_path),
            },
            start_time=now,
            end_time=now,
            extra={
                "metadata": {
                    "experiment": exp_id,
                    "type": "summary",
                    "project": project,
                }
            },
            tags=tags + ["summary"],
        )

        for key, score in overall.items():
            _safe_feedback(summary_run_id, key, score)
        for key, score in hr_k_results.items():
            _safe_feedback(summary_run_id, key, score)
        for key, score in failure_counts.items():
            _safe_feedback(summary_run_id, f"failure_{key}", score)
    except Exception as e:
        if _is_connection_error(e):
            print("[LangSmith] summary 업로드 오류: 연결성 문제")
            return
        print(f"[LangSmith] summary 업로드 오류: {e}")
        return

    print(f"[LangSmith] 완료 - project: {project}, tag: {exp_id}")


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    # ── 데이터 로딩 ───────────────────────────────────────────────
    source_label: str
    if args.source == "qna":
        cases = load_qna_legacy(QNA_XLSX, args.sample)
        source_label = "국가 제공 QnA"
    else:
        cases = load_chatbot_final(CHATBOT_XLSX, args.sample)
        source_label = "챗봇 평가용 최종자료"

    faq_cases = load_lease_faq(LEASE_FAQ_JSONL, 0)
    print(f"[Data] 챗봇 평가: {len(cases)}개 | lease_faq HR@K: {len(faq_cases)}개")

    collections = [c.strip() for c in args.collections.split(",") if c.strip()]
    db = embeddings = reranker = llm = None
    failure_counts = {
        "chat_cases": 0,
        "faq_cases": 0,
        "legal_fit": 0,
        "ragas": 0,
    }

    if not args.mock:
        from app.core.dependencies import get_llm, get_vector_db
        from app.rag.embedding.kure import KUREEmbeddings

        db         = get_vector_db()
        embeddings = KUREEmbeddings(model_name="nlpai-lab/KURE-v1")
        # chat_rag()가 HyDE(기본 활성화)에 LLM 필요 — 항상 초기화
        llm        = get_llm()

    # ── 챗봇 QA 검색 + (선택) 답변 생성 ──────────────────────────
    print(f"\n[Mode] {'Mock' if args.mock else '실제 Pinecone'} 검색"
          + (" + 답변 생성" if args.answer else ""))
    rows: list[dict[str, Any]] = []

    for idx, case in enumerate(cases):
        ctx_texts: list[str]
        cited_laws: list[dict]
        response: str | None
        latency_ms: float
        error_message: str | None = None

        try:
            if args.mock:
                ctx_texts = _mock_docs(case, idx)
                cited_laws = []
                response = _mock_answer(case, idx) if args.answer else None
                latency_ms = 1.0
            else:
                from app.rag.chain.chain import chat_rag

                t0 = time.perf_counter()
                result = await chat_rag(
                    message=case["question"],
                    history=[],
                    client=db,
                    embeddings=embeddings,
                    llm=llm,
                    use_multiquery=args.multiquery,
                    use_compression=args.compression,
                )
                src_docs = result["source_documents"]
                ctx_texts = [d.page_content for d in src_docs] or ["(검색 결과 없음)"]
                cited_laws = _extract_cited_laws([d.metadata for d in src_docs])
                response = result["answer"] if args.answer else None
                latency_ms = (time.perf_counter() - t0) * 1000
                law_summary = ", ".join(c["law_name"] for c in cited_laws) if cited_laws else "없음"
                print(f"  [{idx+1}/{len(cases)}] {case['question'][:40]} ... {latency_ms:.0f}ms")
                print(f"    근거 법령: {law_summary}")
        except Exception as e:
            failure_counts["chat_cases"] += 1
            error_message = str(e)
            ctx_texts = ["(case processing failed)"]
            cited_laws = []
            response = None
            latency_ms = 0.0
            print(f"  [{idx+1}/{len(cases)}] CASE ERROR: {case['question'][:40]} -> {e}")

        legal_fit: int | None = None
        legal_fit_reason: str | None = None
        if args.llm:
            try:
                fit_result = (
                    _mock_legal_fit(idx, cited_laws)
                    if args.mock
                    else await _judge_legal_fit(
                        question=case["question"],
                        response=response,
                        cited_laws=cited_laws,
                        contexts=ctx_texts,
                        llm=llm,
                    )
                )
                legal_fit = int(fit_result["score"])
                legal_fit_reason = str(fit_result["reason"])
            except Exception as e:
                failure_counts["legal_fit"] += 1
                legal_fit_reason = f"judge_error: {e}"

        rows.append({
            "분류":               case.get("분류", "기타"),
            "세부유형":           case.get("세부유형", "-"),
            "question":          case["question"],
            "answer":            case.get("answer", ""),
            "retrieved_contexts": ctx_texts,
            "cited_laws":        cited_laws,
            "response":          response,
            "latency_ms":        latency_ms,
            "legal_fit":         legal_fit,
            "legal_fit_reason":  legal_fit_reason,
            "error":             error_message,
        })

    # ── lease_faq HR@K 검색 ────────────────────────────────────────
    print(f"\n[HR@K] lease_faq {len(faq_cases)}개 검색 중...")
    faq_docs_per_case:  list[list[str]]  = []
    faq_metas_per_case: list[list[dict]] = []

    for idx, faq in enumerate(faq_cases):
        try:
            if args.mock:
                faq_docs_per_case.append(_mock_hr_docs(faq, idx))
                faq_metas_per_case.append([])
            else:
                from app.rag.chain.chain import chat_rag

                result = await chat_rag(
                    message=faq["question"],
                    history=[],
                    client=db,
                    embeddings=embeddings,
                    llm=llm,
                    use_multiquery=args.multiquery,
                    use_compression=args.compression,
                )
                src_docs = result["source_documents"]
                doc_metas = [d.metadata for d in src_docs]
                faq_docs_per_case.append([d.page_content for d in src_docs] or ["(검색 결과 없음)"])
                faq_metas_per_case.append(doc_metas)
                cited = _extract_cited_laws(doc_metas)
                law_summary = ", ".join(c["law_name"] for c in cited) if cited else "없음"
                print(f"  [{idx+1}/{len(faq_cases)}] {faq['question'][:40]}")
                print(f"    근거 법령: {law_summary}")
        except Exception as e:
            failure_counts["faq_cases"] += 1
            faq_docs_per_case.append(["(faq processing failed)"])
            faq_metas_per_case.append([])
            print(f"  [{idx+1}/{len(faq_cases)}] FAQ ERROR: {faq['question'][:40]} -> {e}")

    hr_k_results = compute_hr_k_mrr(faq_cases, faq_docs_per_case)
    print(f"[HR@K] 결과: {hr_k_results}")

    # ── RAGAS 평가 ─────────────────────────────────────────────────
    print("\n[RAGAS] 지표 계산 중...")
    try:
        dataset = _build_ragas_dataset(rows)
        scores_df = _run_ragas(dataset, use_llm=args.llm, has_response=args.answer)
    except Exception as e:
        failure_counts["ragas"] += 1
        print(f"[WARN] RAGAS 계산 실패: {e}")
        scores_df = pd.DataFrame(index=range(len(rows)))

    if len(scores_df) != len(rows):
        scores_df = scores_df.reset_index(drop=True)
        scores_df = scores_df.reindex(range(len(rows)))

    scores_df["legal_fit"] = pd.Series([r.get("legal_fit") for r in rows], dtype="float")

    metric_cols = _metric_cols(scores_df)
    overall: dict[str, float] = {
        col: round(float(scores_df[col].mean(skipna=True)), 4)
        for col in metric_cols
    }

    # ── 결과 출력 ──────────────────────────────────────────────────
    W = 60
    print("\n" + "=" * W)
    print(f"  RAG 평가 결과  |  {source_label}")
    print(f"  챗봇 케이스: {len(rows)}   lease_faq: {len(faq_cases)}")
    print("-" * W)
    print("  [RAGAS 검색/답변 품질]")
    for col, val in overall.items():
        print(f"  {METRIC_LABELS.get(col, col):<40} {val:.4f}")
    if hr_k_results:
        print("  [HR@K / MRR - lease_faq]")
        for k, v in hr_k_results.items():
            print(f"  {k:<40} {v:.4f}")
    print("  [실패 카운트]")
    for key, value in failure_counts.items():
        print(f"  {key:<40} {value}")
    print("=" * W)

    # ── 저장 ──────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "created_at":     datetime.now().isoformat(timespec="seconds"),
        "source":         source_label,
        "mode":           "mock" if args.mock else "real",
        "llm_judge":      args.llm,
        "answer_eval":    args.answer,
        "chatbot_count":  len(rows),
        "faq_count":      len(faq_cases),
        "overall_ragas":  overall,
        "hr_k_mrr":       hr_k_results,
        "failure_counts": failure_counts,
        "cases": [
            {
                "분류":     r["분류"],
                "세부유형": r["세부유형"],
                "question": r["question"],
                "answer":   r["answer"],
                "response": r.get("response"),
                "retrieved_contexts": r["retrieved_contexts"],
                "cited_laws": r.get("cited_laws", []),
                "latency_ms": round(r["latency_ms"], 2),
                "legal_fit_reason": r.get("legal_fit_reason"),
                "error": r.get("error"),
                **{
                    col: round(float(scores_df.iloc[i][col]), 4)
                    for col in metric_cols
                    if i < len(scores_df) and scores_df.iloc[i][col] == scores_df.iloc[i][col]
                },
            }
            for i, r in enumerate(rows)
        ],
        "faq_hr_k_cases": [
            {
                "question":   fc["question"],
                "gt_laws":    list(fc["gt_laws"]),
                "retrieved":  faq_docs_per_case[i][:3],
                "cited_laws": _extract_cited_laws(faq_metas_per_case[i]),
            }
            for i, fc in enumerate(faq_cases)
        ],
    }
    json_path = RESULTS_DIR / f"ragas_qna_eval_{ts}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[JSON] 저장: {json_path}")

    plot_path = RESULTS_DIR / f"ragas_qna_eval_{ts}.png"
    _make_plots(rows, scores_df, hr_k_results, plot_path, source_label)

    if args.langsmith:
        log_eval_to_langsmith(
            rows=rows,
            faq_cases=faq_cases,
            faq_docs_per_case=faq_docs_per_case,
            faq_metas_per_case=faq_metas_per_case,
            metric_cols=metric_cols,
            scores_df=scores_df,
            overall=overall,
            hr_k_results=hr_k_results,
            failure_counts=failure_counts,
            source_label=source_label,
            mode="mock" if args.mock else "real",
            llm_judge=args.llm,
            answer_eval=args.answer,
            json_path=json_path,
            plot_path=plot_path,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAGAS 검색 품질 + HR@K + 답변 정확도 평가")
    p.add_argument("--mock",   action="store_true", help="실제 검색 없이 mock 사용")
    p.add_argument(
        "--llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="LLM 기반 RAGAS 지표와 legal_fit 평가 사용 (기본: 사용)",
    )
    p.add_argument(
        "--answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="LLM 답변 생성 + answer_correctness 사용 (기본: 사용)",
    )
    p.add_argument("--source", default="chatbot", choices=["chatbot", "qna"],
                   help="chatbot=챗봇_최종자료(기본) / qna=국가_QnA(레거시)")
    p.add_argument("--sample", type=int, default=0,  help="챗봇 데이터 처음 N개 (0=전체)")
    p.add_argument("--collections", default=",".join(DEFAULT_COLLECTIONS))
    p.add_argument("--k-per-collection", type=int, default=3)
    p.add_argument("--rerank",           action="store_true")
    p.add_argument("--rerank-top-n",     type=int, default=5)
    p.add_argument("--multiquery",       action="store_true",
                   help="쿼리 3개 확장 후 결과 합산 (LLM 호출 추가)")
    p.add_argument("--compression",      action="store_true",
                   help="Contextual Compression — 긴 문서에서 관련 부분만 추출")
    p.add_argument(
        "--langsmith",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="LangSmith에 케이스별/집계 metric 업로드 (기본: 사용)",
    )
    return p.parse_args()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run(parse_args()))
