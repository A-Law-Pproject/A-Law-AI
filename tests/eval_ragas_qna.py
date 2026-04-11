"""RAGAS 기반 retrieval 품질 평가 — 국가 제공 QnA 자료

국가_제공_QnA_자료.xlsx를 바탕으로 retrieval 품질을 RAGAS 지표로 측정하고
결과를 plot으로 시각화합니다.

지표:
  - non_llm_context_precision   : 검색된 컨텍스트 중 정답 관련 비율 (키워드 기반)
  - non_llm_context_recall      : 정답이 검색된 컨텍스트로 얼마나 커버되는지
  - llm_context_precision       : LLM 기반 컨텍스트 정밀도 (--llm 옵션 시)
  - llm_context_recall          : LLM 기반 컨텍스트 재현율 (--llm 옵션 시)

Examples:
    # 모의(mock) 검색, NonLLM 지표
    .venv\\Scripts\\python.exe tests\\eval_ragas_qna.py --mock

    # 실제 Pinecone 검색, NonLLM 지표
    .venv\\Scripts\\python.exe tests\\eval_ragas_qna.py

    # 실제 검색 + LLM 지표
    .venv\\Scripts\\python.exe tests\\eval_ragas_qna.py --llm

    # 샘플 10개만
    .venv\\Scripts\\python.exe tests\\eval_ragas_qna.py --mock --sample 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# .env 로드 (OPENAI_API_KEY 등)
load_dotenv(ROOT / ".env")

QNA_XLSX = ROOT / "tests" / "평가데이터셋" / "국가_제공_QnA_자료.xlsx"
RESULTS_DIR = ROOT / "results"

DEFAULT_COLLECTIONS = [
    "law_database",
    "law_statutes",
    "contracts",
    "special_clauses_illegal",
    "special_clauses_normal",
]

# ──────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────

def load_qna(path: Path, sample: int) -> list[dict[str, Any]]:
    """국가 제공 QnA xlsx를 로드하고, 헤더 반복 행을 제거한다."""
    df = pd.read_excel(path, engine="openpyxl")
    # 컬럼 정규화
    df.columns = ["분류", "question", "answer", "notes"]

    # 헤더 반복 행 제거 (분류 열이 '상세 분류' 또는 '분류'인 경우)
    df = df[~df["분류"].isin(["분류", "상세 분류"])].reset_index(drop=True)
    df = df.dropna(subset=["question", "answer"])

    cases = df.to_dict("records")
    if sample > 0:
        cases = cases[:sample]
    return cases


# ──────────────────────────────────────────────
# Mock 검색
# ──────────────────────────────────────────────

def _mock_docs(case: dict[str, Any], idx: int) -> list[str]:
    """간단한 mock: 일부 케이스는 정답을 포함한 컨텍스트 반환."""
    answer = str(case.get("answer", ""))
    notes = str(case.get("notes", ""))
    good_ctx = f"{answer}. {notes}"
    noise_ctx = "임대차 관련 일반 안내 사항입니다. 계약서 작성 시 주의하세요."

    if idx % 5 == 0:
        # 완전 미스
        return [noise_ctx, noise_ctx]
    if idx % 3 == 0:
        # 부분 히트
        return [noise_ctx, good_ctx]
    # 정상 히트
    return [good_ctx, noise_ctx]


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
) -> list[str]:
    from app.rag.retriever.multi_retriever import search_multi_index

    docs = search_multi_index(
        db,
        embeddings,
        question,
        collections=collections,
        k_per_collection=k_per_collection,
        reranker=reranker,
        rerank_top_n=rerank_top_n,
    )
    texts: list[str] = []
    for doc in docs:
        if hasattr(doc, "page_content"):
            texts.append(str(doc.page_content))
        elif isinstance(doc, dict):
            texts.append(str(doc.get("content") or doc.get("page_content") or ""))
        else:
            texts.append(str(getattr(doc, "content", "")))
    return [t for t in texts if t.strip()]


# ──────────────────────────────────────────────
# RAGAS 평가
# ──────────────────────────────────────────────

def _build_ragas_dataset(rows: list[dict[str, Any]]):
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    samples = []
    for row in rows:
        samples.append(
            SingleTurnSample(
                user_input=row["question"],
                retrieved_contexts=row["retrieved_contexts"],
                # NonLLMContextPrecisionWithReference 는 reference_contexts 필요
                reference_contexts=[row["answer"]],
                reference=row["answer"],
            )
        )
    return EvaluationDataset(samples=samples)


def _run_nonllm_metrics(dataset) -> dict[str, float]:
    from ragas import evaluate
    from ragas.metrics import _NonLLMContextPrecisionWithReference, _NonLLMContextRecall

    result = evaluate(
        dataset,
        metrics=[
            _NonLLMContextPrecisionWithReference(),
            _NonLLMContextRecall(),
        ],
    )
    return dict(result)


def _run_llm_metrics(dataset) -> dict[str, float]:
    """LLM 기반 RAGAS 지표 (OPENAI_API_KEY 필요)."""
    from ragas import evaluate
    from ragas.metrics import _LLMContextPrecisionWithReference, _LLMContextRecall

    result = evaluate(
        dataset,
        metrics=[
            _LLMContextPrecisionWithReference(),
            _LLMContextRecall(),
        ],
    )
    return dict(result)


def _per_sample_scores(dataset, use_llm: bool) -> pd.DataFrame:
    """샘플별 점수 DataFrame 반환."""
    from ragas import evaluate
    from ragas.metrics import (
        _NonLLMContextPrecisionWithReference,
        _NonLLMContextRecall,
    )

    metrics = [
        _NonLLMContextPrecisionWithReference(),
        _NonLLMContextRecall(),
    ]
    if use_llm:
        from ragas.metrics import _LLMContextPrecisionWithReference, _LLMContextRecall
        metrics += [_LLMContextPrecisionWithReference(), _LLMContextRecall()]

    result = evaluate(dataset, metrics=metrics)
    return result.to_pandas()


# ──────────────────────────────────────────────
# Plot 생성
# ──────────────────────────────────────────────

METRIC_LABELS = {
    "non_llm_context_precision_with_reference": "Context Precision\n(Non-LLM)",
    "non_llm_context_recall": "Context Recall\n(Non-LLM)",
    "llm_context_precision_with_reference": "Context Precision\n(LLM)",
    "context_recall": "Context Recall\n(LLM)",
}

COLORS = {
    "non_llm_context_precision_with_reference": "#4C72B0",
    "non_llm_context_recall": "#55A868",
    "llm_context_precision_with_reference": "#C44E52",
    "context_recall": "#8172B2",
}


def _metric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c in METRIC_LABELS]


def _make_plots(
    rows: list[dict[str, Any]],
    scores_df: pd.DataFrame,
    save_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.font_manager as fm
    import numpy as np

    # Windows 한글 폰트 설정
    _korean_fonts = ["Malgun Gothic", "NanumGothic", "Apple SD Gothic Neo", "UnDotum"]
    _available = {f.name for f in fm.fontManager.ttflist}
    for _fn in _korean_fonts:
        if _fn in _available:
            plt.rcParams["font.family"] = _fn
            break
    plt.rcParams["axes.unicode_minus"] = False

    metric_cols = _metric_cols(scores_df)
    if not metric_cols:
        print("[WARN] 지표 컬럼이 없어 plot을 생성하지 못했습니다.")
        return

    # 분류 컬럼 추가
    cats = [r["분류"] for r in rows]
    scores_df = scores_df.copy()
    scores_df["분류"] = cats[: len(scores_df)]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "국가 제공 QnA — RAG Retrieval 품질 평가 (RAGAS)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # ── 1. 전체 평균 지표 바 차트 ─────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    overall_means = {col: float(scores_df[col].mean()) for col in metric_cols}
    labels = [METRIC_LABELS.get(k, k) for k in overall_means]
    values = list(overall_means.values())
    bar_colors = [COLORS.get(k, "#999999") for k in overall_means]

    bars = ax1.barh(labels, values, color=bar_colors, edgecolor="white", height=0.6)
    for bar, v in zip(bars, values):
        ax1.text(
            min(v + 0.02, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )
    ax1.set_xlim(0, 1.1)
    ax1.set_xlabel("Score", fontsize=10)
    ax1.set_title("전체 평균 RAGAS 지표", fontsize=12, fontweight="bold")
    ax1.axvline(0.7, color="red", linestyle="--", alpha=0.5, linewidth=1, label="목표 0.7")
    ax1.legend(fontsize=8)
    ax1.grid(axis="x", alpha=0.3)

    # ── 2. 분류별 평균 ────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    cat_group = scores_df.groupby("분류")[metric_cols].mean()
    cat_group = cat_group.sort_values(metric_cols[0], ascending=False)

    n_cats = len(cat_group)
    n_metrics = len(metric_cols)
    x = np.arange(n_cats)
    width = 0.8 / n_metrics

    for i, col in enumerate(metric_cols):
        offsets = x + (i - n_metrics / 2 + 0.5) * width
        ax2.bar(
            offsets,
            cat_group[col].values,
            width=width,
            color=COLORS.get(col, "#999999"),
            label=METRIC_LABELS.get(col, col),
            alpha=0.85,
            edgecolor="white",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_group.index, rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_title("분류별 RAGAS 지표", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.axhline(0.7, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax2.grid(axis="y", alpha=0.3)

    # ── 3. 점수 분포 (박스 플롯) ──────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    box_data = [scores_df[col].dropna().values for col in metric_cols]
    bp = ax3.boxplot(
        box_data,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    for patch, col in zip(bp["boxes"], metric_cols):
        patch.set_facecolor(COLORS.get(col, "#999999"))
        patch.set_alpha(0.75)

    ax3.set_xticks(range(1, len(metric_cols) + 1))
    ax3.set_xticklabels(
        [METRIC_LABELS.get(c, c) for c in metric_cols], fontsize=9
    )
    ax3.set_ylim(-0.05, 1.1)
    ax3.set_ylabel("Score", fontsize=10)
    ax3.set_title("점수 분포 (Box Plot)", fontsize=12, fontweight="bold")
    ax3.axhline(0.7, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax3.grid(axis="y", alpha=0.3)

    # ── 4. 질문별 히트맵 ─────────────────────────────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    heat_df = scores_df[metric_cols].copy()
    heat_df.index = [
        f"Q{i+1}: {rows[i]['question'][:22]}…" if len(rows[i]["question"]) > 22 else f"Q{i+1}: {rows[i]['question']}"
        for i in range(min(len(rows), len(heat_df)))
    ]

    # 너무 많으면 20개만 표시
    if len(heat_df) > 20:
        heat_df = heat_df.iloc[:20]

    im = ax4.imshow(heat_df.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax4.set_xticks(range(len(metric_cols)))
    ax4.set_xticklabels(
        [METRIC_LABELS.get(c, c) for c in metric_cols], fontsize=8, rotation=20, ha="right"
    )
    ax4.set_yticks(range(len(heat_df)))
    ax4.set_yticklabels(heat_df.index, fontsize=7)
    ax4.set_title(
        f"질문별 점수 히트맵 (상위 {len(heat_df)}개)",
        fontsize=12,
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    # 셀 값 표시
    for row_i in range(len(heat_df)):
        for col_i in range(len(metric_cols)):
            val = heat_df.values[row_i, col_i]
            ax4.text(
                col_i,
                row_i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="black" if 0.3 < val < 0.8 else "white",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] 저장: {save_path}")


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    cases = load_qna(QNA_XLSX, args.sample)
    print(f"[Data] 총 {len(cases)}개 QnA 케이스 로드됨")

    # ── 검색 실행 ──────────────────────────────────────────────────
    if args.mock:
        print("[Mode] Mock 검색")
        rows: list[dict[str, Any]] = []
        for idx, case in enumerate(cases):
            rows.append(
                {
                    "분류": case["분류"],
                    "question": case["question"],
                    "answer": case["answer"],
                    "notes": case.get("notes", ""),
                    "retrieved_contexts": _mock_docs(case, idx),
                    "latency_ms": 1.0,
                }
            )
    else:
        from app.core.dependencies import get_embeddings, get_vector_db
        from app.rag.retriever.reranker import get_reranker

        print("[Mode] 실제 Pinecone 검색")
        db = get_vector_db()
        embeddings = get_embeddings()
        reranker = get_reranker() if args.rerank else None
        collections = [c.strip() for c in args.collections.split(",") if c.strip()]

        rows = []
        for case in cases:
            start = time.perf_counter()
            ctx_texts = _real_docs(
                case["question"],
                db,
                embeddings,
                collections,
                args.k_per_collection,
                reranker,
                args.rerank_top_n,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            rows.append(
                {
                    "분류": case["분류"],
                    "question": case["question"],
                    "answer": case["answer"],
                    "notes": case.get("notes", ""),
                    "retrieved_contexts": ctx_texts or ["(검색 결과 없음)"],
                    "latency_ms": latency_ms,
                }
            )
            print(f"  [{len(rows)}/{len(cases)}] {case['question'][:40]} … {latency_ms:.0f}ms")

    # ── RAGAS 평가 ─────────────────────────────────────────────────
    print("\n[RAGAS] 지표 계산 중…")
    dataset = _build_ragas_dataset(rows)
    scores_df = _per_sample_scores(dataset, use_llm=args.llm)

    # 전체 평균 출력
    metric_cols = _metric_cols(scores_df)
    overall: dict[str, float] = {col: float(scores_df[col].mean()) for col in metric_cols}

    print("\n" + "=" * 50)
    print("RAGAS Retrieval 평가 결과 (국가 제공 QnA 자료)")
    print(f"케이스 수: {len(rows)}")
    print("-" * 50)
    for col, val in overall.items():
        print(f"  {METRIC_LABELS.get(col, col):<35} {val:.4f}")
    print("=" * 50)

    # ── 결과 저장 ─────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "mock" if args.mock else "real",
        "llm_judge": args.llm,
        "case_count": len(rows),
        "overall_metrics": overall,
        "cases": [
            {
                "분류": r["분류"],
                "question": r["question"],
                "answer": r["answer"],
                "retrieved_contexts": r["retrieved_contexts"],
                "latency_ms": round(r["latency_ms"], 2),
                **{
                    col: round(float(scores_df.iloc[i][col]), 4)
                    for col in metric_cols
                    if i < len(scores_df)
                },
            }
            for i, r in enumerate(rows)
        ],
    }
    json_path = RESULTS_DIR / f"ragas_qna_eval_{ts}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[JSON] 저장: {json_path}")

    # Plot
    plot_path = RESULTS_DIR / f"ragas_qna_eval_{ts}.png"
    _make_plots(rows, scores_df, plot_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGAS retrieval 평가 — 국가 제공 QnA")
    parser.add_argument("--mock", action="store_true", help="실제 검색 없이 mock 데이터 사용")
    parser.add_argument("--llm", action="store_true", help="LLM 기반 RAGAS 지표 추가 (OPENAI_API_KEY 필요)")
    parser.add_argument("--sample", type=int, default=0, help="처음 N개 케이스만 사용 (0=전체)")
    parser.add_argument("--collections", default=",".join(DEFAULT_COLLECTIONS))
    parser.add_argument("--k-per-collection", type=int, default=3)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-top-n", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))
