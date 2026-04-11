"""
RAG A/B 실험 비교 스크립트 — 포트폴리오용

핵심 아이디어:
  모든 개선 기능이 '선택적 파라미터'로 구현되어 있으므로,
  Before  = 파라미터 비활성화 (reranker=None, HyDE 미사용 등)
  After   = 파라미터 활성화

실험 목록:
  --exp 1  Reranker Before/After   (HitRate@3/5, MRR)
  --exp 3  Score Threshold 분포    (컬렉션별 히스토그램 + Precision 비교)
  --exp 4  HyDE Before/After       (구어체 질문 vs 법률체 질문 분리)
  --exp 5  Multi-query Before/After (HitRate, Recall, Latency 트레이드오프)

실행 예시:
  # 전체 실험 (mock)
  python scripts/eval_ab_compare.py --all --mock

  # 실험 1만, 샘플 15개
  python scripts/eval_ab_compare.py --exp 1 --sample 15

  # 전체 실험 + 플롯 저장
  python scripts/eval_ab_compare.py --all --save-plots
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경에서도 동작
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as _fm

# 한국어 폰트 설정 (Windows: Malgun Gothic, 없으면 AppleGothic → DejaVu 순)
def _set_korean_font():
    _candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim"]
    _available = {f.name for f in _fm.fontManager.ttflist}
    for _name in _candidates:
        if _name in _available:
            matplotlib.rcParams["font.family"] = _name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

_set_korean_font()
import numpy as np
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── 경로 ──────────────────────────────────────────────────
DATASET_PATH = Path(__file__).parent.parent / "tests" / "eval_dataset.json"
RESULTS_DIR  = Path(__file__).parent.parent / "results"
PLOTS_DIR    = RESULTS_DIR / "plots"

# ── 컬렉션 설정 ───────────────────────────────────────────
QA_COLLECTIONS  = ["law_database", "contracts"]
ALL_COLLECTIONS = ["law_database", "contracts", "special_clauses_illegal", "special_clauses_normal"]

# Exp 3: 컬렉션별 튜닝 threshold 후보 (분포 분석 후 조정)
TUNED_THRESHOLDS: dict[str, float] = {
    "law_database":             0.40,
    "contracts":                0.38,
    "special_clauses_illegal":  0.50,
    "special_clauses_normal":   0.45,
    "default":                  0.30,
}


# ══════════════════════════════════════════════════════════
# 데이터 유틸
# ══════════════════════════════════════════════════════════

def load_questions(sample: int = 0) -> list[dict]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    qs = [q for q in data["questions"] if len(q.get("expected_keywords", [])) >= 2]
    if sample > 0:
        qs = random.sample(qs, min(sample, len(qs)))
    return qs


def load_colloquial_questions() -> tuple[list[dict], list[dict]]:
    """구어체(C*)와 법률체(D*) 질문 분리 — Exp 4 전용."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    colloquial = [q for q in data["questions"] if q.get("is_colloquial")]
    formal     = [q for q in data["questions"] if not q.get("is_colloquial")
                  and q["category"] == "difficulty"]
    return colloquial, formal


# ══════════════════════════════════════════════════════════
# 평가 지표
# ══════════════════════════════════════════════════════════

def _is_relevant(doc, keywords: list[str]) -> bool:
    """문서가 expected_keywords 중 하나 이상 포함하면 관련 문서로 판단."""
    content = doc.page_content.lower()
    return any(kw.lower() in content for kw in keywords)


def hit_rate_at_k(docs_list: list, keywords_list: list[list[str]], k: int) -> float:
    if not docs_list:
        return 0.0
    hits = sum(
        1 for docs, kws in zip(docs_list, keywords_list)
        if any(_is_relevant(d, kws) for d in docs[:k])
    )
    return hits / len(docs_list)


def mrr(docs_list: list, keywords_list: list[list[str]]) -> float:
    if not docs_list:
        return 0.0
    rr_sum = 0.0
    for docs, kws in zip(docs_list, keywords_list):
        for i, doc in enumerate(docs, 1):
            if _is_relevant(doc, kws):
                rr_sum += 1.0 / i
                break
    return rr_sum / len(docs_list)


def precision_at_k(docs_list: list, keywords_list: list[list[str]], k: int) -> float:
    """Top-K 문서 중 관련 문서 비율."""
    if not docs_list:
        return 0.0
    p_sum = 0.0
    for docs, kws in zip(docs_list, keywords_list):
        top = docs[:k]
        if top:
            p_sum += sum(1 for d in top if _is_relevant(d, kws)) / len(top)
    return p_sum / len(docs_list)


# ══════════════════════════════════════════════════════════
# 실험 1 — Reranker Before / After
# ══════════════════════════════════════════════════════════

def run_exp1_reranker(questions, db, emb, mock: bool = False):
    """
    Before : search_multi_index(..., reranker=None)
    After  : search_multi_index(..., reranker=BGEReranker)
    """
    from app.rag.retriever.multi_retriever import search_multi_index
    from app.rag.retriever.reranker import get_reranker

    reranker = None if mock else get_reranker()
    K = 5

    before_docs, after_docs, keywords_all = [], [], []
    latency_before, latency_after = [], []

    for q in questions:
        kws = q.get("expected_keywords", [])
        keywords_all.append(kws)

        if mock:
            before_docs.append([])
            after_docs.append([])
            latency_before.append(0.1)
            latency_after.append(0.25)
            continue

        # Before
        t0 = time.perf_counter()
        docs_b = search_multi_index(db, emb, q["question"], QA_COLLECTIONS,
                                    k_per_collection=K, reranker=None)
        latency_before.append(time.perf_counter() - t0)
        before_docs.append(docs_b)

        # After
        t0 = time.perf_counter()
        docs_a = search_multi_index(db, emb, q["question"], QA_COLLECTIONS,
                                    k_per_collection=K, reranker=reranker,
                                    rerank_top_n=K)
        latency_after.append(time.perf_counter() - t0)
        after_docs.append(docs_a)

    results = {
        "before": {
            "hr3":  round(hit_rate_at_k(before_docs, keywords_all, 3), 3),
            "hr5":  round(hit_rate_at_k(before_docs, keywords_all, 5), 3),
            "mrr":  round(mrr(before_docs, keywords_all), 3),
            "p3":   round(precision_at_k(before_docs, keywords_all, 3), 3),
            "avg_latency": round(sum(latency_before) / len(latency_before), 3),
        },
        "after": {
            "hr3":  round(hit_rate_at_k(after_docs, keywords_all, 3), 3),
            "hr5":  round(hit_rate_at_k(after_docs, keywords_all, 5), 3),
            "mrr":  round(mrr(after_docs, keywords_all), 3),
            "p3":   round(precision_at_k(after_docs, keywords_all, 3), 3),
            "avg_latency": round(sum(latency_after) / len(latency_after), 3),
        },
        "n": len(questions),
    }
    if mock:
        # Mock 시 샘플 수치로 채우기
        results["before"] = {"hr3": 0.62, "hr5": 0.73, "mrr": 0.61, "p3": 0.48, "avg_latency": 1.1}
        results["after"]  = {"hr3": 0.79, "hr5": 0.88, "mrr": 0.80, "p3": 0.67, "avg_latency": 2.4}
    return results


# ══════════════════════════════════════════════════════════
# 실험 3 — Score Threshold 분포 분석
# ══════════════════════════════════════════════════════════

def run_exp3_threshold(questions, db, emb, mock: bool = False):
    """
    Step 1: 각 컬렉션의 score 분포 수집 (threshold=0.0 으로 전부 가져옴)
    Step 2: uniform 0.3 vs TUNED_THRESHOLDS 비교
    """
    from app.rag.retriever.multi_retriever import search_multi_index, search_collection

    score_dist: dict[str, list[float]] = {c: [] for c in ALL_COLLECTIONS}
    before_docs, tuned_docs, keywords_all = [], [], []

    for q in questions:
        kws = q.get("expected_keywords", [])
        keywords_all.append(kws)

        if mock:
            for c in ALL_COLLECTIONS:
                # 가짜 정규분포 점수 생성
                mu = {"law_database": 0.45, "contracts": 0.40,
                      "special_clauses_illegal": 0.55, "special_clauses_normal": 0.50}.get(c, 0.4)
                score_dist[c].extend(list(np.random.normal(mu, 0.12, 8).clip(0, 1)))
            before_docs.append([])
            tuned_docs.append([])
            continue

        qv = emb.embed_query(q["question"])
        for coll in ALL_COLLECTIONS:
            docs = search_collection(db, emb, q["question"], coll,
                                     k=10, score_threshold=0.0, query_vector=qv)
            score_dist[coll].extend(
                [d.metadata.get("score", 0) for d in docs]
            )

        # Before: uniform 0.3
        docs_b = search_multi_index(db, emb, q["question"], QA_COLLECTIONS,
                                    k_per_collection=5, score_threshold=0.3)
        before_docs.append(docs_b)

        # After: per-collection tuned
        docs_t = search_multi_index(db, emb, q["question"], QA_COLLECTIONS,
                                    k_per_collection=5, score_threshold=TUNED_THRESHOLDS)
        tuned_docs.append(docs_t)

    results = {
        "score_distributions": {c: v for c, v in score_dist.items()},
        "before_uniform_0.3": {
            "hr3": round(hit_rate_at_k(before_docs, keywords_all, 3), 3),
            "p3":  round(precision_at_k(before_docs, keywords_all, 3), 3),
            "avg_docs": round(sum(len(d) for d in before_docs) / max(len(before_docs), 1), 1),
        },
        "after_tuned": {
            "hr3": round(hit_rate_at_k(tuned_docs, keywords_all, 3), 3),
            "p3":  round(precision_at_k(tuned_docs, keywords_all, 3), 3),
            "avg_docs": round(sum(len(d) for d in tuned_docs) / max(len(tuned_docs), 1), 1),
        },
        "tuned_thresholds": TUNED_THRESHOLDS,
        "n": len(questions),
    }
    if mock:
        results["before_uniform_0.3"] = {"hr3": 0.64, "p3": 0.41, "avg_docs": 7.2}
        results["after_tuned"]        = {"hr3": 0.71, "p3": 0.56, "avg_docs": 5.1}
    return results


# ══════════════════════════════════════════════════════════
# 실험 4 — HyDE Before / After
# ══════════════════════════════════════════════════════════

def run_exp4_hyde(colloquial_qs, formal_qs, db, emb, llm, mock: bool = False):
    """
    Before: embed(원문 질문)
    After : embed(LLM 생성 가상 법률 문서)
    구어체 vs 법률체 질문 각각 측정하여 HyDE 효과 비교.
    """
    from app.rag.retriever.multi_retriever import search_multi_index
    from app.rag.retriever.query_expansion import expand_query_hyde

    def _run_group(qs, label):
        before_docs, after_docs, kws_list = [], [], []
        lat_b, lat_a = [], []
        for q in qs:
            kws = q.get("expected_keywords", [])
            kws_list.append(kws)
            if mock:
                before_docs.append([])
                after_docs.append([])
                lat_b.append(0.9)
                lat_a.append(2.1)
                continue
            t0 = time.perf_counter()
            docs_b = search_multi_index(db, emb, q["question"], QA_COLLECTIONS,
                                        k_per_collection=5)
            lat_b.append(time.perf_counter() - t0)
            before_docs.append(docs_b)

            t0 = time.perf_counter()
            hyde_text = expand_query_hyde(q["question"], llm)
            docs_a = search_multi_index(db, emb, hyde_text, QA_COLLECTIONS,
                                        k_per_collection=5)
            lat_a.append(time.perf_counter() - t0)
            after_docs.append(docs_a)

        return {
            "group": label,
            "n": len(qs),
            "before": {
                "hr3": round(hit_rate_at_k(before_docs, kws_list, 3), 3),
                "hr5": round(hit_rate_at_k(before_docs, kws_list, 5), 3),
                "mrr": round(mrr(before_docs, kws_list), 3),
                "avg_latency": round(sum(lat_b) / max(len(lat_b), 1), 3),
            },
            "after": {
                "hr3": round(hit_rate_at_k(after_docs, kws_list, 3), 3),
                "hr5": round(hit_rate_at_k(after_docs, kws_list, 5), 3),
                "mrr": round(mrr(after_docs, kws_list), 3),
                "avg_latency": round(sum(lat_a) / max(len(lat_a), 1), 3),
            },
        }

    r_colloquial = _run_group(colloquial_qs, "구어체")
    r_formal     = _run_group(formal_qs,     "법률체")

    if mock:
        r_colloquial["before"] = {"hr3": 0.55, "hr5": 0.68, "mrr": 0.56, "avg_latency": 0.9}
        r_colloquial["after"]  = {"hr3": 0.78, "hr5": 0.88, "mrr": 0.77, "avg_latency": 2.1}
        r_formal["before"]     = {"hr3": 0.74, "hr5": 0.83, "mrr": 0.72, "avg_latency": 0.9}
        r_formal["after"]      = {"hr3": 0.77, "hr5": 0.85, "mrr": 0.74, "avg_latency": 2.1}

    return {"colloquial": r_colloquial, "formal": r_formal}


# ══════════════════════════════════════════════════════════
# 실험 5 — Multi-query Before / After
# ══════════════════════════════════════════════════════════

def run_exp5_multiquery(questions, db, emb, llm, mock: bool = False):
    """
    Before: 단일 쿼리, k_per_collection=3
    After : 멀티쿼리(원문+변형 3개), 각 k_per_collection=2, 중복 제거
    """
    from app.rag.retriever.multi_retriever import search_multi_index, _deduplicate
    from app.rag.retriever.query_expansion import expand_query_multi

    before_docs, after_docs, kws_list = [], [], []
    lat_b, lat_a, recall_gain = [], [], []

    for q in questions:
        kws = q.get("expected_keywords", [])
        kws_list.append(kws)

        if mock:
            before_docs.append([])
            after_docs.append([])
            lat_b.append(0.8)
            lat_a.append(2.9)
            recall_gain.append(random.uniform(0.1, 0.4))
            continue

        # Before
        t0 = time.perf_counter()
        docs_b = search_multi_index(db, emb, q["question"], QA_COLLECTIONS,
                                    k_per_collection=3)
        lat_b.append(time.perf_counter() - t0)
        before_docs.append(docs_b)

        # After: 멀티쿼리
        t0 = time.perf_counter()
        queries = expand_query_multi(q["question"], llm, n=3)
        all_docs = []
        for qv in queries:
            all_docs.extend(
                search_multi_index(db, emb, qv, QA_COLLECTIONS, k_per_collection=2)
            )
        docs_a = _deduplicate(all_docs)
        lat_a.append(time.perf_counter() - t0)
        after_docs.append(docs_a)

        # Recall gain: 추가로 커버된 관련 문서 비율
        before_kw_covered = set(
            kw for doc in docs_b for kw in kws if kw.lower() in doc.page_content.lower()
        )
        after_kw_covered = set(
            kw for doc in docs_a for kw in kws if kw.lower() in doc.page_content.lower()
        )
        if kws:
            gain = (len(after_kw_covered) - len(before_kw_covered)) / len(kws)
            recall_gain.append(max(0.0, gain))
        else:
            recall_gain.append(0.0)

    results = {
        "before": {
            "hr3": round(hit_rate_at_k(before_docs, kws_list, 3), 3),
            "hr5": round(hit_rate_at_k(before_docs, kws_list, 5), 3),
            "mrr": round(mrr(before_docs, kws_list), 3),
            "avg_latency": round(sum(lat_b) / max(len(lat_b), 1), 3),
            "avg_docs": round(sum(len(d) for d in before_docs) / max(len(before_docs), 1), 1),
        },
        "after": {
            "hr3": round(hit_rate_at_k(after_docs, kws_list, 3), 3),
            "hr5": round(hit_rate_at_k(after_docs, kws_list, 5), 3),
            "mrr": round(mrr(after_docs, kws_list), 3),
            "avg_latency": round(sum(lat_a) / max(len(lat_a), 1), 3),
            "avg_docs": round(sum(len(d) for d in after_docs) / max(len(after_docs), 1), 1),
        },
        "avg_recall_gain": round(sum(recall_gain) / max(len(recall_gain), 1), 3),
        "latency_tradeoff": list(zip(
            [round(x, 2) for x in lat_b],
            [round(x, 2) for x in lat_a],
        )),
        "n": len(questions),
    }
    if mock:
        results["before"] = {"hr3": 0.63, "hr5": 0.74, "mrr": 0.64, "avg_latency": 0.8,  "avg_docs": 6.0}
        results["after"]  = {"hr3": 0.76, "hr5": 0.86, "mrr": 0.77, "avg_latency": 3.0, "avg_docs": 9.4}
        results["avg_recall_gain"] = 0.21
        results["latency_tradeoff"] = [(0.8, 3.0)] * results["n"]
    return results


# ══════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════

_COLORS = {"before": "#6c8ebf", "after": "#82b366"}


def _bar_comparison(ax, metrics: list[str], b_vals: list, a_vals: list,
                    title: str, ylabel: str = "Score") -> None:
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, b_vals, w, label="Before", color=_COLORS["before"])
    ax.bar(x + w/2, a_vals, w, label="After",  color=_COLORS["after"])
    for xi, (bv, av) in enumerate(zip(b_vals, a_vals)):
        ax.text(xi - w/2, bv + 0.01, f"{bv:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w/2, av + 0.01, f"{av:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))


def plot_exp1(results: dict, save_dir: Path | None = None) -> None:
    b, a = results["before"], results["after"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Exp 1: Reranker (BGE CrossEncoder) — n={results['n']}", fontsize=13, fontweight="bold")

    # 왼쪽: Hit Rate / MRR
    _bar_comparison(
        axes[0],
        ["Hit Rate@3", "Hit Rate@5", "MRR", "Precision@3"],
        [b["hr3"], b["hr5"], b["mrr"], b["p3"]],
        [a["hr3"], a["hr5"], a["mrr"], a["p3"]],
        "Retrieval 품질 지표",
    )

    # 오른쪽: Latency
    axes[1].bar(["Before", "After"],
                [b["avg_latency"], a["avg_latency"]],
                color=[_COLORS["before"], _COLORS["after"]], width=0.4)
    for xi, v in enumerate([b["avg_latency"], a["avg_latency"]]):
        axes[1].text(xi, v + 0.02, f"{v:.2f}s", ha="center", fontsize=9)
    axes[1].set_ylabel("평균 응답 시간 (초)")
    axes[1].set_title("응답 시간 트레이드오프")

    fig.tight_layout()
    _save_or_show(fig, save_dir, "exp1_reranker.png")


def plot_exp3(results: dict, save_dir: Path | None = None) -> None:
    dists = results["score_distributions"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Exp 3: Score Threshold 분석", fontsize=13, fontweight="bold")

    # 왼쪽: 컬렉션별 score 분포
    colors = ["#5b8dd9", "#e06c75", "#98c379", "#e5c07b"]
    for (coll, scores), color in zip(dists.items(), colors):
        if scores:
            axes[0].hist(scores, bins=30, alpha=0.6, label=coll, color=color)
            thresh = TUNED_THRESHOLDS.get(coll, 0.3)
            axes[0].axvline(thresh, color=color, linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Cosine Similarity Score")
    axes[0].set_ylabel("빈도")
    axes[0].set_title("컬렉션별 Score 분포 (점선 = 제안 Threshold)")
    axes[0].legend(fontsize=8)

    # 오른쪽: Before vs After 비교
    b = results["before_uniform_0.3"]
    t = results["after_tuned"]
    x = np.arange(3)
    w = 0.35
    b_vals = [b["hr3"], b["p3"], b["avg_docs"] / 10]
    t_vals = [t["hr3"], t["p3"], t["avg_docs"] / 10]
    axes[1].bar(x - w/2, b_vals, w, label="Uniform 0.3",    color=_COLORS["before"])
    axes[1].bar(x + w/2, t_vals, w, label="Per-collection", color=_COLORS["after"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Hit Rate@3", "Precision@3", "Avg Docs (÷10)"])
    axes[1].set_title("Threshold 전략 비교")
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()

    fig.tight_layout()
    _save_or_show(fig, save_dir, "exp3_threshold.png")


def plot_exp4(results: dict, save_dir: Path | None = None) -> None:
    c = results["colloquial"]
    f = results["formal"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Exp 4: HyDE — 구어체 vs 법률체 질문 효과 비교", fontsize=13, fontweight="bold")

    for ax, group, title in [
        (axes[0], c, f"구어체 질문 (n={c['n']})"),
        (axes[1], f, f"법률체 질문 (n={f['n']})"),
    ]:
        b, a = group["before"], group["after"]
        _bar_comparison(
            ax,
            ["HR@3", "HR@5", "MRR"],
            [b["hr3"], b["hr5"], b["mrr"]],
            [a["hr3"], a["hr5"], a["mrr"]],
            title,
        )

    fig.tight_layout()
    _save_or_show(fig, save_dir, "exp4_hyde.png")


def plot_exp5(results: dict, save_dir: Path | None = None) -> None:
    b, a = results["before"], results["after"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Exp 5: Multi-query (원문+변형 3개) — n={results['n']}", fontsize=13, fontweight="bold")

    # 왼쪽: 지표 비교
    _bar_comparison(
        axes[0],
        ["HR@3", "HR@5", "MRR"],
        [b["hr3"], b["hr5"], b["mrr"]],
        [a["hr3"], a["hr5"], a["mrr"]],
        "Retrieval 품질 지표",
    )

    # 오른쪽: Latency vs HR@3 트레이드오프 산점도
    pairs = results.get("latency_tradeoff", [])
    if pairs:
        b_lats = [p[0] for p in pairs]
        a_lats = [p[1] for p in pairs]
        axes[1].scatter(b_lats, [b["hr3"]] * len(b_lats),
                        color=_COLORS["before"], alpha=0.6, label="Before", s=40)
        axes[1].scatter(a_lats, [a["hr3"]] * len(a_lats),
                        color=_COLORS["after"],  alpha=0.6, label="After",  s=40)
    axes[1].set_xlabel("응답 시간 (초)")
    axes[1].set_ylabel("Hit Rate@3")
    axes[1].set_title("Latency–Quality 트레이드오프")
    axes[1].legend()

    fig.tight_layout()
    _save_or_show(fig, save_dir, "exp5_multiquery.png")


def _save_or_show(fig, save_dir: Path | None, filename: str) -> None:
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  💾 플롯 저장: {path}")
    else:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════
# 결과 출력
# ══════════════════════════════════════════════════════════

def _pct(v: float) -> str:
    return f"{v:.1%}"


def print_exp1(results: dict) -> None:
    b, a = results["before"], results["after"]
    print(f"\n{'─'*60}")
    print(f"  실험 1: Reranker (BGE CrossEncoder)  n={results['n']}")
    print(f"{'─'*60}")
    print(f"  {'지표':<18} {'Before':>10} {'After':>10} {'개선':>10}")
    print(f"  {'─'*50}")
    for key, label in [("hr3","Hit Rate@3"),("hr5","Hit Rate@5"),("mrr","MRR"),("p3","Precision@3")]:
        diff = a[key] - b[key]
        sign = "+" if diff >= 0 else ""
        print(f"  {label:<18} {_pct(b[key]):>10} {_pct(a[key]):>10} {sign}{_pct(diff):>9}")
    print(f"  {'Avg Latency':<18} {b['avg_latency']:>9.2f}s {a['avg_latency']:>9.2f}s {'':>9}")


def print_exp3(results: dict) -> None:
    b = results["before_uniform_0.3"]
    t = results["after_tuned"]
    print(f"\n{'─'*60}")
    print(f"  실험 3: Score Threshold  n={results['n']}")
    print(f"{'─'*60}")
    print(f"  {'지표':<18} {'Uniform 0.3':>12} {'Per-Coll.':>12} {'개선':>10}")
    print(f"  {'─'*54}")
    for key, label in [("hr3","Hit Rate@3"),("p3","Precision@3"),("avg_docs","Avg Docs")]:
        bv, tv = b[key], t[key]
        if key == "avg_docs":
            print(f"  {label:<18} {bv:>12.1f} {tv:>12.1f} {tv-bv:>+10.1f}")
        else:
            diff = tv - bv
            sign = "+" if diff >= 0 else ""
            print(f"  {label:<18} {_pct(bv):>12} {_pct(tv):>12} {sign}{_pct(diff):>9}")
    print(f"\n  튜닝 Threshold:")
    for c, v in results["tuned_thresholds"].items():
        print(f"    {c:<38} {v:.2f}")


def print_exp4(results: dict) -> None:
    print(f"\n{'─'*60}")
    print(f"  실험 4: HyDE")
    print(f"{'─'*60}")
    for group in [results["colloquial"], results["formal"]]:
        label = group["group"]
        b, a = group["before"], group["after"]
        print(f"\n  [{label} 질문  n={group['n']}]")
        print(f"  {'지표':<12} {'Before':>10} {'After':>10} {'개선':>10}")
        print(f"  {'─'*44}")
        for key, name in [("hr3","HR@3"),("hr5","HR@5"),("mrr","MRR")]:
            diff = a[key] - b[key]
            sign = "+" if diff >= 0 else ""
            print(f"  {name:<12} {_pct(b[key]):>10} {_pct(a[key]):>10} {sign}{_pct(diff):>9}")
        print(f"  {'Latency':<12} {b['avg_latency']:>9.2f}s {a['avg_latency']:>9.2f}s")


def print_exp5(results: dict) -> None:
    b, a = results["before"], results["after"]
    print(f"\n{'─'*60}")
    print(f"  실험 5: Multi-query  n={results['n']}")
    print(f"{'─'*60}")
    print(f"  {'지표':<18} {'Before':>10} {'After':>10} {'개선':>10}")
    print(f"  {'─'*50}")
    for key, label in [("hr3","Hit Rate@3"),("hr5","Hit Rate@5"),("mrr","MRR")]:
        diff = a[key] - b[key]
        sign = "+" if diff >= 0 else ""
        print(f"  {label:<18} {_pct(b[key]):>10} {_pct(a[key]):>10} {sign}{_pct(diff):>9}")
    print(f"  {'Avg Docs':<18} {b['avg_docs']:>10.1f} {a['avg_docs']:>10.1f}")
    print(f"  {'Avg Latency':<18} {b['avg_latency']:>9.2f}s {a['avg_latency']:>9.2f}s")
    print(f"  평균 Recall 향상: +{results['avg_recall_gain']:.1%}")


# ══════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════

async def main() -> None:
    parser = argparse.ArgumentParser(description="RAG A/B 실험 비교")
    parser.add_argument("--exp",  type=int, choices=[1, 3, 4, 5],
                        help="실행할 실험 번호 (1/3/4/5)")
    parser.add_argument("--all",  action="store_true", help="전체 실험 실행")
    parser.add_argument("--sample", type=int, default=0,
                        help="랜덤 샘플 수 (0=전체, 권장: 15~30)")
    parser.add_argument("--mock",  action="store_true",
                        help="RAG 호출 없이 샘플 수치로 실행 (빠른 확인)")
    parser.add_argument("--save-plots", action="store_true",
                        help=f"플롯을 {PLOTS_DIR} 에 저장")
    args = parser.parse_args()

    run_exps = {1, 3, 4, 5} if args.all else ({args.exp} if args.exp else set())
    if not run_exps:
        parser.print_help()
        return

    save_dir = PLOTS_DIR if args.save_plots else None

    # 의존성 초기화
    db = emb = llm = None
    if not args.mock:
        print("🔌 RAG 의존성 초기화 중...")
        from app.core.dependencies import get_vector_db, get_embeddings, get_llm
        db  = get_vector_db()
        emb = get_embeddings()
        llm = get_llm()
        print("✅ 초기화 완료\n")

    all_questions = load_questions(sample=args.sample)
    colloquial_qs, formal_qs = load_colloquial_questions()
    if args.sample > 0:
        colloquial_qs = colloquial_qs[:min(args.sample // 2, len(colloquial_qs))]
        formal_qs     = formal_qs[:min(args.sample // 2, len(formal_qs))]

    print(f"{'='*60}")
    print(f"  RAG A/B experiments  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  experiments: {sorted(run_exps)}  |  samples: {len(all_questions)}  |  mock: {args.mock}")
    print(f"{'='*60}")

    all_results = {}

    if 1 in run_exps:
        print("\nRunning experiment 1: Reranker ...")
        r1 = run_exp1_reranker(all_questions, db, emb, mock=args.mock)
        all_results["exp1"] = r1
        print_exp1(r1)
        plot_exp1(r1, save_dir)

    if 3 in run_exps:
        print("\nRunning experiment 3: Score Threshold ...")
        r3 = run_exp3_threshold(all_questions, db, emb, mock=args.mock)
        all_results["exp3"] = r3
        print_exp3(r3)
        plot_exp3(r3, save_dir)

    if 4 in run_exps:
        print("\nRunning experiment 4: HyDE ...")
        r4 = run_exp4_hyde(colloquial_qs, formal_qs, db, emb, llm, mock=args.mock)
        all_results["exp4"] = r4
        print_exp4(r4)
        plot_exp4(r4, save_dir)

    if 5 in run_exps:
        print("\nRunning experiment 5: Multi-query ...")
        r5 = run_exp5_multiquery(all_questions, db, emb, llm, mock=args.mock)
        all_results["exp5"] = r5
        print_exp5(r5)
        plot_exp5(r5, save_dir)

    # 결과 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"ab_compare_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        # score_distributions는 리스트가 커서 별도 파일 저장
        save_data = {}
        for k, v in all_results.items():
            if k == "exp3" and "score_distributions" in v:
                # 각 컬렉션의 통계만 저장
                v = {**v, "score_distributions": {
                    c: {"mean": round(float(np.mean(s)), 4),
                        "std":  round(float(np.std(s)), 4),
                        "count": len(s)}
                    for c, s in v["score_distributions"].items() if s
                }}
            save_data[k] = v
        json.dump({"timestamp": datetime.now().isoformat(), **save_data},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved results: {out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
