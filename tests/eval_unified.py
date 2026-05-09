"""tests/eval_unified.py — 통합 RAG 하이퍼파라미터 평가 + Streamlit 대시보드

RAG 파이프라인의 각 기술(HyDE, Query Expansion, Reranker, Threshold)을
하이퍼파라미터로 관리하고, 조합별 성능을 측정·시각화합니다.

평가 지표:
  검색: Hit Rate@3/5, MRR, Precision@3
  RAGAS: context_precision, context_recall, faithfulness(환각), answer_relevancy
  응답 시간: retrieval_ms, llm_ms, total_ms

평가 실행:
  python tests/eval_unified.py --run --mock --sample 20
  python tests/eval_unified.py --run --configs baseline,reranker,hyde --sample 30
  python tests/eval_unified.py --run --ragas --sample 20   # RAGAS + 환각 탐지 포함

Streamlit 대시보드:
  streamlit run tests/eval_unified.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

os.environ.setdefault("LANGCHAIN_PROJECT", "A-LAW-eval")

RETRIEVAL_DATASET = ROOT / "tests" / "평가데이터셋" / "eval_dataset_ver2.json"
RESULTS_DIR = ROOT / "results"
DEFAULT_COLLECTIONS = [
    "law_database", "law_statutes", "contracts",
    "special_clauses_illegal", "special_clauses_normal",
]


# ══════════════════════════════════════════════════════════════════════════
# 1. 하이퍼파라미터 Config
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RAGConfig:
    """RAG 파이프라인 하이퍼파라미터 묶음."""
    name: str
    description: str = ""
    # Stage 1 – Query Processing
    use_hyde: bool = False
    use_query_expansion: bool = False
    n_query_variants: int = 2
    # Stage 3 – Retrieval
    k_per_collection: int = 3
    score_threshold: float = 0.3
    # Stage 4 – Post-Retrieval
    use_reranker: bool = False
    rerank_top_n: int = 5
    # Stage 5 – Generation
    llm_model: str = "gpt-4o-mini"

    def to_dict(self) -> dict:
        return asdict(self)

    def feature_summary(self) -> dict:
        return {
            "HyDE":             "✅" if self.use_hyde else "❌",
            "Query Expansion":  "✅" if self.use_query_expansion else "❌",
            "Reranker":         "✅" if self.use_reranker else "❌",
            "k/collection":     self.k_per_collection,
            "threshold":        self.score_threshold,
            "rerank_top_n":     self.rerank_top_n,
            "LLM":              self.llm_model,
        }


PRESETS: dict[str, RAGConfig] = {
    "baseline": RAGConfig(
        "baseline", "기본 Dense 검색 (개선 없음)",
    ),
    "reranker": RAGConfig(
        "reranker", "+ BGE CrossEncoder Reranker",
        use_reranker=True,
    ),
    "query_expansion": RAGConfig(
        "query_expansion", "+ Multi-Query Expansion (×3)",
        use_query_expansion=True,
    ),
    "reranker_qe": RAGConfig(
        "reranker_qe", "Reranker + Query Expansion",
        use_reranker=True, use_query_expansion=True,
    ),
    "hyde": RAGConfig(
        "hyde", "+ HyDE (Hypothetical Document Embeddings)",
        use_hyde=True, use_reranker=True,
    ),
    "full": RAGConfig(
        "full", "HyDE + Reranker, k=5 (Full Pipeline)",
        use_hyde=True, use_reranker=True,
        k_per_collection=5, rerank_top_n=7,
    ),
}


# ══════════════════════════════════════════════════════════════════════════
# 2. 데이터 로딩
# ══════════════════════════════════════════════════════════════════════════

def load_retrieval_cases(path: Path, sample: int = 0) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("questions", data if isinstance(data, list) else [])
    cases = [q for q in items if q.get("question") and q.get("expected_keywords")]
    return cases[:sample] if sample > 0 else cases


# ══════════════════════════════════════════════════════════════════════════
# 3. 검색 메트릭 유틸
# ══════════════════════════════════════════════════════════════════════════

def _is_relevant(doc: Any, keywords: list[str]) -> bool:
    if hasattr(doc, "page_content"):
        content = doc.page_content
    elif isinstance(doc, dict):
        content = doc.get("content") or doc.get("page_content") or ""
    else:
        content = str(getattr(doc, "content", ""))
    return any(str(kw).lower() in content.lower() for kw in keywords if kw)


def compute_retrieval_metrics(rows: list[dict]) -> dict:
    """rows: [{"docs", "keywords", "retrieval_ms", "llm_ms", "total_ms"}, ...]"""
    if not rows:
        return {}

    hr3 = mean(
        1.0 if any(_is_relevant(d, r["keywords"]) for d in r["docs"][:3]) else 0.0
        for r in rows
    )
    hr5 = mean(
        1.0 if any(_is_relevant(d, r["keywords"]) for d in r["docs"][:5]) else 0.0
        for r in rows
    )
    mrr_val = 0.0
    for r in rows:
        for i, doc in enumerate(r["docs"], 1):
            if _is_relevant(doc, r["keywords"]):
                mrr_val += 1.0 / i
                break
    mrr_val /= len(rows)

    p3_scores = []
    for r in rows:
        top3 = r["docs"][:3]
        if top3:
            p3_scores.append(
                sum(1 for d in top3 if _is_relevant(d, r["keywords"])) / len(top3)
            )
    p3 = mean(p3_scores) if p3_scores else 0.0

    return {
        "hit_rate_at_3":   round(hr3, 4),
        "hit_rate_at_5":   round(hr5, 4),
        "mrr":             round(mrr_val, 4),
        "precision_at_3":  round(p3, 4),
        "avg_docs":        round(mean(len(r["docs"]) for r in rows), 2),
        "avg_retrieval_ms": round(mean(r.get("retrieval_ms", 0) for r in rows), 1),
        "avg_llm_ms":      round(mean(r.get("llm_ms", 0) for r in rows), 1),
        "avg_total_ms":    round(mean(r.get("total_ms", 0) for r in rows), 1),
    }


def _row_to_case_summary(r: dict) -> dict:
    return {
        "question":     r["question"],
        "hit_at_3":     any(_is_relevant(d, r["keywords"]) for d in r["docs"][:3]),
        "hit_at_5":     any(_is_relevant(d, r["keywords"]) for d in r["docs"][:5]),
        "answer_snippet": r.get("answer", "")[:120],
        "retrieval_ms": round(r.get("retrieval_ms", 0), 1),
        "llm_ms":       round(r.get("llm_ms", 0), 1),
        "total_ms":     round(r.get("total_ms", 0), 1),
    }


# ══════════════════════════════════════════════════════════════════════════
# 4. RAGAS 평가
# ══════════════════════════════════════════════════════════════════════════

def run_ragas_eval(
    rows: list[dict],
    use_llm_metrics: bool = False,
) -> dict:
    """rows: [{"question", "retrieved_texts", "answer", "reference", "keywords"}, ...]

    context_precision/recall: 키워드 기반 자체 구현
        - RAGAS NonLLM 메트릭은 Levenshtein 정규화 거리를 사용하는데,
          한국어 법률 도메인에서 reference(~150자) vs retrieved chunk(~1000자) 길이 차이로
          항상 0.0이 나오는 문제가 있어 키워드 AP/coverage 방식으로 대체.
    faithfulness/answer_relevancy: RAGAS LLM 메트릭 (LLM+embeddings 명시 전달)
    """
    result: dict = {}

    # ── 자체 키워드 기반 context_precision / context_recall ──────────────
    cp_scores, cr_scores = [], []
    for r in rows:
        kws = [str(k) for k in (r.get("keywords") or []) if k]
        texts = [t for t in (r.get("retrieved_texts") or []) if t.strip()]
        if not kws or not texts:
            continue
        hits, ap_sum = 0, 0.0
        for i, txt in enumerate(texts, 1):
            if any(kw.lower() in txt.lower() for kw in kws):
                hits += 1
                ap_sum += hits / i
        cp_scores.append(ap_sum / hits if hits else 0.0)
        covered = sum(1 for kw in kws if any(kw.lower() in t.lower() for t in texts))
        cr_scores.append(covered / len(kws))

    if cp_scores:
        result["context_precision"] = round(mean(cp_scores), 4)
    if cr_scores:
        result["context_recall"] = round(mean(cr_scores), 4)

    if not use_llm_metrics:
        return result

    # ── LLM 기반 faithfulness / answer_relevancy ─────────────────────────
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness, AnswerRelevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError:
        result["ragas_error"] = "ragas/langchain_openai not installed"
        return result

    valid_rows = [r for r in rows if r.get("answer", "").strip()]
    if not valid_rows:
        return result

    try:
        samples = [
            SingleTurnSample(
                user_input=r["question"],
                retrieved_contexts=r.get("retrieved_texts") or ["(없음)"],
                reference=r.get("reference", ""),
                response=r["answer"],
            )
            for r in valid_rows
        ]
        dataset = EvaluationDataset(samples=samples)
        ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
        ragas_emb = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small")
        )
        faith_metric = Faithfulness(llm=ragas_llm)
        rel_metric   = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)
        llm_result = evaluate(
            dataset,
            metrics=[faith_metric, rel_metric],
            raise_exceptions=False,
        )
        df = llm_result.to_pandas()
        if "faithfulness" in df.columns:
            result["faithfulness"] = round(float(df["faithfulness"].mean(skipna=True)), 4)
        if "answer_relevancy" in df.columns:
            result["answer_relevancy"] = round(
                float(df["answer_relevancy"].mean(skipna=True)), 4
            )
    except Exception as e:
        result["ragas_error"] = str(e)

    return result


# ══════════════════════════════════════════════════════════════════════════
# 5. Mock 실행기
# ══════════════════════════════════════════════════════════════════════════

def _make_mock_docs(keywords: list[str], hit: bool) -> list:
    from langchain_core.documents import Document
    if hit:
        return [
            Document(
                page_content=f"{' '.join(keywords[:3])} 주택임대차보호법 제3조 관련 근거",
                metadata={"collection": "law_statutes", "score": 0.87},
            ),
            Document(
                page_content="임대차 계약 일반 안내 문서",
                metadata={"collection": "contracts", "score": 0.44},
            ),
        ]
    return [
        Document(
            page_content="관련 없는 일반 문서",
            metadata={"collection": "law_database", "score": 0.31},
        ),
    ]


# Config별 성능 오프셋 (mock에서 차이를 시각적으로 보여주기 위함)
_MOCK_OFFSETS: dict[str, dict] = {
    "baseline":        {"hr": 0.00, "mrr": 0.00, "ret_ms": 90,  "llm_ms": 1200},
    "reranker":        {"hr": 0.12, "mrr": 0.14, "ret_ms": 350, "llm_ms": 1200},
    "query_expansion": {"hr": 0.09, "mrr": 0.08, "ret_ms": 900, "llm_ms": 1200},
    "reranker_qe":     {"hr": 0.17, "mrr": 0.18, "ret_ms": 1100,"llm_ms": 1200},
    "hyde":            {"hr": 0.19, "mrr": 0.22, "ret_ms": 1300,"llm_ms": 1200},
    "full":            {"hr": 0.23, "mrr": 0.25, "ret_ms": 1500,"llm_ms": 1200},
}


def run_mock(cases: list[dict], cfg: RAGConfig, ragas_enabled: bool) -> dict:
    import random
    rng = random.Random(42)
    off = _MOCK_OFFSETS.get(cfg.name, {"hr": 0, "mrr": 0, "ret_ms": 200, "llm_ms": 1200})
    base_hr = 0.60

    rows, ragas_rows = [], []
    for case in cases:
        kws = case.get("expected_keywords", [])
        hit = rng.random() < (base_hr + off["hr"])
        docs = _make_mock_docs(kws, hit)
        ret_ms = rng.uniform(off["ret_ms"] * 0.8, off["ret_ms"] * 1.2)
        llm_ms = rng.uniform(off["llm_ms"] * 0.7, off["llm_ms"] * 1.3)
        answer = f"[mock] {case['question'][:40]}에 대한 답변입니다."
        rows.append({
            "question":    case["question"],
            "docs":        docs,
            "keywords":    kws,
            "retrieval_ms": ret_ms,
            "llm_ms":      llm_ms,
            "total_ms":    ret_ms + llm_ms,
            "answer":      answer,
        })
        if ragas_enabled:
            ragas_rows.append({
                "question":        case["question"],
                "retrieved_texts": [d.page_content for d in docs],
                "answer":          answer,
                "reference":       case.get("expected_answer") or " ".join(case.get("expected_keywords", [])),
                "keywords":        case.get("expected_keywords", []),
            })

    metrics = compute_retrieval_metrics(rows)

    if ragas_enabled and ragas_rows:
        ragas_metrics = run_ragas_eval(ragas_rows, use_llm_metrics=False)
        ragas_metrics.setdefault("faithfulness", round(0.73 + off["hr"] * 0.5, 4))
        ragas_metrics.setdefault("answer_relevancy", round(0.70 + off["hr"] * 0.6, 4))
        metrics.update(ragas_metrics)

    return {
        "metrics": metrics,
        "cases":   [_row_to_case_summary(r) for r in rows],
    }


# ══════════════════════════════════════════════════════════════════════════
# 6. 실제 RAG 실행기 (async)
# ══════════════════════════════════════════════════════════════════════════

async def run_real(cases: list[dict], cfg: RAGConfig, ragas_enabled: bool) -> dict:
    from langchain_openai import ChatOpenAI
    from app.core.dependencies import get_vector_db, get_embeddings, get_llm
    from app.rag.retriever.multi_retriever import async_search_multi_index, _deduplicate
    from app.rag.retriever.reranker import get_reranker
    from app.rag.retriever.query_expansion import (
        async_expand_query_hyde,
        async_expand_query_multi,
    )
    from app.rag.chain.chain import build_context
    from app.rag.chain.prompts import CHAT_PROMPT

    db  = get_vector_db()
    emb = get_embeddings()
    llm: ChatOpenAI = get_llm()
    if cfg.llm_model != "gpt-4o-mini":
        llm = ChatOpenAI(model=cfg.llm_model, temperature=0)
    reranker = get_reranker() if cfg.use_reranker else None

    rows, ragas_rows = [], []

    for case in cases:
        kws = case.get("expected_keywords", [])
        t_start = time.perf_counter()

        # Stage 1: Query Processing
        t_ret = time.perf_counter()
        if cfg.use_hyde:
            hyde_text = await async_expand_query_hyde(case["question"], llm)
            queries   = [hyde_text]
        elif cfg.use_query_expansion:
            queries = await async_expand_query_multi(
                case["question"], llm, n=cfg.n_query_variants
            )
        else:
            queries = [case["question"]]

        # Stage 3: Retrieval (병렬)
        all_docs: list = []
        for q in queries:
            docs = await async_search_multi_index(
                db, emb, q,
                collections=DEFAULT_COLLECTIONS,
                k_per_collection=cfg.k_per_collection,
                score_threshold=cfg.score_threshold,
            )
            all_docs.extend(docs)
        all_docs = _deduplicate(all_docs)

        # Stage 4: Reranker (원문 질문 기준으로 재정렬)
        if reranker and all_docs:
            all_docs = await reranker.async_rerank(
                case["question"], all_docs, top_n=cfg.rerank_top_n
            )
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        # Stage 5: Generation
        t_llm = time.perf_counter()
        context = build_context(all_docs, max_length=2000)
        prompt_msgs = CHAT_PROMPT.format_messages(
            context=context, history=[], question=case["question"]
        )
        llm_response = await llm.ainvoke(prompt_msgs)
        answer   = llm_response.content
        llm_ms   = (time.perf_counter() - t_llm) * 1000
        total_ms = (time.perf_counter() - t_start) * 1000

        rows.append({
            "question":    case["question"],
            "docs":        all_docs,
            "keywords":    kws,
            "retrieval_ms": retrieval_ms,
            "llm_ms":      llm_ms,
            "total_ms":    total_ms,
            "answer":      answer,
        })

        if ragas_enabled:
            ragas_rows.append({
                "question":        case["question"],
                "retrieved_texts": [t for t in (d.page_content for d in all_docs[:5]) if t.strip()],
                "answer":          answer,
                "reference":       case.get("expected_answer") or " ".join(case.get("expected_keywords", [])),
                "keywords":        case.get("expected_keywords", []),
            })

    metrics = compute_retrieval_metrics(rows)
    if ragas_enabled and ragas_rows:
        metrics.update(run_ragas_eval(ragas_rows, use_llm_metrics=True))

    return {
        "metrics": metrics,
        "cases":   [_row_to_case_summary(r) for r in rows],
    }


# ══════════════════════════════════════════════════════════════════════════
# 7. 결과 저장/로드
# ══════════════════════════════════════════════════════════════════════════

def save_results(data: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"unified_eval_{ts}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_result_files() -> list[Path]:
    return sorted(RESULTS_DIR.glob("unified_eval_*.json"), reverse=True)


# ══════════════════════════════════════════════════════════════════════════
# 8. CLI 메인
# ══════════════════════════════════════════════════════════════════════════

async def cli_main() -> None:
    parser = argparse.ArgumentParser(description="통합 RAG 하이퍼파라미터 평가")
    parser.add_argument("--run",     action="store_true", help="평가 실행")
    parser.add_argument("--configs", default="",
                        help="comma-separated config 이름 (기본: 전체)")
    parser.add_argument("--sample",  type=int, default=20, help="케이스 수")
    parser.add_argument("--mock",    action="store_true", help="mock 모드")
    parser.add_argument("--ragas",   action="store_true",
                        help="RAGAS + 환각 탐지 포함 (LLM 필요)")
    args = parser.parse_args()

    if not args.run:
        parser.print_help()
        return

    selected = (
        {k: PRESETS[k] for k in args.configs.split(",") if k in PRESETS}
        if args.configs else PRESETS
    )

    if not RETRIEVAL_DATASET.exists():
        print(f"[ERROR] 데이터셋 없음: {RETRIEVAL_DATASET}")
        return

    cases = load_retrieval_cases(RETRIEVAL_DATASET, args.sample)
    print(f"\n{'='*64}")
    print(f"  통합 RAG 평가  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Configs : {list(selected.keys())}")
    print(f"  Cases   : {len(cases)}  |  Mock: {args.mock}  |  RAGAS: {args.ragas}")
    print(f"{'='*64}\n")

    payload: dict[str, Any] = {
        "metadata": {
            "created_at":       datetime.now().isoformat(),
            "mock":             args.mock,
            "ragas_enabled":    args.ragas,
            "sample_count":     len(cases),
            "configs_tested":   list(selected.keys()),
            "langsmith_project": os.getenv("LANGCHAIN_PROJECT", "A-LAW-eval"),
        },
        "configs": {name: cfg.to_dict() for name, cfg in selected.items()},
        "results": {},
    }

    for name, cfg in selected.items():
        print(f"  ▶ [{name}] {cfg.description} ...")
        t0 = time.perf_counter()
        result = run_mock(cases, cfg, args.ragas) if args.mock \
            else await run_real(cases, cfg, args.ragas)
        elapsed = time.perf_counter() - t0
        payload["results"][name] = result

        m = result["metrics"]
        print(
            f"     HR@3={m.get('hit_rate_at_3',0):.3f}  "
            f"HR@5={m.get('hit_rate_at_5',0):.3f}  "
            f"MRR={m.get('mrr',0):.3f}  "
            f"P@3={m.get('precision_at_3',0):.3f}"
            + (f"  faith={m.get('faithfulness','-')}" if args.ragas else "")
            + f"  [{elapsed:.1f}s]\n"
        )

    path = save_results(payload)
    print(f"✅ 결과 저장: {path}")
    print(f"📊 대시보드: streamlit run tests/eval_unified.py\n")

    ls_key = os.getenv("LANGCHAIN_API_KEY")
    if ls_key:
        proj = os.getenv("LANGCHAIN_PROJECT", "A-LAW-eval")
        print(f"🔗 LangSmith: https://smith.langchain.com (project: {proj})\n")


# ══════════════════════════════════════════════════════════════════════════
# 9. Streamlit 대시보드
# ══════════════════════════════════════════════════════════════════════════

def run_dashboard() -> None:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(
        page_title="A-LAW RAG 평가 대시보드",
        page_icon="⚖️",
        layout="wide",
    )
    st.title("⚖️ A-LAW RAG 하이퍼파라미터 평가 대시보드")

    COLORS = px.colors.qualitative.Set2

    METRIC_LABELS: dict[str, str] = {
        "hit_rate_at_3":    "Hit Rate@3",
        "hit_rate_at_5":    "Hit Rate@5",
        "mrr":              "MRR",
        "precision_at_3":   "Precision@3",
        "context_precision": "Context Precision",
        "context_recall":   "Context Recall",
        "faithfulness":     "Faithfulness (반환각↓)",
        "answer_relevancy": "Answer Relevancy",
        "avg_retrieval_ms": "검색 시간 (ms)",
        "avg_llm_ms":       "LLM 시간 (ms)",
        "avg_total_ms":     "전체 시간 (ms)",
    }
    RETRIEVAL_M = ["hit_rate_at_3", "hit_rate_at_5", "mrr", "precision_at_3"]
    RAGAS_M     = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    LATENCY_M   = ["avg_retrieval_ms", "avg_llm_ms", "avg_total_ms"]

    # ── 사이드바 ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📁 결과 파일")
        result_files = list_result_files()
        if not result_files:
            st.warning(
                "results/ 에 평가 결과가 없습니다.\n\n"
                "```\npython tests/eval_unified.py --run --mock\n```"
            )
            st.stop()

        file_opts = {f.name: f for f in result_files}
        sel_file  = st.selectbox("결과 파일 선택", list(file_opts.keys()))
        data      = json.loads(file_opts[sel_file].read_text(encoding="utf-8"))

        meta = data.get("metadata", {})
        st.caption(f"생성: {meta.get('created_at','')[:19]}")
        st.caption(
            f"케이스: {meta.get('sample_count')}개  "
            f"| Mock: {meta.get('mock')}  "
            f"| RAGAS: {meta.get('ragas_enabled')}"
        )

        all_cfgs = list(data["results"].keys())
        sel_cfgs = st.multiselect("비교할 Config", all_cfgs, default=all_cfgs)
        if not sel_cfgs:
            st.warning("Config를 하나 이상 선택하세요.")
            st.stop()

        st.divider()
        ls_project = meta.get("langsmith_project", "")
        if ls_project and os.getenv("LANGCHAIN_API_KEY"):
            st.markdown(f"🔗 [LangSmith 트레이스](https://smith.langchain.com)")
        st.caption(f"LangSmith project: {ls_project}")

        st.divider()
        st.markdown("**새 평가 실행**")
        st.code(
            "python tests/eval_unified.py\n"
            "  --run --mock --sample 20\n"
            "  --configs baseline,reranker,hyde",
            language="bash",
        )

    results     = {k: data["results"][k] for k in sel_cfgs if k in data["results"]}
    configs_raw = data.get("configs", {})

    def gm(cfg_name: str, metric: str) -> float | None:
        return results.get(cfg_name, {}).get("metrics", {}).get(metric)

    # ── 탭 ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 종합 비교",
        "🔍 검색 품질",
        "🧪 RAGAS / 환각",
        "⏱️ 응답 시간",
        "⚙️ Config 파라미터",
    ])

    # ─── Tab 1: 종합 비교 ────────────────────────────────────────────────
    with tab1:
        st.subheader("Config × 메트릭 종합 비교표")

        summary_rows = []
        for cfg in sel_cfgs:
            row: dict[str, Any] = {"Config": cfg}
            for m in RETRIEVAL_M + RAGAS_M:
                v = gm(cfg, m)
                row[METRIC_LABELS[m]] = round(v, 4) if v is not None else None
            summary_rows.append(row)

        df_sum = pd.DataFrame(summary_rows).set_index("Config")
        num_cols = [c for c in df_sum.columns if df_sum[c].notna().any()]

        st.dataframe(
            df_sum.style
                  .highlight_max(axis=0, subset=num_cols, color="#c3e6cb")
                  .highlight_min(axis=0, subset=num_cols, color="#f5c6cb")
                  .format({c: "{:.4f}" for c in num_cols}, na_rep="-"),
            use_container_width=True,
        )

        # 레이더 차트 (검색 품질)
        avail_rm = [m for m in RETRIEVAL_M if any(gm(c, m) is not None for c in sel_cfgs)]
        if len(sel_cfgs) >= 2 and len(avail_rm) >= 3:
            st.subheader("레이더 차트 — 검색 품질")
            fig_radar = go.Figure()
            for i, cfg in enumerate(sel_cfgs):
                vals   = [gm(cfg, m) or 0.0 for m in avail_rm]
                labels = [METRIC_LABELS[m] for m in avail_rm]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=labels + [labels[0]],
                    fill="toself",
                    name=cfg,
                    line_color=COLORS[i % len(COLORS)],
                    opacity=0.65,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                height=420, showlegend=True,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # ─── Tab 2: 검색 품질 ────────────────────────────────────────────────
    with tab2:
        st.subheader("검색 품질 지표")

        bar_data = [
            {"지표": METRIC_LABELS[m], "Config": cfg, "Score": gm(cfg, m)}
            for m in RETRIEVAL_M for cfg in sel_cfgs
            if gm(cfg, m) is not None
        ]
        if bar_data:
            fig_ret = px.bar(
                pd.DataFrame(bar_data),
                x="지표", y="Score", color="Config",
                barmode="group", range_y=[0, 1.1],
                text_auto=".3f",
                color_discrete_sequence=COLORS,
            )
            fig_ret.update_traces(textposition="outside")
            fig_ret.update_layout(height=420)
            st.plotly_chart(fig_ret, use_container_width=True)

        # baseline 대비 개선폭
        if "baseline" in sel_cfgs and len(sel_cfgs) > 1:
            st.subheader("baseline 대비 개선폭")
            others = [c for c in sel_cfgs if c != "baseline"]
            cols   = st.columns(len(others))
            for col, cfg in zip(cols, others):
                with col:
                    st.markdown(f"**{cfg}**")
                    for m in RETRIEVAL_M:
                        base = gm("baseline", m) or 0.0
                        curr = gm(cfg, m) or 0.0
                        st.metric(METRIC_LABELS[m], f"{curr:.3f}", f"{curr-base:+.3f}")

        # 질문별 Hit@3 히트맵
        st.subheader("질문별 Hit@3 히트맵")
        case_lists = {c: results[c].get("cases", []) for c in sel_cfgs}
        n_q = max((len(v) for v in case_lists.values()), default=0)
        if n_q > 0:
            q_labels = []
            for i in range(n_q):
                q = next(
                    (case_lists[c][i]["question"][:28] for c in sel_cfgs if i < len(case_lists[c])),
                    f"Q{i+1}",
                )
                q_labels.append(f"Q{i+1}: {q}")

            heat: dict[str, list] = {}
            for cfg in sel_cfgs:
                lst = case_lists[cfg]
                heat[cfg] = [
                    1.0 if (i < len(lst) and lst[i].get("hit_at_3")) else 0.0
                    for i in range(n_q)
                ]

            df_heat = pd.DataFrame(heat, index=q_labels[:20]).T
            fig_heat = px.imshow(
                df_heat,
                color_continuous_scale=["#ffcccc", "#ccffcc"],
                aspect="auto", zmin=0, zmax=1,
                labels={"color": "Hit@3"},
            )
            fig_heat.update_layout(height=max(180, len(sel_cfgs) * 45 + 80))
            st.plotly_chart(fig_heat, use_container_width=True)

    # ─── Tab 3: RAGAS / 환각 ─────────────────────────────────────────────
    with tab3:
        has_ragas = any(gm(c, "context_precision") is not None for c in sel_cfgs)

        if not has_ragas:
            st.info(
                "RAGAS 지표가 없습니다. `--ragas` 옵션으로 재실행하세요.\n\n"
                "```\npython tests/eval_unified.py --run --ragas --sample 20\n```"
            )
        else:
            st.subheader("RAGAS 지표 비교")
            ragas_data = [
                {"지표": METRIC_LABELS[m], "Config": cfg, "Score": gm(cfg, m)}
                for m in RAGAS_M for cfg in sel_cfgs
                if gm(cfg, m) is not None
            ]
            if ragas_data:
                fig_ragas = px.bar(
                    pd.DataFrame(ragas_data),
                    x="지표", y="Score", color="Config",
                    barmode="group", range_y=[0, 1.1],
                    text_auto=".3f",
                    color_discrete_sequence=COLORS,
                )
                fig_ragas.update_traces(textposition="outside")
                fig_ragas.update_layout(height=420)
                st.plotly_chart(fig_ragas, use_container_width=True)

            # 환각 탐지 강조 카드
            faith_vals = {c: gm(c, "faithfulness") for c in sel_cfgs}
            if any(v is not None for v in faith_vals.values()):
                st.subheader("🔍 환각 탐지 (Faithfulness)")
                st.caption("1.0 = 검색 근거 기반 답변 (환각 없음)  |  0.0 = 환각 가능성 높음")
                cols = st.columns(len(sel_cfgs))
                for col, cfg in zip(cols, sel_cfgs):
                    v = faith_vals.get(cfg)
                    if v is not None:
                        icon = "🟢" if v >= 0.8 else ("🟡" if v >= 0.6 else "🔴")
                        col.metric(f"{icon} {cfg}", f"{v:.3f}")

            # RAGAS baseline 대비
            if "baseline" in sel_cfgs:
                st.subheader("baseline 대비 RAGAS 개선폭")
                others = [c for c in sel_cfgs if c != "baseline"]
                if others:
                    cols = st.columns(len(others))
                    for col, cfg in zip(cols, others):
                        with col:
                            st.markdown(f"**{cfg}**")
                            for m in RAGAS_M:
                                base = gm("baseline", m)
                                curr = gm(cfg, m)
                                if curr is not None and base is not None:
                                    st.metric(METRIC_LABELS[m], f"{curr:.3f}", f"{curr-base:+.3f}")

    # ─── Tab 4: 응답 시간 ─────────────────────────────────────────────────
    with tab4:
        st.subheader("응답 시간 분석")

        lat_rows = []
        for cfg in sel_cfgs:
            r_ms  = gm(cfg, "avg_retrieval_ms") or 0
            l_ms  = gm(cfg, "avg_llm_ms") or 0
            lat_rows.append({"Config": cfg, "검색 (ms)": r_ms, "LLM (ms)": l_ms})
        df_lat = pd.DataFrame(lat_rows)

        # 스택 막대
        fig_lat = go.Figure()
        for col_name, color in [("검색 (ms)", "#6c8ebf"), ("LLM (ms)", "#82b366")]:
            fig_lat.add_trace(go.Bar(
                name=col_name,
                x=df_lat["Config"],
                y=df_lat[col_name],
                marker_color=color,
                text=df_lat[col_name].apply(lambda v: f"{v:.0f}"),
                textposition="inside",
            ))
        fig_lat.update_layout(
            barmode="stack", height=380,
            yaxis_title="평균 응답 시간 (ms)",
        )
        st.plotly_chart(fig_lat, use_container_width=True)

        # 품질 vs 속도 산점도
        st.subheader("검색 품질 vs 응답 시간 트레이드오프")
        sc_data = [
            {
                "Config":       cfg,
                "Hit Rate@3":   gm(cfg, "hit_rate_at_3") or 0,
                "Total (ms)":   gm(cfg, "avg_total_ms") or 0,
            }
            for cfg in sel_cfgs
        ]
        df_sc = pd.DataFrame(sc_data)
        fig_sc = px.scatter(
            df_sc, x="Total (ms)", y="Hit Rate@3",
            text="Config", color="Config",
            color_discrete_sequence=COLORS,
        )
        fig_sc.update_traces(
            textposition="top center",
            marker=dict(size=14),
        )
        fig_sc.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig_sc, use_container_width=True)

        # 수치 테이블
        st.dataframe(df_lat.set_index("Config"), use_container_width=True)

    # ─── Tab 5: Config 파라미터 ──────────────────────────────────────────
    with tab5:
        st.subheader("Config 파라미터 비교")

        param_rows = []
        for cfg_name in sel_cfgs:
            raw = configs_raw.get(cfg_name, {})
            try:
                cfg_obj = RAGConfig(
                    **{k: v for k, v in raw.items() if k in RAGConfig.__dataclass_fields__}
                )
            except Exception:
                cfg_obj = PRESETS.get(cfg_name, RAGConfig(cfg_name))
            row = {"Config": cfg_name, "설명": cfg_obj.description}
            row.update(cfg_obj.feature_summary())
            param_rows.append(row)

        st.dataframe(
            pd.DataFrame(param_rows).set_index("Config"),
            use_container_width=True,
        )

        st.divider()
        st.subheader("새 Config 추가 방법")
        st.code(
            '# eval_unified.py의 PRESETS 딕셔너리에 추가\n'
            'PRESETS["my_config"] = RAGConfig(\n'
            '    name="my_config",\n'
            '    description="커스텀 설정",\n'
            '    use_hyde=True,\n'
            '    use_reranker=True,\n'
            '    k_per_collection=5,\n'
            '    score_threshold=0.35,\n'
            '    llm_model="gpt-4o",\n'
            ')\n\n'
            '# 실행\n'
            'python tests/eval_unified.py --run \\\n'
            '  --configs my_config,baseline --sample 30',
            language="python",
        )

        st.divider()
        st.subheader("전체 파이프라인 단계별 기술 옵션")
        pipeline_df = pd.DataFrame([
            {"단계": "Query Processing", "기술": "HyDE",             "파라미터": "use_hyde=True",             "효과": "쿼리→가상 답변 임베딩 → Precision↑"},
            {"단계": "Query Processing", "기술": "Query Expansion",   "파라미터": "use_query_expansion=True", "효과": "쿼리 3개 변형 → Recall↑"},
            {"단계": "Retrieval",        "기술": "k 조정",            "파라미터": "k_per_collection=5",        "효과": "더 많은 후보 → Recall↑, 속도↓"},
            {"단계": "Retrieval",        "기술": "Threshold 조정",    "파라미터": "score_threshold=0.4",       "효과": "노이즈 감소 → Precision↑"},
            {"단계": "Post-Retrieval",   "기술": "BGE Reranker",      "파라미터": "use_reranker=True",         "효과": "관련성 재정렬 → Precision↑↑"},
            {"단계": "Post-Retrieval",   "기술": "CRAG",              "파라미터": "(자동 적용)",               "효과": "점수 낮으면 쿼리 재작성 → 실패 복구"},
            {"단계": "Generation",       "기술": "LLM 교체",          "파라미터": "llm_model='gpt-4o'",        "효과": "응답 품질↑, 비용↑"},
        ])
        st.dataframe(pipeline_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════

def _is_streamlit() -> bool:
    if "streamlit" in sys.modules:
        return True
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _is_streamlit():
    run_dashboard()
elif __name__ == "__main__":
    asyncio.run(cli_main())
