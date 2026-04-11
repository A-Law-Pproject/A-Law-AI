"""Risk and retrieval evaluation runner for A-LAW RAG.

This script keeps two evaluations separate:
- retrieval: checks whether expected legal keywords appear in Top-K evidence.
- risk: checks whether risk level, score band, legal grounds, and rationale match.

Examples:
    .venv\\Scripts\\python.exe tests\\eval_risk_retrieval.py --mode all --mock --sample 3
    .venv\\Scripts\\python.exe tests\\eval_risk_retrieval.py --mode retrieval --save
    .venv\\Scripts\\python.exe tests\\eval_risk_retrieval.py --mode risk --save
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_RETRIEVAL_DATASET = ROOT / "tests" / "eval_dataset_ver2.json"
DEFAULT_RISK_DATASET = ROOT / "tests" / "risk_eval_dataset.json"
RESULTS_DIR = ROOT / "results"

DEFAULT_COLLECTIONS = [
    "law_database",
    "law_statutes",
    "contracts",
    "special_clauses_illegal",
    "special_clauses_normal",
]

LEVEL_TO_SCORE = {
    "안전": 20,
    "주의": 55,
    "위험": 85,
}


@dataclass
class EvalDoc:
    content: str
    metadata: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _take_sample(items: list[dict[str, Any]], sample: int) -> list[dict[str, Any]]:
    if sample <= 0:
        return items
    return items[:sample]


def _doc_content(doc: Any) -> str:
    if hasattr(doc, "page_content"):
        return str(doc.page_content)
    if isinstance(doc, dict):
        return str(doc.get("content") or doc.get("page_content") or "")
    return str(getattr(doc, "content", ""))


def _doc_metadata(doc: Any) -> dict[str, Any]:
    if hasattr(doc, "metadata"):
        return dict(doc.metadata or {})
    if isinstance(doc, dict):
        return dict(doc.get("metadata") or {})
    return dict(getattr(doc, "metadata", {}) or {})


def _keyword_hit(text: str, keywords: list[str]) -> bool:
    text_l = text.lower()
    return any(str(keyword).lower() in text_l for keyword in keywords if keyword)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple | set):
        return [str(item) for item in value if str(item).strip()]
    if str(value).strip():
        return [str(value)]
    return []


def _norm(value: Any) -> str:
    return "".join(str(value or "").lower().split())


def _metadata_match(metadata: dict[str, Any], expected_values: list[str], metadata_keys: list[str]) -> bool:
    if not expected_values:
        return False
    expected = {_norm(value) for value in expected_values if _norm(value)}
    actual = {_norm(metadata.get(key)) for key in metadata_keys if _norm(metadata.get(key))}
    if not expected or not actual:
        return False
    return bool(expected & actual)


def _text_or_metadata_hit(doc: Any, expected_values: list[str], metadata_keys: list[str] | None = None) -> bool:
    content = _doc_content(doc)
    metadata = _doc_metadata(doc)
    selected_meta = metadata
    if metadata_keys:
        selected_meta = {key: metadata.get(key) for key in metadata_keys}
    meta_text = " ".join(str(v) for v in selected_meta.values())
    combined = f"{content}\n{meta_text}"
    return _keyword_hit(combined, expected_values)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _semantic_reference(item: dict[str, Any]) -> str:
    parts = [
        item.get("expected_answer", ""),
        item.get("question", ""),
        " ".join(_as_list(item.get("expected_keywords"))),
        " ".join(_as_list(item.get("relevant_law"))),
    ]
    return "\n".join(part for part in parts if part)


def _semantic_scores_for_docs(
    item: dict[str, Any],
    docs: list[Any],
    embeddings: Any,
    cache: dict[str, list[float]],
) -> list[float]:
    reference = _semantic_reference(item)
    if not reference.strip():
        return [0.0 for _ in docs]

    def embed(text: str) -> list[float]:
        key = text[:4000]
        if key not in cache:
            cache[key] = embeddings.embed_query(key)
        return cache[key]

    ref_vec = embed(reference)
    scores = []
    for doc in docs:
        doc_text = _doc_content(doc)[:4000]
        if not doc_text.strip():
            scores.append(0.0)
            continue
        scores.append(_cosine_similarity(ref_vec, embed(doc_text)))
    return scores


def _doc_match_reasons(doc: Any, row: dict[str, Any], index: int) -> list[str]:
    metadata = _doc_metadata(doc)
    reasons: list[str] = []

    if _metadata_match(metadata, row.get("expected_chunk_ids", []), ["chunk_id", "id", "vector_id"]):
        reasons.append("chunk_id")
    if _metadata_match(metadata, row.get("expected_doc_ids", []), ["doc_id", "source_id"]):
        reasons.append("doc_id")
    if _metadata_match(metadata, row.get("expected_law_ids", []), ["law_id", "law_name", "title"]):
        reasons.append("law_id")
    if _metadata_match(metadata, row.get("expected_articles", []), ["article", "article_no"]):
        reasons.append("article")

    if _text_or_metadata_hit(doc, row.get("expected_keywords", [])):
        reasons.append("keyword")
    if _text_or_metadata_hit(doc, row.get("expected_laws", []), ["law_id", "law_name", "title", "article", "article_no", "source"]):
        reasons.append("law_text")

    semantic_scores = row.get("semantic_scores") or []
    if index < len(semantic_scores) and semantic_scores[index] >= row.get("semantic_threshold", 1.0):
        reasons.append("semantic")

    return reasons


def _is_relevant(doc: Any, row: dict[str, Any], index: int) -> bool:
    return bool(_doc_match_reasons(doc, row, index))


def _hit_rate_at_k(rows: list[dict[str, Any]], k: int) -> float:
    if not rows:
        return 0.0
    hits = 0
    for row in rows:
        docs = row["docs"][:k]
        if any(_is_relevant(doc, row, index) for index, doc in enumerate(docs)):
            hits += 1
    return hits / len(rows)


def _precision_at_k(rows: list[dict[str, Any]], k: int) -> float:
    if not rows:
        return 0.0
    scores = []
    for row in rows:
        docs = row["docs"][:k]
        if not docs:
            scores.append(0.0)
            continue
        relevant = sum(
            1
            for index, doc in enumerate(docs)
            if _is_relevant(doc, row, index)
        )
        scores.append(relevant / len(docs))
    return mean(scores)


def _mrr(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    reciprocal_ranks = []
    for row in rows:
        rank = 0
        for zero_index, doc in enumerate(row["docs"]):
            if _is_relevant(doc, row, zero_index):
                rank = zero_index + 1
                break
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return mean(reciprocal_ranks)


def _field_hit_rate_at_k(rows: list[dict[str, Any]], reason: str, k: int) -> float | None:
    expected_field_by_reason = {
        "chunk_id": "expected_chunk_ids",
        "doc_id": "expected_doc_ids",
        "law_id": "expected_law_ids",
        "article": "expected_articles",
    }
    if reason == "semantic":
        applicable = [row for row in rows if row.get("semantic_scores")]
    else:
        expected_field = expected_field_by_reason.get(reason)
        applicable = [row for row in rows if row.get(expected_field, [])]

    if not applicable:
        return None
    hits = 0
    for row in applicable:
        docs = row["docs"][:k]
        if any(reason in _doc_match_reasons(doc, row, index) for index, doc in enumerate(docs)):
            hits += 1
    return hits / len(applicable)


def _serialize_doc(doc: Any, row: dict[str, Any] | None = None, index: int = 0) -> dict[str, Any]:
    metadata = _doc_metadata(doc)
    serialized = {
        "content": _doc_content(doc)[:500],
        "collection": metadata.get("collection"),
        "score": metadata.get("score"),
        "rerank_score": metadata.get("rerank_score"),
        "doc_id": metadata.get("doc_id"),
        "chunk_id": metadata.get("chunk_id"),
        "law_id": metadata.get("law_id"),
        "law_name": metadata.get("law_name"),
        "article": metadata.get("article"),
        "article_no": metadata.get("article_no"),
        "title": metadata.get("title"),
        "category": metadata.get("category"),
    }
    if row is not None:
        semantic_scores = row.get("semantic_scores") or []
        serialized["match_reasons"] = _doc_match_reasons(doc, row, index)
        if index < len(semantic_scores):
            serialized["semantic_score"] = round(semantic_scores[index], 4)
    return serialized


def load_retrieval_cases(path: Path, sample: int) -> list[dict[str, Any]]:
    data = _load_json(path)
    questions = data.get("questions", data if isinstance(data, list) else [])
    cases = [
        item
        for item in questions
        if item.get("question") and len(item.get("expected_keywords", [])) > 0
    ]
    return _take_sample(cases, sample)


def load_risk_cases(path: Path, sample: int) -> list[dict[str, Any]]:
    data = _load_json(path)
    cases = data.get("cases", data if isinstance(data, list) else [])
    return _take_sample(cases, sample)


def _mock_retrieval_docs(item: dict[str, Any], index: int) -> list[EvalDoc]:
    expected = " ".join(item.get("expected_keywords", [])[:3])
    law = " ".join(item.get("relevant_law", [])[:2])
    irrelevant = EvalDoc("계약 일반 안내 문서", {"collection": "mock", "score": 0.31})
    relevant = EvalDoc(
        f"{expected} {law} 관련 법률 근거 문서",
        {"collection": "mock_law", "score": 0.87},
    )

    if index % 5 == 0:
        return [irrelevant]
    if index % 3 == 0:
        return [irrelevant, relevant]
    return [relevant, irrelevant]


async def evaluate_retrieval(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_retrieval_cases(Path(args.retrieval_dataset), args.sample)
    rows: list[dict[str, Any]] = []
    semantic_cache: dict[str, list[float]] = {}

    if args.mock:
        for index, item in enumerate(cases):
            rows.append(
                {
                    "id": item.get("id"),
                    "question": item["question"],
                    "expected_keywords": item.get("expected_keywords", []),
                    "expected_laws": item.get("relevant_law", []),
                    "expected_doc_ids": _as_list(item.get("expected_doc_ids")),
                    "expected_chunk_ids": _as_list(item.get("expected_chunk_ids")),
                    "expected_law_ids": _as_list(item.get("expected_law_ids")),
                    "expected_articles": _as_list(item.get("expected_articles")),
                    "docs": _mock_retrieval_docs(item, index),
                    "semantic_scores": [],
                    "semantic_threshold": args.semantic_threshold,
                    "latency_ms": 1.0,
                }
            )
    else:
        from app.core.dependencies import get_embeddings, get_vector_db
        from app.rag.retriever.multi_retriever import search_multi_index
        from app.rag.retriever.reranker import get_reranker

        db = get_vector_db()
        embeddings = get_embeddings()
        reranker = get_reranker() if args.rerank else None
        collections = [item.strip() for item in args.collections.split(",") if item.strip()]

        for item in cases:
            start = time.perf_counter()
            docs = search_multi_index(
                db,
                embeddings,
                item["question"],
                collections=collections,
                k_per_collection=args.k_per_collection,
                reranker=reranker,
                rerank_top_n=args.rerank_top_n,
            )
            semantic_scores = (
                _semantic_scores_for_docs(item, docs, embeddings, semantic_cache)
                if args.semantic_judge
                else []
            )
            rows.append(
                {
                    "id": item.get("id"),
                    "question": item["question"],
                    "expected_keywords": item.get("expected_keywords", []),
                    "expected_laws": item.get("relevant_law", []),
                    "expected_doc_ids": _as_list(item.get("expected_doc_ids")),
                    "expected_chunk_ids": _as_list(item.get("expected_chunk_ids")),
                    "expected_law_ids": _as_list(item.get("expected_law_ids")),
                    "expected_articles": _as_list(item.get("expected_articles")),
                    "docs": docs,
                    "semantic_scores": semantic_scores,
                    "semantic_threshold": args.semantic_threshold,
                    "latency_ms": (time.perf_counter() - start) * 1000,
                }
            )

    metrics = {
        "case_count": len(rows),
        "hit_rate_at_3": round(_hit_rate_at_k(rows, 3), 4),
        "hit_rate_at_5": round(_hit_rate_at_k(rows, 5), 4),
        "mrr": round(_mrr(rows), 4),
        "precision_at_3": round(_precision_at_k(rows, 3), 4),
        "avg_docs": round(mean([len(row["docs"]) for row in rows]) if rows else 0.0, 4),
        "avg_latency_ms": round(mean([row["latency_ms"] for row in rows]) if rows else 0.0, 2),
    }
    field_metrics = {
        "chunk_id_hit_rate_at_5": _field_hit_rate_at_k(rows, "chunk_id", 5),
        "doc_id_hit_rate_at_5": _field_hit_rate_at_k(rows, "doc_id", 5),
        "law_id_hit_rate_at_5": _field_hit_rate_at_k(rows, "law_id", 5),
        "article_hit_rate_at_5": _field_hit_rate_at_k(rows, "article", 5),
        "semantic_hit_rate_at_5": _field_hit_rate_at_k(rows, "semantic", 5),
    }
    metrics.update(
        {
            key: round(value, 4)
            for key, value in field_metrics.items()
            if value is not None
        }
    )

    return {
        "metrics": metrics,
        "judge": {
            "keyword_match": True,
            "metadata_fields": ["doc_id", "chunk_id", "law_id", "article", "article_no"],
            "semantic_judge": args.semantic_judge,
            "semantic_threshold": args.semantic_threshold,
        },
        "cases": [
            {
                "id": row["id"],
                "question": row["question"],
                "expected_keywords": row["expected_keywords"],
                "expected_laws": row.get("expected_laws", []),
                "expected_doc_ids": row.get("expected_doc_ids", []),
                "expected_chunk_ids": row.get("expected_chunk_ids", []),
                "expected_law_ids": row.get("expected_law_ids", []),
                "expected_articles": row.get("expected_articles", []),
                "latency_ms": round(row["latency_ms"], 2),
                "docs": [
                    _serialize_doc(doc, row, index)
                    for index, doc in enumerate(row["docs"][:5])
                ],
            }
            for row in rows
        ],
    }


def _normalize_level(value: Any, score: float | None = None) -> str:
    text = str(value or "").lower()
    if "위험" in text or text == "risk" or "risk" in text:
        return "위험"
    if "주의" in text or text == "caution" or "caution" in text:
        return "주의"
    if "안전" in text or text == "safety" or "safe" in text:
        return "안전"
    if score is not None:
        if score >= 70:
            return "위험"
        if score >= 40:
            return "주의"
        return "안전"
    return ""


def _risk_score_from_result(result: dict[str, Any]) -> float:
    score = result.get("overall_risk_score")
    if isinstance(score, (int, float)):
        return float(score)
    clauses = result.get("clauses") or []
    clause_scores = [
        float(clause.get("score"))
        for clause in clauses
        if isinstance(clause.get("score"), (int, float))
    ]
    return max(clause_scores) if clause_scores else 0.0


def _representative_clause(result: dict[str, Any]) -> dict[str, Any]:
    clauses = result.get("clauses") or []
    if not clauses:
        return {}
    return max(clauses, key=lambda clause: float(clause.get("score") or 0))


def _combined_risk_text(result: dict[str, Any]) -> str:
    clauses = result.get("clauses") or []
    parts = [
        str(result.get("overall_analysis", "")),
        str(result.get("analysis", "")),
        str(result.get("recommendation", "")),
    ]
    for clause in clauses:
        parts.extend(
            [
                str(clause.get("text", "")),
                str(clause.get("analysis", "")),
                str(clause.get("related_law", "")),
                str(clause.get("category", "")),
            ]
        )
    return "\n".join(parts)


def _mock_risk_result(item: dict[str, Any]) -> dict[str, Any]:
    level = item["expected_risk_level"]
    score = LEVEL_TO_SCORE[level]
    return {
        "overall_risk_score": score,
        "risk_summary": {
            "Risk": 1 if level == "위험" else 0,
            "Caution": 1 if level == "주의" else 0,
            "Safety": 1 if level == "안전" else 0,
        },
        "total_clauses": 1,
        "clauses": [
            {
                "text": item.get("target_clause", item["contract_text"]),
                "risk_level": level,
                "category": item.get("category", ""),
                "analysis": f"{item.get('reason', '')} {' '.join(item.get('expected_keywords', []))}",
                "related_law": ", ".join(item.get("expected_laws", [])),
                "score": score,
            }
        ],
    }


async def evaluate_risk(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_risk_cases(Path(args.risk_dataset), args.sample)
    rows: list[dict[str, Any]] = []

    if args.mock:
        for item in cases:
            rows.append({"item": item, "result": _mock_risk_result(item), "latency_ms": 1.0})
    else:
        from app.core.dependencies import get_embeddings, get_llm, get_vector_db
        from app.rag.chain.chain import detect_risk_contract

        db = get_vector_db()
        embeddings = get_embeddings()
        llm = get_llm()

        for item in cases:
            start = time.perf_counter()
            result = await detect_risk_contract(
                user_clause=item["contract_text"],
                client=db,
                embeddings=embeddings,
                llm=llm,
            )
            rows.append(
                {
                    "item": item,
                    "result": result,
                    "latency_ms": (time.perf_counter() - start) * 1000,
                }
            )

    evaluated = []
    for row in rows:
        item = row["item"]
        result = row["result"]
        score = _risk_score_from_result(result)
        rep_clause = _representative_clause(result)
        predicted_level = _normalize_level(rep_clause.get("risk_level"), score)
        expected_level = item["expected_risk_level"]
        content = _combined_risk_text(result)

        keywords = item.get("expected_keywords", [])
        keyword_hits = [
            keyword for keyword in keywords if keyword and keyword.lower() in content.lower()
        ]
        laws = item.get("expected_laws", [])
        law_hits = [law for law in laws if law and law.lower() in content.lower()]
        score_min = item.get("expected_score_min", 0)
        score_max = item.get("expected_score_max", 100)

        evaluated.append(
            {
                "id": item["id"],
                "category": item.get("category"),
                "expected_level": expected_level,
                "predicted_level": predicted_level,
                "level_match": predicted_level == expected_level,
                "predicted_score": score,
                "score_range": [score_min, score_max],
                "score_in_range": score_min <= score <= score_max,
                "keyword_recall": len(keyword_hits) / len(keywords) if keywords else 1.0,
                "keyword_hits": keyword_hits,
                "law_hit": bool(law_hits) if laws else True,
                "law_hits": law_hits,
                "empty_result": not bool(result.get("clauses")),
                "latency_ms": round(row["latency_ms"], 2),
                "representative_clause": rep_clause,
            }
        )

    metrics = {
        "case_count": len(evaluated),
        "risk_level_accuracy": round(mean([1.0 if row["level_match"] else 0.0 for row in evaluated]) if evaluated else 0.0, 4),
        "score_range_pass_rate": round(mean([1.0 if row["score_in_range"] else 0.0 for row in evaluated]) if evaluated else 0.0, 4),
        "law_hit_rate": round(mean([1.0 if row["law_hit"] else 0.0 for row in evaluated]) if evaluated else 0.0, 4),
        "rationale_keyword_recall": round(mean([row["keyword_recall"] for row in evaluated]) if evaluated else 0.0, 4),
        "empty_result_rate": round(mean([1.0 if row["empty_result"] else 0.0 for row in evaluated]) if evaluated else 0.0, 4),
        "avg_latency_ms": round(mean([row["latency_ms"] for row in evaluated]) if evaluated else 0.0, 2),
    }

    return {"metrics": metrics, "cases": evaluated}


def _save_result(payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"risk_retrieval_eval_{ts}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _print_report(payload: dict[str, Any]) -> None:
    print("\nA-LAW Risk/Retrieval Evaluation")
    print(f"mode: {payload['mode']}")
    print(f"mock: {payload['mock']}")

    if "retrieval" in payload:
        print("\n[retrieval]")
        for key, value in payload["retrieval"]["metrics"].items():
            print(f"- {key}: {value}")

    if "risk" in payload:
        print("\n[risk]")
        for key, value in payload["risk"]["metrics"].items():
            print(f"- {key}: {value}")

    if payload.get("saved_path"):
        print(f"\nsaved: {payload['saved_path']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate A-LAW retrieval and risk analysis.")
    parser.add_argument("--mode", choices=["retrieval", "risk", "all"], default="all")
    parser.add_argument("--mock", action="store_true", help="Run without Pinecone/KURE/OpenAI calls.")
    parser.add_argument("--sample", type=int, default=0, help="Use only the first N cases.")
    parser.add_argument("--save", action="store_true", help="Save JSON report under results/.")
    parser.add_argument("--retrieval-dataset", default=str(DEFAULT_RETRIEVAL_DATASET))
    parser.add_argument("--risk-dataset", default=str(DEFAULT_RISK_DATASET))
    parser.add_argument("--collections", default=",".join(DEFAULT_COLLECTIONS))
    parser.add_argument("--k-per-collection", type=int, default=3)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-top-n", type=int, default=5)
    parser.add_argument("--semantic-judge", action="store_true", help="Use embedding cosine similarity as an additional retrieval judge.")
    parser.add_argument("--semantic-threshold", type=float, default=0.7)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    payload: dict[str, Any] = {
        "mode": args.mode,
        "mock": args.mock,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "retrieval_dataset": str(Path(args.retrieval_dataset)),
        "risk_dataset": str(Path(args.risk_dataset)),
    }

    if args.mode in {"retrieval", "all"}:
        payload["retrieval"] = await evaluate_retrieval(args)
    if args.mode in {"risk", "all"}:
        payload["risk"] = await evaluate_risk(args)

    if args.save:
        payload["saved_path"] = str(_save_result(payload))

    _print_report(payload)


if __name__ == "__main__":
    asyncio.run(main())
