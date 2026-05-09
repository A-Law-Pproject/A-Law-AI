from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "llmops"

load_dotenv(ROOT / ".env")

METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)$"
)
LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prometheus metrics 기반 드리프트 판정")
    parser.add_argument("--metrics-url", default="http://localhost:8001/metrics")
    parser.add_argument("--endpoint", default="", help="특정 endpoint 라벨만 집계")
    parser.add_argument("--min-sample-size", type=int, default=30)
    parser.add_argument(
        "--min-legal-citation-rate",
        type=float,
        default=float(os.getenv("LLMOPS_MIN_LEGAL_CITATION_RATE", "0.60")),
    )
    parser.add_argument(
        "--max-rejection-rate",
        type=float,
        default=float(os.getenv("LLMOPS_MAX_REJECTION_RATE", "0.10")),
    )
    parser.add_argument(
        "--max-empty-context-rate",
        type=float,
        default=float(os.getenv("LLMOPS_MAX_EMPTY_CONTEXT_RATE", "0.20")),
    )
    parser.add_argument(
        "--min-avg-reranker-score",
        type=float,
        default=float(os.getenv("LLMOPS_MIN_AVG_RERANKER_SCORE", "-3.0")),
    )
    parser.add_argument(
        "--log-mlflow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="드리프트 요약을 MLflow에 로깅 (기본: 사용)",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
        help="MLflow Tracking URI",
    )
    parser.add_argument(
        "--experiment",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "a-law-llmops"),
        help="MLflow experiment name",
    )
    return parser.parse_args()


def fetch_metrics(url: str) -> str:
    with httpx.Client(timeout=10.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def parse_metric_lines(text: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        labels = {
            label: value
            for label, value in LABEL_RE.findall(match.group("labels") or "")
        }
        parsed.append(
            {
                "name": match.group("name"),
                "labels": labels,
                "value": float(match.group("value")),
            }
        )
    return parsed


def metric_sum(metrics: list[dict[str, Any]], name: str, **label_filters: str) -> float:
    total = 0.0
    for metric in metrics:
        if metric["name"] != name:
            continue
        if any(metric["labels"].get(key) != value for key, value in label_filters.items()):
            continue
        total += metric["value"]
    return total


def maybe_log_to_mlflow(
    *,
    args: argparse.Namespace,
    summary_path: Path,
    status: str,
    ratios: dict[str, Any],
) -> str | None:
    try:
        import mlflow
    except ImportError:
        print("[MLflow] mlflow 미설치 - 로깅 건너뜀")
        return None

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=f"drift-check-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
        mlflow.log_params(
            {
                "metrics_url": args.metrics_url,
                "endpoint": args.endpoint or "all",
                "min_sample_size": args.min_sample_size,
                "min_legal_citation_rate": args.min_legal_citation_rate,
                "max_rejection_rate": args.max_rejection_rate,
                "max_empty_context_rate": args.max_empty_context_rate,
                "min_avg_reranker_score": args.min_avg_reranker_score,
            }
        )
        for key, value in ratios.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))
        mlflow.set_tag("drift_status", status)
        mlflow.log_artifact(str(summary_path))
        return run.info.run_id


def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_text = fetch_metrics(args.metrics_url)
    metrics = parse_metric_lines(metrics_text)
    endpoint_filter = {"endpoint": args.endpoint} if args.endpoint else {}

    citation_true = metric_sum(
        metrics,
        "rag_legal_citation_total",
        has_citation="true",
        **endpoint_filter,
    )
    citation_false = metric_sum(
        metrics,
        "rag_legal_citation_total",
        has_citation="false",
        **endpoint_filter,
    )
    refusal_true = metric_sum(
        metrics,
        "rag_refusal_response_total",
        is_refusal="true",
        **endpoint_filter,
    )
    refusal_false = metric_sum(
        metrics,
        "rag_refusal_response_total",
        is_refusal="false",
        **endpoint_filter,
    )
    empty_true = metric_sum(
        metrics,
        "rag_empty_context_total",
        is_empty_context="true",
        **endpoint_filter,
    )
    empty_false = metric_sum(
        metrics,
        "rag_empty_context_total",
        is_empty_context="false",
        **endpoint_filter,
    )
    retrieval_true = metric_sum(
        metrics,
        "rag_retrieval_hit_total",
        has_documents="true",
        **endpoint_filter,
    )
    retrieval_false = metric_sum(
        metrics,
        "rag_retrieval_hit_total",
        has_documents="false",
        **endpoint_filter,
    )
    reranker_sum = metric_sum(metrics, "rag_reranker_score_sum", **endpoint_filter)
    reranker_count = metric_sum(metrics, "rag_reranker_score_count", **endpoint_filter)

    sample_size = int(citation_true + citation_false)
    ratios = {
        "sample_size": sample_size,
        "legal_citation_rate": round(citation_true / sample_size, 4) if sample_size else None,
        "rejection_rate": round(refusal_true / (refusal_true + refusal_false), 4)
        if (refusal_true + refusal_false)
        else None,
        "empty_context_rate": round(empty_true / (empty_true + empty_false), 4)
        if (empty_true + empty_false)
        else None,
        "retrieval_hit_rate": round(retrieval_true / (retrieval_true + retrieval_false), 4)
        if (retrieval_true + retrieval_false)
        else None,
        "avg_reranker_score": round(reranker_sum / reranker_count, 4) if reranker_count else None,
    }

    reasons: list[str] = []
    status = "healthy"
    if sample_size < args.min_sample_size:
        status = "insufficient_sample"
        reasons.append(
            f"sample_size {sample_size} < {args.min_sample_size}"
        )
    else:
        if ratios["legal_citation_rate"] is not None and ratios["legal_citation_rate"] < args.min_legal_citation_rate:
            reasons.append("legal_citation_rate_low")
        if ratios["rejection_rate"] is not None and ratios["rejection_rate"] > args.max_rejection_rate:
            reasons.append("rejection_rate_high")
        if ratios["empty_context_rate"] is not None and ratios["empty_context_rate"] > args.max_empty_context_rate:
            reasons.append("empty_context_rate_high")
        if ratios["avg_reranker_score"] is not None and ratios["avg_reranker_score"] < args.min_avg_reranker_score:
            reasons.append("avg_reranker_score_low")
        if reasons:
            status = "drift"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metrics_url": args.metrics_url,
        "endpoint": args.endpoint or "all",
        "status": status,
        "reasons": reasons,
        "thresholds": {
            "min_sample_size": args.min_sample_size,
            "min_legal_citation_rate": args.min_legal_citation_rate,
            "max_rejection_rate": args.max_rejection_rate,
            "max_empty_context_rate": args.max_empty_context_rate,
            "min_avg_reranker_score": args.min_avg_reranker_score,
        },
        "ratios": ratios,
        "raw_counts": {
            "citation_true": citation_true,
            "citation_false": citation_false,
            "refusal_true": refusal_true,
            "refusal_false": refusal_false,
            "empty_true": empty_true,
            "empty_false": empty_false,
            "retrieval_true": retrieval_true,
            "retrieval_false": retrieval_false,
            "reranker_count": reranker_count,
        },
    }

    summary_path = RESULTS_DIR / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mlflow_run_id = None
    if args.log_mlflow:
        mlflow_run_id = maybe_log_to_mlflow(
            args=args,
            summary_path=summary_path,
            status=status,
            ratios=ratios,
        )

    print("\n[LLMOps] 드리프트 점검 완료")
    print(f"  summary : {summary_path}")
    print(f"  status  : {status}")
    print(f"  reasons : {', '.join(reasons) if reasons else 'none'}")
    if mlflow_run_id:
        print(f"  mlflow  : {mlflow_run_id}")

    return 0 if status in {"healthy", "insufficient_sample"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
