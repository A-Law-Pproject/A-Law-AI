from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
LLMOPS_RESULTS_DIR = RESULTS_DIR / "llmops"

load_dotenv(ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A-LAW LLMOps 평가 스위트 실행")
    parser.add_argument("--sample", type=int, default=20, help="통합 평가 케이스 수")
    parser.add_argument("--ragas-sample", type=int, default=20, help="RAGAS 평가 케이스 수")
    parser.add_argument(
        "--configs",
        default="baseline,reranker,hyde,full",
        help="eval_unified.py config 목록",
    )
    parser.add_argument("--mock", action="store_true", help="mock 모드로 평가 실행")
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="eval_ragas_qna.py 실행 생략",
    )
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help="RAGAS 평가 결과를 LangSmith에 업로드",
    )
    parser.add_argument(
        "--log-mlflow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="요약 결과를 MLflow에 로깅 (기본: 사용)",
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
    parser.add_argument("--min-hit-rate-at-3", type=float, default=0.60)
    parser.add_argument("--min-mrr", type=float, default=0.55)
    parser.add_argument("--max-avg-total-ms", type=float, default=12000.0)
    parser.add_argument("--min-faq-hr3", type=float, default=0.70)
    parser.add_argument("--min-faithfulness", type=float, default=0.75)
    parser.add_argument("--min-chunk-hr3", type=float, default=0.65,
                        help="chunk_hr@3 최소값 (--dataset-v3 사용 시 적용)")
    parser.add_argument("--dataset-v3", action="store_true", dest="dataset_v3",
                        help="eval_dataset_v3_with_gt.json 사용 (chunk GT 기반 지표)")
    parser.add_argument("--use-failure-analyzer", action="store_true", dest="use_failure_analyzer",
                        help="평가 완료 후 eval_failure_analyzer.py 실행")
    return parser.parse_args()


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    merged_env["PYTHONIOENCODING"] = "utf-8"
    merged_env["PYTHONUTF8"] = "1"
    if env:
        merged_env.update(env)
    print(f"\n[RUN] {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, env=merged_env, check=True)


def find_latest(pattern: str, started_at: float) -> Path | None:
    candidates = [
        path for path in RESULTS_DIR.glob(pattern)
        if path.stat().st_mtime >= started_at - 1
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def summarize_unified(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    leaderboard: list[dict[str, Any]] = []
    for config_name, config_result in payload.get("results", {}).items():
        metrics = config_result.get("metrics", {})
        leaderboard.append(
            {
                "name": config_name,
                "hit_rate_at_3": _safe_float(metrics.get("hit_rate_at_3")) or 0.0,
                "hit_rate_at_5": _safe_float(metrics.get("hit_rate_at_5")) or 0.0,
                "mrr": _safe_float(metrics.get("mrr")) or 0.0,
                "precision_at_3": _safe_float(metrics.get("precision_at_3")) or 0.0,
                "avg_total_ms": _safe_float(metrics.get("avg_total_ms")) or 0.0,
                "faithfulness": _safe_float(metrics.get("faithfulness")),
                "answer_relevancy": _safe_float(metrics.get("answer_relevancy")),
            }
        )

    leaderboard.sort(
        key=lambda item: (
            item["hit_rate_at_3"],
            item["mrr"],
            -item["avg_total_ms"],
        ),
        reverse=True,
    )
    return {
        "path": str(path),
        "metadata": payload.get("metadata", {}),
        "leaderboard": leaderboard,
        "best_config": leaderboard[0] if leaderboard else None,
    }


def summarize_ragas(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    overall = payload.get("overall_ragas", {})
    hr_metrics = payload.get("hr_k_mrr", {})
    chunk_hr_metrics = payload.get("chunk_hr_k_mrr", {})
    failures = payload.get("failure_counts", {})
    return {
        "path": str(path),
        "mode": payload.get("mode"),
        "dataset_v3": payload.get("dataset_v3", False),
        "overall_ragas": overall,
        "faq_hr_k_mrr": hr_metrics,
        "chunk_hr_k_mrr": chunk_hr_metrics,
        "failure_counts": failures,
        "chatbot_count": payload.get("chatbot_count", 0),
        "faq_count": payload.get("faq_count", 0),
    }


def build_gate(args: argparse.Namespace, unified: dict[str, Any], ragas: dict[str, Any] | None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    best = unified.get("best_config") or {}

    hit_rate_at_3 = _safe_float(best.get("hit_rate_at_3"))
    mrr = _safe_float(best.get("mrr"))
    avg_total_ms = _safe_float(best.get("avg_total_ms"))
    faithfulness = None
    faq_hr3 = None
    if ragas:
        faithfulness = _safe_float(ragas.get("overall_ragas", {}).get("faithfulness"))
        faq_hr3 = _safe_float(ragas.get("faq_hr_k_mrr", {}).get("hr@3"))

    def add_check(name: str, passed: bool, value: Any, threshold: Any) -> None:
        checks.append(
            {
                "name": name,
                "passed": passed,
                "value": value,
                "threshold": threshold,
            }
        )

    add_check(
        "best_config.hit_rate_at_3",
        hit_rate_at_3 is not None and hit_rate_at_3 >= args.min_hit_rate_at_3,
        hit_rate_at_3,
        f">= {args.min_hit_rate_at_3}",
    )
    add_check(
        "best_config.mrr",
        mrr is not None and mrr >= args.min_mrr,
        mrr,
        f">= {args.min_mrr}",
    )
    add_check(
        "best_config.avg_total_ms",
        avg_total_ms is not None and avg_total_ms <= args.max_avg_total_ms,
        avg_total_ms,
        f"<= {args.max_avg_total_ms}",
    )

    if ragas:
        add_check(
            "ragas.hr@3",
            faq_hr3 is not None and faq_hr3 >= args.min_faq_hr3,
            faq_hr3,
            f">= {args.min_faq_hr3}",
        )
        if faithfulness is not None:
            add_check(
                "ragas.faithfulness",
                faithfulness >= args.min_faithfulness,
                faithfulness,
                f">= {args.min_faithfulness}",
            )
        # v3 데이터셋 사용 시 chunk_hr@3 gate 추가
        chunk_hr3 = _safe_float(ragas.get("chunk_hr_k_mrr", {}).get("chunk_hr@3"))
        if ragas.get("dataset_v3") and chunk_hr3 is not None:
            add_check(
                "ragas.chunk_hr@3",
                chunk_hr3 >= args.min_chunk_hr3,
                chunk_hr3,
                f">= {args.min_chunk_hr3}",
            )

    failed = [check["name"] for check in checks if not check["passed"]]
    return {
        "status": "pass" if not failed else "fail",
        "failed_checks": failed,
        "checks": checks,
    }


def log_to_mlflow(
    *,
    args: argparse.Namespace,
    summary_path: Path,
    unified: dict[str, Any],
    ragas: dict[str, Any] | None,
    gate: dict[str, Any],
) -> str | None:
    try:
        import mlflow
    except ImportError:
        print("[MLflow] mlflow 미설치 - 로깅 건너뜀")
        return None

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    best = unified.get("best_config") or {}
    metrics: dict[str, float] = {}
    for key in ("hit_rate_at_3", "hit_rate_at_5", "mrr", "precision_at_3", "avg_total_ms"):
        value = _safe_float(best.get(key))
        if value is not None:
            metrics[f"best_{key}"] = value

    if ragas:
        for key, value in ragas.get("overall_ragas", {}).items():
            numeric = _safe_float(value)
            if numeric is not None:
                metrics[f"ragas_{key}"] = numeric
        for key, value in ragas.get("faq_hr_k_mrr", {}).items():
            numeric = _safe_float(value)
            if numeric is not None:
                metrics[f"faq_{key.replace('@', '_at_')}"] = numeric
        for key, value in ragas.get("chunk_hr_k_mrr", {}).items():
            if not key.startswith("_"):
                numeric = _safe_float(value)
                if numeric is not None:
                    metrics[f"chunk_{key.replace('@', '_at_')}"] = numeric

    with mlflow.start_run(run_name=f"eval-suite-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
        mlflow.log_params(
            {
                "mock": args.mock,
                "configs": args.configs,
                "sample": args.sample,
                "ragas_sample": args.ragas_sample,
                "skip_ragas": args.skip_ragas,
                "langsmith": args.langsmith,
            }
        )
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.set_tag("deploy_ready", "yes" if gate["status"] == "pass" else "no")
        mlflow.set_tag("gate_failed_checks", ",".join(gate["failed_checks"]) or "none")
        mlflow.log_artifact(str(summary_path))
        unified_path = unified.get("path")
        if unified_path:
            mlflow.log_artifact(unified_path)
        if ragas and ragas.get("path"):
            mlflow.log_artifact(ragas["path"])
        return run.info.run_id


def main() -> int:
    args = parse_args()
    LLMOPS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    use_v3 = getattr(args, "dataset_v3", False)

    unified_cmd = [
        sys.executable,
        "tests/eval_unified.py",
        "--run",
        "--sample",
        str(args.sample),
        "--configs",
        args.configs,
    ]
    if args.mock:
        unified_cmd.append("--mock")
    if use_v3:
        unified_cmd.append("--dataset-v3")

    run_command(unified_cmd)
    unified_path = find_latest("unified_eval_*.json", started_at)
    if unified_path is None:
        raise FileNotFoundError("unified_eval 결과 파일을 찾지 못했습니다.")
    unified_summary = summarize_unified(unified_path)

    ragas_summary: dict[str, Any] | None = None
    ragas_path: Path | None = None
    if not args.skip_ragas:
        ragas_cmd = [
            sys.executable,
            "tests/eval_ragas_qna.py",
            "--sample",
            str(args.ragas_sample),
        ]
        if args.mock:
            ragas_cmd.append("--mock")
        ragas_cmd.append("--langsmith" if args.langsmith else "--no-langsmith")
        if use_v3:
            ragas_cmd.append("--dataset-v3")
        run_command(ragas_cmd)
        ragas_path = find_latest("ragas_qna_eval_*.json", started_at)
        if ragas_path is None:
            raise FileNotFoundError("ragas_qna_eval 결과 파일을 찾지 못했습니다.")
        ragas_summary = summarize_ragas(ragas_path)

    gate = build_gate(args, unified_summary, ragas_summary)

    # --use-failure-analyzer: 실패 분석 실행
    failure_analysis_path: Path | None = None
    if getattr(args, "use_failure_analyzer", False) and ragas_path and ragas_path.exists():
        v3_path = ROOT / "tests" / "평가데이터셋" / "eval_dataset_v3_with_gt.json"
        if v3_path.exists():
            failure_cmd = [
                sys.executable,
                "tests/eval_failure_analyzer.py",
                "--eval-result", str(ragas_path),
                "--dataset", str(v3_path),
            ]
            try:
                run_command(failure_cmd)
                failure_analysis_path = find_latest("failure_analysis_*.json", started_at)
            except subprocess.CalledProcessError as e:
                print(f"[WARN] 실패 분석 실행 오류: {e}")
        else:
            print("[WARN] --use-failure-analyzer: v3 데이터셋 없음, 건너뜀")

    payload = normalize_json(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "mock" if args.mock else "real",
            "dataset_v3": use_v3,
            "commands": {
                "eval_unified": unified_cmd,
                "eval_ragas_qna": None if args.skip_ragas else ragas_cmd,
            },
            "unified": unified_summary,
            "ragas": ragas_summary,
            "gate": gate,
            "failure_analysis": str(failure_analysis_path) if failure_analysis_path else None,
        }
    )

    summary_path = LLMOPS_RESULTS_DIR / f"eval_suite_{timestamp}.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mlflow_run_id = None
    if args.log_mlflow:
        mlflow_run_id = log_to_mlflow(
            args=args,
            summary_path=summary_path,
            unified=unified_summary,
            ragas=ragas_summary,
            gate=gate,
        )

    best = unified_summary.get("best_config") or {}
    print("\n[LLMOps] 평가 스위트 완료")
    print(f"  summary : {summary_path}")
    print(f"  best    : {best.get('name')} (HR@3={best.get('hit_rate_at_3')}, MRR={best.get('mrr')})")
    print(f"  gate    : {gate['status']}")
    if gate["failed_checks"]:
        print(f"  failed  : {', '.join(gate['failed_checks'])}")
    if mlflow_run_id:
        print(f"  mlflow  : {mlflow_run_id}")

    return 0 if gate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
