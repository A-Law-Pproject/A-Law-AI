from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "llmops"

load_dotenv(ROOT / ".env")

LOAD_PROFILES = {
    "smoke": [
        {"cumulative": 30, "users": 2, "spawn_rate": 1},
        {"cumulative": 90, "users": 4, "spawn_rate": 1},
        {"cumulative": 150, "users": 6, "spawn_rate": 1},
    ],
    "staged": [
        {"cumulative": 120, "users": 3, "spawn_rate": 1},
        {"cumulative": 420, "users": 10, "spawn_rate": 1},
        {"cumulative": 780, "users": 20, "spawn_rate": 2},
    ],
    "peak": [
        {"cumulative": 60, "users": 5, "spawn_rate": 1},
        {"cumulative": 240, "users": 15, "spawn_rate": 2},
        {"cumulative": 480, "users": 30, "spawn_rate": 3},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A-LAW Locust 부하테스트 실행")
    parser.add_argument("--host", default="http://localhost:8001", help="테스트 대상 호스트")
    parser.add_argument(
        "--profile",
        choices=sorted(LOAD_PROFILES),
        default="smoke",
        help="Locust 부하 프로필",
    )
    parser.add_argument("--max-p95-ms", type=float, default=3000.0)
    parser.add_argument("--max-failure-ratio", type=float, default=0.01)
    parser.add_argument("--min-request-count", type=int, default=10)
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
    return parser.parse_args()


def _to_number(value: str | None, *, integer: bool = False) -> float | int:
    if value in (None, ""):
        return 0 if integer else 0.0
    number = float(value)
    return int(number) if integer else number


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _extract_aggregate_row(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if row.get("Name") == "Aggregated" or row.get("Type") == "Aggregated":
            return row
    if not rows:
        raise ValueError("Locust stats CSV가 비어 있습니다.")
    return rows[-1]


def build_gate(args: argparse.Namespace, aggregate: dict[str, Any]) -> dict[str, Any]:
    checks = [
        {
            "name": "request_count",
            "passed": aggregate["request_count"] >= args.min_request_count,
            "value": aggregate["request_count"],
            "threshold": f">= {args.min_request_count}",
        },
        {
            "name": "failure_ratio",
            "passed": aggregate["failure_ratio"] <= args.max_failure_ratio,
            "value": aggregate["failure_ratio"],
            "threshold": f"<= {args.max_failure_ratio}",
        },
        {
            "name": "p95_ms",
            "passed": aggregate["p95_ms"] <= args.max_p95_ms,
            "value": aggregate["p95_ms"],
            "threshold": f"<= {args.max_p95_ms}",
        },
    ]
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
    html_path: Path,
    csv_prefix: Path,
    aggregate: dict[str, Any],
    gate: dict[str, Any],
) -> str | None:
    try:
        import mlflow
    except ImportError:
        print("[MLflow] mlflow 미설치 - 로깅 건너뜀")
        return None

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=f"load-test-{args.profile}-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
        mlflow.log_params(
            {
                "host": args.host,
                "profile": args.profile,
                "max_p95_ms": args.max_p95_ms,
                "max_failure_ratio": args.max_failure_ratio,
                "min_request_count": args.min_request_count,
            }
        )
        mlflow.log_metrics(
            {
                "request_count": aggregate["request_count"],
                "failure_count": aggregate["failure_count"],
                "failure_ratio": aggregate["failure_ratio"],
                "rps": aggregate["rps"],
                "avg_response_ms": aggregate["avg_response_ms"],
                "p50_ms": aggregate["p50_ms"],
                "p95_ms": aggregate["p95_ms"],
                "p99_ms": aggregate["p99_ms"],
            }
        )
        mlflow.set_tag("perf_gate", gate["status"])
        mlflow.set_tag("perf_failed_checks", ",".join(gate["failed_checks"]) or "none")
        for suffix in ("_stats.csv", "_failures.csv", "_exceptions.csv", "_stats_history.csv"):
            artifact = csv_prefix.with_name(csv_prefix.name + suffix)
            if artifact.exists():
                mlflow.log_artifact(str(artifact))
        if html_path.exists():
            mlflow.log_artifact(str(html_path))
        mlflow.log_artifact(str(summary_path))
        return run.info.run_id


def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = RESULTS_DIR / f"locust_{args.profile}_{timestamp}.html"
    csv_prefix = RESULTS_DIR / f"locust_{args.profile}_{timestamp}"

    env = os.environ.copy()
    env["ALAW_LOCUST_PROFILE"] = args.profile
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    runtime_seconds = LOAD_PROFILES[args.profile][-1]["cumulative"] + 5
    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        "locustfile.py",
        "--host",
        args.host,
        "--headless",
        "--run-time",
        f"{runtime_seconds}s",
        "--only-summary",
        "--html",
        str(html_path),
        "--csv",
        str(csv_prefix),
    ]

    print(f"\n[RUN] {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, env=env, check=True)

    stats_path = csv_prefix.with_name(csv_prefix.name + "_stats.csv")
    rows = _read_csv_rows(stats_path)
    aggregate_row = _extract_aggregate_row(rows)
    endpoint_rows = [
        row for row in rows
        if row.get("Name") not in {"Aggregated", None, ""}
    ]

    aggregate = {
        "request_count": _to_number(aggregate_row.get("Request Count"), integer=True),
        "failure_count": _to_number(aggregate_row.get("Failure Count"), integer=True),
        "failure_ratio": 0.0,
        "rps": _to_number(aggregate_row.get("Requests/s")),
        "avg_response_ms": _to_number(aggregate_row.get("Average Response Time")),
        "p50_ms": _to_number(aggregate_row.get("50%")),
        "p95_ms": _to_number(aggregate_row.get("95%")),
        "p99_ms": _to_number(aggregate_row.get("99%")),
    }
    if aggregate["request_count"]:
        aggregate["failure_ratio"] = round(
            aggregate["failure_count"] / aggregate["request_count"],
            4,
        )

    failures_rows = _read_csv_rows(csv_prefix.with_name(csv_prefix.name + "_failures.csv"))
    gate = build_gate(args, aggregate)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": args.host,
        "profile": args.profile,
        "runtime_seconds": runtime_seconds,
        "locust_html": str(html_path),
        "locust_csv_prefix": str(csv_prefix),
        "aggregate": aggregate,
        "gate": gate,
        "endpoints": [
            {
                "name": row.get("Name"),
                "request_count": _to_number(row.get("Request Count"), integer=True),
                "failure_count": _to_number(row.get("Failure Count"), integer=True),
                "avg_response_ms": _to_number(row.get("Average Response Time")),
                "p95_ms": _to_number(row.get("95%")),
                "rps": _to_number(row.get("Requests/s")),
            }
            for row in endpoint_rows
        ],
        "failures": failures_rows[:10],
    }

    summary_path = RESULTS_DIR / f"load_test_{args.profile}_{timestamp}.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mlflow_run_id = None
    if args.log_mlflow:
        mlflow_run_id = log_to_mlflow(
            args=args,
            summary_path=summary_path,
            html_path=html_path,
            csv_prefix=csv_prefix,
            aggregate=aggregate,
            gate=gate,
        )

    print("\n[LLMOps] 부하테스트 완료")
    print(f"  summary : {summary_path}")
    print(f"  p95     : {aggregate['p95_ms']} ms")
    print(f"  error   : {aggregate['failure_ratio'] * 100:.2f}%")
    print(f"  gate    : {gate['status']}")
    if gate["failed_checks"]:
        print(f"  failed  : {', '.join(gate['failed_checks'])}")
    if mlflow_run_id:
        print(f"  mlflow  : {mlflow_run_id}")

    return 0 if gate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
