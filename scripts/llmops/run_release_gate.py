from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "llmops"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLMOps 평가 + 부하테스트 릴리스 게이트")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--ragas-sample", type=int, default=20)
    parser.add_argument("--configs", default="baseline,reranker,hyde,full")
    parser.add_argument("--profile", default="smoke", choices=["smoke", "staged", "peak"])
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--skip-load-test", action="store_true")
    parser.add_argument("--skip-drift-check", action="store_true")
    parser.add_argument("--langsmith", action="store_true")
    parser.add_argument(
        "--log-mlflow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="하위 스크립트들의 MLflow 로깅 사용 여부",
    )
    parser.add_argument("--drift-endpoint", default="chat_rag")
    return parser.parse_args()


def run_command(command: list[str]) -> int:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    print(f"\n[RUN] {' '.join(command)}")
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    return completed.returncode


def find_latest(pattern: str, started_at: float) -> Path | None:
    candidates = [
        path for path in RESULTS_DIR.glob(pattern)
        if path.stat().st_mtime >= started_at - 1
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    commands: dict[str, list[str]] = {}
    return_codes: dict[str, int] = {}

    eval_cmd = [
        sys.executable,
        "scripts/llmops/run_eval_suite.py",
        "--sample",
        str(args.sample),
        "--ragas-sample",
        str(args.ragas_sample),
        "--configs",
        args.configs,
        "--log-mlflow" if args.log_mlflow else "--no-log-mlflow",
    ]
    if args.mock:
        eval_cmd.append("--mock")
    if args.skip_ragas:
        eval_cmd.append("--skip-ragas")
    if args.langsmith:
        eval_cmd.append("--langsmith")
    commands["eval_suite"] = eval_cmd
    return_codes["eval_suite"] = run_command(eval_cmd)

    if not args.skip_load_test:
        load_cmd = [
            sys.executable,
            "scripts/llmops/run_load_test.py",
            "--profile",
            args.profile,
            "--log-mlflow" if args.log_mlflow else "--no-log-mlflow",
        ]
        commands["load_test"] = load_cmd
        return_codes["load_test"] = run_command(load_cmd)

    if not args.skip_drift_check:
        drift_cmd = [
            sys.executable,
            "scripts/monitor/drift_detector.py",
            "--endpoint",
            args.drift_endpoint,
            "--log-mlflow" if args.log_mlflow else "--no-log-mlflow",
        ]
        commands["drift_check"] = drift_cmd
        return_codes["drift_check"] = run_command(drift_cmd)

    eval_summary_path = find_latest("eval_suite_*.json", started_at)
    load_summary_path = None if args.skip_load_test else find_latest("load_test_*.json", started_at)
    drift_summary_path = None if args.skip_drift_check else find_latest("drift_report_*.json", started_at)

    eval_summary = load_json(eval_summary_path)
    load_summary = load_json(load_summary_path)
    drift_summary = load_json(drift_summary_path)

    overall_status = "pass"
    failed_steps = [name for name, code in return_codes.items() if code != 0]
    if failed_steps:
        overall_status = "fail"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "overall_status": overall_status,
        "failed_steps": failed_steps,
        "commands": commands,
        "return_codes": return_codes,
        "artifacts": {
            "eval_suite": str(eval_summary_path) if eval_summary_path else None,
            "load_test": str(load_summary_path) if load_summary_path else None,
            "drift_check": str(drift_summary_path) if drift_summary_path else None,
        },
        "summaries": {
            "eval_suite": eval_summary,
            "load_test": load_summary,
            "drift_check": drift_summary,
        },
    }

    output_path = RESULTS_DIR / f"release_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[LLMOps] 릴리스 게이트 완료")
    print(f"  summary : {output_path}")
    print(f"  status  : {overall_status}")
    print(f"  failed  : {', '.join(failed_steps) if failed_steps else 'none'}")

    return 0 if overall_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
