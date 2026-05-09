# LLMOps Runbook

## 목적

- `.claude/agents/llmops-monitor.md`의 모니터링 방향과 `.claude/agents/llmops-evaluator.md`의 평가 흐름을 현재 레포에 맞는 실행 단위로 묶는다.
- RAG 품질, 지연시간, 부하테스트 결과를 같은 `results/llmops/` 아래에서 관리한다.
- LoRA/파인튜닝이 아직 없어도 동일한 운영 루프를 먼저 구축해두고, 이후 모델 변경 시 그대로 재사용한다.

## 구성 요소

- `app/monitoring/llmops_metrics.py`
  - 법령 인용률, 회피 응답률, 빈 컨텍스트 비율, 문서 수, reranker 점수를 Prometheus로 수집
- `scripts/llmops/run_eval_suite.py`
  - `tests/eval_unified.py` + `tests/eval_ragas_qna.py` 실행
  - 결과 요약 JSON 생성
  - 선택적 MLflow 로깅
- `scripts/llmops/run_load_test.py`
  - Locust headless 실행
  - HTML/CSV/JSON 결과 저장
  - P95/실패율 기반 성능 게이트 판정
- `scripts/llmops/run_release_gate.py`
  - 평가 스위트 + 부하테스트 + 드리프트 체크를 한 번에 실행
- `scripts/monitor/drift_detector.py`
  - `/metrics` 기반 드리프트 판정

## 빠른 시작

```powershell
# 1. FastAPI 실행
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8001

# 2. 모니터링 스택 실행
docker-compose -f docker-compose.monitoring.yml up -d

# 3. 평가 스위트 실행
.venv\Scripts\python.exe scripts\llmops\run_eval_suite.py --mock --sample 20 --ragas-sample 20

# 4. 빠른 부하테스트 실행
.venv\Scripts\python.exe scripts\llmops\run_load_test.py --profile smoke

# 5. 드리프트 판정
.venv\Scripts\python.exe scripts\monitor\drift_detector.py --endpoint chat_rag

# 6. 릴리스 게이트 일괄 실행
.venv\Scripts\python.exe scripts\llmops\run_release_gate.py --profile staged
```

## 운영 시퀀스

1. 기능 수정 후 `run_eval_suite.py`로 retrieval/RAG 품질 회귀를 확인한다.
2. 성능 영향이 있는 변경이면 `run_load_test.py --profile smoke`를 먼저 실행한다.
3. 릴리스 직전에는 `--profile staged` 또는 `--profile peak`로 부하 게이트를 확인한다.
4. 운영 중에는 Grafana와 `drift_detector.py` 결과로 품질 드리프트를 추적한다.

## 게이트 기준

- 평가 게이트
  - `hit_rate_at_3 >= 0.60`
  - `mrr >= 0.55`
  - `avg_total_ms <= 12000`
  - `faq hr@3 >= 0.70`
  - `faithfulness >= 0.75` (값이 존재할 때)
- 부하 게이트
  - `P95 <= 3000ms`
  - `failure_ratio <= 1%`
  - `request_count >= 10`
- 드리프트 기준
  - `legal_citation_rate < 0.60`
  - `rejection_rate > 0.10`
  - `empty_context_rate > 0.20`
  - `avg_reranker_score < -3.0`

## Locust 프로필

- `smoke`
  - 빠른 사전 점검용
  - 총 150초
- `staged`
  - 기본 단계형 부하
  - 총 780초
- `peak`
  - 공격적인 피크 부하
  - 총 480초

프로필 선택은 `ALAW_LOCUST_PROFILE` 환경변수 또는 `run_load_test.py --profile ...`로 제어한다.

## 결과 저장 경로

- 평가 요약: `results/llmops/eval_suite_*.json`
- 부하테스트 요약: `results/llmops/load_test_*.json`
- 릴리스 게이트 요약: `results/llmops/release_gate_*.json`
- Locust 리포트: `results/llmops/locust_*.html`, `results/llmops/locust_*_stats.csv`
- 드리프트 리포트: `results/llmops/drift_report_*.json`

## Grafana에서 볼 것

- HTTP RPS / 5xx / P95 응답시간
- `rag_embedding_latency_seconds`
- `rag_llm_latency_seconds`
- `rag_legal_citation_total`
- `rag_refusal_response_total`
- `rag_empty_context_total`
- `rag_reranker_score`

## 참고

- MLflow는 선택적이다. 설치되어 있으면 스크립트가 자동으로 run/artifact를 남긴다.
- LangSmith 업로드는 `run_eval_suite.py --langsmith`일 때만 수행한다.
- 현재 레포에서 `scripts/start-task.sh`는 찾지 못해 `exec-plans/active/llmops-loadtest-20260507.md`로 수동 계획을 남겼다.
