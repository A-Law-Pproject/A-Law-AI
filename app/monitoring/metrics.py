"""
Prometheus 커스텀 메트릭 정의

RAG 파이프라인 구간별 지연시간을 측정하기 위한 메트릭.
prometheus-fastapi-instrumentator 가 HTTP 레벨 메트릭을 자동 수집하고,
여기서는 RAG 내부 구간(임베딩, LLM)을 추가로 측정한다.
"""
from prometheus_client import Histogram

# KURE embed_query / embed_documents 실행 시간
# CPU 연산이므로 부하 시 200ms → 수 초로 늘어나는 구간 확인용
EMBEDDING_LATENCY = Histogram(
    "rag_embedding_latency_seconds",
    "KURE embed_query 실행 시간 (CPU)",
    buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0],
)

# OpenAI LLM 응답 대기 시간 (외부 네트워크 요인 분리)
LLM_LATENCY = Histogram(
    "rag_llm_latency_seconds",
    "OpenAI LLM 응답 대기 시간",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0],
)
