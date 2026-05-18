"""
Prometheus metrics for RAG internals.

The HTTP layer is instrumented elsewhere. This module tracks internal RAG
stages so we can see whether latency comes from embedding, expansion,
retrieval, reranking, or the LLM call itself.
"""
from prometheus_client import Counter, Histogram


EMBEDDING_LATENCY = Histogram(
    "rag_embedding_latency_seconds",
    "Latency of embedding generation on the CPU path",
    buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0],
)

LLM_LATENCY = Histogram(
    "rag_llm_latency_seconds",
    "Latency of OpenAI LLM responses",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0],
)

RAG_PIPELINE_STAGE_LATENCY = Histogram(
    "rag_pipeline_stage_latency_seconds",
    "Latency of internal RAG pipeline stages",
    ["stage"],
    buckets=[0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
)

RAG_RETRIEVAL_CACHE_TOTAL = Counter(
    "rag_retrieval_cache_total",
    "Redis retrieval cache lookups grouped by result",
    ["result"],
)
