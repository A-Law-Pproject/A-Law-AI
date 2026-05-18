from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_core.documents import Document
from loguru import logger


SEARCH_CACHE_TTL_SECONDS = 600
SEARCH_CACHE_NAMESPACE = "rag:search:v1"


def _stable_payload(data: Any) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_search_cache_key(
    *,
    query: str,
    collections: list[str],
    k_per_collection: int | dict[str, int],
    score_threshold: float | dict[str, float],
    collection_filters: dict[str, dict] | None,
    rerank_top_n: int,
    use_hyde: bool,
    use_multiquery: bool,
) -> str:
    raw_key = _stable_payload(
        {
            "query": query,
            "collections": collections,
            "k_per_collection": k_per_collection,
            "score_threshold": score_threshold,
            "collection_filters": collection_filters or {},
            "rerank_top_n": rerank_top_n,
            "use_hyde": use_hyde,
            "use_multiquery": use_multiquery,
        }
    )
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return f"{SEARCH_CACHE_NAMESPACE}:{digest}"


def serialize_documents(documents: list[Document]) -> str:
    payload = [
        {
            "page_content": document.page_content,
            "metadata": document.metadata,
        }
        for document in documents
    ]
    return _stable_payload(payload)


def deserialize_documents(raw: str | None) -> list[Document] | None:
    if not raw:
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("[SearchCache] Invalid cache payload skipped: {}", exc)
        return None

    documents: list[Document] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        documents.append(
            Document(
                page_content=str(item.get("page_content") or ""),
                metadata=item.get("metadata") or {},
            )
        )
    return documents
