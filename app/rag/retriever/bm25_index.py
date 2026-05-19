"""코퍼스 전체 BM25 인덱스.

기존 hybrid는 BM25가 dense 후보 풀(4×k) 안에서만 재정렬되어, dense 임베딩이
애초에 정답 문서를 못 가져오면 lexical로도 회수 불가능했다. 이 모듈은
namespace별 전체 코퍼스에 대한 독립 BM25 인덱스를 제공해, dense가 놓친
문서를 lexical 검색으로 직접 회수한 뒤 RRF로 융합할 수 있게 한다.

코퍼스 입력: scripts/export_pinecone_corpus.py 가 생성한
    data/bm25_corpus/{namespace}.jsonl   (각 라인 {"content","metadata"})

아티팩트가 없으면 search()는 빈 리스트를 반환 → 기존 동작과 100% 동일(무회귀).
"""
from __future__ import annotations

import json
import math
import threading
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger

from app.core.config import settings
from app.rag.retriever.multi_retriever import (
    _exact_legal_boost,
    _tokenize_lexical,
)

_BM25_K1 = 1.5
_BM25_B = 0.75


class _NamespaceBM25:
    """단일 namespace의 사전 계산된 BM25 통계."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.contents: list[str] = []
        self.metadatas: list[dict] = []
        self.counters: list[Counter] = []
        self.doc_lengths: list[int] = []
        self.document_frequency: Counter[str] = Counter()
        self.avgdl: float = 1.0
        self.total_docs: int = 0

    def load(self, path: Path) -> None:
        if not path.exists():
            logger.info(f"[CorpusBM25] 아티팩트 없음, namespace={self.namespace} 건너뜀 ({path})")
            return

        with path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = rec.get("content") or ""
                if not content:
                    continue
                meta = dict(rec.get("metadata") or {})
                # PineconeAdapter.search()와 동일하게 collection 부여 → _document_key 정합
                meta["collection"] = self.namespace
                # 어휘 소스: 메타(법령명/조문/사건번호) + 본문
                lexical_src = "\n".join(
                    part
                    for part in [
                        meta.get("law_name"),
                        meta.get("title"),
                        meta.get("article"),
                        meta.get("case_no"),
                        content,
                    ]
                    if part
                )
                tokens = _tokenize_lexical(lexical_src)
                counter = Counter(tokens)
                self.contents.append(content)
                self.metadatas.append(meta)
                self.counters.append(counter)
                self.doc_lengths.append(max(len(tokens), 1))
                self.document_frequency.update(counter.keys())

        self.total_docs = len(self.contents)
        if self.total_docs:
            self.avgdl = sum(self.doc_lengths) / self.total_docs
        logger.info(f"[CorpusBM25] namespace={self.namespace} 적재 완료: {self.total_docs}개")

    def search(self, query: str, k: int) -> list[Document]:
        if self.total_docs == 0 or k <= 0:
            return []
        query_tokens = _tokenize_lexical(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, int]] = []
        for idx in range(self.total_docs):
            counter = self.counters[idx]
            doc_len = self.doc_lengths[idx]
            score = 0.0
            for token in query_tokens:
                tf = counter.get(token, 0)
                if tf <= 0:
                    continue
                df = self.document_frequency.get(token, 0)
                idf = math.log(1 + (self.total_docs - df + 0.5) / (df + 0.5))
                denom = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * (doc_len / max(self.avgdl, 1.0)))
                score += idf * (tf * (_BM25_K1 + 1)) / max(denom, 1e-9)
            if score <= 0:
                continue
            scored.append((score, idx))

        if not scored:
            return []

        scored.sort(key=lambda item: item[0], reverse=True)
        # 정확매칭(조문번호/사건번호/법령명) 부스트는 상위 후보에만 적용 (비용 절감)
        prelim = scored[: max(k * 4, k)]

        boosted: list[tuple[float, Document]] = []
        for base_score, idx in prelim:
            doc = Document(
                page_content=self.contents[idx],
                metadata=dict(self.metadatas[idx]),
            )
            final_score = base_score + _exact_legal_boost(query, doc)
            doc.metadata["lexical_score"] = final_score
            boosted.append((final_score, doc))

        boosted.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in boosted[:k]]


_indexes: dict[str, _NamespaceBM25] = {}
_lock = threading.Lock()


def _get_namespace_index(namespace: str) -> _NamespaceBM25:
    idx = _indexes.get(namespace)
    if idx is not None:
        return idx
    with _lock:
        idx = _indexes.get(namespace)
        if idx is None:
            idx = _NamespaceBM25(namespace)
            idx.load(Path(settings.BM25_CORPUS_DIR) / f"{namespace}.jsonl")
            _indexes[namespace] = idx
    return idx


def corpus_bm25_search(query: str, namespace: str, k: int) -> list[Document]:
    """namespace 전체 코퍼스에서 BM25 lexical 검색.

    아티팩트가 없으면 [] 반환 → 호출부는 기존 dense 동작으로 폴백.
    반환 Document는 PineconeAdapter.search()와 동일한 page_content/metadata
    구조이므로 _rrf_fuse_documents의 _document_key가 dense 결과와 정합된다.
    """
    if not settings.ENABLE_CORPUS_BM25:
        return []
    try:
        return _get_namespace_index(namespace).search(query, k)
    except Exception as exc:  # 어떤 경우에도 검색 파이프라인을 막지 않음
        logger.warning(f"[CorpusBM25] search 실패 namespace={namespace}: {exc}")
        return []
