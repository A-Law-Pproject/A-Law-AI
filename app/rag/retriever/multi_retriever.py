import asyncio
import math
import re
from collections import Counter

from langchain_core.documents import Document
from loguru import logger

from app.core.config import settings
from app.rag.embedding.kure import KUREEmbeddings
from app.rag.retriever.law_graph import build_multi_law_filter, get_related_laws
from app.rag.vector_store.base import VectorDB


def _resolve_threshold(score_threshold: float | dict[str, float], collection: str) -> float:
    """컬렉션별 threshold 해소.

    Args:
        score_threshold: 단일 float 또는 {컬렉션명: float} 딕셔너리.
        collection: 조회할 컬렉션 이름.

    Returns:
        해당 컬렉션의 threshold. dict에 없으면 "default" 키 → 0.0 순으로 fallback.
    """
    if isinstance(score_threshold, dict):
        return score_threshold.get(collection, score_threshold.get("default", 0.0))
    return score_threshold


def _resolve_k(k_per_collection: int | dict[str, int], collection: str) -> int:
    """컬렉션별 top-k 해소."""
    if isinstance(k_per_collection, dict):
        return k_per_collection.get(collection, k_per_collection.get("default", 3))
    return k_per_collection


def _deduplicate(documents: list[Document]) -> list[Document]:
    """page_content 앞 100자 기준 중복 문서 제거 (첫 등장 유지).

    멀티 컬렉션 검색 시 동일 내용이 다른 namespace에 중복 저장된 경우를 처리한다.
    """
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in documents:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


_HYBRID_TARGET_COLLECTIONS = {
    "law_database",
    "law_statutes",
    "special_clauses_illegal",
    "special_clauses_normal",
}
_ARTICLE_RE = re.compile(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?")
_CASE_RE = re.compile(r"\d{4}[가-힣]{1,4}\d+")
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z0-9]+)*|\d+|[가-힣]{2,}")


def _normalize_article_token(text: str | None) -> str:
    match = _ARTICLE_RE.search(text or "")
    if not match:
        return ""
    return f"제{match.group(1)}조" + (f"의{match.group(2)}" if match.group(2) else "")


def _normalize_case_token(text: str | None) -> str:
    match = _CASE_RE.search(text or "")
    return re.sub(r"\s+", "", match.group(0)) if match else ""


def _compact_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _compact_no_space(text: str | None) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def _document_key(doc: Document) -> str:
    meta = doc.metadata
    stable_bits = [
        str(meta.get("collection") or ""),
        str(meta.get("id") or meta.get("doc_id") or meta.get("chunk_id") or meta.get("case_no") or ""),
        str(meta.get("law_name") or meta.get("title") or ""),
        str(meta.get("article") or ""),
        doc.page_content[:160],
    ]
    return "|".join(stable_bits)


def _copy_document(doc: Document) -> Document:
    return Document(page_content=doc.page_content, metadata=dict(doc.metadata))


def _lexical_source_text(doc: Document) -> str:
    meta = doc.metadata
    return "\n".join(
        part
        for part in [
            meta.get("law_name"),
            meta.get("title"),
            meta.get("article"),
            meta.get("case_no"),
            doc.page_content,
        ]
        if part
    )


def _tokenize_lexical(text: str) -> list[str]:
    normalized = _compact_text(text)
    if not normalized:
        return []

    article_tokens = [_normalize_article_token(match.group(0)) for match in _ARTICLE_RE.finditer(normalized)]
    case_tokens = [_normalize_case_token(match.group(0)) for match in _CASE_RE.finditer(normalized)]

    stripped = _ARTICLE_RE.sub(" ", normalized)
    stripped = _CASE_RE.sub(" ", stripped)
    tokens = [token.lower() for token in _TOKEN_RE.findall(stripped)]
    return [token for token in [*article_tokens, *case_tokens, *tokens] if token]


def _exact_legal_boost(query: str, doc: Document) -> float:
    boost = 0.0
    meta = doc.metadata

    query_articles = {
        normalized
        for normalized in (_normalize_article_token(match.group(0)) for match in _ARTICLE_RE.finditer(query or ""))
        if normalized
    }
    doc_article = _normalize_article_token(str(meta.get("article") or ""))
    page_article = _normalize_article_token(doc.page_content[:120])
    if query_articles:
        if doc_article and doc_article in query_articles:
            boost += 7.0
        elif page_article and page_article in query_articles:
            boost += 4.0

    query_cases = {
        normalized
        for normalized in (_normalize_case_token(match.group(0)) for match in _CASE_RE.finditer(query or ""))
        if normalized
    }
    doc_case = _normalize_case_token(str(meta.get("case_no") or ""))
    page_case = _normalize_case_token(doc.page_content[:160])
    if query_cases:
        if doc_case and doc_case in query_cases:
            boost += 8.0
        elif page_case and page_case in query_cases:
            boost += 4.5

    query_compact = _compact_no_space(query)
    law_name = _compact_no_space(str(meta.get("law_name") or meta.get("title") or ""))
    if law_name and law_name in query_compact:
        boost += 2.0

    if query_compact and query_compact in _compact_no_space(doc.page_content[:220]):
        boost += 1.5

    return boost


def _bm25_rank_candidates(
    query: str,
    candidates: list[Document],
    collection_name: str,
    top_k: int,
) -> list[Document]:
    if not candidates or top_k <= 0:
        return []

    query_tokens = _tokenize_lexical(query)
    if not query_tokens:
        return []

    doc_tokens = [_tokenize_lexical(_lexical_source_text(doc)) for doc in candidates]
    doc_counters = [Counter(tokens) for tokens in doc_tokens]
    doc_lengths = [max(len(tokens), 1) for tokens in doc_tokens]
    avgdl = sum(doc_lengths) / len(doc_lengths)

    document_frequency: Counter[str] = Counter()
    for tokens in doc_tokens:
        document_frequency.update(set(tokens))

    scored: list[tuple[float, Document]] = []
    total_docs = len(candidates)
    k1 = 1.5
    b = 0.75

    for doc, token_counter, doc_len in zip(candidates, doc_counters, doc_lengths):
        score = 0.0
        for token in query_tokens:
            tf = token_counter.get(token, 0)
            if tf <= 0:
                continue
            df = document_frequency.get(token, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1 - b + b * (doc_len / max(avgdl, 1.0)))
            score += idf * (tf * (k1 + 1)) / max(denominator, 1e-9)

        score += _exact_legal_boost(query, doc)
        if score <= 0:
            continue

        ranked_doc = _copy_document(doc)
        ranked_doc.metadata["collection"] = collection_name
        ranked_doc.metadata["lexical_score"] = score
        scored.append((score, ranked_doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def _rrf_fuse_documents(
    ranked_lists: dict[str, list[Document]],
    *,
    final_k: int,
    rrf_k: int,
) -> list[Document]:
    if not ranked_lists:
        return []

    merged_scores: dict[str, float] = {}
    merged_docs: dict[str, Document] = {}
    merged_modes: dict[str, set[str]] = {}

    for mode, docs in ranked_lists.items():
        for rank, doc in enumerate(docs, start=1):
            key = _document_key(doc)
            merged_scores[key] = merged_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            merged_modes.setdefault(key, set()).add(mode)
            if key not in merged_docs:
                merged_docs[key] = _copy_document(doc)
            if mode == "dense":
                merged_docs[key].metadata["dense_score"] = doc.metadata.get("score", 0.0)
            if mode == "bm25":
                merged_docs[key].metadata["lexical_score"] = doc.metadata.get("lexical_score", 0.0)

    fused: list[Document] = []
    for key, doc in merged_docs.items():
        doc.metadata["rrf_score"] = merged_scores[key]
        doc.metadata["score"] = merged_scores[key]
        doc.metadata["retrieval_modes"] = sorted(merged_modes.get(key, []))
        fused.append(doc)

    fused.sort(
        key=lambda item: (
            item.metadata.get("rrf_score", 0.0),
            item.metadata.get("lexical_score", 0.0),
            item.metadata.get("dense_score", 0.0),
        ),
        reverse=True,
    )
    return fused[:final_k]


def _should_use_hybrid_search(collection_name: str, query: str) -> bool:
    return bool(
        settings.ENABLE_HYBRID_SEARCH
        and collection_name in _HYBRID_TARGET_COLLECTIONS
        and _compact_text(query)
    )


def _matches_filter(meta: dict, filter_dict: dict | None) -> bool:
    """Pinecone 메타데이터 필터를 Python에서 검사.

    infer_law_statutes_filter / build_multi_law_filter / _merge_filter_dict가
    생성하는 구조($and, $or, {key: value} 동등비교)만 지원한다. 코퍼스 BM25가
    dense와 동일한 후보 범위를 유지하도록 해 원래 버그(엉뚱한 법령 회수) 재발을 막는다.
    알 수 없는 연산자는 통과(True) 처리해 과도한 배제를 피한다.
    """
    if not filter_dict:
        return True
    for key, value in filter_dict.items():
        if key == "$and":
            if not all(_matches_filter(meta, sub) for sub in value):
                return False
        elif key == "$or":
            if not any(_matches_filter(meta, sub) for sub in value):
                return False
        elif key.startswith("$"):
            continue  # 미지원 연산자는 통과
        else:
            mv = meta.get(key)
            if isinstance(value, dict):
                if "$eq" in value and mv != value["$eq"]:
                    return False
                if "$ne" in value and mv == value["$ne"]:
                    return False
                if "$in" in value and mv not in value["$in"]:
                    return False
            elif mv != value:
                return False
    return True


_LAW_NAME_KEYWORDS = {
    # 주택 임대차 관련 (가장 우선 매칭 — 길이 내림차순 정렬로 인해 긴 이름이 먼저 매칭됨)
    "주택임대차계약증서의 확정일자 부여 및 정보제공에 관한 규칙": "주택임대차계약증서의 확정일자 부여 및 정보제공에 관한 규칙",
    "주택임대차보호법 시행령": "주택임대차보호법 시행령",
    "주택임대차보호법": "주택임대차보호법",
    # 상가 임대차 관련
    "상가건물 임대차보호법 시행령": "상가건물 임대차보호법 시행령",
    "상가건물 임대차보호법": "상가건물 임대차보호법",
    # 민간임대주택 관련 (data/raw/학습법률문서/민간임대주택/ 소스)
    "민간임대주택에 관한 특별법 시행규칙": "민간임대주택에 관한 특별법 시행규칙",
    "민간임대주택에 관한 특별법 시행령": "민간임대주택에 관한 특별법 시행령",
    "민간임대주택에 관한 특별법": "민간임대주택에 관한 특별법",
    # 전세사기 관련 (data/raw/학습법률문서/전세사기/ 소스)
    "전세사기피해자 지원 및 주거안정에 관한 특별법": "전세사기피해자 지원 및 주거안정에 관한 특별법",
    # 부동산 거래 관련
    "부동산 거래신고 등에 관한 법률 시행령": "부동산 거래신고 등에 관한 법률 시행령",
    "부동산 거래신고 등에 관한 법률": "부동산 거래신고 등에 관한 법률",
    # 집합건물·공동주택 관련
    "집합건물의 소유 및 관리에 관한 법률": "집합건물의 소유 및 관리에 관한 법률",
    "공동주택관리법 시행규칙": "공동주택관리법 시행규칙",
    "공동주택관리법 시행령": "공동주택관리법 시행령",
    "공동주택관리법": "공동주택관리법",
    "공공주택 특별법 시행령": "공공주택 특별법 시행령",
    "공공주택 특별법": "공공주택 특별법",
    # 중개 관련
    "공인중개사법 시행규칙": "공인중개사법 시행규칙",
    "공인중개사법 시행령": "공인중개사법 시행령",
    "공인중개사법": "공인중개사법",
    # 절차·등기 관련
    "임차권등기명령 절차에 관한 규칙": "임차권등기명령 절차에 관한 규칙",
    "주택공급에 관한 규칙": "주택공급에 관한 규칙",
    "민사집행법": "민사집행법",
    # 세금 관련
    "종합부동산세법 시행령": "종합부동산세법 시행령",
    "국세기본법": "국세기본법",
    "지방세법": "지방세법",
    # 민법 (가장 범용 — 마지막에 매칭)
    "민법": "민법",
}

_SOURCE_DIR_KEYWORDS = {
    "주택임대차": [
        # 기본 임대차 용어
        "주택임대차", "임대차", "임차권", "확정일자", "전세", "월세", "보증금",
        # 법적 권리·절차 — 이 키워드들이 포함된 질의는 주택임대차 법령으로 라우팅
        "전입신고", "주민등록전입", "대항력", "우선변제권", "최우선변제",
        "소액임차인", "임차권등기", "묵시적갱신", "계약갱신청구권", "갱신청구",
        # 분쟁·종료 관련
        "명도", "퇴거", "보증금반환", "반환청구", "경매", "강제집행",
        "임대인", "임차인", "전대", "전대차",
        # 전세사기 관련 — 이 키워드도 주택임대차 법령으로 라우팅
        "전세사기", "전세사기피해", "깡통전세",
        # 민간임대주택 관련
        "민간임대주택", "임대사업자", "민간임대",
    ],
    "상가건물": ["상가", "상가건물", "점포", "상업용"],
    "공동주택": ["공동주택", "공공주택", "아파트", "관리비", "입주자대표회의"],
    "공인중개사법": ["공인중개사", "중개사", "중개보수", "중개업", "중개대상물"],
}


def _merge_filter_dict(base_filter: dict | None, extra_filter: dict | None) -> dict | None:
    if not base_filter:
        return extra_filter
    if not extra_filter:
        return base_filter
    return {"$and": [base_filter, extra_filter]}


def infer_law_statutes_filter(query: str, expand_related: bool = True) -> dict | None:
    """질의에서 law_statutes namespace용 메타데이터 필터를 추론.

    Args:
        query: 사용자 질의 또는 조항 텍스트.
        expand_related: True이면 law_graph를 이용해 관련 법령도 OR 필터로 확장.
            시행령/시행규칙 타입 필터가 적용되는 경우엔 확장하지 않는다.

    Returns:
        Pinecone 메타데이터 필터 딕셔너리 또는 None.
    """
    normalized = re.sub(r"\s+", " ", query).strip()
    clauses: list[dict] = []

    matched_law_name = None
    for keyword, law_name in sorted(_LAW_NAME_KEYWORDS.items(), key=lambda item: len(item[0]), reverse=True):
        if keyword in normalized:
            matched_law_name = law_name
            break

    # 시행령/시행규칙 타입 필터 여부 확인 (관련법 확장과 함께 쓰면 노이즈 발생)
    has_law_type_filter = "시행규칙" in normalized or "시행령" in normalized or "규칙" in normalized

    if matched_law_name:
        if expand_related and not has_law_type_filter:
            related = get_related_laws(matched_law_name, query)
            law_filter = build_multi_law_filter(matched_law_name, related)
        else:
            law_filter = {"law_name": matched_law_name}
        clauses.append(law_filter)

    matched_source_dir = None
    for source_dir, keywords in _SOURCE_DIR_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            matched_source_dir = source_dir
            break

    if matched_source_dir and not matched_law_name:
        clauses.append({"source_dir": matched_source_dir})

    if "시행규칙" in normalized:
        clauses.append({"law_type": "enforcement_rule"})
    elif "시행령" in normalized:
        clauses.append({"law_type": "enforcement_decree"})
    elif "규칙" in normalized:
        clauses.append({"$or": [{"law_type": "rule"}, {"law_type": "enforcement_rule"}]})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def search_collection(
    db: VectorDB,
    embeddings: KUREEmbeddings,
    query: str,
    collection_name: str,
    k: int = 4,
    filter_dict: dict | None = None,
    score_threshold: float | dict[str, float] = 0.0,
    query_vector: list[float] | None = None,
    sparse_vector: dict | None = None,
) -> list[Document]:
    """단일 컬렉션(namespace)에서 유사도 검색.

    Args:
        db: VectorDB 인스턴스 (PineconeAdapter).
        embeddings: 임베딩 모델.
        query: 검색 쿼리.
        collection_name: 컬렉션/namespace 이름.
        k: 반환할 문서 개수.
        filter_dict: 메타데이터 필터.
        score_threshold: 최소 유사도 점수. float 또는 {컬렉션명: float} 딕셔너리.
        query_vector: 미리 계산된 쿼리 벡터. 전달 시 embed_query 생략.

    Returns:
        검색된 Document 리스트 (metadata에 score, collection 포함).
    """
    if query_vector is None:
        query_vector = embeddings.embed_query(query)

    threshold = _resolve_threshold(score_threshold, collection_name)
    dense_k = k
    if _should_use_hybrid_search(collection_name, query):
        dense_k = max(k, k * settings.HYBRID_DENSE_CANDIDATE_MULTIPLIER)

    dense_results = db.search(
        query_vector,
        collection_name,
        dense_k,
        filter_dict,
        threshold,
        sparse_vector,
    )
    dense_results.sort(key=lambda doc: doc.metadata.get("score", 0.0), reverse=True)

    if not _should_use_hybrid_search(collection_name, query):
        return dense_results[:k]

    ranked_lists: dict[str, list[Document]] = {}
    if dense_results:
        ranked_lists["dense"] = dense_results

    # dense 후보 풀 내부 재정렬 BM25 (기존 동작)
    if len(dense_results) > 1:
        lexical_k = max(k, k * settings.HYBRID_LEXICAL_CANDIDATE_MULTIPLIER)
        lexical_results = _bm25_rank_candidates(
            query,
            dense_results,
            collection_name,
            min(lexical_k, len(dense_results)),
        )
        if lexical_results:
            ranked_lists["bm25"] = lexical_results

    # 코퍼스 전체 BM25 — dense가 놓친 문서를 직접 회수 (아티팩트 없으면 [] → 무회귀)
    from app.rag.retriever.bm25_index import corpus_bm25_search

    corpus_k = max(k, k * settings.HYBRID_CORPUS_BM25_MULTIPLIER)
    corpus_results = corpus_bm25_search(query, collection_name, corpus_k)
    if corpus_results and filter_dict:
        corpus_results = [
            doc for doc in corpus_results if _matches_filter(doc.metadata, filter_dict)
        ]
    if corpus_results:
        ranked_lists["corpus_bm25"] = corpus_results

    # 융합할 lexical 신호가 전혀 없으면 기존 dense 동작 유지 (무회귀)
    if set(ranked_lists) <= {"dense"}:
        return dense_results[:k]

    fused_results = _rrf_fuse_documents(
        ranked_lists,
        final_k=k,
        rrf_k=settings.HYBRID_RRF_K,
    )
    if not fused_results:
        logger.debug("[HybridSearch] RRF fusion empty for collection={}", collection_name)
        return dense_results[:k]
    return fused_results


def search_multi_index(
    db: VectorDB,
    embeddings: KUREEmbeddings,
    query: str,
    collections: list[str],
    k_per_collection: int | dict[str, int] = 3,
    score_threshold: float | dict[str, float] = 0.0,
    collection_filters: dict[str, dict] | None = None,
    reranker=None,
    rerank_top_n: int | None = None,
    query_vector: list[float] | None = None,
) -> list[Document]:
    """여러 컬렉션에서 검색 후 중복 제거 → (reranker 있으면) 재정렬.

    embed_query는 1회만 호출하고 모든 컬렉션 검색에 재사용한다.

    Args:
        k_per_collection: 단일 int 또는 {컬렉션명: k} 딕셔너리.
        score_threshold: float 또는 {컬렉션명: float} 딕셔너리.
            딕셔너리 사용 예::

                {
                    "law_database": 0.4,
                    "special_clauses_illegal": 0.6,
                    "default": 0.3,
                }

        reranker: BGEReranker 인스턴스 (선택). 전달 시 재정렬 수행.
        rerank_top_n: reranker 적용 후 반환할 문서 수. None이면 전체 반환.

    Returns:
        reranker 있으면 rerank_score 내림차순, 없으면 score 내림차순 Document 리스트.
    """
    if query_vector is None:
        query_vector = embeddings.embed_query(query)
    all_results: list[Document] = []

    for coll in collections:
        filter_dict = collection_filters.get(coll) if collection_filters else None
        k = _resolve_k(k_per_collection, coll)
        results = search_collection(
            db, embeddings, query, coll,
            k=k,
            filter_dict=filter_dict,
            score_threshold=score_threshold,
            query_vector=query_vector,
        )
        all_results.extend(results)

    all_results = _deduplicate(all_results)
    all_results.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)

    if reranker is not None:
        all_results = reranker.rerank(query, all_results, top_n=rerank_top_n)

    return all_results


async def async_search_multi_index(
    db: VectorDB,
    embeddings: KUREEmbeddings,
    query: str,
    collections: list[str],
    k_per_collection: int | dict[str, int] = 3,
    score_threshold: float | dict[str, float] = 0.0,
    collection_filters: dict[str, dict] | None = None,
    reranker=None,
    rerank_top_n: int | None = None,
    query_vector: list[float] | None = None,
) -> list[Document]:
    """여러 컬렉션을 asyncio.gather로 병렬 검색 후 중복 제거 → (reranker 있으면) 재정렬.

    embed_query는 1회만 호출하고, 컬렉션 검색은 병렬로 실행한다.

    Args:
        k_per_collection: 단일 int 또는 {컬렉션명: k} 딕셔너리.
        score_threshold: float 또는 {컬렉션명: float} 딕셔너리.
        reranker: BGEReranker 인스턴스 (선택).
        rerank_top_n: reranker 적용 후 반환할 문서 수.

    Returns:
        reranker 있으면 rerank_score 내림차순, 없으면 score 내림차순 Document 리스트.
    """
    if query_vector is None:
        query_vector = await asyncio.to_thread(embeddings.embed_query, query)

    tasks = [
        asyncio.to_thread(
            search_collection,
            db, embeddings, query, coll,
            _resolve_k(k_per_collection, coll),
            collection_filters.get(coll) if collection_filters else None,
            score_threshold,
            query_vector,
        )
        for coll in collections
    ]
    results_per_collection: list[list[Document]] = await asyncio.gather(*tasks)

    all_results: list[Document] = [
        doc for docs in results_per_collection for doc in docs
    ]
    all_results = _deduplicate(all_results)
    all_results.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)

    if reranker is not None:
        all_results = await reranker.async_rerank(query, all_results, top_n=rerank_top_n)

    return all_results
