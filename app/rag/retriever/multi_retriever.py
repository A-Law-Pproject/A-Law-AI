import asyncio
import re

from langchain_core.documents import Document

from app.rag.embedding.kure import KUREEmbeddings
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


_LAW_NAME_KEYWORDS = {
    "주택임대차계약증서의 확정일자 부여 및 정보제공에 관한 규칙": "주택임대차계약증서의 확정일자 부여 및 정보제공에 관한 규칙",
    "부동산 거래신고 등에 관한 법률": "부동산 거래신고 등에 관한 법률",
    "집합건물의 소유 및 관리에 관한 법률": "집합건물의 소유 및 관리에 관한 법률",
    "임차권등기명령 절차에 관한 규칙": "임차권등기명령 절차에 관한 규칙",
    "종합부동산세법 시행령": "종합부동산세법 시행령",
    "공동주택관리법 시행규칙": "공동주택관리법 시행규칙",
    "공동주택관리법 시행령": "공동주택관리법 시행령",
    "공동주택관리법": "공동주택관리법",
    "공공주택 특별법 시행령": "공공주택 특별법 시행령",
    "공공주택 특별법": "공공주택 특별법",
    "공인중개사법 시행규칙": "공인중개사법 시행규칙",
    "공인중개사법 시행령": "공인중개사법 시행령",
    "공인중개사법": "공인중개사법",
    "상가건물 임대차보호법 시행령": "상가건물 임대차보호법 시행령",
    "상가건물 임대차보호법": "상가건물 임대차보호법",
    "주택임대차보호법 시행령": "주택임대차보호법 시행령",
    "주택임대차보호법": "주택임대차보호법",
    "주택공급에 관한 규칙": "주택공급에 관한 규칙",
    "민사집행법": "민사집행법",
    "국세기본법": "국세기본법",
    "지방세법": "지방세법",
    "민법": "민법",
}

_SOURCE_DIR_KEYWORDS = {
    "주택임대차": ["주택임대차", "임대차", "임차권", "확정일자", "전세", "월세", "보증금"],
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


def infer_law_statutes_filter(query: str) -> dict | None:
    """질의에서 law_statutes namespace용 메타데이터 필터를 추론."""
    normalized = re.sub(r"\s+", " ", query).strip()
    clauses: list[dict] = []

    matched_law_name = None
    for keyword, law_name in sorted(_LAW_NAME_KEYWORDS.items(), key=lambda item: len(item[0]), reverse=True):
        if keyword in normalized:
            matched_law_name = law_name
            break

    if matched_law_name:
        clauses.append({"law_name": matched_law_name})

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
    return db.search(query_vector, collection_name, k, filter_dict, threshold)


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
