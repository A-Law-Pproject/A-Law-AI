import asyncio

from langchain_core.documents import Document

from app.rag.embedding.kure import KUREEmbeddings
from app.rag.vector_store.base import VectorDB


def search_collection(
    db: VectorDB,
    embeddings: KUREEmbeddings,
    query: str,
    collection_name: str,
    k: int = 4,
    filter_dict: dict | None = None,
    score_threshold: float = 0.0,
    query_vector: list[float] | None = None,
) -> list[Document]:
    """단일 컬렉션(namespace)에서 유사도 검색.

    Args:
        db: VectorDB 인스턴스 (QdrantAdapter 또는 PineconeAdapter).
        embeddings: 임베딩 모델.
        query: 검색 쿼리.
        collection_name: 컬렉션/namespace 이름.
        k: 반환할 문서 개수.
        filter_dict: 메타데이터 필터.
        score_threshold: 최소 유사도 점수.
        query_vector: 미리 계산된 쿼리 벡터. 전달 시 embed_query 생략.

    Returns:
        검색된 Document 리스트 (metadata에 score, collection 포함).
    """
    if query_vector is None:
        query_vector = embeddings.embed_query(query)

    return db.search(query_vector, collection_name, k, filter_dict, score_threshold)


def search_multi_index(
    db: VectorDB,
    embeddings: KUREEmbeddings,
    query: str,
    collections: list[str],
    k_per_collection: int = 3,
    score_threshold: float = 0.0,
) -> list[Document]:
    """여러 컬렉션에서 검색 후 score 기준 통합 정렬.

    embed_query는 1회만 호출하고 모든 컬렉션 검색에 재사용한다.

    Returns:
        score 내림차순 정렬된 Document 리스트.
    """
    query_vector = embeddings.embed_query(query)
    all_results: list[Document] = []

    for coll in collections:
        results = search_collection(
            db, embeddings, query, coll,
            k=k_per_collection,
            score_threshold=score_threshold,
            query_vector=query_vector,
        )
        all_results.extend(results)

    all_results.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)
    return all_results


async def async_search_multi_index(
    db: VectorDB,
    embeddings: KUREEmbeddings,
    query: str,
    collections: list[str],
    k_per_collection: int = 3,
    score_threshold: float = 0.0,
) -> list[Document]:
    """여러 컬렉션을 asyncio.gather로 병렬 검색 후 score 기준 통합 정렬.

    embed_query는 1회만 호출하고, 컬렉션 검색은 병렬로 실행한다.

    Returns:
        score 내림차순 정렬된 Document 리스트.
    """
    query_vector = await asyncio.to_thread(embeddings.embed_query, query)

    tasks = [
        asyncio.to_thread(
            search_collection,
            db, embeddings, query, coll,
            k_per_collection, None, score_threshold, query_vector,
        )
        for coll in collections
    ]
    results_per_collection: list[list[Document]] = await asyncio.gather(*tasks)

    all_results: list[Document] = [
        doc for docs in results_per_collection for doc in docs
    ]
    all_results.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)
    return all_results
