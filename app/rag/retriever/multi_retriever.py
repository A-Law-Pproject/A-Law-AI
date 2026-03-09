import asyncio

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.rag.embedding.kure import KUREEmbeddings


def search_collection(
    client: QdrantClient,
    embeddings: KUREEmbeddings,
    query: str,
    collection_name: str,
    k: int = 4,
    filter_dict: dict | None = None,
    score_threshold: float = 0.0,
    query_vector: list[float] | None = None,
) -> list[Document]:
    """단일 Qdrant 컬렉션에서 유사도 검색.

    Args:
        client: QdrantClient 인스턴스.
        embeddings: 임베딩 모델.
        query: 검색 쿼리.
        collection_name: 컬렉션 이름.
        k: 반환할 문서 개수.
        filter_dict: 메타데이터 필터 (예: {"category": "보증금"}).
        score_threshold: 최소 유사도 점수.
        query_vector: 미리 계산된 쿼리 벡터. 전달 시 embed_query 호출 생략.

    Returns:
        검색된 Document 리스트 (metadata에 score, collection 포함).
    """
    if query_vector is None:
        query_vector = embeddings.embed_query(query)

    search_filter = None
    if filter_dict:
        conditions = [
            FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
            for key, value in filter_dict.items()
        ]
        search_filter = Filter(must=conditions)

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=k,
        query_filter=search_filter,
        score_threshold=score_threshold,
    )

    documents: list[Document] = []
    for point in response.points:
        metadata = point.payload.get("metadata", {})
        metadata["score"] = point.score
        metadata["collection"] = collection_name
        documents.append(
            Document(page_content=point.payload.get("content", ""), metadata=metadata)
        )
    return documents


def search_multi_index(
    client: QdrantClient,
    embeddings: KUREEmbeddings,
    query: str,
    collections: list[str],
    k_per_collection: int = 3,
    score_threshold: float = 0.0,
) -> list[Document]:
    """여러 컬렉션에서 검색 후 score 기준 통합 정렬.

    embed_query는 1회만 호출하고 모든 컬렉션 검색에 재사용한다.

    Args:
        client: QdrantClient 인스턴스.
        embeddings: 임베딩 모델.
        query: 검색 쿼리.
        collections: 검색할 컬렉션 이름 리스트.
        k_per_collection: 컬렉션당 반환 문서 수.
        score_threshold: 최소 유사도 점수.

    Returns:
        score 내림차순 정렬된 Document 리스트.
    """
    query_vector = embeddings.embed_query(query)
    all_results: list[Document] = []

    for coll in collections:
        results = search_collection(
            client, embeddings, query, coll,
            k=k_per_collection,
            score_threshold=score_threshold,
            query_vector=query_vector,
        )
        all_results.extend(results)

    all_results.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)
    return all_results


async def async_search_multi_index(
    client: QdrantClient,
    embeddings: KUREEmbeddings,
    query: str,
    collections: list[str],
    k_per_collection: int = 3,
    score_threshold: float = 0.0,
) -> list[Document]:
    """여러 컬렉션을 asyncio.gather로 병렬 검색 후 score 기준 통합 정렬.

    embed_query는 1회만 호출하고, 컬렉션 검색은 병렬로 실행한다.

    Args:
        client: QdrantClient 인스턴스.
        embeddings: 임베딩 모델.
        query: 검색 쿼리.
        collections: 검색할 컬렉션 이름 리스트.
        k_per_collection: 컬렉션당 반환 문서 수.
        score_threshold: 최소 유사도 점수.

    Returns:
        score 내림차순 정렬된 Document 리스트.
    """
    # 쿼리 벡터 1회 계산 (CPU 연산이므로 to_thread 사용)
    query_vector = await asyncio.to_thread(embeddings.embed_query, query)

    tasks = [
        asyncio.to_thread(
            search_collection,
            client, embeddings, query, coll,
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
