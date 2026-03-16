"""
Qdrant VectorDB 어댑터 (개발/로컬 환경)
"""
from langchain_core.documents import Document
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


class QdrantAdapter:
    """QdrantClient를 VectorDB 인터페이스로 감싸는 어댑터."""

    def __init__(self, url: str, api_key: str | None = None, timeout: int = 60):
        self._client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
        logger.info(f"QdrantAdapter connected: {url}")

    def search(
        self,
        query_vector: list[float],
        namespace: str,
        k: int = 4,
        filter_dict: dict | None = None,
        score_threshold: float = 0.0,
    ) -> list[Document]:
        search_filter = None
        if filter_dict:
            conditions = [
                FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                for key, value in filter_dict.items()
            ]
            search_filter = Filter(must=conditions)

        response = self._client.query_points(
            collection_name=namespace,
            query=query_vector,
            limit=k,
            query_filter=search_filter,
            score_threshold=score_threshold,
        )

        documents: list[Document] = []
        for point in response.points:
            metadata = point.payload.get("metadata", {})
            metadata["score"] = point.score
            metadata["collection"] = namespace
            documents.append(
                Document(page_content=point.payload.get("content", ""), metadata=metadata)
            )
        return documents

    # 데이터 로딩용 (개발 시 사용)
    @property
    def raw_client(self) -> QdrantClient:
        return self._client
