"""
VectorDB 추상 인터페이스

QdrantAdapter / PineconeAdapter가 이 프로토콜을 구현한다.
실제 코드는 이 타입만 바라보므로, 환경변수 VECTOR_DB 변경만으로 백엔드를 교체할 수 있다.
"""
from typing import Protocol, runtime_checkable

from langchain_core.documents import Document


@runtime_checkable
class VectorDB(Protocol):
    """벡터 DB 검색 인터페이스."""

    def search(
        self,
        query_vector: list[float],
        namespace: str,
        k: int = 4,
        filter_dict: dict | None = None,
        score_threshold: float = 0.0,
    ) -> list[Document]:
        """
        Args:
            query_vector: 쿼리 임베딩 벡터.
            namespace: 컬렉션 이름 (Qdrant: collection, Pinecone: namespace).
            k: 반환할 문서 수.
            filter_dict: 메타데이터 필터.
            score_threshold: 최소 유사도 점수.

        Returns:
            score, collection 메타데이터가 포함된 Document 리스트.
        """
        ...
