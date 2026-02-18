# ============================================================
# [LEGACY] 이 파일은 더 이상 사용되지 않습니다.
# OpenAI 임베딩 + 구 qdrant client.search() API 기반입니다.
# 새로운 구현: app/rag/vector_store/multi_index.py (KURE 임베딩 + query_points API)
# TODO: 안정화 후 삭제 예정
# ============================================================
"""
Qdrant Vector Store 관리 (LEGACY)
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from loguru import logger
from app.core.config import settings


class QdrantVectorStore:
    """Qdrant 벡터 스토어 관리 클래스"""

    def __init__(
        self,
        collection_name: str = None,
        url: str = None,
        api_key: str = None
    ):
        """
        Args:
            collection_name: Qdrant 컬렉션 이름
            url: Qdrant 서버 URL
            api_key: Qdrant API 키
        """
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY

        # Qdrant 클라이언트 초기화
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=60
        )

        # OpenAI Embeddings 초기화
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )

        logger.info(f"Qdrant Vector Store initialized: {self.collection_name}")

    def create_collection(self, vector_size: int = 1536, force: bool = False):
        """
        컬렉션 생성

        Args:
            vector_size: 벡터 차원 (text-embedding-3-small: 1536)
            force: 기존 컬렉션 삭제 후 재생성
        """
        try:
            # 기존 컬렉션 확인
            collections = self.client.get_collections().collections
            collection_exists = any(
                c.name == self.collection_name for c in collections
            )

            if collection_exists:
                if force:
                    logger.warning(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Collection already exists: {self.collection_name}")
                    return

            # 컬렉션 생성
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Collection created: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> int:
        """
        문서 벡터화 및 저장

        Args:
            documents: LangChain Document 리스트
            batch_size: 배치 크기

        Returns:
            저장된 문서 개수
        """
        if not documents:
            logger.warning("No documents to add")
            return 0

        total_added = 0

        try:
            # 배치 단위로 처리
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]

                # 텍스트 추출
                texts = [doc.page_content for doc in batch]

                # 임베딩 생성
                logger.info(f"Embedding batch {i//batch_size + 1} ({len(batch)} docs)")
                vectors = self.embeddings.embed_documents(texts)

                # PointStruct 생성
                points = []
                for idx, (doc, vector) in enumerate(zip(batch, vectors)):
                    point_id = i + idx
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={
                                "content": doc.page_content,
                                "metadata": doc.metadata
                            }
                        )
                    )

                # Qdrant에 저장
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )

                total_added += len(batch)
                logger.info(f"Added {total_added}/{len(documents)} documents")

            logger.info(f"Successfully added {total_added} documents to Qdrant")
            return total_added

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict] = None,
        score_threshold: float = 0.0
    ) -> List[Document]:
        """
        유사도 검색

        Args:
            query: 검색 쿼리
            k: 반환할 문서 개수
            filter_dict: 메타데이터 필터 (예: {"document_type": "주거용부동산임대계약서"})
            score_threshold: 최소 유사도 점수

        Returns:
            검색된 Document 리스트
        """
        try:
            # 쿼리 임베딩
            query_vector = self.embeddings.embed_query(query)

            # 필터 설정
            search_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=value)
                        )
                    )
                search_filter = Filter(must=conditions)

            # 검색 수행
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k,
                query_filter=search_filter,
                score_threshold=score_threshold
            )

            # Document 객체로 변환
            documents = []
            for scored_point in search_result:
                metadata = scored_point.payload.get("metadata", {})
                metadata["score"] = scored_point.score

                doc = Document(
                    page_content=scored_point.payload.get("content", ""),
                    metadata=metadata
                )
                documents.append(doc)

            logger.info(f"Found {len(documents)} documents for query: {query[:50]}...")
            return documents

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def get_collection_info(self) -> Dict:
        """
        컬렉션 정보 조회

        Returns:
            컬렉션 정보 딕셔너리
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def delete_collection(self):
        """컬렉션 삭제"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection deleted: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise


# 싱글톤 인스턴스
_vector_store_instance = None


def get_vector_store() -> QdrantVectorStore:
    """
    벡터 스토어 싱글톤 인스턴스 반환

    Returns:
        QdrantVectorStore 인스턴스
    """
    global _vector_store_instance

    if _vector_store_instance is None:
        _vector_store_instance = QdrantVectorStore()

    return _vector_store_instance
"""
