from langchain_core.documents import Document
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.rag.embedding.kure import KUREEmbeddings


class MultiIndexStore:
    """Qdrant Multi-Index 벡터 스토어 관리.

    4개 컬렉션 분리 관리:
    - law_database: 약관 + 판결문
    - contracts: 임대차 계약서
    - special_clauses_illegal: 독소 특약사항
    - special_clauses_normal: 정상 특약사항
    """

    def __init__(self, qdrant_url: str, vector_dim: int, timeout: int = 60):
        self.client = QdrantClient(url=qdrant_url, timeout=timeout)
        self.vector_dim = vector_dim
        logger.info(f"MultiIndexStore connected to {qdrant_url} (dim={vector_dim})")

    def create_collections(self, collection_names: list[str]) -> None:
        """컬렉션이 없으면 생성.

        Args:
            collection_names: 생성할 컬렉션 이름 리스트.
        """
        existing = [c.name for c in self.client.get_collections().collections]

        for name in collection_names:
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self.vector_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created collection: {name}")
            else:
                info = self.client.get_collection(name)
                logger.info(f"Collection '{name}' exists (points: {info.points_count})")

    def upload_documents(
        self,
        collection_docs: dict[str, list[Document]],
        embeddings: KUREEmbeddings,
        batch_size: int = 50,
    ) -> dict[str, int]:
        """여러 컬렉션에 문서 배치 업로드.

        Args:
            collection_docs: {컬렉션명: Document 리스트} 딕셔너리.
            embeddings: 임베딩 모델.
            batch_size: 배치 크기.

        Returns:
            {컬렉션명: 업로드된 문서 수} 딕셔너리.
        """
        results: dict[str, int] = {}

        for collection_name, docs in collection_docs.items():
            if not docs:
                logger.warning(f"[{collection_name}] No documents, skipping")
                results[collection_name] = 0
                continue

            logger.info(f"Uploading to '{collection_name}' ({len(docs)} docs)")

            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                texts = [doc.page_content for doc in batch]
                vectors = embeddings.embed_documents(texts)

                points = [
                    PointStruct(
                        id=i + idx,
                        vector=vector,
                        payload={"content": doc.page_content, "metadata": doc.metadata},
                    )
                    for idx, (doc, vector) in enumerate(zip(batch, vectors))
                ]

                self.client.upsert(collection_name=collection_name, points=points)

                batch_num = i // batch_size + 1
                total_batches = (len(docs) - 1) // batch_size + 1
                logger.debug(f"  [{collection_name}] Batch {batch_num}/{total_batches}")

            info = self.client.get_collection(collection_name)
            results[collection_name] = info.points_count
            logger.info(f"  [{collection_name}] Total points: {info.points_count}")

        return results

    def get_collection_info(self, collection_name: str) -> dict:
        """컬렉션 정보 조회."""
        info = self.client.get_collection(collection_name)
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status,
        }

    def delete_collection(self, collection_name: str) -> None:
        """컬렉션 삭제."""
        self.client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
