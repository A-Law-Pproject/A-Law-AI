"""
Pinecone VectorDB 어댑터 (배포/프로덕션 환경)

Pinecone 구조:
- 인덱스 1개: PINECONE_INDEX (예: "alaw-legal")
- namespace = 컬렉션 구분자
  - law_database
  - contracts
  - special_clauses_illegal
  - special_clauses_normal

메타데이터 저장 포맷:
  {"content": "...", "category": "...", "type": "...", ...}  (flat)

KURE-v1 임베딩 차원: 1024
Pinecone 인덱스 생성 시 dimension=1024, metric="cosine" 으로 설정할 것
"""
from langchain_core.documents import Document
from loguru import logger


class PineconeAdapter:
    """Pinecone을 VectorDB 인터페이스로 감싸는 어댑터."""

    def __init__(self, api_key: str, index_name: str):
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError("pinecone 패키지가 필요합니다: pip install pinecone-client")

        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(index_name)
        self._index_name = index_name
        logger.info(f"PineconeAdapter connected: index={index_name}")

    def search(
        self,
        query_vector: list[float],
        namespace: str,
        k: int = 4,
        filter_dict: dict | None = None,
        score_threshold: float = 0.0,
        sparse_vector: dict | None = None,
    ) -> list[Document]:
        query_kwargs = {
            "vector": query_vector,
            "top_k": k,
            "namespace": namespace,
            "include_metadata": True,
            "filter": filter_dict,
        }
        if sparse_vector:
            query_kwargs["sparse_vector"] = sparse_vector

        response = self._index.query(**query_kwargs)

        documents: list[Document] = []
        for match in response.matches:
            if match.score < score_threshold:
                continue

            raw_meta = match.metadata or {}
            content = raw_meta.pop("content", "")
            metadata = dict(raw_meta)
            metadata["id"] = getattr(match, "id", metadata.get("id", ""))
            metadata["score"] = match.score
            metadata["collection"] = namespace

            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def upsert(
        self,
        namespace: str,
        vectors: list[dict],
    ) -> None:
        """데이터 로딩용 upsert.

        Args:
            namespace: 컬렉션 이름.
            vectors: [{"id": str, "values": list[float], "metadata": dict}, ...]
        """
        self._index.upsert(vectors=vectors, namespace=namespace)
