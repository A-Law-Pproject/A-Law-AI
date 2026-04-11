"""BGE CrossEncoder 기반 Reranker.

멀티 컬렉션 검색 결과를 코사인 유사도 score만으로 정렬하면
컬렉션마다 score 분포가 달라 순위가 왜곡될 수 있다.
CrossEncoder는 (query, document) 쌍을 직접 평가하므로 컬렉션 간 score 차이를 무시하고
실제 관련성 기준으로 재정렬한다.

사용 모델: BAAI/bge-reranker-v2-m3
- 다국어 지원 (한국어 포함)
- sentence-transformers CrossEncoder API 사용
"""
import asyncio
import threading

from langchain_core.documents import Document
from loguru import logger


class BGEReranker:
    """BAAI/bge-reranker-v2-m3 CrossEncoder 재정렬기."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        logger.info(f"BGEReranker initialized: {model_name}")

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        """(query, document) 쌍 점수로 문서를 재정렬.

        Args:
            query: 사용자 질문 원문.
            documents: 1차 검색된 Document 리스트.
            top_n: 반환할 최대 문서 수. None이면 전체 반환.

        Returns:
            rerank_score 내림차순 정렬된 Document 리스트.
            각 Document의 metadata에 "rerank_score" 키가 추가된다.
        """
        if not documents:
            return documents

        pairs = [(query, doc.page_content) for doc in documents]
        scores: list[float] = self._model.predict(pairs).tolist()

        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = score

        reranked = sorted(documents, key=lambda d: d.metadata["rerank_score"], reverse=True)
        return reranked[:top_n] if top_n is not None else reranked

    async def async_rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        """rerank의 비동기 버전 (asyncio.to_thread 래핑)."""
        return await asyncio.to_thread(self.rerank, query, documents, top_n)


_reranker_instance: BGEReranker | None = None
_reranker_lock = threading.Lock()


def get_reranker(model_name: str = "BAAI/bge-reranker-v2-m3") -> BGEReranker:
    global _reranker_instance
    if _reranker_instance is None:
        with _reranker_lock:
            if _reranker_instance is None:
                _reranker_instance = BGEReranker(model_name)
    return _reranker_instance
