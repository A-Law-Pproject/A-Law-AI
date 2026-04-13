import threading

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from loguru import logger

from app.monitoring.metrics import EMBEDDING_LATENCY


class KUREEmbeddings(Embeddings):
    """KURE-v1 한국어 임베딩 모델 (LangChain Embeddings 인터페이스 준수).

    langchain_core.embeddings.Embeddings를 상속하므로 LangServe,
    LangChain 파이프라인, LCEL 등 LangChain 생태계와 완전 호환된다.

    SentenceTransformer는 내부 PyTorch/Rust 바인딩이 멀티스레드 동시 접근에 안전하지
    않으므로 (_lock으로 직렬화) asyncio.to_thread로 병렬 호출해도 충돌하지 않는다.
    """

    def __init__(self, model_name: str = "nlpai-lab/KURE-v1"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self._lock = threading.Lock()
        logger.info(f"KUREEmbeddings initialized: {model_name} (dim={self.dimension})")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            with EMBEDDING_LATENCY.time():
                return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            with EMBEDDING_LATENCY.time():
                return self.model.encode(text, normalize_embeddings=True).tolist()


_embeddings_instance: KUREEmbeddings | None = None


def get_embeddings(model_name: str = "nlpai-lab/KURE-v1") -> KUREEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = KUREEmbeddings(model_name)
    return _embeddings_instance
