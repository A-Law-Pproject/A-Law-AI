from sentence_transformers import SentenceTransformer
from loguru import logger

from app.monitoring.metrics import EMBEDDING_LATENCY


class KUREEmbeddings:
    """KURE-v1 한국어 임베딩 모델 (LangChain 호환 인터페이스)"""

    def __init__(self, model_name: str = "nlpai-lab/KURE-v1"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"KUREEmbeddings initialized: {model_name} (dim={self.dimension})")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with EMBEDDING_LATENCY.time():
            return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        with EMBEDDING_LATENCY.time():
            return self.model.encode(text, normalize_embeddings=True).tolist()


_embeddings_instance: KUREEmbeddings | None = None


def get_embeddings(model_name: str = "nlpai-lab/KURE-v1") -> KUREEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = KUREEmbeddings(model_name)
    return _embeddings_instance
