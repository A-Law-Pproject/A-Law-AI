import sys
import types

from langchain_core.documents import Document

if "sentence_transformers" not in sys.modules:
    stub_module = types.ModuleType("sentence_transformers")

    class _DummySentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

    stub_module.SentenceTransformer = _DummySentenceTransformer
    sys.modules["sentence_transformers"] = stub_module

from app.rag.retriever import multi_retriever


class _FakeVectorDB:
    def __init__(self, documents: list[Document]):
        self._documents = documents
        self.last_k = 0

    def search(
        self,
        query_vector,
        namespace,
        k=4,
        filter_dict=None,
        score_threshold=0.0,
        sparse_vector=None,
    ):
        self.last_k = k
        return [
            Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            for doc in self._documents[:k]
        ]


class _FakeEmbeddings:
    def embed_query(self, query: str):
        return [0.0]


def test_search_collection_hybrid_rrf_promotes_exact_article_match(monkeypatch):
    documents = [
        Document(
            page_content="주택임대차보호법 제8조 우선변제권 관련 설명",
            metadata={
                "id": "doc-8",
                "collection": "law_statutes",
                "law_name": "주택임대차보호법",
                "article": "제8조",
                "score": 0.95,
            },
        ),
        Document(
            page_content="주택임대차보호법 제3조 임대차기간은 2년으로 본다.",
            metadata={
                "id": "doc-3",
                "collection": "law_statutes",
                "law_name": "주택임대차보호법",
                "article": "제3조",
                "score": 0.90,
            },
        ),
        Document(
            page_content="주택임대차보호법 제4조 차임 증감 청구 관련 규정",
            metadata={
                "id": "doc-4",
                "collection": "law_statutes",
                "law_name": "주택임대차보호법",
                "article": "제4조",
                "score": 0.88,
            },
        ),
    ]

    db = _FakeVectorDB(documents)
    embeddings = _FakeEmbeddings()

    monkeypatch.setattr(multi_retriever.settings, "ENABLE_HYBRID_SEARCH", True)
    monkeypatch.setattr(multi_retriever.settings, "HYBRID_RRF_K", 20)
    monkeypatch.setattr(multi_retriever.settings, "HYBRID_DENSE_CANDIDATE_MULTIPLIER", 3)
    monkeypatch.setattr(multi_retriever.settings, "HYBRID_LEXICAL_CANDIDATE_MULTIPLIER", 2)

    results = multi_retriever.search_collection(
        db,
        embeddings,
        "주택임대차보호법 제3조 내용이 뭐야",
        "law_statutes",
        k=2,
        query_vector=[0.0],
    )

    assert db.last_k > 2
    assert results[0].metadata["article"] == "제3조"
    assert "bm25" in results[0].metadata["retrieval_modes"]
    assert results[0].metadata["rrf_score"] >= results[1].metadata["rrf_score"]
