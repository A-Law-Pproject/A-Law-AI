import openai
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from qdrant_client import QdrantClient

from app.rag.chain.prompts import CONTRACT_QA_PROMPT, RISK_PROMPT
from app.rag.embedding.kure import KUREEmbeddings
from app.rag.retriever.multi_retriever import search_collection, search_multi_index


def build_context(documents: list[Document], max_length: int = 2000) -> str:
    """검색된 문서들로 LLM 컨텍스트 문자열 생성.

    Args:
        documents: 검색된 Document 리스트.
        max_length: 최대 컨텍스트 문자 수.

    Returns:
        포맷된 컨텍스트 문자열.
    """
    context_parts: list[str] = []
    current_length = 0

    for i, doc in enumerate(documents, 1):
        coll = doc.metadata.get("collection", "")
        header = f"[문서 {i} - {coll}]"
        if doc.metadata.get("article"):
            header += f" {doc.metadata['article']}"
        if doc.metadata.get("title"):
            header += f" - {doc.metadata['title']}"
        if doc.metadata.get("category"):
            header += f" ({doc.metadata['category']})"
        header += "\n"

        content = doc.page_content
        part_length = len(header) + len(content)

        if current_length + part_length > max_length:
            remaining = max_length - current_length
            if remaining > 100:
                content = content[:remaining - len(header) - 3] + "..."
                context_parts.append(header + content)
            break

        context_parts.append(header + content)
        current_length += part_length

    return "\n\n".join(context_parts)


def rag_query(
    question: str,
    client: QdrantClient,
    embeddings: KUREEmbeddings,
    llm: ChatOpenAI,
    collections: list[str],
    k_per_collection: int = 3,
) -> dict:
    """Multi-Index RAG 파이프라인.

    Args:
        question: 사용자 질문.
        client: QdrantClient 인스턴스.
        embeddings: 임베딩 모델.
        llm: ChatOpenAI 인스턴스.
        collections: 검색할 컬렉션 리스트.
        k_per_collection: 컬렉션당 검색 수.

    Returns:
        {"answer": str, "source_documents": list, "context": str}
    """
    docs = search_multi_index(
        client, embeddings, question,
        collections=collections,
        k_per_collection=k_per_collection,
    )
    context = build_context(docs)
    prompt_text = CONTRACT_QA_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt_text)

    return {
        "answer": response.content,
        "source_documents": docs,
        "context": context,
    }


def detect_risk(
    user_clause: str,
    client: QdrantClient,
    embeddings: KUREEmbeddings,
    llm: ChatOpenAI,
) -> dict:
    """Multi-Index 독소조항 위험 탐지.

    Args:
        user_clause: 사용자 계약 조항 텍스트.
        client: QdrantClient 인스턴스.
        embeddings: 임베딩 모델.
        llm: ChatOpenAI 인스턴스.

    Returns:
        {"illegal_similarity", "normal_similarity", "risk_delta", "analysis",
         "illegal_matches", "normal_matches", "law_matches"}
    """
    illegal_results = search_collection(
        client, embeddings, user_clause, "special_clauses_illegal", k=3,
    )
    normal_results = search_collection(
        client, embeddings, user_clause, "special_clauses_normal", k=2,
    )
    law_results = search_collection(
        client, embeddings, user_clause, "law_database", k=2,
    )

    illegal_score = (
        max(d.metadata.get("score", 0) for d in illegal_results)
        if illegal_results else 0
    )
    normal_score = (
        max(d.metadata.get("score", 0) for d in normal_results)
        if normal_results else 0
    )

    illegal_text = "\n".join(
        f"- ({d.metadata.get('category', '')}) {d.page_content}"
        for d in illegal_results
    ) or "해당 없음"

    normal_text = "\n".join(
        f"- ({d.metadata.get('category', '')}) {d.page_content}"
        for d in normal_results
    ) or "해당 없음"

    law_text = "\n".join(
        d.page_content for d in law_results
    ) or "관련 법률 없음"

    analysis = llm.invoke(
        RISK_PROMPT.format(
            clause=user_clause,
            illegal_matches=illegal_text,
            normal_matches=normal_text,
            law_context=law_text,
        )
    )

    return {
        "illegal_similarity": illegal_score,
        "normal_similarity": normal_score,
        "risk_delta": illegal_score - normal_score,
        "analysis": analysis.content,
        "illegal_matches": illegal_results,
        "normal_matches": normal_results,
        "law_matches": law_results,
    }


class RagBot:
    """Multi-Index RAG 봇 (LangSmith 추적 지원).

    Args:
        client: QdrantClient 인스턴스.
        embeddings: 임베딩 모델.
        collections: 검색 대상 컬렉션 리스트.
        model: OpenAI 모델명.
    """

    def __init__(
        self,
        client: QdrantClient,
        embeddings: KUREEmbeddings,
        collections: list[str],
        model: str = "gpt-4o-mini",
    ):
        self._openai_client = wrap_openai(openai.Client())
        self._qdrant_client = client
        self._embeddings = embeddings
        self._collections = collections
        self._model = model

    @traceable()
    def retrieve_docs(self, question: str) -> list[Document]:
        return search_multi_index(
            self._qdrant_client, self._embeddings, question,
            collections=self._collections, k_per_collection=3,
        )

    @traceable()
    def invoke_llm(self, question: str, docs: list[Document]) -> dict:
        context = build_context(docs)
        response = self._openai_client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 한국의 임대차 계약 전문가입니다. "
                        "아래 법률 문서와 특약사항을 참고해서 사용자의 질문에 답변해주세요.\n\n"
                        f"## 참고 문서\n\n{context}"
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        return {
            "answer": response.choices[0].message.content,
            "contexts": [str(doc) for doc in docs],
        }

    @traceable()
    def get_answer(self, question: str) -> dict:
        docs = self.retrieve_docs(question)
        return self.invoke_llm(question, docs)
