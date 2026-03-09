import json
import time

import openai
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from qdrant_client import QdrantClient

from langchain_core.messages import HumanMessage, AIMessage

from app.monitoring.metrics import LLM_LATENCY
from app.rag.chain.prompts import CONTRACT_QA_PROMPT, RISK_PROMPT, CHAT_PROMPT, TERM_EXPLANATION_PROMPT
from app.rag.embedding.kure import KUREEmbeddings
from app.rag.retriever.multi_retriever import search_collection, search_multi_index, async_search_multi_index


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


@traceable()
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
    with LLM_LATENCY.time():
        response = llm.invoke(prompt_text)

    return {
        "answer": response.content,
        "source_documents": docs,
        "context": context,
    }


@traceable()
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
    # 쿼리 벡터 1회 계산 후 3개 컬렉션에 재사용
    query_vector = embeddings.embed_query(user_clause)
    illegal_results = search_collection(
        client, embeddings, user_clause, "special_clauses_illegal", k=3,
        query_vector=query_vector,
    )
    normal_results = search_collection(
        client, embeddings, user_clause, "special_clauses_normal", k=2,
        query_vector=query_vector,
    )
    law_results = search_collection(
        client, embeddings, user_clause, "law_database", k=2,
        query_vector=query_vector,
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

    with LLM_LATENCY.time():
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


_CHAT_COLLECTIONS = [
    "law_database",
    "contracts",
    "special_clauses_illegal",
    "special_clauses_normal",
]


@traceable()
async def chat_rag(
    message: str,
    history: list[dict],
    client: QdrantClient,
    embeddings: KUREEmbeddings,
    llm: ChatOpenAI,
    contract_context: str | None = None,
    collections: list[str] | None = None,
    k_per_collection: int = 2,
) -> dict:
    """대화 이력을 포함한 RAG 챗봇 (병렬 검색).

    Args:
        message: 현재 사용자 메시지.
        history: 이전 대화 이력 [{"role": "user"|"assistant", "content": str}, ...].
                 최근 10턴만 사용됨.
        client: QdrantClient 인스턴스.
        embeddings: 임베딩 모델.
        llm: ChatOpenAI 인스턴스.
        contract_context: 사용자가 현재 보고 있는 계약서 텍스트 (선택).
        collections: 검색할 컬렉션 리스트. None이면 전체 4개 컬렉션 사용.
        k_per_collection: 컬렉션당 검색 수.

    Returns:
        {"answer": str, "sources": list[str], "context": str}
    """
    if collections is None:
        collections = _CHAT_COLLECTIONS

    # 1. 병렬 RAG 검색
    docs = await async_search_multi_index(
        client, embeddings, message,
        collections=collections,
        k_per_collection=k_per_collection,
        score_threshold=0.3,  # 관련성 낮은 문서 제거
    )
    context = build_context(docs, max_length=3000)

    # 계약서 컨텍스트가 있으면 앞에 붙임
    if contract_context:
        context = f"[사용자 계약서 원문 요약]\n{contract_context[:800]}\n\n{context}"

    # 2. 대화 이력 → LangChain 메시지 변환 (최근 10턴)
    lc_history = []
    for msg in history[-10:]:
        if msg.get("role") == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            lc_history.append(AIMessage(content=msg["content"]))

    # 3. 프롬프트 구성 및 LLM 비동기 호출
    prompt_messages = CHAT_PROMPT.format_messages(
        context=context,
        history=lc_history,
        question=message,
    )
    _llm_start = time.perf_counter()
    response = await llm.ainvoke(prompt_messages)
    LLM_LATENCY.observe(time.perf_counter() - _llm_start)

    # 4. 출처 요약 (상위 3개)
    sources = []
    for doc in docs[:3]:
        meta = doc.metadata
        label = meta.get("article") or meta.get("title") or meta.get("category") or ""
        coll = meta.get("collection", "")
        sources.append(f"[{coll}] {label}".strip(" []"))

    return {
        "answer": response.content,
        "sources": sources,
        "context": context,
    }


@traceable()
async def explain_term_rag(
    term: str,
    client: QdrantClient,
    embeddings: KUREEmbeddings,
    llm: ChatOpenAI,
    context: str = "",
    surrounding_text: str = "",
) -> dict:
    """RAG 기반 법률 용어 해설.

    law_database 컬렉션에서 관련 법조문을 검색한 뒤 LLM으로 용어를 설명한다.

    Args:
        term: 해설할 법률 용어.
        client: QdrantClient 인스턴스.
        embeddings: 임베딩 모델.
        llm: ChatOpenAI 인스턴스.
        context: 용어가 등장한 문맥 (예: "주택임대차보호법").
        surrounding_text: 용어 주변 문장.

    Returns:
        {"simple_explanation": str, "legal_definition": str, "examples": list[str]}
    """
    search_query = " ".join(filter(None, [term, context, surrounding_text]))
    docs = await async_search_multi_index(
        client, embeddings, search_query,
        collections=["law_database"],
        k_per_collection=4,
        score_threshold=0.3,
    )
    law_context = build_context(docs, max_length=1500) or "관련 법률 문서를 찾을 수 없습니다."

    prompt_text = TERM_EXPLANATION_PROMPT.format(
        term=term,
        context=context or "임대차 계약",
        surrounding_text=surrounding_text or "없음",
        law_context=law_context,
    )

    _llm_start = time.perf_counter()
    response = await llm.ainvoke(prompt_text)
    LLM_LATENCY.observe(time.perf_counter() - _llm_start)

    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 전체 응답을 simple_explanation으로 폴백
        return {
            "simple_explanation": response.content,
            "legal_definition": "",
            "examples": [],
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
