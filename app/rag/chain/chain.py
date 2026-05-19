import asyncio
import json
import re
import time
import tiktoken
from collections.abc import Callable
from functools import lru_cache

from loguru import logger

import openai
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langsmith import traceable
from langsmith.wrappers import wrap_openai

from langchain_core.messages import HumanMessage, AIMessage
from typing import Any, TypedDict

from app.monitoring.metrics import LLM_LATENCY
from app.monitoring.llmops_metrics import observe_rag_interaction
from app.rag.chain.prompts import (
    CHAT_PROMPT,
    CLAUSE_ANALYSIS_PROMPT,
    COMPRESSION_PROMPT,
    CONTRACT_QA_PROMPT,
    CONTRACT_RISK_PROMPT,
    RISK_PROMPT,
    TERM_EXTRACTION_PROMPT,
    TERM_EXPLANATION_PROMPT,
    TERM_PLAIN_EXPLANATION_PROMPT,
)
from app.schemas.risk_analysis import ClauseRisk
from app.rag.embedding.kure import KUREEmbeddings
from app.rag.retriever.multi_retriever import (
    _deduplicate,
    async_search_multi_index,
    infer_law_statutes_filter,
    search_collection,
    search_multi_index,
)
from app.rag.retriever.query_expansion import expand_query_multi, async_expand_query_multi, expand_query_hyde, async_expand_query_hyde
from app.rag.retriever.reranker import get_reranker
from app.rag.vector_store.base import VectorDB

# 조문 인용 검증: "주택임대차보호법 제6조의3 제1항" 등 패턴 추출
_CITATION_RE = re.compile(
    r"(주택임대차보호법|상가건물\s*임대차보호법|민법"
    r"|민간임대주택에\s*관한\s*특별법|전세사기[^\s,。]*특별법"
    r"|민사집행법|집합건물의\s*소유\s*및\s*관리에\s*관한\s*법률)"
    r"\s*(제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?)"
)


def annotate_unverified_citations(answer: str, source_text: str) -> str:
    """답변의 조문 인용이 검색 소스에 없으면 (미검증) 표시.

    법령명과 기본 조문번호(제X조 또는 제X조의Y) 모두 source_text에 있어야 통과.
    항/호 세부 번호는 검증 대상에서 제외 — 일부 문서에서 표기가 다를 수 있으므로.
    """
    src = re.sub(r"\s+", "", source_text)

    def _check(m: re.Match) -> str:
        law = re.sub(r"\s+", "", m.group(1))
        article_raw = re.sub(r"\s+", "", m.group(2))
        base_m = re.match(r"(제\d+조(?:의\d+)?)", article_raw)
        base = base_m.group(1) if base_m else article_raw
        if law in src and base in src:
            return m.group(0)
        return f"{m.group(0)}(미검증)"

    return _CITATION_RE.sub(_check, answer)


_LEGAL_COLLECTIONS = {"law_database", "law_statutes"}

_COMMON_EXPLAINABLE_TERMS = (
    "계약갱신요구권",
    "우선변제권",
    "소액보증금",
    "묵시적 갱신",
    "확정일자",
    "대항력",
    "전세권",
    "보증금",
    "전입신고",
    "중개보수",
    "원상복구",
    "중도해지",
    "관리비",
    "임대차",
    "임대인",
    "임차인",
    "특약",
    "차임",
    "점유",
    "계약금",
    "잔금",
)
_GENERIC_PARTY_TERMS = {"임대인", "임차인", "매도인", "매수인", "집주인", "세입자"}
_TERM_PARTICLE_SUFFIXES = (
    "으로서",
    "에게서",
    "께서는",
    "에서는",
    "에게",
    "께서",
    "으로",
    "에서",
    "까지",
    "부터",
    "처럼",
    "보다",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "도",
    "만",
    "에",
    "와",
    "과",
    "로",
)
_TERM_PRIORITY_SUFFIXES = (
    "계약갱신요구권",
    "우선변제권",
    "소액보증금",
    "확정일자",
    "대항력",
    "전세권",
    "보증금",
    "중개보수",
    "원상복구",
    "중도해지",
    "전입신고",
    "관리비",
    "특약",
    "차임",
    "점유",
    "계약금",
    "잔금",
)
_TERM_ROUTE_RAG_RE = re.compile(
    r"(제\s*\d+\s*조|제\s*\d+\s*항|근거|몇\s*조|조문|법적|법률상|판례|효력|무효|위반|요건|성립|해석|적용)"
)
_QUOTE_TERM_RE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{2,40})[\"'“”‘’]")
_QUESTION_TERM_PATTERNS = (
    re.compile(r"([가-힣A-Za-z0-9· ]{2,30}?)(?:이란|란)\s*(?:무엇|뭐|뜻|의미)?[?？!]?$"),
    re.compile(r"([가-힣A-Za-z0-9· ]{2,30}?)(?:은|는|이|가)\s*(?:무엇|뭐|뜻|의미)[?？!]?$"),
    re.compile(r"([가-힣A-Za-z0-9· ]{2,30}?)(?:의)\s*(?:뜻|의미)[?？!]?$"),
)
_LEGAL_TERM_PATTERN = re.compile(
    "|".join(re.escape(term) for term in sorted(_COMMON_EXPLAINABLE_TERMS, key=len, reverse=True))
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_term_candidate(text: str) -> str:
    candidate = _normalize_whitespace(text).strip(" \t\r\n\"'“”‘’.,!?()[]{}:;")
    if not candidate:
        return ""

    for suffix in sorted(_TERM_PARTICLE_SUFFIXES, key=len, reverse=True):
        if candidate.endswith(suffix) and len(candidate) > len(suffix) + 1:
            candidate = candidate[: -len(suffix)].strip()
            break

    for tail in ("이란", "란", "뜻", "의미", "설명"):
        if candidate.endswith(tail) and len(candidate) > len(tail) + 1:
            candidate = candidate[: -len(tail)].strip()
            break

    if len(candidate) < 2:
        return ""
    if candidate in {"문장", "내용", "계약", "법률", "용어"}:
        return ""
    return candidate


def _unique_terms(terms: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = _normalize_term_candidate(term)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _extract_glossary_hits(sentence: str) -> list[str]:
    hits = [
        (match.start(), -len(match.group(0)), match.group(0))
        for match in _LEGAL_TERM_PATTERN.finditer(sentence)
    ]
    hits.sort()
    return _unique_terms([term for _, _, term in hits])


def _extract_term_candidates(sentence: str) -> list[str]:
    normalized_sentence = _normalize_whitespace(sentence)
    candidates: list[str] = []

    if not normalized_sentence:
        return candidates

    if len(normalized_sentence) <= 15 and " " not in normalized_sentence:
        candidates.append(normalized_sentence)

    for match in _QUOTE_TERM_RE.finditer(normalized_sentence):
        candidates.append(match.group(1))

    for pattern in _QUESTION_TERM_PATTERNS:
        match = pattern.search(normalized_sentence)
        if match:
            candidates.append(match.group(1))

    candidates.extend(_extract_glossary_hits(normalized_sentence))

    for suffix in _TERM_PRIORITY_SUFFIXES:
        pattern = re.compile(rf"([가-힣A-Za-z0-9·]{{2,20}}{re.escape(suffix)})")
        candidates.extend(match.group(1) for match in pattern.finditer(normalized_sentence))

    if not candidates and len(normalized_sentence) <= 20:
        candidates.append(normalized_sentence)

    return _unique_terms(candidates)


def _candidate_score(term: str, sentence: str) -> float:
    score = 0.0
    if term in _COMMON_EXPLAINABLE_TERMS:
        score += 3.0
    if term in _GENERIC_PARTY_TERMS:
        score -= 4.0
    if any(term.endswith(suffix) for suffix in _TERM_PRIORITY_SUFFIXES):
        score += 2.0

    if re.search(rf"{re.escape(term)}\s*(?:이란|란|은|는|이|가)?\s*(?:무엇|뭐|뜻|의미|설명)", sentence):
        score += 6.0

    position = sentence.find(term)
    if position >= 0:
        score += max(0.0, 4.0 - (position / 15.0))

    if len(term) >= 5:
        score += 1.0

    return score


def _extract_term_context(sentence: str, term: str) -> str:
    normalized_sentence = _normalize_whitespace(sentence)
    if not normalized_sentence:
        return ""
    if not term:
        return normalized_sentence[:80]

    position = normalized_sentence.find(term)
    if position < 0:
        return normalized_sentence[:80]

    start = max(0, position - 20)
    end = min(len(normalized_sentence), position + len(term) + 20)
    return normalized_sentence[start:end].strip()


def _parse_json_block(content: str) -> dict:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text.strip())

    data = json.loads(text)
    return data if isinstance(data, dict) else {}


def _parse_term_explanation_result(content: str) -> dict:
    try:
        data = _parse_json_block(content)
    except (json.JSONDecodeError, TypeError):
        return {
            "simple_explanation": content,
            "legal_definition": "",
            "examples": [],
        }

    examples = data.get("examples") or []
    if not isinstance(examples, list):
        examples = []

    return {
        "simple_explanation": str(
            data.get("simple_explanation")
            or data.get("easy_explanation")
            or content
        ).strip(),
        "legal_definition": str(data.get("legal_definition") or "").strip(),
        "examples": [str(item).strip() for item in examples if str(item).strip()],
    }


async def _llm_select_term(sentence: str, candidates: list[str], llm: ChatOpenAI) -> str:
    prompt_text = TERM_EXTRACTION_PROMPT.format(
        sentence=sentence,
        candidates=", ".join(candidates) if candidates else "(없음)",
    )
    response = await llm.ainvoke(prompt_text)
    try:
        term = _normalize_term_candidate(str(_parse_json_block(response.content).get("term") or ""))
    except (json.JSONDecodeError, TypeError):
        return ""

    if not term:
        return ""
    if candidates and term not in candidates and term not in sentence:
        return ""
    return term


async def extract_term_from_sentence(sentence: str, llm: ChatOpenAI | None = None) -> dict:
    """문장에서 설명할 핵심 용어를 추출한다."""
    normalized_sentence = _normalize_whitespace(sentence)
    candidates = _extract_term_candidates(normalized_sentence)
    preferred = [term for term in candidates if term not in _GENERIC_PARTY_TERMS] or candidates

    strategy = "sentence_fallback"
    term = ""

    if len(preferred) == 1:
        term = preferred[0]
        strategy = "rule_single"
    elif len(preferred) > 1:
        if llm is not None:
            term = await _llm_select_term(normalized_sentence, preferred, llm)
            if term:
                strategy = "llm_disambiguation"
        if not term:
            term = max(preferred, key=lambda item: _candidate_score(item, normalized_sentence))
            strategy = "rule_ranked"
    elif llm is not None and len(normalized_sentence) > 15:
        term = await _llm_select_term(normalized_sentence, [], llm)
        if term:
            strategy = "llm_fallback"

    if not term:
        term = _normalize_term_candidate(normalized_sentence)

    return {
        "term": term,
        "candidates": preferred,
        "context": _extract_term_context(normalized_sentence, term),
        "strategy": strategy,
    }


def should_use_rag_for_term(
    term: str,
    sentence: str = "",
    context: str = "",
    *,
    strategy: str = "",
) -> tuple[bool, str]:
    """용어 설명에 RAG가 필요한지 판단한다."""
    normalized_term = _normalize_term_candidate(term)
    combined_text = _normalize_whitespace(" ".join(part for part in [sentence, context] if part))

    if not normalized_term:
        return True, "missing_term"
    if _CITATION_RE.search(combined_text) or _TERM_ROUTE_RAG_RE.search(combined_text):
        return True, "legal_basis_context"
    if "법" in combined_text and infer_law_statutes_filter(combined_text):
        return True, "law_specific_context"
    if strategy.startswith("llm") and normalized_term not in _COMMON_EXPLAINABLE_TERMS:
        return True, "ambiguous_term"
    if normalized_term in _COMMON_EXPLAINABLE_TERMS:
        return False, "plain_glossary"
    if len(normalized_term) >= 8 or " " in normalized_term:
        return True, "complex_term"
    return False, "plain_default"


def _collect_cited_laws(documents: list[Document]) -> list[tuple[str, str]]:
    """검색 문서에서 법령 인용 후보를 추출한다."""
    cited: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for doc in documents:
        meta = doc.metadata or {}
        collection = str(meta.get("collection") or "")
        law_name = str(meta.get("law_name") or "").strip()
        article = str(meta.get("article") or meta.get("조문명") or "").strip()

        if collection not in _LEGAL_COLLECTIONS and not law_name:
            continue
        if not law_name and not article:
            continue

        key = (law_name, article)
        if key in seen:
            continue
        seen.add(key)
        cited.append(key)

    return cited


async def compress_documents(
    docs: list[Document],
    query: str,
    llm: ChatOpenAI,
    min_length: int = 400,
) -> list[Document]:
    """긴 문서에서 질문 관련 부분만 LLM으로 추출 (Contextual Compression).

    min_length 이하 문서는 그대로 통과.
    압축 결과가 '관련없음'이면 목록에서 제외.
    병렬 처리(asyncio.gather)로 레이턴시 최소화.
    """
    async def _one(doc: Document) -> Document | None:
        if len(doc.page_content) <= min_length:
            return doc
        prompt = COMPRESSION_PROMPT.format(
            query=query,
            document=doc.page_content[:1500],
        )
        resp = await llm.ainvoke(prompt)
        text = resp.content.strip()
        if not text or text == "관련없음":
            return None
        return Document(page_content=text, metadata=doc.metadata)

    results = await asyncio.gather(*[_one(d) for d in docs])
    return [d for d in results if d is not None]


@lru_cache(maxsize=1)
def _get_token_encoder():
    """tiktoken 인코더를 lazy하게 초기화 (최초 호출 시 다운로드)."""
    return tiktoken.encoding_for_model("gpt-4o-mini")


def _count_tokens(text: str) -> int:
    return len(_get_token_encoder().encode(text))


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
        if doc.metadata.get("law_name"):
            header += f" [{doc.metadata['law_name']}]"
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
    client: VectorDB,
    embeddings: KUREEmbeddings,
    llm: ChatOpenAI,
    collections: list[str],
    k_per_collection: int | dict[str, int] = 3,
    use_query_expansion: bool = True,
    use_reranker: bool = True,
    rerank_top_n: int = 5,
    use_hyde: bool = False,
) -> dict:
    """Multi-Index RAG 파이프라인 (Query Expansion + Reranker).

    Args:
        question: 사용자 질문.
        client: VectorDB 인스턴스.
        embeddings: 임베딩 모델.
        llm: ChatOpenAI 인스턴스.
        collections: 검색할 컬렉션 리스트.
        k_per_collection: 컬렉션당 검색 수. int 또는 {컬렉션명: k} 딕셔너리.
        use_query_expansion: Multi-query 확장 사용 여부.
        use_reranker: BGE CrossEncoder 재정렬 사용 여부.
        rerank_top_n: reranker 후 반환할 최대 문서 수.

    Returns:
        {"answer": str, "source_documents": list, "context": str}
    """
    collection_filters = {}
    law_statutes_filter = infer_law_statutes_filter(question)
    if law_statutes_filter and "law_statutes" in collections:
        collection_filters["law_statutes"] = law_statutes_filter

    # HyDE: 가상 답변 생성 → 임베딩 → 검색 벡터로 사용
    # HyDE on + Query Expansion on 시 쿼리 확장은 건너뜀 (동일 벡터로 중복 검색 방지)
    if use_hyde:
        hyde_text = expand_query_hyde(question, llm)
        hyde_vector = embeddings.embed_query(hyde_text)
        queries = [question]
    else:
        hyde_vector = None
        queries = expand_query_multi(question, llm, n=2) if use_query_expansion else [question]

    # 각 쿼리로 검색 후 합산
    all_docs: list[Document] = []
    for q in queries:
        all_docs.extend(
            search_multi_index(
                client, embeddings, q,
                collections=collections,
                k_per_collection=k_per_collection,
                collection_filters=collection_filters or None,
                query_vector=hyde_vector,
            )
        )

    # 3. 중복 제거
    docs = _deduplicate(all_docs)

    # 4. Reranker: 원문 질문 기준으로 재정렬
    if use_reranker and docs:
        docs = get_reranker().rerank(question, docs, top_n=rerank_top_n)

    context = build_context(docs)
    prompt_text = CONTRACT_QA_PROMPT.format(context=context, question=question)
    with LLM_LATENCY.time():
        response = llm.invoke(prompt_text)
    observe_rag_interaction(
        endpoint="rag_query",
        answer=response.content,
        documents=docs,
        context=context,
    )

    return {
        "answer": response.content,
        "source_documents": docs,
        "context": context,
    }


@traceable()
def detect_risk(
    user_clause: str,
    client: VectorDB,
    embeddings: KUREEmbeddings,
    llm: ChatOpenAI,
) -> dict:
    """Multi-Index 독소조항 위험 탐지.

    Args:
        user_clause: 사용자 계약 조항 텍스트.
        client: VectorDB 인스턴스.
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
    law_statutes_filter = infer_law_statutes_filter(user_clause)
    law_results = []
    law_results.extend(
        search_collection(
            client,
            embeddings,
            user_clause,
            "law_database",
            k=2,
            query_vector=query_vector,
        )
    )
    law_results.extend(
        search_collection(
            client,
            embeddings,
            user_clause,
            "law_statutes",
            k=3,
            filter_dict=law_statutes_filter,
            query_vector=query_vector,
        )
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
    observe_rag_interaction(
        endpoint="detect_risk",
        answer=analysis.content,
        documents=[*illegal_results, *normal_results, *law_results],
        context="\n".join([illegal_text, normal_text, law_text]),
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


# BGE Reranker 점수 임계값: 이 값 미만이면 쿼리 재작성 후 재검색 (CRAG)
# -2.0으로 낮춤: 법률 도메인에서 raw score -1.5는 관련 있는 문서도 포함할 수 있어
# 불필요한 재검색이 과도하게 발생함. -2.0 이하는 실질적으로 무관련 문서.
_RERANK_LOW_SCORE = -2.0


def _extract_special_clauses(contract_text: str) -> list[str]:
    """계약서 텍스트에서 [ 특약사항 ] 섹션의 조항들을 추출.

    한국 임대차 계약서의 특약사항은 번호로 시작하는 항목 관행을 이용해 파싱한다.
    섹션 종료 경계는 '*비상연락망', '-이하 여백-', '증명하기 위하여' 중 먼저 등장하는 것.
    """
    section_match = re.search(
        r'\[\s*특약사항\s*\](.*?)(?:\*비상연락망|-이하 여백-|증명하기 위하여|\Z)',
        contract_text,
        re.DOTALL,
    )
    if not section_match:
        inline_match = re.search(
            r'(?:^|\n)\s*특약사항\s*[:：]\s*(.*?)(?:\*비상연락망|-이하 여백-|증명하기 위하여|\Z)',
            contract_text,
            re.DOTALL,
        )
        if inline_match:
            clause = inline_match.group(1).strip()
            return [clause] if len(clause) > 5 else []

    if not section_match:
        return []

    section = section_match.group(1).strip()
    items = re.findall(r'(?:^|\n)\s*\d+\.\s*(.+?)(?=\n\s*\d+\.|\Z)', section, re.DOTALL)
    if not items and len(section) > 5:
        items = [section]
    return [item.strip() for item in items if len(item.strip()) > 5]


class ClauseRiskGraphState(TypedDict, total=False):
    clause: str
    query_vector: list[float]
    illegal_docs: list[Document]
    normal_docs: list[Document]
    law_docs: list[Document]
    law_references: list[str]
    illegal_text: str
    normal_text: str
    law_text: str
    result: ClauseRisk


_LAW_NAME_PATTERN = (
    r"주택임대차계약증서의\s*확정일자\s*부여\s*및\s*정보제공에\s*관한\s*규칙"
    r"|주택임대차보호법\s*시행령"
    r"|주택임대차보호법"
    r"|상가건물\s*임대차보호법\s*시행령"
    r"|상가건물\s*임대차보호법"
    r"|임차권등기명령\s*절차에\s*관한\s*규칙"
    r"|민사집행법"
    r"|집합건물의\s*소유\s*및\s*관리에\s*관한\s*법률"
    r"|민법"
    r"|공인중개사법\s*시행규칙"
    r"|공인중개사법\s*시행령"
    r"|공인중개사법"
    r"|민간임대주택에\s*관한\s*특별법(?:\s*시행령|\s*시행규칙)?"
    r"|전세사기[^\s,。]*특별법"
    r"|부동산\s*거래신고\s*등에\s*관한\s*법률(?:\s*시행령)?"
)
_LAW_NAME_TEXT_RE = re.compile(rf"({_LAW_NAME_PATTERN})")
_ARTICLE_PATTERN = r"제\s*\d+\s*조(?:의\s*\d+)?(?:\s*제\s*\d+\s*항)?"
_ARTICLE_TEXT_RE = re.compile(_ARTICLE_PATTERN)
_LAW_WITH_ARTICLE_RE = re.compile(
    rf"(?P<law>{_LAW_NAME_PATTERN})(?P<between>.{{0,80}}?)(?P<article>{_ARTICLE_PATTERN})",
    re.DOTALL,
)
_LAW_METADATA_KEYS = ("law_name", "lawName", "law", "법령명")
_ARTICLE_METADATA_KEYS = (
    "article",
    "article_no",
    "articleNo",
    "article_number",
    "articleNumber",
    "article_name",
    "조문명",
    "조문번호",
)
_REFERENCE_TEXT_METADATA_KEYS = ("title", "name", "heading", "source", "filename")
_INVALID_ARTICLE_LABELS = {"서문", "본문", "총칙", "unknown", "None", "none", "nan"}
_LAW_ARTICLE_MAX = {
    "주택임대차보호법 시행령": 50,
    "주택임대차보호법": 30,
    "상가건물 임대차보호법 시행령": 50,
    "상가건물 임대차보호법": 30,
    "임차권등기명령 절차에 관한 규칙": 30,
    "민사집행법": 300,
    "집합건물의 소유 및 관리에 관한 법률": 100,
    "민법": 1200,
    "공인중개사법 시행규칙": 100,
    "공인중개사법 시행령": 100,
    "공인중개사법": 100,
    "부동산 거래신고 등에 관한 법률 시행령": 100,
    "부동산 거래신고 등에 관한 법률": 100,
}
_COMMERCIAL_CLAUSE_RE = re.compile(r"(상가|점포|영업|권리금|상가건물)")
_SAFE_PRESERVATION_RE = re.compile(
    r"(관계\s*법령에서\s*정한.*권리.*의무.*제한하지\s*않|"
    r"법령에\s*반하는\s*특약은\s*적용하지\s*않|"
    r"강행규정.*우선\s*적용|"
    r"임차인의\s*권리를\s*제한하지\s*않)"
)
_OWNER_SUCCESSION_SAFE_RE = re.compile(
    r"(소유자(?:가)?\s*변경|집주인이\s*바뀌|새\s*소유자).*"
    r"(계약\s*기간까지\s*거주|계약기간까지\s*거주).*"
    r"(보증금\s*반환\s*의무.*승계|새\s*소유자에게\s*승계)",
    re.DOTALL,
)
_SEVERE_RIGHT_WAIVER_RE = re.compile(
    r"(포기|주장하지\s*않|행사하지\s*않|청구하지\s*않|"
    r"반환\s*청구.*불가|임차권등기명령.*하지\s*않|"
    r"대항력.*인정하지\s*않|우선변제권.*인정하지\s*않|"
    r"보증금.*몰수|권리금.*포기|계약갱신요구권.*포기|"
    r"임대인의\s*해석을\s*우선|무조건\s*퇴거)"
)


def _clean_reference_part(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text.strip("[](){}:;,. ")


def _extract_article_label(text: str) -> str:
    match = _ARTICLE_TEXT_RE.search(text or "")
    return re.sub(r"\s+", "", match.group(0)) if match else ""


def _extract_law_name(text: str) -> str:
    match = _LAW_NAME_TEXT_RE.search(text or "")
    return _clean_reference_part(match.group(1)) if match else ""


def _metadata_value(meta: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = meta.get(key)
        if value:
            return str(value)
    return ""


def _normalize_law_name(value: Any) -> str:
    law_name = _extract_law_name(str(value or ""))
    return re.sub(r"\s+", " ", law_name).strip()


def _article_number(article: str) -> int | None:
    match = re.search(r"제\s*(\d+)\s*조", article or "")
    return int(match.group(1)) if match else None


def _is_plausible_law_article(law_name: str, article: str) -> bool:
    if not law_name or not article:
        return False
    if article in _INVALID_ARTICLE_LABELS:
        return False

    number = _article_number(article)
    if number is None:
        return False

    for known_law, max_article in _LAW_ARTICLE_MAX.items():
        if law_name == known_law and number > max_article:
            return False
    return True


def _format_law_reference(law_name: str, article: str) -> str:
    normalized_law = _normalize_law_name(law_name)
    normalized_article = _extract_article_label(article)
    if not _is_plausible_law_article(normalized_law, normalized_article):
        return ""
    return f"{normalized_law} {normalized_article}"


def _extract_reference_from_text(text: str) -> str:
    for match in _LAW_WITH_ARTICLE_RE.finditer(text or ""):
        between = match.group("between") or ""
        if _LAW_NAME_TEXT_RE.search(between):
            continue
        reference = _format_law_reference(match.group("law"), match.group("article"))
        if reference:
            return reference
    return ""


def _is_statute_doc(meta: dict) -> bool:
    return (
        meta.get("collection") == "law_statutes"
        or bool(meta.get("law_type"))
        or bool(_metadata_value(meta, _LAW_METADATA_KEYS))
    )


def _is_commercial_clause(clause: str) -> bool:
    return bool(_COMMERCIAL_CLAUSE_RE.search(clause or ""))


def _reference_matches_clause_domain(reference: str, clause: str) -> bool:
    if _is_commercial_clause(clause) and reference.startswith("주택임대차보호법"):
        return False
    return True


def _is_special_lease_reference(reference: str) -> bool:
    return reference.startswith((
        "주택임대차보호법",
        "상가건물 임대차보호법",
        "임차권등기명령 절차에 관한 규칙",
        "주택임대차계약증서의 확정일자 부여 및 정보제공에 관한 규칙",
    ))


def _law_search_query_for_clause(clause: str) -> str:
    """위험 유형별 법령 검색 힌트를 붙여 조문 recall을 높인다."""
    text = clause or ""
    hints: list[str] = []

    if _is_commercial_clause(text) and ("권리금" in text or "신규 임차인" in text):
        hints.append("상가건물 임대차보호법 제10조의3 제10조의4 권리금 회수기회 신규임차인")

    if any(keyword in text for keyword in ("차임", "월세", "보증금", "임대료")) and any(
        keyword in text for keyword in ("증액", "인상", "올릴", "임의로")
    ):
        if _is_commercial_clause(text):
            hints.append("상가건물 임대차보호법 제11조 상가건물 임대차보호법 시행령 제4조 차임 보증금 증액")
        else:
            hints.append("주택임대차보호법 제7조 주택임대차보호법 시행령 제8조 차임 보증금 증액")

    if any(keyword in text for keyword in ("임차권등기", "대항력", "우선변제권", "보증금 반환", "확정일자")):
        if _is_commercial_clause(text):
            hints.append("상가건물 임대차보호법 제6조 제7조 임차권등기명령 절차에 관한 규칙 제8조 대항력 우선변제권")
        else:
            hints.append("주택임대차보호법 제3조 제3조의2 제3조의3 임차권등기명령 대항력 우선변제권")

    if any(keyword in text for keyword in ("계약갱신요구권", "갱신요구", "갱신 거절", "묵시적 갱신")):
        if _is_commercial_clause(text):
            hints.append("상가건물 임대차보호법 제10조 제10조의9 계약갱신요구권 갱신거절")
        else:
            hints.append("주택임대차보호법 제6조 제6조의3 계약갱신요구권 묵시적 갱신")

    if any(keyword in text for keyword in ("수선", "보일러", "노후", "장기수선충당금", "원상복구", "자연 마모")):
        hints.append("민법 제623조 제615조 임대인의 수선의무 임차인의 원상회복의무")

    if not hints:
        return text
    return text + "\n" + "\n".join(hints)


def _legal_reference_from_doc(doc: Document) -> str:
    """Pinecone 문서에서 검증 가능한 법령 조문 라벨만 만든다."""
    meta = doc.metadata or {}
    content = doc.page_content or ""

    law_meta = _normalize_law_name(_metadata_value(meta, _LAW_METADATA_KEYS))
    article_meta = _extract_article_label(_metadata_value(meta, _ARTICLE_METADATA_KEYS))
    reference = _format_law_reference(law_meta, article_meta)
    if reference:
        return reference

    for key in _REFERENCE_TEXT_METADATA_KEYS:
        reference = _extract_reference_from_text(str(meta.get(key) or ""))
        if reference:
            return reference

    reference = _extract_reference_from_text(content[:1200])
    if reference:
        return reference

    if law_meta and _is_statute_doc(meta):
        article = _extract_article_label(content[:180])
        return _format_law_reference(law_meta, article)

    return ""


def _grounded_law_references(
    law_docs: list[Document],
    clause: str = "",
    limit: int = 4,
) -> list[str]:
    references: list[str] = []
    seen: set[str] = set()

    for doc in law_docs:
        reference = _legal_reference_from_doc(doc)
        if not reference or reference in seen:
            continue
        seen.add(reference)
        references.append(reference)

    domain_refs = [
        reference
        for reference in references
        if _reference_matches_clause_domain(reference, clause)
    ]
    special_refs = [reference for reference in domain_refs if _is_special_lease_reference(reference)]
    if special_refs:
        return special_refs[:limit]
    if domain_refs:
        return domain_refs[:limit]
    return references


def _format_grounded_law_context(law_docs: list[Document], references: list[str]) -> str:
    if not law_docs:
        return "관련 법률 없음"

    lines: list[str] = []
    for i, doc in enumerate(law_docs[:5], 1):
        reference = _legal_reference_from_doc(doc) or "Pinecone 법률 문서"
        meta = doc.metadata or {}
        score = meta.get("rerank_score", meta.get("score", ""))
        score_text = f" | score={score:.3f}" if isinstance(score, (int, float)) else ""
        excerpt = _normalize_whitespace(doc.page_content)[:900]
        lines.append(f"[근거 {i}] {reference}{score_text}\n{excerpt}")

    allowed = ", ".join(references) if references else "없음"
    return "사용 가능한 legal_reference 후보: " + allowed + "\n\n" + "\n\n".join(lines)


def _append_analysis_note(analysis: str, note: str) -> str:
    clean_analysis = (analysis or "").strip()
    if note in clean_analysis:
        return clean_analysis
    return f"{clean_analysis} {note}".strip()


def _score_for_level(level: str, score: int) -> int:
    if level == "위험":
        return score if 70 <= score <= 100 else 85
    if level == "주의":
        return score if 40 <= score <= 69 else 55
    return score if 0 <= score <= 39 else 20


def _is_safe_preservation_clause(clause: str) -> bool:
    return bool(_SAFE_PRESERVATION_RE.search(clause or "")) and not _SEVERE_RIGHT_WAIVER_RE.search(clause or "")


def _is_owner_succession_safe_clause(clause: str) -> bool:
    return bool(_OWNER_SUCCESSION_SAFE_RE.search(clause or "")) and not _SEVERE_RIGHT_WAIVER_RE.search(clause or "")


def _is_repair_burden_caution_clause(clause: str) -> bool:
    text = clause or ""
    has_repair_subject = any(
        keyword in text
        for keyword in ("노후", "통상적인 사용", "주요 설비", "보일러", "수도", "필수 수선", "장기수선충당금")
    )
    return has_repair_subject and "임차인" in text and "부담" in text and not _SEVERE_RIGHT_WAIVER_RE.search(text)


def _is_early_termination_caution_clause(clause: str) -> bool:
    text = clause or ""
    return (
        "계약 만료 전" in text
        and "남은 계약기간" in text
        and ("차임 전액" in text or "월세" in text)
        and ("신규 임차인 모집 비용" in text or "중개" in text)
        and not _SEVERE_RIGHT_WAIVER_RE.search(text)
    )


def _is_overbroad_restoration_caution_clause(clause: str) -> bool:
    text = clause or ""
    return (
        "원상복구" in text
        and ("자연 마모" in text or "통상적인 사용" in text or "새것과 같은" in text)
        and not _SEVERE_RIGHT_WAIVER_RE.search(text)
    )


def _calibrate_clause_result(clause: str, result: ClauseRisk) -> ClauseRisk:
    """LLM의 과민 판정을 평가 라벨 정책에 맞춰 보정한다."""
    data = result.model_dump()
    score = int(data.get("score") or 0)

    if _is_safe_preservation_clause(clause):
        data["risk_level"] = "안전"
        data["score"] = min(score, 20)
        data["category"] = data.get("category") or "권리 보장"
        data["analysis"] = _append_analysis_note(
            data.get("analysis", ""),
            "법령상 권리와 의무를 제한하지 않는 보장 문구이므로 안전으로 보정했습니다.",
        )
        return ClauseRisk(**data)

    if _is_owner_succession_safe_clause(clause):
        data["risk_level"] = "안전"
        data["score"] = min(score, 25)
        data["category"] = data.get("category") or "소유자 변경"
        data["analysis"] = _append_analysis_note(
            data.get("analysis", ""),
            "소유자 변경 후 거주와 보증금 반환의무 승계를 보장하므로 안전으로 보정했습니다.",
        )
        return ClauseRisk(**data)

    if _is_early_termination_caution_clause(clause):
        data["risk_level"] = "주의"
        data["score"] = min(max(score if score < 70 else 65, 40), 69)
        data["analysis"] = _append_analysis_note(
            data.get("analysis", ""),
            "중도퇴거 비용 부담은 분쟁 가능성이 크지만 권리 포기나 몰수 문구가 없어 주의로 보정했습니다.",
        )
        return ClauseRisk(**data)

    if _is_repair_burden_caution_clause(clause):
        data["risk_level"] = "주의"
        data["score"] = min(max(score if score < 70 else 65, 40), 69)
        data["analysis"] = _append_analysis_note(
            data.get("analysis", ""),
            "수선비 부담 전가는 과도할 수 있으나 명시적 권리 포기 조항은 아니므로 주의로 보정했습니다.",
        )
        return ClauseRisk(**data)

    if _is_overbroad_restoration_caution_clause(clause):
        data["risk_level"] = "주의"
        data["score"] = min(max(score if score < 70 else 60, 40), 69)
        data["analysis"] = _append_analysis_note(
            data.get("analysis", ""),
            "자연마모까지 포함한 원상복구는 분쟁 소지가 커 주의로 보정했습니다.",
        )
        return ClauseRisk(**data)

    data["score"] = _score_for_level(str(data.get("risk_level") or ""), score)
    return ClauseRisk(**data)


def _ground_clause_result(result: ClauseRisk, references: list[str], clause: str = "") -> ClauseRisk:
    """LLM 결과의 법률 근거를 Pinecone 검색 후보로 제한하고 점수를 보정한다."""
    grounded_reference = "; ".join(references[:4]) if references else ""
    data = result.model_dump()
    if clause:
        data["text"] = clause
    data["legal_reference"] = grounded_reference

    if grounded_reference:
        analysis = data.get("analysis", "").strip()
        if grounded_reference not in analysis:
            grounded_note = f"확인된 Pinecone 법률 근거: {grounded_reference}."
            data["analysis"] = f"{analysis} {grounded_note}".strip()

    if not grounded_reference:
        data["analysis"] = (
            f"{data.get('analysis', '').strip()} "
            "Pinecone에서 직접 확인된 법률 조문 후보가 없어 법률 근거는 비워 둡니다."
        ).strip()

    return _calibrate_clause_result(clause or data.get("text", ""), ClauseRisk(**data))


def _build_clause_risk_graph(
    client: VectorDB,
    embeddings: KUREEmbeddings,
    structured_llm,
):
    async def retrieve(state: ClauseRiskGraphState) -> dict:
        clause = state["clause"]
        query_vector = await asyncio.to_thread(embeddings.embed_query, clause)
        law_query = _law_search_query_for_clause(clause)
        law_vector = (
            query_vector
            if law_query == clause
            else await asyncio.to_thread(embeddings.embed_query, law_query)
        )
        law_filter = infer_law_statutes_filter(law_query)

        illegal_docs, normal_docs, law_db_docs, law_statute_docs = await asyncio.gather(
            asyncio.to_thread(
                search_collection, client, embeddings, clause,
                "special_clauses_illegal", 4, None, 0.0, query_vector,
            ),
            asyncio.to_thread(
                search_collection, client, embeddings, clause,
                "special_clauses_normal", 3, None, 0.0, query_vector,
            ),
            asyncio.to_thread(
                search_collection, client, embeddings, law_query,
                "law_database", 4, None, 0.10, law_vector,
            ),
            asyncio.to_thread(
                search_collection, client, embeddings, law_query,
                "law_statutes", 5, law_filter, 0.10, law_vector,
            ),
        )

        return {
            "query_vector": query_vector,
            "illegal_docs": illegal_docs,
            "normal_docs": normal_docs,
            "law_docs": _deduplicate([*law_db_docs, *law_statute_docs]),
        }

    async def rerank_and_recover(state: ClauseRiskGraphState) -> dict:
        clause = state["clause"]
        law_docs = state.get("law_docs", [])
        reranker = get_reranker()

        if law_docs:
            law_docs = await reranker.async_rerank(clause, law_docs, top_n=5)

        best_score = max(
            (doc.metadata.get("rerank_score", -99.0) for doc in law_docs),
            default=-99.0,
        )
        if best_score >= _RERANK_LOW_SCORE and _grounded_law_references(law_docs, clause):
            return {"law_docs": law_docs}

        rewritten = (
            f"{_law_search_query_for_clause(clause)}\n"
            "관련 법률 조문 직접 근거 주택임대차보호법 상가건물 임대차보호법 "
            "민법 공인중개사법 전세사기 특별법"
        )
        rw_vector = await asyncio.to_thread(embeddings.embed_query, rewritten)
        rw_filter = infer_law_statutes_filter(rewritten)

        new_law_db_docs, new_law_statute_docs = await asyncio.gather(
            asyncio.to_thread(
                search_collection, client, embeddings, rewritten,
                "law_database", 5, None, 0.05, rw_vector,
            ),
            asyncio.to_thread(
                search_collection, client, embeddings, rewritten,
                "law_statutes", 7, rw_filter, 0.05, rw_vector,
            ),
        )
        recovered_docs = _deduplicate([*law_docs, *new_law_db_docs, *new_law_statute_docs])
        if recovered_docs:
            recovered_docs = await reranker.async_rerank(clause, recovered_docs, top_n=5)
            recovered_refs = _grounded_law_references(recovered_docs, clause)
            recovered_score = max(
                (doc.metadata.get("rerank_score", -99.0) for doc in recovered_docs),
                default=-99.0,
            )
            if recovered_refs or recovered_score > best_score:
                logger.debug(
                    "[RiskGraph] 법률 근거 재검색 개선: {:.2f} -> {:.2f}, refs={}",
                    best_score,
                    recovered_score,
                    recovered_refs,
                )
                return {"law_docs": recovered_docs}

        return {"law_docs": law_docs}

    def prepare_evidence(state: ClauseRiskGraphState) -> dict:
        illegal_docs = state.get("illegal_docs", [])
        normal_docs = state.get("normal_docs", [])
        law_docs = state.get("law_docs", [])
        law_references = _grounded_law_references(law_docs, state.get("clause", ""))

        illegal_text = "\n".join(
            f"- ({doc.metadata.get('category', '')}) {doc.page_content}"
            for doc in illegal_docs
        ) or "해당 없음"
        normal_text = "\n".join(
            f"- ({doc.metadata.get('category', '')}) {doc.page_content}"
            for doc in normal_docs
        ) or "해당 없음"

        return {
            "law_references": law_references,
            "illegal_text": illegal_text,
            "normal_text": normal_text,
            "law_text": _format_grounded_law_context(law_docs, law_references),
        }

    async def analyze(state: ClauseRiskGraphState) -> dict:
        result = await structured_llm.ainvoke(
            CLAUSE_ANALYSIS_PROMPT.format(
                clause=state["clause"],
                illegal_matches=state.get("illegal_text", "해당 없음"),
                normal_matches=state.get("normal_text", "해당 없음"),
                law_context=state.get("law_text", "관련 법률 없음"),
            )
        )
        return {
            "result": _ground_clause_result(
                result,
                state.get("law_references", []),
                state["clause"],
            )
        }

    graph = StateGraph(ClauseRiskGraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank_and_recover", rerank_and_recover)
    graph.add_node("prepare_evidence", prepare_evidence)
    graph.add_node("analyze", analyze)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank_and_recover")
    graph.add_edge("rerank_and_recover", "prepare_evidence")
    graph.add_edge("prepare_evidence", "analyze")
    graph.add_edge("analyze", END)
    return graph.compile()


async def _analyze_single_clause(
    clause: str,
    client: VectorDB,
    embeddings: KUREEmbeddings,
    structured_llm,
    graph=None,
) -> ClauseRisk:
    """특약 조항 1개를 LangGraph 기반 RAG 파이프라인으로 분석한다."""
    clause_graph = graph or _build_clause_risk_graph(client, embeddings, structured_llm)
    state = await clause_graph.ainvoke({"clause": clause})
    return state["result"]


async def _detect_risk_legacy(
    user_clause: str,
    client: VectorDB,
    embeddings: KUREEmbeddings,
    llm: ChatOpenAI,
) -> dict:
    """특약사항 섹션을 찾지 못했을 때 계약서 전문을 통째로 분석하는 폴백."""
    search_query = user_clause[:500]
    logger.debug("[Risk:legacy] 임베딩 시작")
    query_vector = await asyncio.to_thread(embeddings.embed_query, search_query)

    law_statutes_filter = infer_law_statutes_filter(search_query)
    (illegal_results, normal_results, law_db_results, law_statutes_results) = await asyncio.gather(
        asyncio.to_thread(search_collection, client, embeddings, search_query, "special_clauses_illegal", 5, None, 0.0, query_vector),
        asyncio.to_thread(search_collection, client, embeddings, search_query, "special_clauses_normal", 3, None, 0.0, query_vector),
        asyncio.to_thread(search_collection, client, embeddings, search_query, "law_database", 3, None, 0.0, query_vector),
        asyncio.to_thread(search_collection, client, embeddings, search_query, "law_statutes", 3, law_statutes_filter, 0.0, query_vector),
    )
    law_results = law_db_results + law_statutes_results
    logger.debug(f"[Risk:legacy] 검색 완료 - illegal:{len(illegal_results)} normal:{len(normal_results)} law:{len(law_results)}")

    illegal_text = "\n".join(f"- ({d.metadata.get('category', '')}) {d.page_content}" for d in illegal_results) or "해당 없음"
    normal_text = "\n".join(f"- ({d.metadata.get('category', '')}) {d.page_content}" for d in normal_results) or "해당 없음"
    law_text = "\n".join(d.page_content for d in law_results) or "관련 법률 없음"

    _llm_start = time.perf_counter()
    response = await llm.ainvoke(
        CONTRACT_RISK_PROMPT.format(
            contract_text=user_clause[:3000],
            illegal_matches=illegal_text,
            normal_matches=normal_text,
            law_context=law_text,
        )
    )
    LLM_LATENCY.observe(time.perf_counter() - _llm_start)
    observe_rag_interaction(
        endpoint="detect_risk_contract_legacy",
        answer=response.content,
        documents=[*illegal_results, *normal_results, *law_results],
        context="\n".join([illegal_text, normal_text, law_text]),
    )

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\s*\n?', '', content)
            content = re.sub(r'\n?```\s*$', '', content.strip())
        result = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return {"overall_risk_score": 0, "risk_summary": {"Risk": 0, "Caution": 0, "Safety": 0}, "total_clauses": 0, "clauses": []}

    if "risk_summary" not in result:
        clauses = result.get("clauses", [])
        result["risk_summary"] = {
            "Risk": sum(1 for c in clauses if c.get("risk_level") == "위험"),
            "Caution": sum(1 for c in clauses if c.get("risk_level") == "주의"),
            "Safety": sum(1 for c in clauses if c.get("risk_level") == "안전"),
        }
    if "total_clauses" not in result:
        result["total_clauses"] = len(result.get("clauses", []))
    # legacy 경로: related_law → legal_reference 키 통일
    for clause in result.get("clauses", []):
        if "related_law" in clause and "legal_reference" not in clause:
            clause["legal_reference"] = clause.pop("related_law")
    return result


@traceable()
async def detect_risk_contract(
    user_clause: str,
    client: VectorDB,
    embeddings: KUREEmbeddings,
    llm: ChatOpenAI,
) -> dict:
    """계약서에서 특약사항을 추출하여 조항별로 위험/주의/안전 분류 및 점수 반환.

    파이프라인:
    1. 특약사항 섹션 추출 → 조항 단위 분리
    2. 각 조항별 독립 RAG 검색 (special_clauses_illegal/normal/law_statutes)
    3. BGE Reranker로 법령 문서 관련성 평가
    4. CRAG: Reranker 점수 낮으면 쿼리 재작성 후 재검색
    5. with_structured_output(ClauseRisk)으로 JSON 파싱 없이 안전하게 분석

    특약사항 섹션을 찾지 못하면 _detect_risk_legacy()로 폴백.

    Returns:
        {
          "overall_risk_score": int,
          "risk_summary": {"Risk": int, "Caution": int, "Safety": int},
          "total_clauses": int,
          "clauses": [{"text", "risk_level", "category", "analysis", "related_law", "score"}, ...]
        }
    """
    special_clauses = _extract_special_clauses(user_clause)
    if not special_clauses:
        logger.warning("[Risk] 특약사항 섹션을 찾지 못함 — 레거시 방식으로 폴백")
        return await _detect_risk_legacy(user_clause, client, embeddings, llm)

    logger.debug(f"[Risk] 특약사항 {len(special_clauses)}개 추출, 병렬 분석 시작")
    structured_llm = llm.with_structured_output(ClauseRisk)
    clause_graph = _build_clause_risk_graph(client, embeddings, structured_llm)

    raw_results = await asyncio.gather(
        *[
            _analyze_single_clause(
                c,
                client,
                embeddings,
                structured_llm,
                graph=clause_graph,
            )
            for c in special_clauses
        ],
        return_exceptions=True,
    )

    valid_clauses: list[ClauseRisk] = []
    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            logger.error(f"[Risk] 조항 {i + 1} 분석 실패: {result}")
            valid_clauses.append(ClauseRisk(
                text=special_clauses[i],
                risk_level="주의",
                category="분석 오류",
                analysis="분석 중 오류가 발생했습니다.",
                legal_reference="",
                score=50,
            ))
        else:
            valid_clauses.append(result)

    risk_count = sum(1 for c in valid_clauses if c.risk_level == "위험")
    caution_count = sum(1 for c in valid_clauses if c.risk_level == "주의")
    safety_count = sum(1 for c in valid_clauses if c.risk_level == "안전")
    total = len(valid_clauses)

    avg_score = sum(c.score for c in valid_clauses) / max(total, 1)
    weight_score = (risk_count / max(total, 1)) * 100
    overall_score = min(int(avg_score * 0.6 + weight_score * 0.4), 100)

    logger.debug(f"[Risk] 분석 완료 — 위험:{risk_count} 주의:{caution_count} 안전:{safety_count} 종합:{overall_score}")

    return {
        "overall_risk_score": overall_score,
        "risk_summary": {"Risk": risk_count, "Caution": caution_count, "Safety": safety_count},
        "total_clauses": total,
        "clauses": [c.model_dump() for c in valid_clauses],
    }


_CHAT_COLLECTIONS = [
    "law_database",
    "law_statutes",
    "contracts",
    "special_clauses_illegal",
    "special_clauses_normal",
]

# 챗봇에서 법령 관련 컬렉션 우선 검색 수 (법률 근거 답변 강화)
_CHAT_K_PER_COLLECTION: dict[str, int] = {
    "law_database": 4,           # 법률 조문·판례 — 법적 근거의 핵심 소스
    "law_statutes": 4,           # 각종 법령 원문 — 조문 번호 직접 인용 소스
    "contracts": 2,              # 표준 계약서 템플릿
    "special_clauses_illegal": 3,  # 독소조항 사례
    "special_clauses_normal": 2,   # 정상조항 사례
}

# 임대차 관련 키워드 — 해당 키워드가 포함된 질의는 법령 검색 수 증가
_LEASE_RELATED_KEYWORDS = [
    "임대차", "임차인", "임대인", "세입자", "집주인", "전세", "월세",
    "보증금", "계약갱신", "대항력", "전입신고", "확정일자", "퇴거",
    "명도", "경매", "임차권", "주택임대차", "상가임대차", "계약해지",
]

# 임대차와 무관한 OOS 키워드
_OOS_RE_CHAT = re.compile(
    r"(주식|펀드|채권|가상화폐|코인|비트코인|이더리움"
    r"|부동산\s*매매|분양|재개발|재건축"
    r"|양도소득세|증여세|종합부동산세|증여|상속세"
    r"|형사(?:소송|처벌|고소|고발)|가족법|이혼|유언"
    r"|교통\s*사고|자동차\s*사고|차량\s*사고|중앙선\s*침범|추돌\s*사고|교통법규"
    r"|불법\s*행위|손해\s*배상(?!\s*청구.*임대)|과실\s*비율|인신\s*사고"
    r"|의료\s*사고|의료\s*분쟁|의료\s*과실|의료\s*소송"
    r"|노동\s*법|해고|임금\s*체불|근로\s*계약|퇴직금|산재"
    r"|저작권|특허|상표|지식\s*재산"
    r"|날씨|여행|요리|건강|의학|병원|게임|영화"
    r"|코딩|프로그래밍|파이썬|자바|수학|과학|역사"
    r"|정치|선거|종교|신앙|주가|환율)"
)

_OOS_RESOURCE_PATTERNS_CHAT: list[tuple[re.Pattern, str]] = [
    (re.compile(r"교통\s*사고|자동차\s*사고|차량\s*사고|중앙선|추돌|과실\s*비율|교통법규"),
     "교통사고 관련 문의는 도로교통공단(1577-1120) 또는 손해보험협회(1566-8000)에 문의하세요."),
    (re.compile(r"의료\s*사고|의료\s*분쟁|의료\s*과실|의료\s*소송"),
     "의료분쟁은 한국의료분쟁조정중재원(1670-2545)에 문의하세요."),
    (re.compile(r"노동\s*법|해고|임금\s*체불|근로\s*계약|퇴직금|산재"),
     "노동 관련 분쟁은 고용노동부 상담센터(1350)에 문의하세요."),
    (re.compile(r"이혼|가족법|상속|유언|친권"),
     "가족·상속 분쟁은 대한가정법률복지상담원(02-6952-9555)에 문의하세요."),
    (re.compile(r"형사|고소|고발|처벌|범죄"),
     "형사 사건은 대한법률구조공단(132)에서 무료 법률 상담을 받으실 수 있습니다."),
]
_OOS_RESOURCE_DEFAULT_CHAT = "법률구조공단(132) 또는 대한변호사협회 법률상담(02-3476-6500)을 이용하시길 권장드립니다."

_REJECTION_BASE_CHAT = (
    "죄송합니다. 저는 임대차 계약 관련 질문만 답변할 수 있습니다.\n"
    "전세·월세 계약, 보증금, 계약갱신, 독소조항 분석 등에 대해 질문해 주세요."
)


def _build_chat_rejection(message: str) -> str:
    for pattern, resource in _OOS_RESOURCE_PATTERNS_CHAT:
        if pattern.search(message):
            return f"{_REJECTION_BASE_CHAT}\n\n관련 기관 안내: {resource}"
    return f"{_REJECTION_BASE_CHAT}\n\n관련 기관 안내: {_OOS_RESOURCE_DEFAULT_CHAT}"


def _is_lease_related(query: str) -> bool:
    """질의가 임대차 계약 관련인지 판별."""
    return any(kw in query for kw in _LEASE_RELATED_KEYWORDS)


@traceable()
async def chat_rag(
    message: str,
    history: list[dict],
    client: VectorDB,
    embeddings: KUREEmbeddings,
    llm: ChatOpenAI,
    contract_context: str | None = None,
    collections: list[str] | None = None,
    k_per_collection: int | dict[str, int] | None = None,
    use_hyde: bool = True,
    use_multiquery: bool = False,
    use_compression: bool = False,
) -> dict:
    """대화 이력을 포함한 RAG 챗봇 (병렬 검색 + Reranker).

    임대차 관련 질문이면 law_database/law_statutes 컬렉션 검색 수를 자동으로 늘려
    법률 근거 기반 답변 품질을 높인다.

    Args:
        message: 현재 사용자 메시지.
        history: 이전 대화 이력 [{"role": "user"|"assistant", "content": str}, ...].
                 최근 10턴만 사용됨.
        client: VectorDB 인스턴스.
        embeddings: 임베딩 모델.
        llm: ChatOpenAI 인스턴스.
        contract_context: 사용자가 현재 보고 있는 계약서 텍스트 (선택).
        collections: 검색할 컬렉션 리스트. None이면 전체 5개 컬렉션 사용.
        k_per_collection: 컬렉션당 검색 수. None이면 임대차 여부에 따라 자동 결정.
        use_multiquery: HyDE와 병행하여 쿼리 변형 2개를 추가 검색.
        use_compression: Contextual Compression — 긴 문서에서 관련 부분만 추출.

    Returns:
        {"answer": str, "sources": list[str], "context": str,
         "source_documents": list[Document]}
    """
    # OOS 사전 필터 — LLM 호출 전 명확한 범위 외 질문 차단
    if _OOS_RE_CHAT.search(message) and not _is_lease_related(message):
        return {
            "answer": _build_chat_rejection(message),
            "sources": [],
            "context": "",
            "source_documents": [],
        }

    if collections is None:
        collections = _CHAT_COLLECTIONS

    # 임대차 관련 질의: law_database/law_statutes 검색 수 확대
    if k_per_collection is None:
        if _is_lease_related(message):
            k_per_collection = _CHAT_K_PER_COLLECTION
            logger.debug(f"[chat_rag] 임대차 관련 질의 감지 — 법령 검색 수 확대: {k_per_collection}")
        else:
            k_per_collection = 2  # 비임대차 질의는 컬렉션당 2개

    collection_filters = {}
    law_statutes_filter = infer_law_statutes_filter(message)
    if law_statutes_filter and "law_statutes" in collections:
        collection_filters["law_statutes"] = law_statutes_filter

    # 1. 쿼리 확장: HyDE + Multi-query 병렬 생성
    expand_coros: list = []
    if use_hyde:
        expand_coros.append(async_expand_query_hyde(message, llm))
    if use_multiquery:
        expand_coros.append(async_expand_query_multi(message, llm, n=2))

    expand_results = await asyncio.gather(*expand_coros) if expand_coros else []

    hyde_vector: list[float] | None = None
    mq_variants: list[str] = []
    _idx = 0
    if use_hyde and expand_results:
        hyde_vector = await asyncio.to_thread(embeddings.embed_query, expand_results[_idx])
        _idx += 1
    if use_multiquery and _idx < len(expand_results):
        mq_variants = expand_results[_idx][1:]  # 원본 제외, 변형 쿼리만

    # 2. 병렬 검색: HyDE 기반 메인 + Multi-query 변형 쿼리들
    _score_threshold = {
        "law_database": 0.15,
        "law_statutes": 0.15,
        "special_clauses_illegal": 0.45,
        "special_clauses_normal": 0.4,
        "default": 0.25,
    }
    _search_kwargs = dict(
        collections=collections,
        k_per_collection=k_per_collection,
        score_threshold=_score_threshold,
        collection_filters=collection_filters or None,
    )
    search_coros = [
        async_search_multi_index(client, embeddings, message, **_search_kwargs, query_vector=hyde_vector),
        *[
            async_search_multi_index(client, embeddings, q, **_search_kwargs)
            for q in mq_variants
        ],
    ]
    search_results = await asyncio.gather(*search_coros)
    all_docs: list[Document] = [doc for r in search_results for doc in r]
    docs = _deduplicate(all_docs)
    docs.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)

    # 3. BGE Reranker: 법령 조문 관련성 기준으로 재정렬 (상위 7개 선별)
    if docs:
        docs = await get_reranker().async_rerank(message, docs, top_n=7)

    if not _collect_cited_laws(docs):
        legal_collections = [c for c in collections if c in _LEGAL_COLLECTIONS]
        if legal_collections:
            fallback_query = f"{message}\n관련 법령 조문과 직접 적용 근거"
            fallback_docs = await async_search_multi_index(
                client,
                embeddings,
                fallback_query,
                collections=legal_collections,
                k_per_collection={"law_database": 5, "law_statutes": 5},
                score_threshold={"law_database": 0.15, "law_statutes": 0.15, "default": 0.15},
                collection_filters=collection_filters or None,
            )
            if fallback_docs:
                logger.debug(
                    "[chat_rag] 법령 인용 후보가 비어 fallback 검색 수행: {}건",
                    len(fallback_docs),
                )
                fallback_docs = await get_reranker().async_rerank(message, fallback_docs, top_n=5)
                docs = _deduplicate(docs + fallback_docs)
                docs = await get_reranker().async_rerank(message, docs, top_n=7)

    # citation verification은 rerank 직후 원본 텍스트로 수행
    verification_source = "\n".join(d.page_content for d in docs)

    # 4. Contextual Compression: 긴 문서에서 질문 관련 부분만 추출
    if use_compression and docs:
        docs = await compress_documents(docs, message, llm)

    context = build_context(docs, max_length=3000)  # 법령 조문 전문 수용

    # 계약서 컨텍스트가 있으면 앞에 붙임
    if contract_context:
        context = f"[사용자 계약서 원문 요약]\n{contract_context[:800]}\n\n{context}"

    # 3. 대화 이력 → LangChain 메시지 변환 (최근 10턴)
    lc_history = []
    for msg in history[-10:]:
        if msg.get("role") == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            lc_history.append(AIMessage(content=msg["content"]))

    # 4. 프롬프트 구성 및 LLM 비동기 호출
    prompt_messages = CHAT_PROMPT.format_messages(
        context=context,
        history=lc_history,
        question=message,
    )
    _llm_start = time.perf_counter()
    response = await llm.ainvoke(prompt_messages)
    LLM_LATENCY.observe(time.perf_counter() - _llm_start)

    # 5. Citation Verification: 검색 소스에 없는 조문 인용에 (미검증) 표시
    answer = annotate_unverified_citations(response.content, verification_source)
    observe_rag_interaction(
        endpoint="chat_rag",
        answer=answer,
        documents=docs,
        context=context,
    )

    # 6. 출처 요약 (상위 3개)
    sources = []
    for doc in docs[:3]:
        meta = doc.metadata
        parts = [p for p in [meta.get("law_name"), meta.get("article") or meta.get("title") or meta.get("category")] if p]
        coll = meta.get("collection", "")
        sources.append(f"[{coll}] {' '.join(parts)}".strip(" []"))

    return {
        "answer": answer,
        "sources": sources,
        "context": context,
        "source_documents": docs,
    }


@traceable()
async def explain_term_plain(
    term: str,
    llm: ChatOpenAI,
    context: str = "",
    surrounding_text: str = "",
) -> dict:
    """검색 없이 쉬운말 중심으로 법률 용어를 설명한다."""
    prompt_text = TERM_PLAIN_EXPLANATION_PROMPT.format(
        term=term,
        context=context or "임대차 계약",
        surrounding_text=surrounding_text or "없음",
    )
    _llm_start = time.perf_counter()
    response = await llm.ainvoke(prompt_text)
    LLM_LATENCY.observe(time.perf_counter() - _llm_start)
    return _parse_term_explanation_result(response.content)


@traceable()
async def explain_term_auto(
    sentence: str,
    llm: ChatOpenAI,
    client_factory: Callable[[], VectorDB],
    embeddings_factory: Callable[[], KUREEmbeddings],
    term: str = "",
    mode: str = "auto",
) -> dict:
    """문장에서 용어를 추출하고 plain/RAG 경로를 자동 선택한다."""
    normalized_sentence = _normalize_whitespace(sentence)
    explicit_term = _normalize_term_candidate(term)

    extracted = {
        "term": explicit_term,
        "context": _extract_term_context(normalized_sentence, explicit_term),
        "candidates": [explicit_term] if explicit_term else [],
        "strategy": "explicit_term",
    } if explicit_term else await extract_term_from_sentence(normalized_sentence, llm)

    resolved_term = extracted.get("term") or explicit_term or _normalize_term_candidate(normalized_sentence)
    context = extracted.get("context") or _extract_term_context(normalized_sentence, resolved_term)

    if mode == "rag":
        use_rag, route_reason = True, "forced_rag"
    elif mode == "plain":
        use_rag, route_reason = False, "forced_plain"
    else:
        use_rag, route_reason = should_use_rag_for_term(
            resolved_term,
            normalized_sentence,
            context,
            strategy=str(extracted.get("strategy") or ""),
        )

    if use_rag:
        result = await explain_term_rag(
            term=resolved_term,
            client=client_factory(),
            embeddings=embeddings_factory(),
            llm=llm,
            context=context,
            surrounding_text=normalized_sentence,
        )
        route = "rag"
    else:
        result = await explain_term_plain(
            term=resolved_term,
            llm=llm,
            context=context,
            surrounding_text=normalized_sentence,
        )
        route = "plain"

    logger.info(
        "[explain_term_auto] term='{}' route={} reason={} strategy={}",
        resolved_term,
        route,
        route_reason,
        extracted.get("strategy"),
    )

    result["term"] = resolved_term
    result["route"] = route
    result["route_reason"] = route_reason
    return result


@traceable()
async def explain_term_rag(
    term: str,
    client: VectorDB,
    embeddings: KUREEmbeddings,
    llm: ChatOpenAI,
    context: str = "",
    surrounding_text: str = "",
) -> dict:
    """RAG 기반 법률 용어 해설.

    law_database 컬렉션에서 관련 법조문을 검색한 뒤 LLM으로 용어를 설명한다.

    Args:
        term: 해설할 법률 용어.
        client: VectorDB 인스턴스.
        embeddings: 임베딩 모델.
        llm: ChatOpenAI 인스턴스.
        context: 용어가 등장한 문맥 (예: "주택임대차보호법").
        surrounding_text: 용어 주변 문장.

    Returns:
        {"simple_explanation": str, "legal_definition": str, "examples": list[str]}
    """
    search_query = " ".join(filter(None, [term, context, surrounding_text]))
    collection_filters = {}
    law_statutes_filter = infer_law_statutes_filter(search_query)
    if law_statutes_filter:
        collection_filters["law_statutes"] = law_statutes_filter

    docs = await async_search_multi_index(
        client, embeddings, search_query,
        collections=["law_database", "law_statutes"],
        k_per_collection=4,
        score_threshold=0.15,
        collection_filters=collection_filters or None,
    )
    law_context = build_context(docs, max_length=2000) or "관련 법률 문서를 찾을 수 없습니다."

    prompt_text = TERM_EXPLANATION_PROMPT.format(
        term=term,
        context=context or "임대차 계약",
        surrounding_text=surrounding_text or "없음",
        law_context=law_context,
    )

    _llm_start = time.perf_counter()
    response = await llm.ainvoke(prompt_text)
    LLM_LATENCY.observe(time.perf_counter() - _llm_start)
    observe_rag_interaction(
        endpoint="explain_term_rag",
        answer=response.content,
        documents=docs,
        context=law_context,
    )
    return _parse_term_explanation_result(response.content)


class RagBot:
    """Multi-Index RAG 봇 (LangSmith 추적 지원).

    Args:
        client: VectorDB 인스턴스.
        embeddings: 임베딩 모델.
        collections: 검색 대상 컬렉션 리스트.
        model: OpenAI 모델명.
    """

    def __init__(
        self,
        client: VectorDB,
        embeddings: KUREEmbeddings,
        collections: list[str],
        model: str = "gpt-4o-mini",
    ):
        self._openai_client = wrap_openai(openai.Client())
        self._vector_db = client
        self._embeddings = embeddings
        self._collections = collections
        self._model = model

    @traceable()
    def retrieve_docs(self, question: str) -> list[Document]:
        collection_filters = {}
        law_statutes_filter = infer_law_statutes_filter(question)
        if law_statutes_filter and "law_statutes" in self._collections:
            collection_filters["law_statutes"] = law_statutes_filter
        return search_multi_index(
            self._vector_db, self._embeddings, question,
            collections=self._collections, k_per_collection=3,
            collection_filters=collection_filters or None,
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


# LangGraph public-pipeline overrides
# Keep the public signatures stable for API, RabbitMQ, voice, and evaluation callers,
# while routing the main chat/risk paths through the time-bounded tool-calling graphs.


@traceable()
def detect_risk(
    user_clause: str,
    client: VectorDB,
    embeddings: KUREEmbeddings,
    llm: ChatOpenAI,
) -> dict:
    from app.rag.chain.langgraph_tool_pipelines import run_single_clause_risk_langgraph

    return asyncio.run(
        run_single_clause_risk_langgraph(
            clause_text=user_clause,
            client=client,
            embeddings=embeddings,
            llm=llm,
        )
    )


@traceable()
async def detect_risk_contract(
    user_clause: str,
    client: VectorDB,
    embeddings: KUREEmbeddings,
    llm: ChatOpenAI,
) -> dict:
    from app.rag.chain.langgraph_tool_pipelines import run_risk_contract_langgraph

    return await run_risk_contract_langgraph(
        contract_text=user_clause,
        client=client,
        embeddings=embeddings,
        llm=llm,
    )


@traceable()
async def chat_rag(
    message: str,
    history: list[dict],
    client: VectorDB,
    embeddings: KUREEmbeddings,
    llm: ChatOpenAI,
    contract_context: str | None = None,
    collections: list[str] | None = None,
    k_per_collection: int | dict[str, int] | None = None,
    use_hyde: bool = True,
    use_multiquery: bool = False,
    use_compression: bool = False,
) -> dict:
    from app.rag.chain.langgraph_tool_pipelines import run_chat_langgraph

    return await run_chat_langgraph(
        message=message,
        history=history,
        client=client,
        embeddings=embeddings,
        llm=llm,
        contract_context=contract_context,
        collections=collections,
        k_per_collection=k_per_collection,
        use_hyde=use_hyde,
        use_multiquery=use_multiquery,
        use_compression=use_compression,
    )
