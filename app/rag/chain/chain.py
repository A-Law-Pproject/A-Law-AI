import asyncio
import json
import re
import time
import tiktoken
from functools import lru_cache

from loguru import logger

import openai
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai

from langchain_core.messages import HumanMessage, AIMessage

from app.monitoring.metrics import LLM_LATENCY
from app.monitoring.llmops_metrics import observe_rag_interaction
from app.rag.chain.prompts import (
    CHAT_PROMPT,
    CLAUSE_ANALYSIS_PROMPT,
    COMPRESSION_PROMPT,
    CONTRACT_QA_PROMPT,
    CONTRACT_RISK_PROMPT,
    RISK_PROMPT,
    TERM_EXPLANATION_PROMPT,
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
    r"|민간임대주택에\s*관한\s*특별법|전세사기[^\s,。]*특별법)"
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

    한국 임대차 계약서의 특약사항은 모든 항목이 '1.'으로 시작하는 관행을 이용해 파싱한다.
    섹션 종료 경계는 '*비상연락망', '-이하 여백-', '증명하기 위하여' 중 먼저 등장하는 것.
    """
    section_match = re.search(
        r'\[\s*특약사항\s*\](.*?)(?:\*비상연락망|-이하 여백-|증명하기 위하여|\Z)',
        contract_text,
        re.DOTALL,
    )
    if not section_match:
        return []

    section = section_match.group(1).strip()
    items = re.findall(r'1\.(.+?)(?=\n1\.|\Z)', section, re.DOTALL)
    return [item.strip() for item in items if len(item.strip()) > 5]


async def _analyze_single_clause(
    clause: str,
    client: VectorDB,
    embeddings: KUREEmbeddings,
    structured_llm,
) -> ClauseRisk:
    """특약 조항 1개 분석: 조항별 RAG 검색 → BGE Reranker → CRAG → Structured Output.

    CRAG 패턴: Reranker 점수가 _RERANK_LOW_SCORE 미만이면 법령 검색 특화 쿼리로
    재작성 후 1회 재검색. 재검색 결과가 더 좋을 때만 교체.
    """
    query_vector = await asyncio.to_thread(embeddings.embed_query, clause)
    law_filter = infer_law_statutes_filter(clause)

    illegal_docs, normal_docs, law_docs = await asyncio.gather(
        asyncio.to_thread(
            search_collection, client, embeddings, clause,
            "special_clauses_illegal", 3, None, 0.0, query_vector,
        ),
        asyncio.to_thread(
            search_collection, client, embeddings, clause,
            "special_clauses_normal", 2, None, 0.0, query_vector,
        ),
        asyncio.to_thread(
            search_collection, client, embeddings, clause,
            "law_statutes", 3, law_filter, 0.0, query_vector,
        ),
    )

    reranker = get_reranker()
    if law_docs:
        law_docs = await reranker.async_rerank(clause, law_docs, top_n=3)

    # CRAG: 법령 관련성 점수가 낮으면 쿼리를 법령 검색에 특화된 형태로 재작성
    best_score = max(
        (d.metadata.get("rerank_score", -99.0) for d in law_docs), default=-99.0
    )
    if best_score < _RERANK_LOW_SCORE:
        rewritten = f"{clause} 관련 법률 위반 여부 주택임대차보호법 민법"
        rw_vector = await asyncio.to_thread(embeddings.embed_query, rewritten)
        rw_filter = infer_law_statutes_filter(rewritten)
        new_law_docs = await asyncio.to_thread(
            search_collection, client, embeddings, rewritten,
            "law_statutes", 5, rw_filter, 0.0, rw_vector,
        )
        if new_law_docs:
            new_law_docs = await reranker.async_rerank(clause, new_law_docs, top_n=3)
            new_best = max(d.metadata.get("rerank_score", -99.0) for d in new_law_docs)
            if new_best > best_score:
                logger.debug(f"[CRAG] 쿼리 재작성 후 법령 점수 개선: {best_score:.2f} → {new_best:.2f}")
                law_docs = new_law_docs

    illegal_text = "\n".join(
        f"- ({d.metadata.get('category', '')}) {d.page_content}" for d in illegal_docs
    ) or "해당 없음"
    normal_text = "\n".join(
        f"- ({d.metadata.get('category', '')}) {d.page_content}" for d in normal_docs
    ) or "해당 없음"
    law_text = "\n".join(d.page_content for d in law_docs[:2]) or "관련 법률 없음"

    return await structured_llm.ainvoke(
        CLAUSE_ANALYSIS_PROMPT.format(
            clause=clause,
            illegal_matches=illegal_text,
            normal_matches=normal_text,
            law_context=law_text,
        )
    )


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

    raw_results = await asyncio.gather(
        *[_analyze_single_clause(c, client, embeddings, structured_llm) for c in special_clauses],
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

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\s*\n?', '', content)
            content = re.sub(r'\n?```\s*$', '', content.strip())
        return json.loads(content)
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
