from __future__ import annotations

import asyncio
import json
import re
import time
from collections import defaultdict
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from loguru import logger

from app.core.dependencies import get_fast_llm, get_law_api_client, get_redis_client
from app.monitoring.llmops_metrics import observe_rag_interaction
from app.monitoring.metrics import (
    LLM_LATENCY,
    RAG_PIPELINE_STAGE_LATENCY,
    RAG_RETRIEVAL_CACHE_TOTAL,
)
from app.rag.retriever.law_graph import get_related_laws
from app.rag.retriever.multi_retriever import (
    _deduplicate,
    async_search_multi_index,
    infer_law_statutes_filter,
    search_collection,
)
from app.rag.retriever.query_expansion import async_expand_query_hyde, async_expand_query_multi
from app.rag.retriever.reranker import get_reranker
from app.rag.retriever.search_cache import (
    SEARCH_CACHE_TTL_SECONDS,
    build_search_cache_key,
    deserialize_documents,
    serialize_documents,
)
from app.rag.vector_store.base import VectorDB
from app.rag.embedding.kure import KUREEmbeddings
from app.schemas.risk_analysis import ClauseRisk, ContractRiskResult

# NOTE:
# This module is imported lazily from app.rag.chain.chain after that module is
# fully loaded. Importing shared helpers back from chain.py is therefore safe.
from app.rag.chain.chain import (  # noqa: E402
    CHAT_PROMPT,
    _CHAT_COLLECTIONS,
    _CHAT_K_PER_COLLECTION,
    _collect_cited_laws,
    _detect_risk_legacy,
    _extract_special_clauses,
    _format_grounded_law_context,
    _ground_clause_result,
    _grounded_law_references,
    _is_lease_related,
    _law_search_query_for_clause,
    _calibrate_clause_result,
    _is_owner_succession_safe_clause,
    _is_overbroad_restoration_caution_clause,
    _is_repair_burden_caution_clause,
    _is_safe_preservation_clause,
    _is_early_termination_caution_clause,
    annotate_unverified_citations,
    build_context,
    compress_documents,
)


CHAT_TIMEOUT_SECONDS = 25.0
RISK_TIMEOUT_SECONDS = 30.0
CHAT_MAX_TOOL_ROUNDS = 1
RISK_MAX_TOOL_ROUNDS = 2
RISK_MAX_LLM_CLAUSES = 6

# 컴파일된 그래프 캐시 — client/embeddings/llm은 싱글톤이므로 요청마다 재빌드 불필요
_CHAT_GRAPH_CACHE: dict[tuple[int, int, int], Any] = {}


def _remaining_seconds(deadline: float) -> float:
    return max(0.0, deadline - time.perf_counter())


def _sort_docs(docs: list[Document]) -> list[Document]:
    return sorted(
        _deduplicate(docs),
        key=lambda doc: doc.metadata.get("rerank_score", doc.metadata.get("score", 0)),
        reverse=True,
    )


_SOURCE_COLLECTION_LABELS = {
    "law_database": "판례·해설",
    "law_statutes": "법령",
    "contracts": "계약서 예시",
    # "special_clauses_illegal": "독소조항 사례",
    # "special_clauses_normal": "일반조항 사례",
}


def _display_source_collection(collection: str) -> str:
    return _SOURCE_COLLECTION_LABELS.get(collection, "참고문서")


def _format_sources(docs: list[Document], limit: int = 3) -> list[str]:
    sources: list[str] = []
    for doc in docs[:limit]:
        meta = doc.metadata
        coll = meta.get("collection", "")
        parts = [
            part
            for part in [
                meta.get("law_name"),
                meta.get("article") or meta.get("title") or meta.get("category"),
            ]
            if part
        ]
        collection_label = _display_source_collection(coll)
        label = f"[{collection_label}] {' '.join(parts)}" if parts else f"[{collection_label}]"

        # law_statutes / law_database: 조항 핵심 내용 첫 문장 추가
        if coll in ("law_statutes", "law_database") and doc.page_content:
            snippet = doc.page_content.split("\n")[0].strip()[:100]
            if snippet:
                label += f" — {snippet}"

        sources.append(label)
    return sources


def _ensure_readable_markdown_answer(answer: str) -> str:
    text = (answer or "").replace("\r\n", "\n").strip()
    if not text:
        return ""

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Separate inline markdown sections and warnings so the renderer can preserve structure.
    text = re.sub(r"(?<!\n)(?=##\s)", "\n\n", text)
    text = re.sub(r"(?<!\n)[ \t]+(?=>\s)", "\n\n", text)

    # Split numbered or bold bullet lists that the model sometimes emits on one long line.
    text = re.sub(r"(?<=:)[ \t]+(?=(?:\d+\.\s+|-\s+\*\*))", "\n\n", text)
    text = re.sub(r"(?<!\n)[ \t]+(?=(?:\d+\.\s+|-\s+\*\*))", "\n", text)

    # Move wrap-up sentences out of the last list item when they start with common connectives.
    text = re.sub(
        r"(?<=[.!?])[ \t]+(?=(?:따라서|정리하면|결론적으로|즉,|즉\s|다만|한편|추가로))",
        "\n\n",
        text,
    )

    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _compact_history(history: list[dict], limit: int = 6) -> str:
    lines: list[str] = []
    for item in history[-limit:]:
        role = "사용자" if item.get("role") == "user" else "도우미"
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content[:280]}")
    return "\n".join(lines) if lines else "(이전 대화 없음)"


def _live_statute_docs_from_payload(payload: dict[str, Any]) -> list[Document]:
    law = payload.get("law") or {}
    law_name = str(law.get("law_name") or "").strip()
    source_url = str(law.get("source_url") or "").strip()

    docs: list[Document] = []
    for snippet in payload.get("snippets") or []:
        content = "\n".join(
            part
            for part in [
                snippet.get("article"),
                snippet.get("title"),
                snippet.get("content"),
            ]
            if part
        ).strip()
        if not content:
            continue
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "collection": "law_mcp",
                    "law_name": law_name,
                    "article": str(snippet.get("article") or "").strip(),
                    "title": str(snippet.get("title") or "").strip(),
                    "source_url": source_url,
                    "source_type": "live_statute",
                },
            )
        )

    if docs:
        return docs

    for item in payload.get("results") or []:
        title = str(item.get("law_name") or "").strip()
        content = " ".join(
            part
            for part in [
                item.get("promulgation_date"),
                item.get("effective_date"),
                item.get("department"),
            ]
            if part
        ).strip()
        if not title:
            continue
        docs.append(
            Document(
                page_content=f"{title}\n{content}".strip(),
                metadata={
                    "collection": "law_mcp",
                    "law_name": title,
                    "source_url": str(item.get("source_url") or "").strip(),
                    "source_type": "live_statute",
                },
            )
        )
    return docs


def _live_precedent_docs_from_payload(payload: dict[str, Any]) -> list[Document]:
    best_match = payload.get("best_match") or {}
    detail = payload.get("detail") or {}

    docs: list[Document] = []
    content_parts = [
        best_match.get("case_name"),
        best_match.get("summary"),
    ]

    for key in ("판시사항", "판결요지", "판례내용", "참조조문", "참조판례"):
        value = detail.get(key)
        if value:
            content_parts.append(f"{key}: {value}")

    content = "\n".join(str(part).strip() for part in content_parts if str(part).strip()).strip()
    if content:
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "collection": "precedent_mcp",
                    "title": str(best_match.get("case_name") or "").strip(),
                    "case_no": str(best_match.get("case_no") or "").strip(),
                    "source_url": str(best_match.get("source_url") or "").strip(),
                    "source_type": "live_precedent",
                },
            )
        )

    for item in payload.get("results") or []:
        title = str(item.get("case_name") or "").strip()
        if not title:
            continue
        summary = str(item.get("summary") or "").strip()
        docs.append(
            Document(
                page_content="\n".join(part for part in [title, summary] if part).strip(),
                metadata={
                    "collection": "precedent_mcp",
                    "title": title,
                    "case_no": str(item.get("case_no") or "").strip(),
                    "source_url": str(item.get("source_url") or "").strip(),
                    "source_type": "live_precedent",
                },
            )
        )
    return _sort_docs(docs)


def _extract_relevant_contract_snippets(contract_context: str, query: str, limit: int = 4) -> list[str]:
    if not contract_context:
        return []

    tokens = [
        token
        for token in re.split(r"\s+", query)
        if len(token) >= 2
    ]
    lines = [
        line.strip()
        for line in re.split(r"[\n\.]", contract_context)
        if line.strip()
    ]
    scored: list[tuple[int, str]] = []
    for line in lines:
        score = sum(1 for token in tokens if token in line)
        if score:
            scored.append((score, line))
    if not scored:
        return [contract_context[:220]]
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return [line for _, line in scored[:limit]]


async def _search_docs(
    client: VectorDB,
    embeddings: KUREEmbeddings,
    query: str,
    *,
    collections: list[str],
    k_per_collection: int | dict[str, int],
    score_threshold: float | dict[str, float],
    rerank_top_n: int,
    collection_filters: dict[str, dict] | None = None,
    use_hyde: bool = False,
    use_multiquery: bool = False,
    llm: ChatOpenAI | None = None,
) -> list[Document]:
    cache_key = build_search_cache_key(
        query=query,
        collections=collections,
        k_per_collection=k_per_collection,
        score_threshold=score_threshold,
        collection_filters=collection_filters,
        rerank_top_n=rerank_top_n,
        use_hyde=use_hyde,
        use_multiquery=use_multiquery,
    )
    redis = None
    try:
        redis = await get_redis_client()
        cached_docs = deserialize_documents(await redis.get(cache_key))
        if cached_docs is not None:
            RAG_RETRIEVAL_CACHE_TOTAL.labels(result="hit").inc()
            return cached_docs
        RAG_RETRIEVAL_CACHE_TOTAL.labels(result="miss").inc()
    except Exception as exc:
        redis = None
        RAG_RETRIEVAL_CACHE_TOTAL.labels(result="error").inc()
        logger.debug("[LangGraphSearch] cache skipped: {}", exc)

    search_kwargs = dict(
        collections=collections,
        k_per_collection=k_per_collection,
        score_threshold=score_threshold,
        collection_filters=collection_filters or None,
    )
    retrieval_start = time.perf_counter()
    search_tasks = [
        asyncio.create_task(
            async_search_multi_index(
                client,
                embeddings,
                query,
                **search_kwargs,
            )
        )
    ]

    if llm is not None and (use_hyde or use_multiquery):
        expansion_start = time.perf_counter()
        expansion_tasks = []
        if use_hyde:
            expansion_tasks.append(async_expand_query_hyde(query, llm))
        if use_multiquery:
            expansion_tasks.append(async_expand_query_multi(query, llm, n=2))
        expansion_results = await asyncio.gather(*expansion_tasks, return_exceptions=True)
        RAG_PIPELINE_STAGE_LATENCY.labels(stage="query_expansion").observe(
            time.perf_counter() - expansion_start
        )

        result_index = 0
        if use_hyde:
            hyde_result = expansion_results[result_index]
            result_index += 1
            if isinstance(hyde_result, Exception):
                logger.debug("[LangGraphSearch] hyde skipped: {}", hyde_result)
            elif hyde_result:
                hyde_vector = await asyncio.to_thread(embeddings.embed_query, hyde_result)
                search_tasks.append(
                    asyncio.create_task(
                        async_search_multi_index(
                            client,
                            embeddings,
                            query,
                            **search_kwargs,
                            query_vector=hyde_vector,
                        )
                    )
                )

        if use_multiquery:
            multi_result = expansion_results[result_index]
            if isinstance(multi_result, Exception):
                logger.debug("[LangGraphSearch] multiquery skipped: {}", multi_result)
            else:
                for variant in multi_result[1:]:
                    if variant and variant != query:
                        search_tasks.append(
                            asyncio.create_task(
                                async_search_multi_index(
                                    client,
                                    embeddings,
                                    variant,
                                    **search_kwargs,
                                )
                            )
                        )

    results = await asyncio.gather(*search_tasks)
    RAG_PIPELINE_STAGE_LATENCY.labels(stage="retrieval").observe(
        time.perf_counter() - retrieval_start
    )
    docs = _sort_docs([doc for bucket in results for doc in bucket])
    if docs:
        rerank_start = time.perf_counter()
        docs = await get_reranker().async_rerank(query, docs, top_n=rerank_top_n)
        RAG_PIPELINE_STAGE_LATENCY.labels(stage="rerank").observe(
            time.perf_counter() - rerank_start
        )

    if redis is not None:
        try:
            await redis.setex(
                cache_key,
                SEARCH_CACHE_TTL_SECONDS,
                serialize_documents(docs),
            )
        except Exception as exc:
            logger.debug("[LangGraphSearch] cache store skipped: {}", exc)

    return docs


def _risk_keyword_score(text: str) -> int:
    keyword_weights = {
        "보증금": 3,
        "반환": 2,
        "퇴거": 4,
        "즉시": 2,
        "포기": 5,
        "권리금": 4,
        "증액": 3,
        "인상": 3,
        "임차권등기": 4,
        "대항력": 3,
        "우선변제권": 3,
        "원상복구": 2,
        "자연 마모": 2,
        "몰수": 5,
        "중개보수": 2,
        "계약갱신": 3,
        "갱신요구": 3,
        "수선": 2,
        "수리": 2,
        "위약금": 3,
        "지체": 2,
        "손해배상": 2,
        "임의로": 4,
        "행사하지": 5,
        "주장하지": 5,
        "무조건": 4,
    }
    return sum(weight for keyword, weight in keyword_weights.items() if keyword in text)


def _build_fast_clause_result(text: str, refs: list[str] | None = None) -> ClauseRisk:
    refs = refs or []

    if _is_safe_preservation_clause(text) or _is_owner_succession_safe_clause(text):
        base = ClauseRisk(
            text=text,
            risk_level="안전",
            category="권리 보장",
            analysis="시간 예산 내 빠른 1차 분류 결과, 임차인 권리를 보장하는 조항으로 보입니다.",
            legal_reference="",
            score=18,
        )
        return _ground_clause_result(base, refs, text)

    if _is_repair_burden_caution_clause(text) or _is_overbroad_restoration_caution_clause(text) or _is_early_termination_caution_clause(text):
        base = ClauseRisk(
            text=text,
            risk_level="주의",
            category="비용 부담",
            analysis="시간 예산 내 빠른 1차 분류 결과, 비용 부담이나 분쟁 가능성이 있어 주의가 필요합니다.",
            legal_reference="",
            score=55,
        )
        return _ground_clause_result(base, refs, text)

    risk_score = _risk_keyword_score(text)
    if risk_score >= 8:
        base = ClauseRisk(
            text=text,
            risk_level="주의",
            category="추가 검토 필요",
            analysis="시간 예산 내 빠른 1차 분류 결과, 임차인 권리에 영향을 줄 수 있는 표현이 포함되어 추가 검토가 필요합니다.",
            legal_reference="",
            score=52,
        )
    else:
        base = ClauseRisk(
            text=text,
            risk_level="안전",
            category="일반 조항",
            analysis="시간 예산 내 빠른 1차 분류 결과, 즉시 위험 신호는 크지 않은 일반 조항으로 보입니다.",
            legal_reference="",
            score=22,
        )
    return _ground_clause_result(base, refs, text)


def _recompute_contract_risk(valid_clauses: list[ClauseRisk]) -> dict:
    risk_count = sum(1 for clause in valid_clauses if clause.risk_level == "위험")
    caution_count = sum(1 for clause in valid_clauses if clause.risk_level == "주의")
    safety_count = sum(1 for clause in valid_clauses if clause.risk_level == "안전")
    total = len(valid_clauses)

    avg_score = sum(clause.score for clause in valid_clauses) / max(total, 1)
    weight_score = (risk_count / max(total, 1)) * 100
    overall_score = min(int(avg_score * 0.6 + weight_score * 0.4), 100)

    return {
        "overall_risk_score": overall_score,
        "risk_summary": {"Risk": risk_count, "Caution": caution_count, "Safety": safety_count},
        "total_clauses": total,
        "clauses": [clause.model_dump() for clause in valid_clauses],
    }


# ── 질문 유형 분류 패턴 ───────────────────────────────────────────────────────

# 임대차와 무관한 OOS 키워드 (임대차 관련 단어가 없는 경우에만 적용)
_OOS_RE = re.compile(
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

# 용어 설명 요청 패턴: "X란?", "X의 뜻", "X를 설명해줘" 등 + 메시지가 짧을 때
_TERM_EXPLAIN_RE = re.compile(
    r"(이란|란)\s*(무엇|뭐|뜻|의미|설명)?\s*[?？]?"
    r"|(?:은|는|이|가)\s*(무엇|뭐|뜻|의미)\s*[?？]"
    r"|(?:의)\s*(뜻|의미|설명)\s*(이\s*)?[?？뭐]?"
    r"|설명\s*(해\s*줘|해\s*주세요|해\s*주십시오)"
    r"|뜻\s*(이\s*)?(뭐야|뭔가요|가\s*뭐)"
    r"|무슨\s*말이야|무슨\s*뜻이야"
)

# 특정 조항 위험도 분석 요청: "이 조항 위험한가요", "특약 검토해줘" 등
_CLAUSE_RISK_RE = re.compile(
    r"(이\s*조항|이\s*특약|이\s*내용|아래\s*조항|다음\s*조항|위\s*조항)"
    r"|(특약\s*조항|특약사항)\s*(분석|검토|확인|위험|괜찮|문제)"
    r"|(독소\s*조항|불공정\s*조항)"
    r"|조항\s*(분석|검토|확인)\s*(해|부탁|요청|줘|주세요)"
    r"|이\s*(내용|문구)\s*(위험|괜찮|문제|유효|무효|불리)"
)

# 실무 절차 안내 요청: 등기, 내용증명, 소송 등
_PROCEDURE_RE = re.compile(
    r"임차권\s*등기\s*명령"
    r"|내용\s*증명"
    r"|지급\s*명령"
    r"|(?:보증금\s*반환|명도)\s*소송"
    r"|(?:법원|관할\s*법원)\s*(신청|접수)"
    r"|신청\s*(방법|절차|서류|서식)"
    r"|어떻게\s*(신청|해야|하면)\s*(되나|되는지|할\s*수\s*있나)"
    r"|강제\s*집행"
    r"|소장\s*(작성|제출)"
)


def _classify_query(message: str) -> str:
    """규칙 기반 질문 유형 분류. LLM 호출 없이 O(n) 시간에 처리.

    Returns:
        "out_of_scope" | "term_explain" | "clause_risk" | "procedure_qa" | "legal_qa"
    """
    has_lease_kw = _is_lease_related(message)

    # 1. 범위 밖 — OOS 패턴이 있고 임대차 키워드가 없으면 거부
    if _OOS_RE.search(message) and not has_lease_kw:
        return "out_of_scope"

    # 2. 용어 설명 — 패턴 + 메시지 길이 ≤ 50자 (긴 문장은 일반 법률 Q&A)
    if _TERM_EXPLAIN_RE.search(message) and len(message) <= 50:
        return "term_explain"

    # 3. 특정 조항 위험도 분석
    if _CLAUSE_RISK_RE.search(message):
        return "clause_risk"

    # 4. 실무 절차 안내
    if _PROCEDURE_RE.search(message):
        return "procedure_qa"

    # 5. 기본: 법률 Q&A
    return "legal_qa"


_REJECTION_BASE = (
    "죄송합니다. 저는 임대차 계약 관련 질문만 답변할 수 있습니다.\n"
    "전세·월세 계약, 보증금, 계약갱신, 독소조항 분석 등에 대해 질문해 주세요."
)

_OOS_RESOURCE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"교통\s*사고|자동차\s*사고|차량\s*사고|중앙선|추돌|과실\s*비율|교통법규"),
        "교통사고 관련 문의는 도로교통공단(1577-1120) 또는 손해보험협회(1566-8000)에 문의하세요.",
    ),
    (
        re.compile(r"의료\s*사고|의료\s*분쟁|의료\s*과실|의료\s*소송"),
        "의료분쟁은 한국의료분쟁조정중재원(1670-2545)에 문의하세요.",
    ),
    (
        re.compile(r"노동\s*법|해고|임금\s*체불|근로\s*계약|퇴직금|산재"),
        "노동 관련 분쟁은 고용노동부 상담센터(1350)에 문의하세요.",
    ),
    (
        re.compile(r"이혼|가족법|상속|유언|친권"),
        "가족·상속 분쟁은 대한가정법률복지상담원(02-6952-9555)에 문의하세요.",
    ),
    (
        re.compile(r"형사|고소|고발|처벌|범죄"),
        "형사 사건은 대한법률구조공단(132)에서 무료 법률 상담을 받으실 수 있습니다.",
    ),
]
_OOS_RESOURCE_DEFAULT = "법률구조공단(132) 또는 대한변호사협회 법률상담(02-3476-6500)을 이용하시길 권장드립니다."


def _get_oos_resource(message: str) -> str:
    """OOS 질문 유형에 맞는 외부 기관 안내 문구를 반환한다."""
    for pattern, resource in _OOS_RESOURCE_PATTERNS:
        if pattern.search(message):
            return resource
    return _OOS_RESOURCE_DEFAULT


def _build_rejection_message(message: str) -> str:
    resource = _get_oos_resource(message)
    return f"{_REJECTION_BASE}\n\n관련 기관 안내: {resource}"


class ChatGraphState(TypedDict, total=False):
    messages: list
    question: str
    history: list[dict]
    contract_context: str | None
    collections: list[str]
    k_per_collection: int | dict[str, int]
    use_hyde: bool
    use_multiquery: bool
    use_compression: bool
    deadline: float
    tool_rounds: int
    evidence: dict[str, list[Document]]
    context_sections: dict[str, str]
    answer: str
    sources: list[str]
    source_documents: list[Document]
    context: str
    query_type: str       # 분류 결과: out_of_scope|term_explain|clause_risk|procedure_qa|legal_qa


def _build_chat_context(state: ChatGraphState) -> tuple[str, list[Document]]:
    sections: list[str] = []
    all_docs: list[Document] = []
    evidence = state.get("evidence", {})

    if state.get("contract_context"):
        snippets = _extract_relevant_contract_snippets(
            state["contract_context"] or "",
            state["question"],
        )
        if snippets:
            sections.append("[계약 조항]\n" + "\n".join(f"- {snippet}" for snippet in snippets))

    label_map = {
        "statute": "법령/조문",
        "live_statute": "실시간 법령 API",
        "precedent": "판례/법리",
        "live_precedent": "실시간 판례 API",
        "procedure": "실무 절차",
        "contract_search": "계약 예시",
    }
    for bucket, label in label_map.items():
        docs = state.get("evidence", {}).get(bucket, [])
        if not docs:
            continue
        all_docs.extend(docs)
        sections.append(f"[{label}]\n{build_context(docs, max_length=1000)}")

    return "\n\n".join(sections)[:3600] if sections else "관련 근거를 찾지 못했습니다.", _sort_docs(all_docs)


def _build_chat_fallback_answer(question: str, context: str, sources: list[str]) -> str:
    source_line = f"\n\n참고 출처: {', '.join(sources[:2])}" if sources else ""
    return _ensure_readable_markdown_answer(
        (
            "죄송합니다. 현재 응답 생성에 시간이 더 걸리고 있습니다. "
            "잠시 후 다시 시도해 주시거나, 질문을 좀 더 구체적으로 입력해 주세요."
            f"{source_line}"
        )
    )


def _is_scope_rejection_answer(answer: str) -> bool:
    return "임대차 계약 관련 질문만 답변할 수 있습니다" in (answer or "")


def _build_in_scope_guardrail_fallback(question: str, sources: list[str]) -> str:
    source_line = f"\n\n참고 출처: {', '.join(sources[:2])}" if sources else ""
    if "보증금" in question:
        guidance = (
            "보증금은 계약서에 정한 지급 시기가 있으면 그 약정이 우선하고, "
            "약정이 불명확하면 계약금·잔금·입주일 약정을 함께 확인하시는 것이 좋습니다."
        )
    else:
        guidance = (
            "구체 판단을 위해서는 계약서의 관련 조항, 핵심 일정, 특약 내용을 함께 확인하시는 것이 좋습니다."
        )
    return (
        _ensure_readable_markdown_answer(
            (
                "질문하신 내용은 임대차 범위 안에 있습니다. "
                f"현재 자동 응답이 범위 밖 질문으로 잘못 처리되었습니다. {guidance}"
                f"{source_line}"
            )
        )
    )


async def _repair_in_scope_rejection(
    *,
    question: str,
    answer: str,
    history: list[dict],
    context: str,
    docs: list[Document],
    llm: ChatOpenAI,
    deadline: float,
) -> str:
    if not (_is_lease_related(question) and _is_scope_rejection_answer(answer)):
        return answer

    if _remaining_seconds(deadline) < 1.0:
        return _build_in_scope_guardrail_fallback(question, _format_sources(docs))

    history_messages = []
    for msg in history[-10:]:
        if msg.get("role") == "user":
            history_messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            history_messages.append(AIMessage(content=msg["content"]))

    prompt_messages = CHAT_PROMPT.format_messages(
        context=context,
        history=history_messages,
        question=question,
    )
    repair_messages = [
        SystemMessage(
            content=(
                "이 질문은 임대차 범위 안에 있습니다. "
                "범위 밖 질문 거절 문구를 쓰지 말고, 참고 문서에 근거한 실질 답변을 작성하세요. "
                "근거가 부족하면 부족하다고 밝히되 일반 원칙과 확인 포인트는 반드시 안내하세요."
            )
        ),
        *prompt_messages,
    ]
    with LLM_LATENCY.time():
        repaired = await llm.ainvoke(repair_messages)
    repaired_answer = repaired.content
    if _is_scope_rejection_answer(repaired_answer):
        return _build_in_scope_guardrail_fallback(question, _format_sources(docs))
    return _ensure_readable_markdown_answer(repaired_answer)


def _should_force_caution(answer: str, evidence: dict[str, list[Document]]) -> bool:
    strong_patterns = (
        "즉시 퇴거해야",
        "무조건 퇴거",
        "거절할 수 없습니다",
        "반드시 먼저 인도",
    )
    if not any(pattern in answer for pattern in strong_patterns):
        return False
    has_statute = bool(evidence.get("statute") or evidence.get("live_statute"))
    has_precedent = bool(evidence.get("precedent") or evidence.get("live_precedent"))
    return not (has_statute and has_precedent)


def _build_chat_graph(
    client: VectorDB,
    embeddings: KUREEmbeddings,
    llm: ChatOpenAI,
):
    # ── [분기 1] 질문 유형 분류 노드 ──────────────────────────────────────────
    async def classify_node(state: ChatGraphState) -> dict:
        query_type = _classify_query(state["question"])
        logger.debug(
            "[ChatGraph] classify query_type={} question={!r}",
            query_type,
            state["question"][:60],
        )
        return {"query_type": query_type}

    def _route_after_classify(state: ChatGraphState) -> str:
        return state.get("query_type", "legal_qa")

    # ── [분기 2] 범위 외 질문 거부 노드 ──────────────────────────────────────
    async def reject_node(state: ChatGraphState) -> dict:
        answer = _build_rejection_message(state["question"])
        observe_rag_interaction(
            endpoint="chat_rag_out_of_scope",
            answer=answer,
            documents=[],
            context="",
        )
        return {
            "answer": answer,
            "sources": [],
            "source_documents": [],
            "context": "",
        }

    # ── [분기 3] 용어 설명 전용 검색 노드 ────────────────────────────────────
    async def term_retrieve_node(state: ChatGraphState) -> dict:
        """법령/조문 컬렉션만 검색하여 용어 설명에 필요한 근거를 수집한다."""
        question = state["question"]
        # 용어 설명 질의는 짧고 구체적 — HyDE/multiquery LLM 호출 불필요
        use_hyde = False
        use_multiquery = False
        cf: dict = {}
        law_filter = infer_law_statutes_filter(question)
        if law_filter:
            cf["law_statutes"] = law_filter

        docs = await _search_docs(
            client,
            embeddings,
            question,
            collections=["law_database", "law_statutes"],
            k_per_collection=5,
            score_threshold={"law_database": 0.10, "law_statutes": 0.10, "default": 0.10},
            rerank_top_n=5,
            collection_filters=cf or None,
            use_hyde=use_hyde,
            use_multiquery=use_multiquery,
            llm=llm if (use_hyde or use_multiquery) else None,
        )
        logger.debug("[ChatGraph:term_retrieve] docs={}", len(docs))
        return {"evidence": {"statute": docs}}

    # ── [분기 4] 조항 위험도 전용 검색 노드 ──────────────────────────────────
    async def clause_retrieve_node(state: ChatGraphState) -> dict:
        """독소조항·정상조항 + 법령을 병렬 검색하여 조항 위험도 판단 근거를 수집한다."""
        question = state["question"]
        # 조항 텍스트는 이미 구체적 — HyDE/multiquery LLM 호출 불필요
        use_hyde = False
        use_multiquery = False
        cf: dict = {}
        law_filter = infer_law_statutes_filter(question)
        if law_filter:
            cf["law_statutes"] = law_filter

        clause_docs = []
        law_docs = await _search_docs(
            client,
            embeddings,
            question,
            collections=["law_database", "law_statutes"],
            k_per_collection=4,
            score_threshold={"law_database": 0.12, "law_statutes": 0.12, "default": 0.12},
            rerank_top_n=4,
            collection_filters=cf or None,
            use_hyde=use_hyde,
            use_multiquery=use_multiquery,
            llm=llm if (use_hyde or use_multiquery) else None,
        )
        logger.debug(
            "[ChatGraph:clause_retrieve] clause_docs={} law_docs={}",
            len(clause_docs),
            len(law_docs),
        )
        return {"evidence": {"contract_search": clause_docs, "statute": law_docs}}

    # ── [기존] 도구 선택 노드 (legal_qa / procedure_qa 공용) ─────────────────
    async def tool_selector(state: ChatGraphState) -> dict:
        if _remaining_seconds(state["deadline"]) < 2.8:
            return {}

        evidence_store: dict[str, list[Document]] = defaultdict(list)
        for bucket, docs in state.get("evidence", {}).items():
            evidence_store[bucket].extend(docs)

        collections = state.get("collections") or _CHAT_COLLECTIONS
        k_per_collection = state.get("k_per_collection")
        if k_per_collection is None:
            k_per_collection = _CHAT_K_PER_COLLECTION if _is_lease_related(state["question"]) else 2

        contract_context = state.get("contract_context") or ""
        use_hyde = bool(state.get("use_hyde"))
        use_multiquery = bool(state.get("use_multiquery"))
        law_api_client = get_law_api_client()

        async def _retrieve_bucket(
            bucket: str,
            query: str,
            *,
            search_collections: list[str],
            score_threshold: float | dict[str, float],
            rerank_top_n: int,
            procedure_hint: str = "",
        ) -> list[Document]:
            effective_query = f"{query}\n{procedure_hint}".strip()
            collection_filters = {}
            law_filter = infer_law_statutes_filter(effective_query)
            if law_filter and "law_statutes" in search_collections:
                collection_filters["law_statutes"] = law_filter

            # 도구 내 검색은 이미 타겟 쿼리이므로 HyDE/multiquery LLM 콜 생략
            # → tool_selector + answer 두 번의 LLM 콜로 전체 지연시간 단축
            docs = await _search_docs(
                client,
                embeddings,
                effective_query,
                collections=search_collections,
                k_per_collection=k_per_collection,
                score_threshold=score_threshold,
                rerank_top_n=rerank_top_n,
                collection_filters=collection_filters or None,
                use_hyde=False,
                use_multiquery=False,
            )
            evidence_store[bucket].extend(docs)
            return docs

        @tool
        async def contract_context_lookup(query: str) -> str:
            """현재 계약서 원문에서 질문과 직접 관련된 조항을 찾습니다."""
            snippets = _extract_relevant_contract_snippets(contract_context, query)
            return json.dumps(
                {
                    "bucket": "contract_context",
                    "snippets": snippets,
                },
                ensure_ascii=False,
            )

        @tool
        async def retrieve_contract_basis(query: str) -> str:
            """계약 예시/특약 예시에서 질문과 가까운 조항을 찾습니다."""
            docs = await _retrieve_bucket(
                "contract_search",
                query,
                search_collections=[collection for collection in collections if collection in {"contracts"}],
                score_threshold={"contracts": 0.25, "default": 0.3},
                rerank_top_n=4,
            )
            return json.dumps(
                {
                    "bucket": "contract_search",
                    "hits": len(docs),
                    "summary": build_context(docs, max_length=700),
                },
                ensure_ascii=False,
            )

        @tool
        async def retrieve_statute_basis(query: str) -> str:
            """질문에 직접 적용될 수 있는 법령 조문과 조문 번호를 찾습니다."""
            docs = await _retrieve_bucket(
                "statute",
                query,
                search_collections=["law_database", "law_statutes"],
                score_threshold={"law_database": 0.15, "law_statutes": 0.15, "default": 0.15},
                rerank_top_n=5,
            )
            refs = [f"{law} {article}".strip() for law, article in _collect_cited_laws(docs)]
            return json.dumps(
                {
                    "bucket": "statute",
                    "hits": len(docs),
                    "references": refs[:5],
                    "summary": build_context(docs, max_length=900),
                },
                ensure_ascii=False,
            )

        @tool
        async def retrieve_precedent_basis(query: str) -> str:
            """질문의 법적 효과나 판례 법리를 설명할 판례성 자료를 찾습니다."""
            docs = await _retrieve_bucket(
                "precedent",
                f"{query}\n판례 법리 동시이행 법적 효과",
                search_collections=["law_database", "law_statutes"],
                score_threshold={"law_database": 0.12, "law_statutes": 0.15, "default": 0.12},
                rerank_top_n=4,
            )
            precedent_docs = [
                doc
                for doc in docs
                if str(doc.metadata.get("source_type") or "").startswith(("precedent", "interpretation"))
                or "판례" in doc.page_content
                or "참조조문" in doc.page_content
            ]
            if precedent_docs:
                evidence_store["precedent"] = precedent_docs
                docs = precedent_docs
            return json.dumps(
                {
                    "bucket": "precedent",
                    "hits": len(docs),
                    "summary": build_context(docs, max_length=900),
                },
                ensure_ascii=False,
            )

        @tool
        async def retrieve_procedure_basis(query: str) -> str:
            """내용증명, 임차권등기명령, 지급명령, 소송 같은 실무 절차 자료를 찾습니다."""
            docs = await _retrieve_bucket(
                "procedure",
                query,
                search_collections=["law_database", "law_statutes"],
                score_threshold={"law_database": 0.12, "law_statutes": 0.15, "default": 0.12},
                rerank_top_n=4,
                procedure_hint="임차권등기명령 내용증명 지급명령 보증금반환청구소송 절차",
            )
            return json.dumps(
                {
                    "bucket": "procedure",
                    "hits": len(docs),
                    "summary": build_context(docs, max_length=900),
                },
                ensure_ascii=False,
            )

        @tool
        async def retrieve_supplementary_law(primary_law: str, query: str) -> str:
            """primary_law의 관련 보충 법령(민법, 민사집행법 등)에서 추가 근거를 찾습니다.
            특별법에 규정이 없거나 집행 절차·손해배상·계약 일반원칙이 필요한 경우 사용하세요.
            예: primary_law='주택임대차보호법', query='보증금 반환 지연이자'
            """
            related = get_related_laws(primary_law, query, max_extra=2)
            if not related:
                return json.dumps({"bucket": "statute", "hits": 0, "summary": "관련 보충 법령 없음"}, ensure_ascii=False)

            # 관련 법령만 타겟 검색
            supplementary_query = f"{query}\n{' '.join(related)}"
            collection_filters: dict = {}
            if related:
                collection_filters["law_statutes"] = {"$or": [{"law_name": law} for law in related]}

            docs = await _search_docs(
                client,
                embeddings,
                supplementary_query,
                collections=["law_database", "law_statutes"],
                k_per_collection=4,
                score_threshold={"law_database": 0.12, "law_statutes": 0.12, "default": 0.12},
                rerank_top_n=5,
                collection_filters=collection_filters or None,
            )
            evidence_store["statute"].extend(docs)
            refs = [f"{law} {article}".strip() for law, article in _collect_cited_laws(docs)]
            return json.dumps(
                {
                    "bucket": "statute",
                    "supplementary_laws": related,
                    "hits": len(docs),
                    "references": refs[:5],
                    "summary": build_context(docs, max_length=900),
                },
                ensure_ascii=False,
            )

        @tool
        async def lookup_live_statute(query: str, article: str = "") -> str:
            """실시간 법령 API에서 최신 법령과 정확한 조문 내용을 조회합니다."""
            payload = await law_api_client.lookup_current_statute(query, article=article)
            docs = _live_statute_docs_from_payload(payload)
            if docs:
                evidence_store["live_statute"].extend(docs)
            law = payload.get("law") or {}
            return json.dumps(
                {
                    "bucket": "live_statute",
                    "law_name": law.get("law_name", ""),
                    "article": payload.get("article", ""),
                    "hits": len(docs),
                    "summary": build_context(docs, max_length=900) if docs else "실시간 법령 결과 없음",
                },
                ensure_ascii=False,
            )

        @tool
        async def lookup_live_precedent(query: str) -> str:
            """실시간 판례 API에서 사건번호·최신 판례 요지를 조회합니다."""
            payload = await law_api_client.lookup_precedent(query)
            docs = _live_precedent_docs_from_payload(payload)
            if docs:
                evidence_store["live_precedent"].extend(docs)
            best_match = payload.get("best_match") or {}
            return json.dumps(
                {
                    "bucket": "live_precedent",
                    "case_name": best_match.get("case_name", ""),
                    "case_no": best_match.get("case_no", ""),
                    "hits": len(docs),
                    "summary": build_context(docs, max_length=900) if docs else "실시간 판례 결과 없음",
                },
                ensure_ascii=False,
            )

        tools = [
            contract_context_lookup,
            retrieve_contract_basis,
            retrieve_statute_basis,
            retrieve_supplementary_law,
            retrieve_precedent_basis,
            retrieve_procedure_basis,
            lookup_live_statute,
            lookup_live_precedent,
        ]
        tools_by_name = {tool_item.name: tool_item for tool_item in tools}

        # procedure_qa 타입이면 절차 도구를 최우선으로 안내
        query_type = state.get("query_type", "legal_qa")
        procedure_hint = (
            "\n- 이 질문은 실무 절차 유형입니다. retrieve_procedure_basis를 반드시 먼저 호출하세요."
            if query_type == "procedure_qa"
            else ""
        )
        system = SystemMessage(
            content=(
                "당신은 한국 임대차 계약 법률 QA를 위한 도구 선택 에이전트입니다.\n"
                "한국 법령 적용 원칙:\n"
                "  - 주택임대차보호법·상가건물임대차보호법은 특별법으로 민법보다 우선 적용\n"
                "  - 특별법에 규정이 없는 부분(계약 일반원칙, 손해배상 등)은 민법이 보충 적용\n"
                "  - 집행 절차(경매 배당, 강제집행)는 민사집행법이 별도 적용\n"
                "도구 선택 규칙:\n"
                "- 최종 답변을 바로 쓰지 말고, 필요한 근거가 있으면 도구를 호출하세요.\n"
                "- 강한 결론을 내려야 하는 질문이면 법령/조문과 판례/법리 중 적어도 하나는 확인하세요.\n"
                "- 현재 계약서 원문이 있으면 contract_context_lookup을 우선 고려하세요.\n"
                "- retrieve_statute_basis로 주 법령을 찾은 뒤 민법/민사집행법이 추가로 필요하면 "
                "retrieve_supplementary_law를 호출하세요.\n"
                "- 조문 번호(예: 제3조), 사건번호, '최신/현재' 같은 최신성이 중요하면 "
                "lookup_live_statute 또는 lookup_live_precedent를 우선 고려하세요.\n"
                "- 시간 예산 때문에 같은 도구를 중복 호출하지 말고, 최대 3개 도구만 고르세요.\n"
                "- 도구 호출이 꼭 필요 없으면 호출 없이 넘겨도 됩니다."
                f"{procedure_hint}"
            )
        )
        human = HumanMessage(
            content=(
                f"질문: {state['question']}\n\n"
                f"이전 대화:\n{_compact_history(state.get('history', []))}\n\n"
                f"계약서 원문 존재 여부: {'있음' if contract_context else '없음'}"
            )
        )

        # 도구 선택은 경량 모델(mini)로 실행 — 속도 우선, 최대 10s
        selector_llm = get_fast_llm()
        try:
            with LLM_LATENCY.time():
                response = await asyncio.wait_for(
                    selector_llm.bind_tools(tools).ainvoke([system, human]),
                    timeout=10.0,
                )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("[ChatGraph:tool_selector] LLM skipped: {} — proceeding without tools", exc)
            return {}

        messages = list(state.get("messages", []))
        messages.append(response)

        if response.tool_calls and state.get("tool_rounds", 0) < CHAT_MAX_TOOL_ROUNDS:
            async def _invoke_one(tool_call: dict) -> ToolMessage:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})
                try:
                    result = await tools_by_name[tool_name].ainvoke(tool_args)
                except Exception as exc:
                    result = json.dumps({"error": str(exc), "tool": tool_name}, ensure_ascii=False)
                return ToolMessage(
                    content=result if isinstance(result, str) else json.dumps(result, ensure_ascii=False),
                    tool_call_id=tool_call["id"],
                )

            tool_messages = list(await asyncio.gather(*[_invoke_one(tc) for tc in response.tool_calls]))
            messages.extend(tool_messages)

        return {
            "messages": messages,
            "tool_rounds": state.get("tool_rounds", 0) + (1 if response.tool_calls else 0),
            "evidence": {bucket: _sort_docs(docs) for bucket, docs in evidence_store.items()},
        }

    async def answer_node(state: ChatGraphState) -> dict:
        context, docs = _build_chat_context(state)
        if state.get("use_compression") and docs and _remaining_seconds(state["deadline"]) > 2.5:
            try:
                docs = await compress_documents(docs, state["question"], llm)
                context = build_context(docs, max_length=3000)
            except Exception as exc:
                logger.debug("[ChatGraph] compression skipped: {}", exc)

        remaining = _remaining_seconds(state["deadline"])
        if remaining < 0.5:
            answer = _build_chat_fallback_answer(state["question"], context, _format_sources(docs))
        else:
            history_messages = []
            for msg in state.get("history", [])[-10:]:
                if msg.get("role") == "user":
                    history_messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    history_messages.append(AIMessage(content=msg["content"]))
            prompt_messages = CHAT_PROMPT.format_messages(
                context=context,
                history=history_messages,
                question=state["question"],
            )
            answer_timeout = min(remaining - 0.3, 18.0)
            try:
                with LLM_LATENCY.time():
                    response = await asyncio.wait_for(
                        llm.ainvoke(prompt_messages),
                        timeout=answer_timeout,
                    )
                answer = response.content
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning("[ChatGraph:answer_node] LLM timeout/error: {}", exc)
                answer = _build_chat_fallback_answer(state["question"], context, _format_sources(docs))

        answer = await _repair_in_scope_rejection(
            question=state["question"],
            answer=answer,
            history=state.get("history", []),
            context=context,
            docs=docs,
            llm=llm,
            deadline=state["deadline"],
        )

        verification_source = "\n".join(doc.page_content for doc in docs)
        answer = annotate_unverified_citations(answer, verification_source)

        if _should_force_caution(answer, state.get("evidence", {})):
            answer += "\n\n다만 이 결론은 조문과 판례 법리를 함께 더 확인해 단정하는 것이 안전합니다."
        answer = _ensure_readable_markdown_answer(answer)

        observe_rag_interaction(
            endpoint="chat_rag_langgraph",
            answer=answer,
            documents=docs,
            context=context,
        )
        return {
            "answer": answer,
            "sources": _format_sources(docs),
            "source_documents": docs,
            "context": context,
        }

    graph = StateGraph(ChatGraphState)

    # 노드 등록
    graph.add_node("classify", classify_node)
    graph.add_node("reject", reject_node)
    graph.add_node("term_retrieve", term_retrieve_node)
    graph.add_node("clause_retrieve", clause_retrieve_node)
    graph.add_node("tool_selector", tool_selector)
    graph.add_node("answer", answer_node)

    # 진입점
    graph.set_entry_point("classify")

    # classify → 질문 유형별 분기
    graph.add_conditional_edges(
        "classify",
        _route_after_classify,
        {
            "out_of_scope": "reject",
            "term_explain": "term_retrieve",
            "clause_risk": "clause_retrieve",
            "procedure_qa": "tool_selector",
            "legal_qa": "tool_selector",
        },
    )

    # 각 브랜치 → 종단
    graph.add_edge("reject", END)
    graph.add_edge("term_retrieve", "answer")
    graph.add_edge("clause_retrieve", "answer")
    graph.add_edge("tool_selector", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


async def run_chat_langgraph(
    *,
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
    cache_key = (id(client), id(embeddings), id(llm))
    graph = _CHAT_GRAPH_CACHE.get(cache_key)
    if graph is None:
        graph = _build_chat_graph(client, embeddings, llm)
        _CHAT_GRAPH_CACHE[cache_key] = graph

    initial_state: ChatGraphState = {
        "messages": [],
        "question": message,
        "history": history,
        "contract_context": contract_context,
        "collections": collections or _CHAT_COLLECTIONS,
        "k_per_collection": k_per_collection or (_CHAT_K_PER_COLLECTION if _is_lease_related(message) else 2),
        "use_hyde": use_hyde,
        "use_multiquery": use_multiquery,
        "use_compression": use_compression,
        "deadline": time.perf_counter() + CHAT_TIMEOUT_SECONDS - 0.2,
        "tool_rounds": 0,
        "evidence": {},
        "query_type": "",  # classify_node에서 채워짐
    }
    try:
        return await asyncio.wait_for(graph.ainvoke(initial_state), timeout=CHAT_TIMEOUT_SECONDS)
    except Exception as exc:
        logger.warning("[ChatGraph] fallback due to error: {} (type={})", exc, type(exc).__name__)
        try:
            fast_docs = await asyncio.wait_for(
                _search_docs(
                    client,
                    embeddings,
                    message,
                    collections=collections or _CHAT_COLLECTIONS,
                    k_per_collection=k_per_collection or (_CHAT_K_PER_COLLECTION if _is_lease_related(message) else 2),
                    score_threshold={"law_database": 0.15, "law_statutes": 0.15, "contracts": 0.25, "default": 0.2},
                    rerank_top_n=5,
                    collection_filters={"law_statutes": infer_law_statutes_filter(message)} if infer_law_statutes_filter(message) else None,
                    use_hyde=False,
                    use_multiquery=False,
                ),
                timeout=8.0,
            )
        except Exception as search_exc:
            logger.warning("[ChatGraph] fallback search also failed: {}", search_exc)
            fast_docs = []
        context = build_context(fast_docs, max_length=2200)
        answer = _build_chat_fallback_answer(message, context, _format_sources(fast_docs))
        observe_rag_interaction(
            endpoint="chat_rag_langgraph_fallback",
            answer=answer,
            documents=fast_docs,
            context=context,
        )
        return {
            "answer": answer,
            "sources": _format_sources(fast_docs),
            "context": context,
            "source_documents": fast_docs,
        }


class RiskGraphState(TypedDict, total=False):
    messages: list
    contract_text: str
    deadline: float
    tool_rounds: int
    clause_candidates: list[dict[str, Any]]
    selected_clause_numbers: list[int]
    evidence_by_clause: dict[int, dict[str, Any]]
    final_result: dict[str, Any]


async def _retrieve_clause_evidence_bundle(
    clause_no: int,
    clause_text: str,
    client: VectorDB,
    embeddings: KUREEmbeddings,
) -> dict[str, Any]:
    query_vector = await asyncio.to_thread(embeddings.embed_query, clause_text)
    law_query = _law_search_query_for_clause(clause_text)
    law_vector = query_vector if law_query == clause_text else await asyncio.to_thread(embeddings.embed_query, law_query)
    law_filter = infer_law_statutes_filter(law_query)

    illegal_docs, normal_docs = [], []
    law_db_docs, law_statute_docs = await asyncio.gather(
        asyncio.to_thread(search_collection, client, embeddings, law_query, "law_database", 4, None, 0.1, law_vector),
        asyncio.to_thread(search_collection, client, embeddings, law_query, "law_statutes", 5, law_filter, 0.1, law_vector),
    )
    law_docs = _sort_docs([*law_db_docs, *law_statute_docs])
    if law_docs:
        law_docs = await get_reranker().async_rerank(clause_text, law_docs, top_n=5)
    refs = _grounded_law_references(law_docs, clause_text)

    illegal_similarity = max((doc.metadata.get("score", 0) for doc in illegal_docs), default=0.0)
    normal_similarity = max((doc.metadata.get("score", 0) for doc in normal_docs), default=0.0)

    return {
        "clause_no": clause_no,
        "text": clause_text,
        "illegal_similarity": illegal_similarity,
        "normal_similarity": normal_similarity,
        "risk_delta": illegal_similarity - normal_similarity,
        "illegal_docs": illegal_docs,
        "normal_docs": normal_docs,
        "law_docs": law_docs,
        "law_references": refs,
        "illegal_text": "\n".join(f"- ({doc.metadata.get('category', '')}) {doc.page_content}" for doc in illegal_docs) or "해당 없음",
        "normal_text": "\n".join(f"- ({doc.metadata.get('category', '')}) {doc.page_content}" for doc in normal_docs) or "해당 없음",
        "law_text": _format_grounded_law_context(law_docs, refs),
    }


def _prioritize_clauses(clauses: list[str]) -> list[dict[str, Any]]:
    prioritized: list[dict[str, Any]] = []
    for index, clause in enumerate(clauses, start=1):
        score = _risk_keyword_score(clause)
        reason = "키워드 밀도 기반"
        if _is_safe_preservation_clause(clause):
            score += 1
            reason = "권리 보장 문구"
        prioritized.append(
            {
                "clause_no": index,
                "text": clause,
                "priority_score": score,
                "priority_reason": reason,
            }
        )
    prioritized.sort(key=lambda item: (-item["priority_score"], item["clause_no"]))
    return prioritized


def _build_risk_graph(
    client: VectorDB,
    embeddings: KUREEmbeddings,
    llm: ChatOpenAI,
):
    async def tool_selector(state: RiskGraphState) -> dict:
        if _remaining_seconds(state["deadline"]) < 4.0:
            return {}

        clause_candidates = list(state.get("clause_candidates", []))
        evidence_by_clause = dict(state.get("evidence_by_clause", {}))

        contract_text = state["contract_text"]

        @tool
        async def extract_contract_clauses() -> str:
            """계약서에서 특약 조항 또는 핵심 검토 조항을 추출하고 우선순위를 매깁니다."""
            clauses = _extract_special_clauses(contract_text)
            if not clauses:
                stripped = contract_text.strip()
                clauses = [stripped[:500]] if stripped else []
            prioritized = _prioritize_clauses(clauses)
            clause_candidates.clear()
            clause_candidates.extend(prioritized)
            return json.dumps(
                {
                    "total_clauses": len(prioritized),
                    "clauses": prioritized[:10],
                },
                ensure_ascii=False,
            )

        @tool
        async def retrieve_clause_evidence(clause_numbers: list[int]) -> str:
            """선택한 조항 번호들에 대해 독소조항 예시, 정상 조항 예시, 관련 법령 근거를 수집합니다."""
            wanted = clause_numbers[:RISK_MAX_LLM_CLAUSES] if clause_numbers else []
            if not wanted and clause_candidates:
                wanted = [item["clause_no"] for item in clause_candidates[:RISK_MAX_LLM_CLAUSES]]

            selected = [item for item in clause_candidates if item["clause_no"] in wanted]
            bundles = await asyncio.gather(
                *[
                    _retrieve_clause_evidence_bundle(item["clause_no"], item["text"], client, embeddings)
                    for item in selected
                ]
            )
            for bundle in bundles:
                evidence_by_clause[bundle["clause_no"]] = bundle
            return json.dumps(
                {
                    "selected_clause_numbers": wanted,
                    "bundles": [
                        {
                            "clause_no": bundle["clause_no"],
                            "text": bundle["text"][:220],
                            "illegal_similarity": round(bundle["illegal_similarity"], 4),
                            "normal_similarity": round(bundle["normal_similarity"], 4),
                            "risk_delta": round(bundle["risk_delta"], 4),
                            "law_references": bundle["law_references"][:4],
                        }
                        for bundle in bundles
                    ],
                },
                ensure_ascii=False,
            )

        tools = [extract_contract_clauses, retrieve_clause_evidence]
        tools_by_name = {tool_item.name: tool_item for tool_item in tools}
        system = SystemMessage(
            content=(
                "당신은 한국 임대차 계약 리스크 분석을 위한 도구 선택 에이전트입니다.\n"
                "규칙:\n"
                "- 먼저 extract_contract_clauses를 호출하세요.\n"
                "- 그 다음 retrieve_clause_evidence를 호출해 우선순위가 높은 조항 최대 6개만 조사하세요.\n"
                "- 시간 예산 때문에 도구는 최대 두 라운드까지만 사용하세요.\n"
                "- 최종 분석 문장은 쓰지 말고 필요한 도구만 선택하세요."
            )
        )
        human = HumanMessage(
            content=f"계약서 원문:\n{contract_text[:2200]}"
        )

        with LLM_LATENCY.time():
            response = await llm.bind_tools(tools).ainvoke([system, human, *state.get("messages", [])])

        messages = list(state.get("messages", []))
        messages.append(response)
        if response.tool_calls and state.get("tool_rounds", 0) < RISK_MAX_TOOL_ROUNDS:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})
                try:
                    result = await tools_by_name[tool_name].ainvoke(tool_args)
                except Exception as exc:
                    result = json.dumps({"error": str(exc), "tool": tool_name}, ensure_ascii=False)
                messages.append(
                    ToolMessage(
                        content=result if isinstance(result, str) else json.dumps(result, ensure_ascii=False),
                        tool_call_id=tool_call["id"],
                    )
                )

        if clause_candidates and not evidence_by_clause and _remaining_seconds(state["deadline"]) > 6.0:
            with LLM_LATENCY.time():
                follow_up = await llm.bind_tools(tools).ainvoke([system, human, *messages])
            messages.append(follow_up)
            if follow_up.tool_calls:
                for tool_call in follow_up.tool_calls:
                    if tool_call["name"] != "retrieve_clause_evidence":
                        continue
                    tool_args = tool_call.get("args", {})
                    try:
                        result = await tools_by_name["retrieve_clause_evidence"].ainvoke(tool_args)
                    except Exception as exc:
                        result = json.dumps({"error": str(exc), "tool": "retrieve_clause_evidence"}, ensure_ascii=False)
                    messages.append(
                        ToolMessage(
                            content=result if isinstance(result, str) else json.dumps(result, ensure_ascii=False),
                            tool_call_id=tool_call["id"],
                        )
                    )

        selected_clause_numbers = sorted(evidence_by_clause.keys())
        return {
            "messages": messages,
            "tool_rounds": state.get("tool_rounds", 0) + (1 if response.tool_calls else 0),
            "clause_candidates": clause_candidates,
            "selected_clause_numbers": selected_clause_numbers,
            "evidence_by_clause": evidence_by_clause,
        }

    async def finalizer(state: RiskGraphState) -> dict:
        clause_candidates = state.get("clause_candidates", [])
        evidence_by_clause = state.get("evidence_by_clause", {})
        if not clause_candidates:
            return {
                "final_result": await _detect_risk_legacy(
                    state["contract_text"],
                    client,
                    embeddings,
                    llm,
                )
            }

        selected = [item for item in clause_candidates if item["clause_no"] in state.get("selected_clause_numbers", [])]
        omitted = [item for item in clause_candidates if item["clause_no"] not in state.get("selected_clause_numbers", [])]

        if _remaining_seconds(state["deadline"]) < 3.5:
            fast_results = []
            for item in clause_candidates:
                refs = evidence_by_clause.get(item["clause_no"], {}).get("law_references", [])
                fast_results.append(_build_fast_clause_result(item["text"], refs))
            return {"final_result": _recompute_contract_risk(fast_results)}

        evidence_payload = []
        for item in selected:
            bundle = evidence_by_clause.get(item["clause_no"], {})
            evidence_payload.append(
                {
                    "clause_no": item["clause_no"],
                    "text": item["text"],
                    "priority_score": item["priority_score"],
                    "illegal_similarity": round(bundle.get("illegal_similarity", 0.0), 4),
                    "normal_similarity": round(bundle.get("normal_similarity", 0.0), 4),
                    "risk_delta": round(bundle.get("risk_delta", 0.0), 4),
                    "illegal_examples": bundle.get("illegal_text", "해당 없음")[:1200],
                    "normal_examples": bundle.get("normal_text", "해당 없음")[:1000],
                    "law_context": bundle.get("law_text", "관련 법률 없음")[:1600],
                    "grounded_refs": bundle.get("law_references", [])[:4],
                }
            )

        structured_llm = llm.with_structured_output(ContractRiskResult)
        prompt = (
            "당신은 한국 임대차 계약 리스크 분석 전문가입니다.\n"
            "아래 조항별 증거를 기반으로 위험/주의/안전을 분류하세요.\n"
            "규칙:\n"
            "- legal_reference에는 각 조항의 grounded_refs에 있는 값을 우선 사용하세요.\n"
            "- grounded_refs가 없으면 해당 조항과 관련된 한국 법령 조문을 판단하여 기재하세요 (예: 주택임대차보호법 제3조).\n"
            "- 과도한 비용 부담, 원상복구 확대, 수선비 전가, 중도퇴거 부담은 우선 '주의'로 보고 명시적 권리 포기/몰수/행사 금지가 있을 때만 '위험'으로 올리세요.\n"
            "- 임차인 권리를 보장하는 문구는 '안전'으로 보세요.\n"
            "- JSON만 반환하세요.\n\n"
            f"조항별 증거:\n{json.dumps(evidence_payload, ensure_ascii=False, indent=2)}\n\n"
            f"이번 분석 대상 전체 조항 수: {len(clause_candidates)}"
        )

        with LLM_LATENCY.time():
            model_result = await structured_llm.ainvoke(prompt)

        result_by_text = {clause.text: clause for clause in model_result.clauses}
        grounded_results: list[ClauseRisk] = []
        for item in selected:
            raw_clause = result_by_text.get(item["text"]) or _build_fast_clause_result(item["text"])
            refs = evidence_by_clause.get(item["clause_no"], {}).get("law_references", [])
            grounded_results.append(_ground_clause_result(raw_clause, refs, item["text"]))

        for item in omitted:
            refs = evidence_by_clause.get(item["clause_no"], {}).get("law_references", [])
            grounded_results.append(_build_fast_clause_result(item["text"], refs))

        grounded_results.sort(
            key=lambda clause: next(
                (
                    candidate["clause_no"]
                    for candidate in clause_candidates
                    if candidate["text"] == clause.text
                ),
                999,
            )
        )
        final_result = _recompute_contract_risk(grounded_results)

        all_docs = []
        for bundle in evidence_by_clause.values():
            all_docs.extend(bundle.get("illegal_docs", []))
            all_docs.extend(bundle.get("normal_docs", []))
            all_docs.extend(bundle.get("law_docs", []))
        observe_rag_interaction(
            endpoint="detect_risk_contract_langgraph",
            answer=json.dumps(final_result, ensure_ascii=False),
            documents=_sort_docs(all_docs),
            context=json.dumps(evidence_payload, ensure_ascii=False),
        )
        return {"final_result": final_result}

    graph = StateGraph(RiskGraphState)
    graph.add_node("tool_selector", tool_selector)
    graph.add_node("finalizer", finalizer)
    graph.set_entry_point("tool_selector")
    graph.add_edge("tool_selector", "finalizer")
    graph.add_edge("finalizer", END)
    return graph.compile()


async def run_risk_contract_langgraph(
    *,
    contract_text: str,
    client: VectorDB,
    embeddings: KUREEmbeddings,
    llm: ChatOpenAI,
) -> dict:
    graph = _build_risk_graph(client, embeddings, llm)
    initial_state: RiskGraphState = {
        "messages": [],
        "contract_text": contract_text,
        "deadline": time.perf_counter() + RISK_TIMEOUT_SECONDS - 0.3,
        "tool_rounds": 0,
        "clause_candidates": [],
        "selected_clause_numbers": [],
        "evidence_by_clause": {},
    }
    try:
        result = await asyncio.wait_for(graph.ainvoke(initial_state), timeout=RISK_TIMEOUT_SECONDS)
        return result["final_result"]
    except Exception as exc:
        logger.warning("[RiskGraph] fallback to legacy due to error: {}", exc)
        return await _detect_risk_legacy(contract_text, client, embeddings, llm)


def _first_clause_from_result(result: dict) -> ClauseRisk:
    clauses = result.get("clauses", [])
    if clauses:
        return ClauseRisk(**clauses[0])
    return ClauseRisk(
        text="",
        risk_level="주의",
        category="분석 오류",
        analysis="분석 결과를 생성하지 못했습니다.",
        legal_reference="",
        score=50,
    )


async def run_single_clause_risk_langgraph(
    *,
    clause_text: str,
    client: VectorDB,
    embeddings: KUREEmbeddings,
    llm: ChatOpenAI,
) -> dict:
    result = await run_risk_contract_langgraph(
        contract_text=f"[특약사항]\n1. {clause_text}",
        client=client,
        embeddings=embeddings,
        llm=llm,
    )
    first_clause = _first_clause_from_result(result)

    bundle = await _retrieve_clause_evidence_bundle(1, clause_text, client, embeddings)
    analysis = (
        f"{first_clause.analysis}\n"
        f"근거 조항: {first_clause.legal_reference or '직접 확인된 조문 없음'}"
    ).strip()
    all_docs = [*bundle["illegal_docs"], *bundle["normal_docs"], *bundle["law_docs"]]
    observe_rag_interaction(
        endpoint="detect_risk_langgraph",
        answer=analysis,
        documents=_sort_docs(all_docs),
        context="\n".join([bundle["illegal_text"], bundle["normal_text"], bundle["law_text"]]),
    )

    return {
        "illegal_similarity": bundle["illegal_similarity"],
        "normal_similarity": bundle["normal_similarity"],
        "risk_delta": bundle["risk_delta"],
        "analysis": analysis,
        "legal_reference": first_clause.legal_reference,
        "illegal_matches": bundle["illegal_docs"],
        "normal_matches": bundle["normal_docs"],
        "law_matches": bundle["law_docs"],
    }
