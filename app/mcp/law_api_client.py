from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings


_ARTICLE_RE = re.compile(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?")
_CASE_RE = re.compile(r"\d{4}[가-힣]{1,4}\d+")

_LAW_NAME_KEYS = ("법령명한글", "법령명", "lawName", "lawNm", "title")
_LAW_ID_KEYS = ("MST", "법령일련번호", "ID", "lawId")
_LAW_URL_KEYS = ("법령상세링크", "상세링크", "link", "url")

_PRECEDENT_ID_KEYS = ("판례정보일련번호", "ID", "precSeq", "caseId")
_PRECEDENT_NAME_KEYS = ("사건명", "판례명", "title")
_PRECEDENT_CASE_NO_KEYS = ("사건번호", "caseNo")

_ARTICLE_NO_KEYS = ("조문번호", "조번호", "article", "articleNo")
_ARTICLE_TITLE_KEYS = ("조문제목", "항제목", "articleTitle")
_ARTICLE_CONTENT_KEYS = ("조문내용", "조문내용전문", "내용", "content", "text")


def _compact(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_article(article: str | None) -> str:
    match = _ARTICLE_RE.search(article or "")
    if not match:
        return ""
    return f"제{match.group(1)}조" + (f"의{match.group(2)}" if match.group(2) else "")


def _extract_query_article(query: str) -> str:
    return _normalize_article(query)


def _extract_case_token(query: str) -> str:
    match = _CASE_RE.search(query or "")
    return _compact(match.group(0)) if match else ""


def _iter_nodes(payload: Any):
    if isinstance(payload, dict):
        yield payload
        for value in payload.values():
            yield from _iter_nodes(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_nodes(item)


def _pick(node: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = node.get(key)
        if value not in (None, ""):
            return _compact(str(value))
    return ""


def _law_result_from_node(node: dict[str, Any]) -> dict[str, str] | None:
    law_name = _pick(node, _LAW_NAME_KEYS)
    law_id = _pick(node, _LAW_ID_KEYS)
    if not law_name or not law_id:
        return None
    return {
        "law_name": law_name,
        "law_id": law_id,
        "promulgation_date": _pick(node, ("공포일자", "공포일", "promulgationDate")),
        "effective_date": _pick(node, ("시행일자", "시행일", "effectiveDate")),
        "department": _pick(node, ("소관부처명", "부서명", "department")),
        "source_url": _pick(node, _LAW_URL_KEYS),
    }


def _precedent_result_from_node(node: dict[str, Any]) -> dict[str, str] | None:
    precedent_id = _pick(node, _PRECEDENT_ID_KEYS)
    case_name = _pick(node, _PRECEDENT_NAME_KEYS)
    if not precedent_id or not case_name:
        return None
    return {
        "precedent_id": precedent_id,
        "case_name": case_name,
        "case_no": _pick(node, _PRECEDENT_CASE_NO_KEYS),
        "court_name": _pick(node, ("법원명", "courtName")),
        "decision_date": _pick(node, ("선고일자", "decisionDate")),
        "summary": _pick(node, ("판결요지", "판시사항", "summary")),
        "source_url": _pick(node, ("판례상세링크", "상세링크", "link", "url")),
    }


def _dedupe_items(items: list[dict[str, str]], key_fields: tuple[str, ...]) -> list[dict[str, str]]:
    seen: set[tuple[str, ...]] = set()
    deduped: list[dict[str, str]] = []
    for item in items:
        key = tuple(item.get(field, "") for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _extract_law_items(payload: Any) -> list[dict[str, str]]:
    items = [item for node in _iter_nodes(payload) if isinstance(node, dict) if (item := _law_result_from_node(node))]
    return _dedupe_items(items, ("law_id", "law_name"))


def _extract_precedent_items(payload: Any) -> list[dict[str, str]]:
    items = [item for node in _iter_nodes(payload) if isinstance(node, dict) if (item := _precedent_result_from_node(node))]
    return _dedupe_items(items, ("precedent_id", "case_name"))


def _extract_article_snippets(payload: Any, requested_article: str = "") -> list[dict[str, str]]:
    requested = _normalize_article(requested_article)
    snippets: list[dict[str, str]] = []
    for node in _iter_nodes(payload):
        if not isinstance(node, dict):
            continue
        article = _normalize_article(_pick(node, _ARTICLE_NO_KEYS))
        title = _pick(node, _ARTICLE_TITLE_KEYS)
        content = _pick(node, _ARTICLE_CONTENT_KEYS)
        if not article or not content:
            continue
        if requested and article != requested:
            continue
        snippets.append(
            {
                "article": article,
                "title": title,
                "content": content,
            }
        )
    return _dedupe_items(snippets, ("article", "content"))


def _best_law_match(items: list[dict[str, str]], query: str) -> dict[str, str] | None:
    normalized_query = _compact(query)
    for item in items:
        if item["law_name"] == normalized_query:
            return item
    return items[0] if items else None


def _best_precedent_match(items: list[dict[str, str]], query: str) -> dict[str, str] | None:
    requested_case = _extract_case_token(query)
    if requested_case:
        for item in items:
            if requested_case and requested_case in item.get("case_no", ""):
                return item
    return items[0] if items else None


@dataclass
class LawApiClient:
    oc: str = settings.LAW_API_OC
    base_url: str = settings.LAW_API_BASE_URL.rstrip("/")
    timeout_seconds: float = settings.LAW_API_TIMEOUT_SECONDS

    async def _request_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.oc:
            raise RuntimeError("LAW_API_OC is not configured")

        request_params = {
            **params,
            "OC": self.oc,
            "type": "JSON",
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(f"{self.base_url}/{path}", params=request_params)
            response.raise_for_status()
            payload = response.json()

        if isinstance(payload, dict) and payload.get("result") not in (None, "", "OK"):
            raise RuntimeError(payload.get("msg") or payload.get("result"))
        return payload

    async def search_laws(self, query: str, *, display: int | None = None, search: int = 1) -> list[dict[str, str]]:
        payload = await self._request_json(
            "lawSearch.do",
            {
                "target": "law",
                "query": query,
                "search": search,
                "display": display or settings.LAW_MCP_MAX_RESULTS,
                "page": 1,
            },
        )
        return _extract_law_items(payload)

    async def fetch_law_detail(self, law_id: str) -> dict[str, Any]:
        return await self._request_json(
            "lawService.do",
            {
                "target": "law",
                "MST": law_id,
            },
        )

    async def lookup_current_statute(self, query: str, article: str = "") -> dict[str, Any]:
        requested_article = _normalize_article(article) or _extract_query_article(query)
        items = await self.search_laws(query, search=1)
        best_match = _best_law_match(items, query)
        if best_match is None:
            return {
                "query": query,
                "article": requested_article,
                "law": None,
                "results": [],
                "snippets": [],
            }

        detail = await self.fetch_law_detail(best_match["law_id"])
        snippets = _extract_article_snippets(detail, requested_article)
        return {
            "query": query,
            "article": requested_article,
            "law": best_match,
            "results": items[: settings.LAW_MCP_MAX_RESULTS],
            "snippets": snippets[: settings.LAW_MCP_MAX_RESULTS],
        }

    async def search_precedents(self, query: str, *, display: int | None = None, search: int = 2) -> list[dict[str, str]]:
        payload = await self._request_json(
            "lawSearch.do",
            {
                "target": "prec",
                "query": query,
                "search": search,
                "display": display or settings.LAW_MCP_MAX_RESULTS,
                "page": 1,
            },
        )
        return _extract_precedent_items(payload)

    async def fetch_precedent_detail(self, precedent_id: str) -> dict[str, Any]:
        return await self._request_json(
            "lawService.do",
            {
                "target": "prec",
                "ID": precedent_id,
            },
        )

    async def lookup_precedent(self, query: str) -> dict[str, Any]:
        items = await self.search_precedents(query)
        best_match = _best_precedent_match(items, query)
        if best_match is None:
            return {
                "query": query,
                "best_match": None,
                "results": [],
                "detail": {},
            }
        detail = await self.fetch_precedent_detail(best_match["precedent_id"])
        return {
            "query": query,
            "best_match": best_match,
            "results": items[: settings.LAW_MCP_MAX_RESULTS],
            "detail": detail,
        }

