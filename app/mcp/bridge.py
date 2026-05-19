from __future__ import annotations

import json
from typing import Any

from app.core.config import settings
from app.mcp.law_api_client import LawApiClient

try:
    from fastmcp import Client
except ImportError:  # pragma: no cover - optional runtime dependency
    Client = None


class LawMCPBridge:
    def __init__(self):
        self._api_client = LawApiClient()
        self._server = None
        self._client = None

        if settings.LAW_MCP_ENABLED and Client is not None:
            try:
                from app.mcp.law_server import create_law_mcp_server

                self._server = create_law_mcp_server()
                self._client = Client(self._server)
            except Exception:
                self._server = None
                self._client = None

    async def _call_mcp_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("MCP client is unavailable")

        async with self._client:
            result = await self._client.call_tool(tool_name, args)

        data = getattr(result, "data", None)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"items": data}

        content = getattr(result, "content", None) or []
        if content:
            text = getattr(content[0], "text", "")
            if text:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return {"text": text}
        return {}

    async def lookup_current_statute(self, query: str, article: str = "") -> dict[str, Any]:
        if self._client is not None:
            return await self._call_mcp_tool(
                "lookup_current_statute",
                {"query": query, "article": article},
            )
        return await self._api_client.lookup_current_statute(query, article=article)

    async def lookup_precedent(self, query: str) -> dict[str, Any]:
        if self._client is not None:
            return await self._call_mcp_tool(
                "lookup_precedent",
                {"query": query},
            )
        return await self._api_client.lookup_precedent(query)
