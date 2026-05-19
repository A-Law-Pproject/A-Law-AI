from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.mcp.law_api_client import LawApiClient

try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover - optional runtime dependency
    FastMCP = None


def create_law_mcp_server():
    if FastMCP is None:
        raise RuntimeError("fastmcp is not installed")

    api_client = LawApiClient()
    mcp = FastMCP(name="ALawLawServer")

    @mcp.tool
    async def lookup_current_statute(query: str, article: str = "") -> dict[str, Any]:
        """Search the live law API and return the best current statute match with article excerpts."""
        return await api_client.lookup_current_statute(query, article=article)

    @mcp.tool
    async def lookup_precedent(query: str) -> dict[str, Any]:
        """Search the live precedent API and return the best precedent match with detail payload."""
        return await api_client.lookup_precedent(query)

    @mcp.tool
    async def search_current_laws(query: str, display: int = settings.LAW_MCP_MAX_RESULTS) -> list[dict[str, str]]:
        """Search current laws without fetching detail content."""
        return await api_client.search_laws(query, display=display)

    return mcp


mcp = create_law_mcp_server() if FastMCP is not None else None


if __name__ == "__main__":  # pragma: no cover - manual server entrypoint
    if mcp is None:
        raise RuntimeError("fastmcp is not installed")
    mcp.run()
