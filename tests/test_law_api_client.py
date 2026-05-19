import asyncio

from app.mcp.law_api_client import LawApiClient


class _StubLawApiClient(LawApiClient):
    async def search_laws(self, query: str, *, display=None, search: int = 1):
        return [
            {
                "law_name": "주택임대차보호법",
                "law_id": "LAW-1",
                "source_url": "https://example.test/law",
            }
        ]

    async def fetch_law_detail(self, law_id: str):
        return {
            "법령": {
                "조문": [
                    {
                        "조문번호": "제3조",
                        "조문제목": "임대차기간",
                        "조문내용": "임대차기간은 2년으로 본다.",
                    },
                    {
                        "조문번호": "제8조",
                        "조문제목": "보증금",
                        "조문내용": "보증금 우선변제 관련 규정이다.",
                    },
                ]
            }
        }


def test_lookup_current_statute_extracts_requested_article():
    client = _StubLawApiClient(oc="dummy")

    result = asyncio.run(client.lookup_current_statute("주택임대차보호법 제3조"))

    assert result["law"]["law_id"] == "LAW-1"
    assert result["article"] == "제3조"
    assert len(result["snippets"]) == 1
    assert result["snippets"][0]["article"] == "제3조"
    assert "임대차기간" in result["snippets"][0]["title"]
