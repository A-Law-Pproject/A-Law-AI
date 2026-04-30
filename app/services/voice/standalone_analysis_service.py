"""
Standalone voice analysis without contract matching.
"""
import asyncio
import json

from loguru import logger

from app.core.config import settings
from app.core.dependencies import get_llm
from app.schemas.voice_shared import AgreementItem, SegmentResult
from app.schemas.voice_standalone import VoiceAnalysisSummary, VoiceRiskItem


_VOICE_ANALYSIS_PROMPT = """You analyze Korean real-estate and legal conversation audio.
Read the transcript and return only JSON with this shape:
{
  "summary": "short Korean summary",
  "keyPoints": ["point 1", "point 2"],
  "riskItems": [
    {
      "riskType": "RISK_TYPE",
      "severity": "low|medium|high|critical",
      "detail": "why this sounds risky",
      "timestampStr": "HH:MM:SS"
    }
  ]
}

Rules:
- Focus on promises, payment terms, dates, penalties, ambiguous conditions, pressure, and one-sided changes.
- If there is no notable risk, return an empty `riskItems` array.
- Keep the summary concise.

[Transcript]
{transcript}

[Extracted structured hints]
{agreements}
"""


async def summarize_voice_analysis(
    transcript: str,
    agreements: list[AgreementItem],
    segments: list[SegmentResult],
) -> VoiceAnalysisSummary:
    llm = get_llm()

    agreements_hint = [
        {
            "type": item.agreement_type,
            "value": item.value,
            "timestamp": item.timestamp_str,
        }
        for item in agreements[:20]
    ]
    prompt = (
        _VOICE_ANALYSIS_PROMPT.replace("{transcript}", transcript[:4000]).replace(
            "{agreements}",
            json.dumps(agreements_hint, ensure_ascii=False),
        )
    )

    try:
        response = await asyncio.wait_for(
            llm.ainvoke(prompt),
            timeout=settings.VOICE_ANALYSIS_TIMEOUT,
        )
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        payload = json.loads(content)
        return VoiceAnalysisSummary(
            summary=payload.get("summary", ""),
            key_points=payload.get("keyPoints", []) or [],
            risk_items=[
                VoiceRiskItem(
                    risk_type=item.get("riskType", "GENERAL_RISK"),
                    severity=item.get("severity", "low"),
                    detail=item.get("detail", ""),
                    timestamp_str=item.get("timestampStr", ""),
                )
                for item in (payload.get("riskItems", []) or [])
            ],
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"[VoiceAnalysis] Summary timeout ({settings.VOICE_ANALYSIS_TIMEOUT}s)"
        )
    except Exception as e:
        logger.error(f"[VoiceAnalysis] Summary parse error: {e}")

    fallback_points = []
    seen = set()
    for item in agreements:
        point = f"{item.agreement_type}: {item.value}"
        if point not in seen:
            seen.add(point)
            fallback_points.append(point)
        if len(fallback_points) >= 5:
            break

    summary_text = transcript[:300].strip()
    if len(transcript) > 300:
        summary_text += "..."

    return VoiceAnalysisSummary(
        summary=summary_text or "음성 분석 결과 요약을 생성하지 못했습니다.",
        key_points=fallback_points,
        risk_items=[],
    )
