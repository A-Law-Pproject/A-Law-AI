"""
Standalone voice analysis without contract matching.
"""
import asyncio
import json

from loguru import logger

from app.core.config import settings
from app.core.dependencies import get_embeddings, get_llm, get_vector_db
from app.rag.chain.chain import _analyze_single_clause, detect_risk_contract
from app.schemas.risk_analysis import ClauseRisk, ContractRiskResult
from app.schemas.voice_shared import AgreementItem, SegmentResult
from app.schemas.voice_standalone import VoiceAnalysisSummary, VoiceClauseRisk


# 요약·핵심포인트만 추출 (위험 판단은 RAG 파이프라인이 담당)
_VOICE_SUMMARY_PROMPT = """당신은 한국 임대차 계약 법률 전문가입니다.
아래 음성 녹음 전사본을 분석하여 JSON만 반환하세요.

반환 형식:
{
  "summary": "전사본 전체 요약 (한국어, 2~3문장)",
  "keyPoints": ["핵심 포인트 1", "핵심 포인트 2"]
}

[전사본]
{transcript}

[추출된 합의 항목]
{agreements}
"""


def _parse_json_content(content: str) -> dict:
    """LLM 응답에서 JSON 파싱 (코드블록 제거 포함)."""
    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1] if len(parts) > 1 else content
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())


async def _generate_summary_keypoints(
    transcript: str,
    agreements: list[AgreementItem],
    llm,
) -> tuple[str, list[str]]:
    """전사본 요약 및 핵심 포인트 생성 (LLM 전용, RAG 불필요)."""
    agreements_hint = [
        {"type": item.agreement_type, "value": item.value, "timestamp": item.timestamp_str}
        for item in agreements[:20]
    ]
    prompt = (
        _VOICE_SUMMARY_PROMPT
        .replace("{transcript}", transcript[:4000])
        .replace("{agreements}", json.dumps(agreements_hint, ensure_ascii=False))
    )
    try:
        response = await asyncio.wait_for(
            llm.ainvoke(prompt),
            timeout=settings.VOICE_ANALYSIS_TIMEOUT,
        )
        payload = _parse_json_content(response.content.strip())
        return payload.get("summary", ""), payload.get("keyPoints", []) or []
    except Exception as e:
        logger.error(f"[VoiceAnalysis] 요약 생성 실패: {e}")
        fallback_points = list({f"{i.agreement_type}: {i.value}" for i in agreements})[:5]
        summary_text = (transcript[:300].strip() + "...") if len(transcript) > 300 else transcript[:300].strip()
        return summary_text or "음성 분석 결과 요약을 생성하지 못했습니다.", fallback_points


async def _rag_risk_from_agreements(
    agreements: list[AgreementItem],
    llm,
    client,
    embeddings,
) -> ContractRiskResult:
    """합의 항목별 RAG 위험 분석.

    각 AgreementItem을 _analyze_single_clause로 분석:
    - special_clauses_illegal / normal / law_statutes 벡터 검색
    - BGE Reranker 법령 관련성 평가
    - CRAG: 낮은 점수면 쿼리 재작성 후 재검색
    - with_structured_output(ClauseRisk)으로 안전한 파싱
    """
    structured_llm = llm.with_structured_output(ClauseRisk)

    # 합의 항목 → 조항 텍스트 (맥락 포함) + 타임스탬프 보존
    clause_texts = [
        f"{item.agreement_type}: {item.value}\n\n맥락: {item.context}"
        for item in agreements[:15]
    ]
    timestamps = [item.timestamp_str for item in agreements[:15]]

    raw_results = await asyncio.gather(
        *[_analyze_single_clause(text, client, embeddings, structured_llm) for text in clause_texts],
        return_exceptions=True,
    )

    valid_clauses: list[VoiceClauseRisk] = []
    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            logger.error(f"[VoiceRisk] 합의항목 {i + 1} 분석 실패: {result}")
            valid_clauses.append(VoiceClauseRisk(
                text=clause_texts[i],
                risk_level="주의",
                category="분석 오류",
                analysis="분석 중 오류가 발생했습니다.",
                legal_reference="",
                score=50,
                timestamp_str=timestamps[i],
            ))
        else:
            valid_clauses.append(VoiceClauseRisk(
                **result.model_dump(),
                timestamp_str=timestamps[i],
            ))

    risk_count = sum(1 for c in valid_clauses if c.risk_level == "위험")
    caution_count = sum(1 for c in valid_clauses if c.risk_level == "주의")
    safety_count = sum(1 for c in valid_clauses if c.risk_level == "안전")
    total = len(valid_clauses)

    # detect_risk_contract와 동일한 종합 점수 산식
    avg_score = sum(c.score for c in valid_clauses) / max(total, 1)
    weight_score = (risk_count / max(total, 1)) * 100
    overall_score = min(int(avg_score * 0.6 + weight_score * 0.4), 100)

    logger.debug(
        f"[VoiceRisk] 분석 완료 — 위험:{risk_count} 주의:{caution_count} 안전:{safety_count} 종합:{overall_score}"
    )

    return ContractRiskResult(
        overall_risk_score=overall_score,
        risk_summary={"Risk": risk_count, "Caution": caution_count, "Safety": safety_count},
        total_clauses=total,
        clauses=valid_clauses,
    )


async def summarize_voice_analysis(
    transcript: str,
    agreements: list[AgreementItem],
    segments: list[SegmentResult],
) -> VoiceAnalysisSummary:
    llm = get_llm()
    client = get_vector_db()
    embeddings = get_embeddings()

    if agreements:
        # 요약(LLM)과 위험 분석(RAG)을 병렬 실행
        (summary_text, key_points), risk_result = await asyncio.gather(
            _generate_summary_keypoints(transcript, agreements, llm),
            _rag_risk_from_agreements(agreements, llm, client, embeddings),
        )
    else:
        # 합의 항목 없으면 전사본 전체를 detect_risk_contract로 분석 (legacy 폴백)
        logger.warning("[VoiceAnalysis] 합의 항목 없음 — 전사본을 detect_risk_contract로 폴백")
        (summary_text, key_points), risk_dict = await asyncio.gather(
            _generate_summary_keypoints(transcript, agreements, llm),
            detect_risk_contract(transcript, client, embeddings, llm),
        )
        risk_result = ContractRiskResult(
            overall_risk_score=risk_dict.get("overall_risk_score", 0),
            risk_summary=risk_dict.get("risk_summary", {"Risk": 0, "Caution": 0, "Safety": 0}),
            total_clauses=risk_dict.get("total_clauses", 0),
            clauses=[ClauseRisk(**c) for c in risk_dict.get("clauses", [])],
        )

    return VoiceAnalysisSummary(
        summary=summary_text,
        key_points=key_points,
        risk_analysis=risk_result,
    )
