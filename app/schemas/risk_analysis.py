from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClauseRisk(BaseModel):
    text: str = Field(description="조항 원문 텍스트")
    risk_level: Literal["위험", "주의", "안전"] = Field(description="위험 수준")
    category: str = Field(description="조항 유형 (예: 원상복구, 관리비, 임차인 의무 등)")
    analysis: str = Field(description="위험/주의/안전 판단 근거 (2~3문장)")
    related_law: str = Field(description="관련 법률 조항 (예: 주택임대차보호법 제3조). 없으면 빈 문자열")
    score: int = Field(ge=0, le=100, description="위험 점수 0~100")


class ContractRiskResult(BaseModel):
    overall_risk_score: int = Field(ge=0, le=100, description="종합 위험도 점수 0~100")
    risk_summary: dict[str, int] = Field(description='{"Risk": n, "Caution": n, "Safety": n}')
    total_clauses: int = Field(description="분석된 전체 조항 수")
    clauses: list[ClauseRisk] = Field(description="조항별 분석 결과")
