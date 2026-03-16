"""
계약서 분석 관련 스키마
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any


class ContractRequest(BaseModel):
    """계약서 분석 요청"""
    text: str = Field(
        ...,
        description="계약서 텍스트",
        examples=["제1조 (목적물) 본 계약의 목적물은 서울특별시 강남구 역삼동 123-45 아파트 101동 1001호로 한다.\n제2조 (계약 기간) 본 계약의 기간은 2024년 1월 1일부터 2026년 12월 31일까지로 한다."]
    )
    contract_id: str = Field(
        ...,
        description="계약서 ID",
        examples=["CONTRACT_20240111_001"]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "제1조 (목적물) 본 계약의 목적물은 서울특별시 강남구 역삼동 123-45 아파트 101동 1001호로 한다.\n제2조 (계약 기간) 본 계약의 기간은 2024년 1월 1일부터 2026년 12월 31일까지로 한다.\n제3조 (보증금 및 차임) 보증금은 1억원으로 하고, 월 차임은 50만원으로 한다.",
                    "contract_id": "CONTRACT_20240111_001"
                }
            ]
        }
    )


class FraudRisk(BaseModel):
    """사기 위험 항목"""
    pattern: str = Field(..., description="탐지된 패턴")
    severity: str = Field(..., description="심각도: high, medium, low")
    description: str = Field(..., description="설명")
    location: Optional[str] = Field(None, description="위치")


class MissingClause(BaseModel):
    """누락된 필수 조항"""
    clause_name: str = Field(..., description="조항명")
    importance: str = Field(..., description="중요도: critical, important, recommended")
    description: str = Field(..., description="설명")
    legal_basis: Optional[str] = Field(None, description="법적 근거")


class IllegalClause(BaseModel):
    """불법 조항"""
    clause_text: str = Field(..., description="조항 내용")
    violation: str = Field(..., description="위반 사항")
    legal_reference: str = Field(..., description="관련 법령")
    recommendation: str = Field(..., description="권고사항")


class FraudDetectionResponse(BaseModel):
    """사기 탐지 응답"""
    fraud_risks: List[Dict[str, Any]] = Field(default_factory=list, description="사기 위험 항목")
    missing_clauses: List[Dict[str, Any]] = Field(default_factory=list, description="누락 조항")
    illegal_clauses: List[Dict[str, Any]] = Field(default_factory=list, description="불법 조항")
    risk_score: float = Field(..., description="종합 위험도 점수 (0-100)")


class TermRequest(BaseModel):
    """법률 용어 해설 요청"""
    sentence: str = Field(..., description="해설이 필요한 문장")


class TermExplanation(BaseModel):
    """법률 용어 해설"""
    easy_explanation: str = Field(..., description="쉬운 설명")
    sentence: str = Field(default="", description="원문 문장")
    examples: List[str] = Field(default_factory=list, description="예시")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "easy_explanation": "보증금을 우선적으로 돌려받을 수 있는 권리를 증명하는 도장입니다.",
                    "sentence": "임차인은 확정일자를 받아야 우선변제권을 행사할 수 있다.",
                    "examples": [
                        "전입신고 + 확정일자 → 우선변제권 획득",
                        "주민센터나 등기소에서 무료로 받을 수 있습니다",
                        "확정일자가 빠를수록 우선순위가 높습니다"
                    ]
                }
            ]
        }
    )


class ClauseAnalysis(BaseModel):
    """조항 분석 결과"""
    title: str = Field(..., description="조항 제목")
    content: str = Field(..., description="조항 내용")
    risk_level: str = Field(..., description="위험도: Risk, Caution, Safety")
    analysis: str = Field(..., description="분석 내용")
    legal_reference: Optional[str] = Field(None, description="관련 법령")


class RiskSummary(BaseModel):
    """위험도 요약"""
    Risk: int = Field(default=0, description="고위험 조항 수")
    Caution: int = Field(default=0, description="주의 조항 수")
    Safety: int = Field(default=0, description="안전 조항 수")


class AnalysisResult(BaseModel):
    """전체 분석 결과"""
    contract_id: str = Field(default="", description="계약서 ID")
    total_clauses: int = Field(default=0, description="총 조항 수")
    risk_summary: RiskSummary = Field(default_factory=RiskSummary, description="위험도 요약")
    clauses: List[ClauseAnalysis] = Field(default_factory=list, description="조항별 분석")
    overall_risk_score: float = Field(default=0.0, description="전체 위험도 점수")
    recommendations: List[str] = Field(default_factory=list, description="권고사항")
