"""
계약서 분석 메시지 DTO
- Spring Boot와 RabbitMQ로 통신하기 위한 메시지 스키마
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from pydantic.alias_generators import to_camel
from datetime import datetime
from enum import Enum


class AnalysisStatus(str, Enum):
    """분석 상태"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RiskLevel(str, Enum):
    """위험도 레벨"""
    RISK = "Risk"
    CAUTION = "Caution"
    SAFETY = "Safety"


# ========================
# RabbitMQ 수신 메시지 (Spring Boot → FastAPI)
# ========================

class ContractAnalysisRequest(BaseModel):
    """
    Spring Boot에서 발행하는 분석 요청 메시지
    Queue: contract-analysis-queue

    camelCase(jobId) / snake_case(job_id) 모두 허용
    """
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    job_id: str = Field(...)
    contract_id: int = Field(...)
    s3_key: str = Field(...)
    user_id: int = Field(...)


# ========================
# RabbitMQ 발행 메시지 (FastAPI → Spring Boot)
# ========================

class ClauseRiskResult(BaseModel):
    """개별 조항 위험도 분석 결과"""
    clause_title: str = Field(..., alias="clauseTitle")
    clause_content: str = Field(..., alias="clauseContent")
    risk_level: str = Field(..., alias="riskLevel")
    legal_reference: str = Field(default="", alias="legalReference")
    recommendation: str = Field(default="")
    reasoning_summary: str = Field(default="", alias="reasoningSummary")

    class Config:
        populate_by_name = True


class RiskAnalysisResult(BaseModel):
    """Risk 분석 전체 결과"""
    total_clauses: int = Field(..., alias="totalClauses")
    risk_count: int = Field(default=0, alias="riskCount")
    caution_count: int = Field(default=0, alias="cautionCount")
    safety_count: int = Field(default=0, alias="safetyCount")
    risk_percentage: float = Field(default=0.0, alias="riskPercentage")
    clause_results: List[ClauseRiskResult] = Field(default_factory=list, alias="clauseResults")

    class Config:
        populate_by_name = True


class ContractSummary(BaseModel):
    """AI 요약 결과"""
    title: str = Field(default="", description="계약서 제목/유형")
    parties: List[str] = Field(default_factory=list, description="계약 당사자")
    key_terms: List[str] = Field(default_factory=list, alias="keyTerms", description="핵심 조건")
    duration: str = Field(default="", description="계약 기간")
    summary_text: str = Field(..., alias="summaryText", description="요약 텍스트")
    important_dates: List[str] = Field(default_factory=list, alias="importantDates")

    class Config:
        populate_by_name = True


class ContractAnalysisResult(BaseModel):
    """
    FastAPI에서 발행하는 분석 결과 메시지
    Exchange: contract.analysis.result
    """
    job_id: str = Field(..., alias="jobId")
    contract_id: int = Field(..., alias="contractId")
    status: AnalysisStatus = Field(default=AnalysisStatus.COMPLETED)

    # AI 요약 결과
    summary: Optional[ContractSummary] = Field(default=None)

    # Risk 분석 결과
    risk_analysis: Optional[RiskAnalysisResult] = Field(default=None, alias="riskAnalysis")

    # 메타데이터
    processing_time_ms: int = Field(default=0, alias="processingTimeMs")
    completed_at: datetime = Field(default_factory=datetime.now, alias="completedAt")
    error_message: Optional[str] = Field(default=None, alias="errorMessage")

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_rabbitmq_message(self) -> Dict[str, Any]:
        """RabbitMQ 발행용 JSON 변환 (camelCase) - mode='json'으로 datetime 자동 직렬화"""
        return self.model_dump(by_alias=True, exclude_none=True, mode='json')
