"""
Schemas for standalone HTTP voice analysis APIs.
"""
from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.risk_analysis import ClauseRisk, ContractRiskResult
from app.schemas.voice_shared import AgreementItem, SegmentResult, VoiceAudioMeta


class VoiceClauseRisk(ClauseRisk):
    """ClauseRisk에 음성 타임스탬프를 추가한 확장형."""
    timestamp_str: str = Field(default="", description="발화 타임스탬프 HH:MM:SS")


class VoiceAnalysisSummary(BaseModel):
    summary: str = Field(..., description="High-level transcript summary")
    key_points: List[str] = Field(default_factory=list, description="Important points")
    risk_analysis: ContractRiskResult = Field(
        default_factory=lambda: ContractRiskResult(
            overall_risk_score=0,
            risk_summary={"Risk": 0, "Caution": 0, "Safety": 0},
            total_clauses=0,
            clauses=[],
        ),
        description="위험도 분석 결과 (기존 risk 서비스와 동일 포맷)",
    )


class VoiceAnalyzeS3Request(BaseModel):
    s3_key: str = Field(..., description="S3 key for the audio file")
    source_id: Optional[str] = Field(
        default="standalone",
        description="Optional source identifier used for metadata grouping",
    )


class VoiceAnalysisResponse(BaseModel):
    success: bool
    transcript: str = Field(default="", description="Combined transcript")
    summary: Optional[VoiceAnalysisSummary] = None
    segments: List[SegmentResult] = Field(default_factory=list)
    agreements: List[AgreementItem] = Field(default_factory=list)
    audio_meta: Optional[VoiceAudioMeta] = None
    processing_time_ms: int = 0
    error_message: Optional[str] = None
