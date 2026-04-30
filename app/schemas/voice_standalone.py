"""
Schemas for standalone HTTP voice analysis APIs.
"""
from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.voice_shared import AgreementItem, SegmentResult, VoiceAudioMeta


class VoiceRiskItem(BaseModel):
    risk_type: str = Field(..., description="Detected risk category")
    severity: str = Field(..., description="low/medium/high/critical")
    detail: str = Field(..., description="Why this may be risky")
    timestamp_str: str = Field(default="", description="Related transcript timestamp")


class VoiceAnalysisSummary(BaseModel):
    summary: str = Field(..., description="High-level transcript summary")
    key_points: List[str] = Field(default_factory=list, description="Important points")
    risk_items: List[VoiceRiskItem] = Field(
        default_factory=list,
        description="Detected risks",
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
