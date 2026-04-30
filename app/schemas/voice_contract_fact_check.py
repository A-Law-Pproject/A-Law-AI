"""
Schemas for Spring-integrated async voice contract fact-check jobs.
"""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class VoiceContractFactCheckRequest(BaseModel):
    """
    Spring Boot -> FastAPI async fact-check request message.

    The contract text is resolved from OCR storage using the contract identifier,
    so the payload only needs job metadata plus the uploaded audio object key.
    """

    model_config = ConfigDict(extra="ignore")

    voiceRecordId: int = Field(..., description="Voice record primary key")
    contractId: int = Field(..., description="Contract primary key")
    userId: int = Field(..., description="User primary key")
    jobId: str = Field(..., description="Analysis job UUID")
    s3Key: str = Field(..., description="S3 key for the uploaded audio file")


class FactCheckItem(BaseModel):
    """Single fact-check result item."""

    claim: str = Field(..., description="Claim extracted from the transcript")
    contractContent: str = Field(..., description="Relevant contract clause content")
    isMatch: bool = Field(..., description="Whether the claim matches the contract")
    severity: Optional[str] = Field(
        default=None,
        description="Mismatch severity (HIGH/MEDIUM/LOW, null when matched)",
    )


class VoiceContractFactCheckResult(BaseModel):
    """Outgoing result message published to `voice.analysis.result`."""

    voiceRecordId: int
    contractId: int
    jobId: str
    status: str  # COMPLETED | FAILED
    transcript: Optional[str] = None
    factCheckItems: List[FactCheckItem] = Field(default_factory=list)
    processingTimeMs: Optional[int] = None
    errorMessage: Optional[str] = None


# Compatibility aliases for the previous mixed naming.
VoiceAnalysisRequest = VoiceContractFactCheckRequest
VoiceFactCheckResult = VoiceContractFactCheckResult
