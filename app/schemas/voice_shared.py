"""
Shared voice schemas used by both standalone HTTP analysis and async fact-check jobs.
"""
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class SegmentResult(BaseModel):
    """Single STT segment."""

    id: str = Field(..., description="Segment identifier")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcript text")
    speaker: Optional[str] = Field(default=None, description="Speaker identifier")
    timestamp_str: str = Field(default="", description="Readable timestamp HH:MM:SS")


class AgreementItem(BaseModel):
    """Structured item extracted from a transcript segment."""

    segment_id: str = Field(..., description="Source segment identifier")
    agreement_type: str = Field(..., description="amount/date/condition/agreement")
    value: str = Field(..., description="Extracted value")
    context: str = Field(..., description="Original transcript context")
    timestamp_str: str = Field(default="", description="Readable timestamp HH:MM:SS")


class VoiceAudioMeta(BaseModel):
    """Stored metadata for uploaded audio files."""

    file_hash: str = Field(..., description="SHA-256 hash with prefix")
    original_filename: str = Field(..., description="Original file name")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC creation timestamp",
    )
    s3_key: str = Field(..., description="S3 object key")
    source_id: str = Field(
        ...,
        description="Owning source identifier such as contract or standalone scope",
    )
    file_size_bytes: int = Field(default=0, description="Uploaded file size in bytes")
    content_type: str = Field(default="", description="MIME content type")
