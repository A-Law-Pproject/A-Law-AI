"""
음성 분석 메시지 DTO
- Spring Boot와 RabbitMQ로 통신하기 위한 메시지 스키마
"""
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class VoiceAnalysisMessage(BaseModel):
    """
    Spring Boot에서 발행하는 음성 분석 요청 메시지
    Queue: voice-record-queue
    """
    model_config = ConfigDict(populate_by_name=True)

    voice_record_id: int = Field(..., alias="voiceRecordId")
    contract_id: int = Field(..., alias="contractId")
    user_id: int = Field(..., alias="userId")
    job_id: str = Field(..., alias="jobId")
    s3_key: str = Field(..., alias="s3Key")
    raw_text: str = Field(..., alias="rawText")
    transcript: Optional[str] = Field(default=None)


class FactCheckItem(BaseModel):
    """개별 팩트체크 항목"""
    model_config = ConfigDict(populate_by_name=True)

    claim: str
    contract_content: str = Field(..., alias="contractContent")
    is_match: bool = Field(..., alias="isMatch")
    severity: Optional[str] = Field(default=None)  # HIGH | MEDIUM | LOW | null


class VoiceFactCheckResultMessage(BaseModel):
    """
    FastAPI에서 발행하는 음성 팩트체크 결과 메시지
    Queue: voice-result-queue
    """
    model_config = ConfigDict(populate_by_name=True)

    voice_record_id: int = Field(..., alias="voiceRecordId")
    contract_id: int = Field(..., alias="contractId")
    job_id: str = Field(..., alias="jobId")
    status: str = Field(...)  # COMPLETED | FAILED
    transcript: Optional[str] = Field(default=None)
    fact_check_items: Optional[List[FactCheckItem]] = Field(default=None, alias="factCheckItems")
    processing_time_ms: Optional[int] = Field(default=None, alias="processingTimeMs")
    error_message: Optional[str] = Field(default=None, alias="errorMessage")

    def to_rabbitmq_message(self) -> dict:
        """RabbitMQ 발행용 JSON 변환 (camelCase)"""
        return self.model_dump(by_alias=True, exclude_none=True)
