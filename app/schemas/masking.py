"""
PII 마스킹 관련 Pydantic 스키마
"""
from dataclasses import dataclass, field
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime, timezone


# ================================================
# 텍스트 마스킹 내부 데이터 클래스 (Pydantic 불필요)
# ================================================

@dataclass
class MaskPosition:
    """개별 마스킹 위치 정보"""
    start: int           # 마스킹 시작 인덱스 (원본 텍스트 기준)
    end: int             # 마스킹 종료 인덱스 (원본 텍스트 기준)
    mask_type: str       # 마스킹 유형: resident_id, phone, address, account
    original_length: int # 원본 문자열 길이


@dataclass
class TextMaskingResult:
    """텍스트 마스킹 처리 결과"""
    masked_text: str                         # 마스킹된 텍스트
    positions: List[MaskPosition] = field(default_factory=list)  # 마스킹 위치 목록
    mask_count: int = 0                      # 총 마스킹 횟수
    mask_types_found: List[str] = field(default_factory=list)    # 발견된 마스킹 유형 목록


# ================================================
# MongoDB 저장용 Pydantic 스키마
# ================================================

class MaskingMetadata(BaseModel):
    """마스킹 처리 메타데이터 (MongoDB maskedAt 필드 저장용)"""
    model_config = ConfigDict(populate_by_name=True)

    masked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        alias="maskedAt",
        description="마스킹 처리 시각 (UTC)"
    )
    mask_count: int = Field(
        default=0,
        alias="maskCount",
        description="총 마스킹 항목 수"
    )
    mask_types: List[str] = Field(
        default_factory=list,
        alias="maskTypes",
        description="마스킹된 PII 유형 목록"
    )
    masked_s3_key: Optional[str] = Field(
        default=None,
        alias="maskedS3Key",
        description="마스킹된 이미지의 S3 키"
    )
    masking_version: str = Field(
        default="1.1",
        alias="maskingVersion",
        description="마스킹 엔진 버전"
    )
    masking_failed: bool = Field(
        default=False,
        alias="maskingFailed",
        description="마스킹 처리 실패 여부 (True면 원본 유지)"
    )


class MaskingStoreResult(BaseModel):
    """마스킹 처리 및 저장 결과"""
    success: bool = Field(..., description="마스킹 성공 여부")
    masked_text: Optional[str] = Field(default=None, description="마스킹된 텍스트")
    masked_s3_key: Optional[str] = Field(default=None, description="마스킹 이미지 S3 키")
    metadata: MaskingMetadata = Field(
        default_factory=MaskingMetadata,
        description="마스킹 메타데이터"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="마스킹 실패 시 오류 메시지"
    )
