"""챗봇 요청/응답 스키마"""
from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """챗봇 메시지 요청"""
    session_id: Optional[str] = Field(
        default=None,
        description="대화 세션 ID. 없으면 새 세션 생성",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    message: str = Field(
        ...,
        description="사용자 메시지",
        examples=["보증금 반환은 언제 받을 수 있나요?"],
    )
    contract_context: Optional[str] = Field(
        default=None,
        description="현재 분석 중인 계약서 원문 (있으면 우선 참고)",
        examples=["제1조 (목적) 이 계약은..."],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "보증금 반환은 언제 받을 수 있나요?",
                },
                {
                    "session_id": "550e8400-e29b-41d4-a716-446655440000",
                    "message": "그럼 이자는 받을 수 있나요?",
                },
            ]
        }
    }


class ChatResponse(BaseModel):
    """챗봇 응답"""
    session_id: str = Field(..., description="대화 세션 ID (Spring Boot에서 관리)")
    answer: str = Field(..., description="AI 답변")
    sources: list[str] = Field(default_factory=list, description="참고한 법률/문서 출처")
    turn_count: int = Field(..., description="현재 세션의 총 대화 수")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "550e8400-e29b-41d4-a716-446655440000",
                    "answer": "주택임대차보호법 제7조에 따르면 임대인은 임대차 종료 후 14일 이내에 보증금을 반환해야 합니다.",
                    "sources": ["[law_database] 제7조(보증금의 반환)"],
                    "turn_count": 1,
                }
            ]
        }
    }


class ChatHistoryResponse(BaseModel):
    """대화 이력 조회 응답"""
    session_id: str
    messages: list[dict]  # [{"role": "user"|"assistant", "content": str}]
    turn_count: int
