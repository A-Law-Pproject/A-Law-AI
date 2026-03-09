"""
RAG 기반 챗봇 API
- Redis로 대화 이력(세션) 관리
- Spring Boot는 session_id만 유지하면 됨
"""
import json
import uuid

import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException
from loguru import logger

from app.core.config import settings
from app.core.dependencies import get_qdrant_client, get_embeddings, get_llm
from app.rag.chain.chain import chat_rag
from app.schemas.chat import ChatRequest, ChatResponse, ChatHistoryResponse

router = APIRouter()

# Redis 클라이언트 (비동기)
_redis: aioredis.Redis | None = None

SESSION_TTL = 3600  # 세션 유효시간 1시간
MAX_HISTORY = 20    # 저장할 최대 메시지 수 (10턴 = 20개)


async def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


async def _load_history(session_id: str) -> list[dict]:
    """Redis에서 대화 이력 로드"""
    try:
        r = await _get_redis()
        raw = await r.get(f"chat:{session_id}")
        return json.loads(raw) if raw else []
    except Exception as e:
        logger.warning(f"[Chat] Redis 이력 로드 실패 (session={session_id}): {e}")
        return []


async def _save_history(session_id: str, messages: list[dict]) -> None:
    """Redis에 대화 이력 저장 (최근 MAX_HISTORY개만 유지)"""
    try:
        r = await _get_redis()
        trimmed = messages[-MAX_HISTORY:]
        await r.setex(f"chat:{session_id}", SESSION_TTL, json.dumps(trimmed, ensure_ascii=False))
    except Exception as e:
        logger.warning(f"[Chat] Redis 이력 저장 실패 (session={session_id}): {e}")


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG 기반 챗봇 메시지 전송

    - session_id 없으면 새 세션 생성
    - Redis에 대화 이력 저장 (1시간 TTL)
    - 병렬 멀티 인덱스 검색으로 관련 법률 문서 검색
    - Spring Boot는 응답의 session_id를 저장해 다음 요청에 재사용
    """
    qdrant = get_qdrant_client()
    embeddings = get_embeddings()
    llm = get_llm()

    if not all([qdrant, embeddings, llm]):
        raise HTTPException(status_code=503, detail="RAG 시스템 초기화 실패")

    # 세션 ID 결정
    session_id = request.session_id or str(uuid.uuid4())

    # 이전 대화 이력 로드
    history = await _load_history(session_id)

    try:
        result = await chat_rag(
            message=request.message,
            history=history,
            client=qdrant,
            embeddings=embeddings,
            llm=llm,
            contract_context=request.contract_context,
        )
    except Exception as e:
        logger.error(f"[Chat] chat_rag 실패 (session={session_id}): {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 이력 업데이트 후 저장
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": result["answer"]})
    await _save_history(session_id, history)

    turn_count = len(history) // 2  # 메시지 쌍 수
    logger.info(f"[Chat] session={session_id}, turn={turn_count}")

    return ChatResponse(
        session_id=session_id,
        answer=result["answer"],
        sources=result["sources"],
        turn_count=turn_count,
    )


@router.get("/{session_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """
    대화 이력 조회

    Spring Boot에서 채팅 UI 복원 시 사용
    """
    history = await _load_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")

    return ChatHistoryResponse(
        session_id=session_id,
        messages=history,
        turn_count=len(history) // 2,
    )


@router.delete("/{session_id}")
async def delete_chat_session(session_id: str):
    """대화 세션 삭제 (로그아웃 / 새 대화 시작)"""
    try:
        r = await _get_redis()
        deleted = await r.delete(f"chat:{session_id}")
        if not deleted:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        return {"message": f"세션 {session_id} 삭제 완료"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Chat] 세션 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
