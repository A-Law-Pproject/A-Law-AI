"""쿼리 확장 전략: HyDE / Multi-query.

사용자 질문이 짧거나 구어체일 때 원문 임베딩과 법조문 임베딩 간
벡터 공간 불일치로 검색 품질이 저하된다.
두 전략 모두 LLM을 활용해 검색 품질을 향상시킨다.

- HyDE (Hypothetical Document Embeddings):
    LLM으로 가상 답변 문서를 생성한 후 그 텍스트를 임베딩.
    질문보다 법조문에 가까운 벡터를 만들어 검색 정확도를 높인다.

- Multi-query:
    하나의 질문을 N가지 표현으로 변형한 뒤 각각 검색 후 합산.
    단일 쿼리가 놓치는 관련 문서를 커버한다.
"""
import asyncio

from langchain_openai import ChatOpenAI
from loguru import logger

_HYDE_PROMPT = """당신은 주택임대차보호법·상가건물 임대차보호법·민법 전문 법률 문서 작성자입니다.
아래 질문에 대한 답변을 반드시 법조문 형식으로 작성하세요.

구어체 → 법률 용어 변환 (질문 해석 전 적용):
- "이미 살고 있는 사람" / "먼저 들어온 사람" → "선행 임차인" / "선순위 임차인"
- "들어가있을때" / "살고 있는 상태에서" → "임차 기간 중" / "점유 중인 상태에서"
- "보증금을 내다" / "돈을 내다" → "보증금을 지급하다" / "임대차 계약을 체결하다"
- "뺏기다" / "못 돌려받다" → "보증금 반환을 거절당하다" / "우선변제권 열위"
- "집주인이 바뀌다" → "임대인 지위 승계"
- "그냥 나가라고 하다" → "명도 청구" / "임차권 대항력"

작성 규칙:
1. "주택임대차보호법 제X조 제X항", "상가건물 임대차보호법 제X조", "민법 제X조" 등
   실제 조문 번호를 반드시 포함할 것
2. "임차인", "임대인", "보증금", "대항력", "우선변제권", "계약갱신청구권" 등
   법률 용어를 그대로 사용할 것
3. 법령 조항을 직접 인용하는 스타일로 작성할 것 (예: "제3조에 따르면 …")
4. 전세사기·민간임대주택 관련 질문이면 해당 특별법 조문도 함께 포함할 것
5. 실제 답이 정확한지는 중요하지 않음 — 법조문과 가장 유사한 벡터를 만드는 것이 목적

질문: {question}
법조문 형식 답변:"""

_MULTI_QUERY_PROMPT = """당신은 임대차 계약 법률 전문가입니다.
아래 질문을 {n}가지 다른 표현으로 바꿔주세요.

변환 규칙:
1. 구어체 → 법률 용어 (예: "집 나가야 해?" → "임대차 계약 종료 후 명도의무")
2. 법률 용어 동의어 활용:
   - 전세 ↔ 보증금 전세계약
   - 전입신고 ↔ 주민등록전입
   - 집주인 ↔ 임대인
   - 세입자 ↔ 임차인
   - 묵시적 갱신 ↔ 계약갱신청구권
3. 핵심 법령 조문 관점에서 재표현
   (예: "주택임대차보호법 제6조 관련 계약갱신 …")
4. 전세사기·깡통전세 관련이면 "전세사기피해자 지원 및 주거안정에 관한 특별법" 언급
5. 한 줄에 하나씩, 번호 없이 출력

질문: {question}"""


def expand_query_hyde(question: str, llm: ChatOpenAI) -> str:
    """HyDE: 가상 답변 문서를 생성해 반환.

    Args:
        question: 사용자 원문 질문.
        llm: ChatOpenAI 인스턴스.

    Returns:
        법조문 스타일 가상 답변 문자열 (이 텍스트를 임베딩에 사용).
    """
    prompt = _HYDE_PROMPT.format(question=question)
    response = llm.invoke(prompt)
    logger.debug(f"HyDE generated for: {question[:40]}...")
    return response.content


async def async_expand_query_hyde(question: str, llm: ChatOpenAI) -> str:
    """expand_query_hyde의 비동기 버전."""
    prompt = _HYDE_PROMPT.format(question=question)
    response = await llm.ainvoke(prompt)
    logger.debug(f"HyDE generated for: {question[:40]}...")
    return response.content


def expand_query_multi(question: str, llm: ChatOpenAI, n: int = 3) -> list[str]:
    """Multi-query: 질문을 n개 변형으로 확장.

    Args:
        question: 사용자 원문 질문.
        llm: ChatOpenAI 인스턴스.
        n: 생성할 변형 쿼리 수.

    Returns:
        원문 포함 (n+1)개 쿼리 리스트. 원문은 항상 첫 번째.
    """
    prompt = _MULTI_QUERY_PROMPT.format(question=question, n=n)
    response = llm.invoke(prompt)
    variants = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
    logger.debug(f"Multi-query expanded {len(variants)} variants for: {question[:40]}...")
    return [question] + variants[:n]


async def async_expand_query_multi(question: str, llm: ChatOpenAI, n: int = 3) -> list[str]:
    """expand_query_multi의 비동기 버전."""
    prompt = _MULTI_QUERY_PROMPT.format(question=question, n=n)
    response = await llm.ainvoke(prompt)
    variants = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
    logger.debug(f"Multi-query expanded {len(variants)} variants for: {question[:40]}...")
    return [question] + variants[:n]
