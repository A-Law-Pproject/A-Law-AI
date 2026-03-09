"""
계약서 분석 API 엔드포인트
"""
from fastapi import APIRouter, HTTPException, Body
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.schemas.contract import (
    ContractRequest,
    FraudDetectionResponse,
    TermRequest,
    TermExplanation,
    AnalysisResult
)
from app.services.analyzer import ContractAnalysisService
from app.rag.chain.chain import explain_term_rag
from app.core.dependencies import get_qdrant_client, get_embeddings, get_llm

router = APIRouter()
executor = ThreadPoolExecutor(max_workers=4)


def get_analysis_service():
    """분석 서비스 인스턴스 반환"""
    return ContractAnalysisService()


# @router.post(
#     "/analyze",
#     response_model=AnalysisResult,
#     summary="계약서 전체 분석",
#     description="""
#     계약서 텍스트를 입력받아 AI 기반 법률 분석을 수행합니다.

#     **분석 항목:** 조항 자동 분리, 위험도 분석, RAG 기반 법률 검증, 불법 조항 탐지, 누락 조항 확인
#     """,
# )
# async def analyze_contract(
#     request: ContractRequest = Body(
#         ...,
#         openapi_examples={
#             "주거용 임대차 계약서": {
#                 "summary": "주거용 임대차 계약서 예시",
#                 "value": {
#                     "text": "제1조 (목적물) 본 계약의 목적물은 서울특별시 강남구 역삼동 123-45 아파트 101동 1001호로 한다.\n제2조 (계약 기간) 본 계약의 기간은 2024년 1월 1일부터 2026년 12월 31일까지 2년으로 한다.\n제3조 (보증금 및 차임) 보증금은 금 일억원정으로 하고, 월 차임은 금 오십만원정으로 한다.\n제4조 (차임 증액) 임대인은 매년 5% 범위 내에서 차임을 증액할 수 있다.",
#                     "contract_id": "CONTRACT_001"
#                 }
#             },
#             "위험 조항 포함": {
#                 "summary": "위험 조항이 포함된 계약서",
#                 "value": {
#                     "text": "제1조 (목적물) 서울시 마포구 OO동 123-45\n제2조 (계약기간) 2024.1.1 ~ 2024.12.31\n제3조 (보증금) 5천만원\n제4조 (특약) 임대인은 언제든지 계약을 해지할 수 있으며, 임차인은 이의를 제기할 수 없다.\n제5조 (차임인상) 임대인은 월세를 매년 10% 인상할 수 있다.",
#                     "contract_id": "CONTRACT_002"
#                 }
#             },
#         }
#     )
# ):
#     try:
#         service = get_analysis_service()
#         result = await service.analyze_contract(request.text, request.contract_id)
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"분석 실패: {str(e)}")


@router.post(
    "/explain/term",
    response_model=TermExplanation,
    summary="법률 용어 해설",
    description="법률 문서 RAG 검색을 통해 임대차 관련 법률 용어를 설명합니다.",
)
async def explain_term(
    request: TermRequest = Body(
        ...,
        openapi_examples={
            "확정일자": {
                "summary": "확정일자란?",
                "value": {
                    "term": "확정일자",
                    "context": "주택임대차보호법",
                    "surrounding_text": "임차인은 확정일자를 받아야 우선변제권을 행사할 수 있다."
                }
            },
            "대항력": {
                "summary": "대항력이란?",
                "value": {
                    "term": "대항력",
                    "context": "임대차 계약서",
                    "surrounding_text": "전입신고와 점유를 통해 대항력을 취득한다."
                }
            },
        }
    )
):
    qdrant = get_qdrant_client()
    emb = get_embeddings()
    llm = get_llm()

    try:
        result = await explain_term_rag(
            term=request.term,
            client=qdrant,
            embeddings=emb,
            llm=llm,
            context=request.context,
            surrounding_text=request.surrounding_text,
        )
        return TermExplanation(
            term=request.term,
            easy_explanation=result.get("simple_explanation", ""),
            original_sentence=request.surrounding_text,
            legal_definition=result.get("legal_definition", ""),
            examples=result.get("examples", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"용어 해설 실패: {str(e)}")


@router.post(
    "/analyze/fraud-detection",
    response_model=FraudDetectionResponse,
    summary="[미구현] 사기 위험 탐지",
    description="""
    **[미구현]** 현재 키워드 매칭 기반 간이 탐지만 수행합니다.
    RAG Multi-Index 기반 독소조항 비교 분석으로 교체 예정입니다.
    """,
)
async def analyze_fraud_detection(
    request: ContractRequest = Body(
        ...,
        openapi_examples={
            "사기 위험 계약서": {
                "summary": "사기 위험이 높은 계약서",
                "value": {
                    "text": "제1조 임대인은 언제든지 계약을 해지할 수 있다.\n제2조 월세는 매년 10% 인상한다.\n제3조 보증금 반환 기한은 명시하지 않는다.\n제4조 임차인은 수리 요청을 할 수 없다.",
                    "contract_id": "FRAUD_TEST_001"
                }
            }
        }
    )
):
    try:
        service = get_analysis_service()
        loop = asyncio.get_running_loop()

        tasks = [
            loop.run_in_executor(executor, service.detect_fraud_patterns, request.text),
            loop.run_in_executor(executor, service.find_missing_clauses, request.text),
            loop.run_in_executor(executor, service.check_illegal_clauses, request.text)
        ]

        fraud_risks, missing_clauses, illegal_clauses = await asyncio.gather(*tasks)

        risk_score = calculate_risk_score(fraud_risks, missing_clauses, illegal_clauses)

        return FraudDetectionResponse(
            fraud_risks=fraud_risks,
            missing_clauses=missing_clauses,
            illegal_clauses=illegal_clauses,
            risk_score=risk_score
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"사기 탐지 실패: {str(e)}")


def calculate_risk_score(fraud_risks: List, missing_clauses: List, illegal_clauses: List) -> float:
    score = 0.0
    score += len(fraud_risks) * 15
    score += len(missing_clauses) * 10
    score += len(illegal_clauses) * 25
    return min(score, 100.0)
