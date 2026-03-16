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
)
from app.rag.chain.chain import explain_term_rag
from app.core.dependencies import get_vector_db, get_embeddings, get_llm

router = APIRouter()
executor = ThreadPoolExecutor(max_workers=4)


def _detect_fraud_patterns(text: str) -> list:
    patterns = []
    if "언제든지 해지" in text or "임의로 해지" in text:
        patterns.append({"pattern": "일방적 해지권", "severity": "high", "description": "임대인이 일방적으로 계약을 해지할 수 있는 조항이 포함되어 있습니다."})
    if "위약금" in text and ("50%" in text or "전액" in text):
        patterns.append({"pattern": "과도한 위약금", "severity": "high", "description": "과도한 위약금 조항이 포함되어 있습니다."})
    if "보증금" in text and "반환" not in text:
        patterns.append({"pattern": "보증금 반환 미명시", "severity": "medium", "description": "보증금 반환 조건이 명시되어 있지 않습니다."})
    return patterns


def _find_missing_clauses(text: str) -> list:
    missing = []
    if "확정일자" not in text:
        missing.append({"clause_name": "확정일자 안내", "importance": "critical", "description": "확정일자 취득 안내가 누락되었습니다.", "legal_basis": "주택임대차보호법 제3조의6"})
    if "수리" not in text and "수선" not in text:
        missing.append({"clause_name": "수리 책임", "importance": "important", "description": "수리 책임에 관한 조항이 누락되었습니다."})
    if "중도 해지" not in text and "중도해지" not in text:
        missing.append({"clause_name": "중도 해지 조건", "importance": "important", "description": "중도 해지 시 조건이 명시되어 있지 않습니다."})
    if "갱신" not in text:
        missing.append({"clause_name": "계약 갱신", "importance": "recommended", "description": "계약 갱신 관련 조항이 없습니다.", "legal_basis": "주택임대차보호법 제6조의3"})
    return missing


def _check_illegal_clauses(text: str) -> list:
    illegal = []
    if "10%" in text and ("인상" in text or "증액" in text):
        illegal.append({"clause_text": "차임 10% 인상 조항", "violation": "주택임대차보호법 차임증액 제한 위반", "legal_reference": "주택임대차보호법 제7조 (5% 제한)", "recommendation": "차임 증액률을 5% 이하로 수정하세요."})
    if "권리를 포기" in text or "이의를 제기할 수 없" in text:
        illegal.append({"clause_text": "임차인 권리 포기 조항", "violation": "강행규정 위반", "legal_reference": "주택임대차보호법 제10조", "recommendation": "임차인의 법적 권리를 제한하는 조항은 무효입니다."})
    if "1년" in text and "기간" in text:
        illegal.append({"clause_text": "1년 계약 기간", "violation": "최소 계약 기간 미달", "legal_reference": "주택임대차보호법 제4조 (2년 보장)", "recommendation": "임차인이 원하면 2년까지 거주할 수 있습니다."})
    return illegal


def calculate_risk_score(fraud_risks: List, missing_clauses: List, illegal_clauses: List) -> float:
    score = 0.0
    score += len(fraud_risks) * 15
    score += len(missing_clauses) * 10
    score += len(illegal_clauses) * 25
    return min(score, 100.0)


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
                    "sentence": "임차인은 확정일자를 받아야 우선변제권을 행사할 수 있다."
                }
            },
            "대항력": {
                "summary": "대항력이란?",
                "value": {
                    "sentence": "전입신고와 점유를 통해 대항력을 취득한다."
                }
            },
        }
    )
):
    db = get_vector_db()
    emb = get_embeddings()
    llm = get_llm()

    try:
        result = await explain_term_rag(
            term=request.sentence,
            client=db,
            embeddings=emb,
            llm=llm,
            surrounding_text=request.sentence,
        )
        return TermExplanation(
            easy_explanation=result.get("simple_explanation", ""),
            sentence=request.sentence,
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
        loop = asyncio.get_running_loop()

        tasks = [
            loop.run_in_executor(executor, _detect_fraud_patterns, request.text),
            loop.run_in_executor(executor, _find_missing_clauses, request.text),
            loop.run_in_executor(executor, _check_illegal_clauses, request.text),
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
