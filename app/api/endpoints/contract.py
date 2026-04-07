"""
계약서 분석 API 엔드포인트
"""
from fastapi import APIRouter, HTTPException, Body

from app.schemas.contract import TermRequest, TermExplanation
from app.rag.chain.chain import explain_term_rag
from app.core.dependencies import get_vector_db, get_embeddings, get_llm

router = APIRouter()


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
