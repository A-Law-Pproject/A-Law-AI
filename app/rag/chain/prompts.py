from langchain_core.prompts import PromptTemplate

CONTRACT_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""당신은 임대차 계약서 분석 전문 AI입니다.

다음 법률 문서, 약관, 특약사항을 참고하여 질문에 답변하세요.

참고 문서:
{context}

질문: {question}

답변 시 주의사항:
1. 독소조항이 있으면 명확히 경고하세요
2. 관련 법률 조항을 인용하세요
3. 사회초년생도 이해하기 쉽게 설명하세요

답변:""",
)

RISK_PROMPT = PromptTemplate(
    input_variables=["clause", "illegal_matches", "normal_matches", "law_context"],
    template="""사용자 계약 조항: {clause}

=== 유사한 독소 특약 사례 ===
{illegal_matches}

=== 유사한 정상 특약 사례 ===
{normal_matches}

=== 관련 법률/약관 ===
{law_context}

위 정보를 바탕으로 사용자 조항의 위험도를 분석하세요.

답변 형식:
- 위험도: [높음/중간/낮음]
- 독소조항 유사도: [유사한 독소조항이 있다면 설명]
- 관련 법률: [조항 인용]
- 설명: [왜 위험한지 또는 안전한지]
- 권장 조치: [구체적 조언]""",
)
