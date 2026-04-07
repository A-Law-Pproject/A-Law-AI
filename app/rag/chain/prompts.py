from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

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

# ── 챗봇 (대화 이력 포함) ──────────────────────────────────────────────────────
CHAT_SYSTEM_PROMPT = """당신은 한국 임대차 계약 전문 AI 어시스턴트입니다.
사용자가 계약서나 법률에 관해 궁금한 점을 대화 형식으로 물어볼 수 있습니다.

아래 법률 문서와 판례를 참고하여 답변하세요.

[참고 문서]
{context}

답변 원칙:
1. 모르는 내용은 솔직하게 "확인이 필요합니다"라고 말하세요.
2. 관련 법조항(주택임대차보호법 등)이 있으면 조문 번호와 함께 인용하세요.
3. 독소조항 위험이 있으면 ⚠️ 표시와 함께 명확히 경고하세요.
4. 사회초년생도 이해할 수 있도록 쉽게 설명하세요.
5. 이전 대화 내용을 참고하여 자연스럽게 이어서 답변하세요."""

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CHAT_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

TERM_EXPLANATION_PROMPT = PromptTemplate(
    input_variables=["term", "context", "surrounding_text", "law_context"],
    template="""당신은 한국 임대차 계약 법률 전문가입니다.
아래 법률 문서를 바탕으로 용어를 설명하세요.

법률 용어: {term}
문맥: {context}
관련 문장: {surrounding_text}

=== 관련 법률 문서 ===
{law_context}

다음 JSON 형식으로만 답변하세요 (한국어로):
{{
  "simple_explanation": "사회초년생도 이해할 수 있는 한 문장 쉬운 설명",
  "legal_definition": "법조문 기반 법률적 정의",
  "examples": ["실생활 예시 1", "실생활 예시 2", "실생활 예시 3"]
}}

JSON만 출력하고 다른 텍스트는 포함하지 마세요.""",
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

CONTRACT_RISK_PROMPT = PromptTemplate(
    input_variables=["contract_text", "illegal_matches", "normal_matches", "law_context"],
    template="""당신은 한국 임대차 계약서 분석 전문 AI입니다.
아래 계약서 전문을 조항 단위로 분석하여 위험 요소를 찾아주세요.

=== 계약서 전문 ===
{contract_text}

=== 유사한 독소조항 사례 (RAG 검색 결과) ===
{illegal_matches}

=== 유사한 정상조항 사례 (RAG 검색 결과) ===
{normal_matches}

=== 관련 법률 (RAG 검색 결과) ===
{law_context}

위 정보를 바탕으로 계약서의 모든 특약 및 주요 조항을 분석하세요.

반드시 아래 JSON 형식으로만 출력하세요. 다른 텍스트는 포함하지 마세요.

{{
  "overall_risk_score": 0~100 사이 정수 (위험도 종합 점수),
  "risk_summary": {{
    "Risk": 위험 조항 개수,
    "Caution": 주의 조항 개수,
    "Safety": 안전 조항 개수
  }},
  "total_clauses": 분석된 전체 조항 개수,
  "clauses": [
    {{
      "text": "조항 원문 텍스트",
      "risk_level": "위험" | "주의" | "안전",
      "category": "조항 유형 (예: 임차인에게 불리한 조항, 보증금 관련, 관리 규정 등)",
      "analysis": "이 조항이 위험/주의/안전한 이유를 2~3문장으로 설명",
      "related_law": "관련 법률 조항 (예: 주택임대차보호법 제3조, 없으면 빈 문자열)",
      "score": 0~100 사이 정수 (해당 조항 위험 점수)
    }}
  ]
}}

판단 기준:
- 위험: 임차인에게 일방적으로 불리하거나 법령 위반 소지가 있는 조항 (score 70~100)
- 주의: 분쟁 가능성이 있거나 주의가 필요한 조항 (score 40~69)
- 안전: 일반적이고 공정한 조항 (score 0~39)

JSON만 출력하고 다른 텍스트는 포함하지 마세요.""",
)
