"""
계약서 분석 서비스
"""
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
from app.schemas.contract import (
    AnalysisResult,
    ClauseAnalysis,
    TermExplanation
)
import logging

logger = logging.getLogger(__name__)


class ContractAnalysisService:
    """계약서 분석 서비스"""

    def __init__(self):
        # 모델 초기화
        self.llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            api_key=settings.OPENAI_API_KEY,
            temperature=0
        )

        # 구조화된 출력을 위한 LLM 설정
        self.structured_llm = self.llm.with_structured_output(AnalysisResult)

        # 프롬프트 템플릿 설정
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 유능한 법률 전문가 AI입니다. 계약서를 분석하여 요약과 위험 요소를 추출해 주세요."),
            ("human", "{text}"),
        ])

        # 체인 구성
        self.chain = self.prompt | self.structured_llm

        self._ready = True
        logger.info("ContractAnalysisService initialized")

    def is_ready(self) -> bool:
        """서비스 준비 상태 확인"""
        return self._ready

    async def analyze_contract(self, text: str, contract_id: str) -> AnalysisResult:
        """
        계약서 전체 분석

        Args:
            text: 계약서 텍스트
            contract_id: 계약서 ID

        Returns:
            AnalysisResult
        """
        try:
            result = await self.chain.ainvoke({"text": text})
            result.contract_id = contract_id
            return result
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            # 기본 응답 반환
            return AnalysisResult(
                contract_id=contract_id,
                total_clauses=0,
                risk_summary={"Risk": 0, "Caution": 0, "Safety": 0},
                clauses=[],
                overall_risk_score=0.0,
                recommendations=["분석 중 오류가 발생했습니다. 전문가 상담을 권장합니다."]
            )

    def detect_fraud_patterns(self, text: str) -> List[Dict]:
        """
        사기 패턴 탐지

        Args:
            text: 계약서 텍스트

        Returns:
            탐지된 패턴 리스트
        """
        logger.info("Detecting fraud patterns")

        # 사기 패턴 키워드
        fraud_patterns = []

        # 일방적 해지 조항
        if "언제든지 해지" in text or "임의로 해지" in text:
            fraud_patterns.append({
                "pattern": "일방적 해지권",
                "severity": "high",
                "description": "임대인이 일방적으로 계약을 해지할 수 있는 조항이 포함되어 있습니다."
            })

        # 과도한 위약금
        if "위약금" in text and ("50%" in text or "전액" in text):
            fraud_patterns.append({
                "pattern": "과도한 위약금",
                "severity": "high",
                "description": "과도한 위약금 조항이 포함되어 있습니다."
            })

        # 보증금 반환 미명시
        if "보증금" in text and "반환" not in text:
            fraud_patterns.append({
                "pattern": "보증금 반환 미명시",
                "severity": "medium",
                "description": "보증금 반환 조건이 명시되어 있지 않습니다."
            })

        return fraud_patterns

    def find_missing_clauses(self, text: str) -> List[Dict]:
        """
        필수 조항 누락 확인

        Args:
            text: 계약서 텍스트

        Returns:
            누락된 조항 리스트
        """
        logger.info("Finding missing clauses")

        missing = []

        # 확정일자 관련
        if "확정일자" not in text:
            missing.append({
                "clause_name": "확정일자 안내",
                "importance": "critical",
                "description": "확정일자 취득 안내가 누락되었습니다.",
                "legal_basis": "주택임대차보호법 제3조의6"
            })

        # 수리 책임
        if "수리" not in text and "수선" not in text:
            missing.append({
                "clause_name": "수리 책임",
                "importance": "important",
                "description": "수리 책임에 관한 조항이 누락되었습니다."
            })

        # 중도 해지
        if "중도 해지" not in text and "중도해지" not in text:
            missing.append({
                "clause_name": "중도 해지 조건",
                "importance": "important",
                "description": "중도 해지 시 조건이 명시되어 있지 않습니다."
            })

        # 계약 갱신
        if "갱신" not in text:
            missing.append({
                "clause_name": "계약 갱신",
                "importance": "recommended",
                "description": "계약 갱신 관련 조항이 없습니다.",
                "legal_basis": "주택임대차보호법 제6조의3"
            })

        return missing

    def check_illegal_clauses(self, text: str) -> List[Dict]:
        """
        불법 조항 검사

        Args:
            text: 계약서 텍스트

        Returns:
            불법 조항 리스트
        """
        logger.info("Checking illegal clauses")

        illegal = []

        # 차임 증액 제한 위반 (5% 초과)
        if "10%" in text and ("인상" in text or "증액" in text):
            illegal.append({
                "clause_text": "차임 10% 인상 조항",
                "violation": "주택임대차보호법 차임증액 제한 위반",
                "legal_reference": "주택임대차보호법 제7조 (5% 제한)",
                "recommendation": "차임 증액률을 5% 이하로 수정하세요."
            })

        # 임차인 권리 포기 강요
        if "권리를 포기" in text or "이의를 제기할 수 없" in text:
            illegal.append({
                "clause_text": "임차인 권리 포기 조항",
                "violation": "강행규정 위반",
                "legal_reference": "주택임대차보호법 제10조",
                "recommendation": "임차인의 법적 권리를 제한하는 조항은 무효입니다."
            })

        # 2년 미만 계약
        if "1년" in text and "기간" in text:
            illegal.append({
                "clause_text": "1년 계약 기간",
                "violation": "최소 계약 기간 미달",
                "legal_reference": "주택임대차보호법 제4조 (2년 보장)",
                "recommendation": "임차인이 원하면 2년까지 거주할 수 있습니다."
            })

        return illegal

    def explain_legal_term(
        self,
        term: str,
        context: str = "",
        surrounding_text: str = ""
    ) -> TermExplanation:
        """
        법률 용어 해설

        Args:
            term: 용어
            context: 문맥
            surrounding_text: 주변 텍스트

        Returns:
            TermExplanation
        """
        logger.info(f"Explaining term: {term}")

        # 주요 법률 용어 사전
        term_dict = {
            "확정일자": TermExplanation(
                term="확정일자",
                simple_explanation="보증금을 우선적으로 돌려받을 수 있는 권리를 증명하는 도장입니다.",
                legal_definition="주택임대차보호법상 대항력과 우선변제권을 갖추기 위한 요건으로, 임대차 계약서에 관할 관청이 날인하는 것을 말합니다.",
                examples=[
                    "전입신고 + 확정일자 → 우선변제권 획득",
                    "주민센터나 등기소에서 무료로 받을 수 있습니다",
                    "확정일자가 빠를수록 우선순위가 높습니다"
                ]
            ),
            "대항력": TermExplanation(
                term="대항력",
                simple_explanation="집주인이 바뀌어도 계속 살 수 있는 권리입니다.",
                legal_definition="임차인이 제3자에 대하여 임대차의 효력을 주장할 수 있는 권리로, 전입신고와 점유를 통해 취득합니다.",
                examples=[
                    "전입신고 + 실제 거주 = 대항력 발생",
                    "집이 경매로 넘어가도 계약 기간 동안 거주 가능",
                    "새 집주인에게 보증금 반환 청구 가능"
                ]
            ),
            "우선변제권": TermExplanation(
                term="우선변제권",
                simple_explanation="다른 채권자보다 먼저 보증금을 돌려받을 수 있는 권리입니다.",
                legal_definition="경매 시 임차인이 후순위 권리자보다 우선하여 보증금을 변제받을 수 있는 권리입니다.",
                examples=[
                    "대항력 + 확정일자 = 우선변제권",
                    "경매 낙찰가에서 보증금 우선 회수",
                    "선순위 근저당이 있으면 후순위가 됨"
                ]
            )
        }

        # 사전에 있으면 반환
        if term in term_dict:
            return term_dict[term]

        # 없으면 기본 응답
        return TermExplanation(
            term=term,
            simple_explanation=f"{term}에 대한 설명입니다.",
            legal_definition="법률적 정의를 찾을 수 없습니다.",
            examples=["예시 정보가 없습니다."]
        )
