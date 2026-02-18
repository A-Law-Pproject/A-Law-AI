# ============================================================
# [LEGACY] 이 파일은 더 이상 사용되지 않습니다.
# 초기 HuggingFace 기반 프로토타입으로, app/rag/chain/chain.py로 대체되었습니다.
# TODO: 안정화 후 삭제 예정
# ============================================================
"""
# services/contract_analyzer.py
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.cache import InMemoryCache
import langchain

# 캐시 활성화
langchain.llm_cache = InMemoryCache()

class ContractAnalyzer:
    def __init__(self):
        # 모델 로드 (서버 시작 시 한 번만)
        self.llm = self._load_model()
        self.fraud_chain = self._create_fraud_chain()
        self.missing_chain = self._create_missing_clause_chain()
        self.illegal_chain = self._create_illegal_clause_chain()
        self.term_chain = self._create_term_explanation_chain()
        
    def _load_model(self):
        """학습된 모델 로드"""
        # 예시: Hugging Face 모델
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        
        model_name = "your-trained-model-path"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # GPU 자동 할당
            load_in_8bit=True   # 메모리 최적화
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,    # 일관된 결과
            do_sample=False     # 결정론적
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    
    def _create_fraud_chain(self):
        """사기 패턴 탐지 체인"""
        prompt = PromptTemplate(
            input_variables=["contract_text"],
            template="""
            다음 계약서에서 사기 위험이 있는 조항을 찾아주세요.
            
            계약서:
            {contract_text}
            
            다음 형식으로 JSON 배열로 응답해주세요:
            [
                {{
                    "clause": "의심스러운 조항 전문",
                    "risk_type": "위험 유형",
                    "explanation": "왜 위험한지 설명",
                    "severity": "high|medium|low"
                }}
            ]
            """
        )
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def _create_missing_clause_chain(self):
        """누락 조항 탐지 체인"""
        prompt = PromptTemplate(
            input_variables=["contract_text"],
            template="""
            다음 계약서에서 필수적으로 포함되어야 하나 누락된 조항을 찾아주세요.
            
            일반적으로 포함되어야 할 조항:
            - 계약 기간 및 해지 조건
            - 손해배상 및 책임 범위
            - 분쟁 해결 방법
            - 비밀유지 조항
            - 불가항력 조항
            
            계약서:
            {contract_text}
            
            JSON 배열로 응답:
            [
                {{
                    "missing_clause": "누락된 조항명",
                    "importance": "high|medium|low",
                    "recommendation": "포함해야 하는 이유"
                }}
            ]
            """
        )
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def _create_illegal_clause_chain(self):
        """위법 조항 탐지 체인"""
        prompt = PromptTemplate(
            input_variables=["contract_text"],
            template="""
            다음 계약서에서 한국 법률에 위반되는 조항을 찾아주세요.
            
            계약서:
            {contract_text}
            
            JSON 배열로 응답:
            [
                {{
                    "clause": "위법 조항",
                    "violated_law": "위반된 법률 (예: 근로기준법 제50조)",
                    "explanation": "왜 위법인지",
                    "severity": "critical|high|medium"
                }}
            ]
            """
        )
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def _create_term_explanation_chain(self):
        """용어 해설 체인 - 경량화"""
        prompt = PromptTemplate(
            input_variables=["term", "context", "surrounding_text"],
            template="""
            법률 용어를 쉽게 설명해주세요.
            
            용어: {term}
            주변 문맥: {surrounding_text}
            계약서 컨텍스트: {context}
            
            JSON 형식으로 응답:
            {{
                "simple_explanation": "초등학생도 이해할 수 있는 쉬운 설명",
                "legal_definition": "정확한 법률적 정의",
                "examples": ["예시1", "예시2"]
            }}
            """
        )
        return LLMChain(llm=self.llm, prompt=prompt)
    
    # 실제 분석 메서드들
    def detect_fraud_patterns(self, contract_text: str):
        """사기 패턴 탐지"""
        result = self.fraud_chain.run(contract_text=contract_text[:4000])  # 토큰 제한
        return self._parse_json_response(result)
    
    def find_missing_clauses(self, contract_text: str):
        """누락 조항 찾기"""
        result = self.missing_chain.run(contract_text=contract_text[:4000])
        return self._parse_json_response(result)
    
    def check_illegal_clauses(self, contract_text: str):
        """위법 조항 체크"""
        result = self.illegal_chain.run(contract_text=contract_text[:4000])
        return self._parse_json_response(result)
    
    def explain_legal_term(self, term: str, context: str, surrounding_text: str):
        """용어 해설 - 빠른 응답"""
        result = self.term_chain.run(
            term=term,
            context=context[:1000],  # 컨텍스트 제한
            surrounding_text=surrounding_text[:500]
        )
        return self._parse_json_response(result)
    
    def _parse_json_response(self, response: str):
        """JSON 파싱"""
        import json
        import re
        
        # JSON 추출
        json_match = re.search(r'\[.*\]|\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return []
    
    def is_ready(self):
        """모델 준비 상태"""
        return self.llm is not None
"""