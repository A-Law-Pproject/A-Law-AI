from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 설정 - 모든 값은 .env 파일에서 로드"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # LANGSMITH_* 등 미정의 환경변수 무시
    )

    # OpenAI API
    OPENAI_API_KEY: str
    MODEL_NAME: str
    EMBEDDING_MODEL: str = "nlpai-lab/KURE-v1"

    # Upstage API (OCR용)
    UPSTAGE_API_KEY: str = ""

    # Naver Clova OCR (대체 OCR 엔진)
    CLOVA_OCR_API_URL: str = ""         # https://{apigw}.ntruss.com/custom/v1/{domain_id}/{invoke_key}/general
    CLOVA_OCR_SECRET_KEY: str = ""      # NCloud Console > CLOVA OCR > Secret Key

    # Pinecone
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX: str = "alaw-legal"

    # RAG Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_DOCUMENTS: int = 5
    ENABLE_HYBRID_SEARCH: bool = True
    HYBRID_RRF_K: int = 60
    HYBRID_DENSE_CANDIDATE_MULTIPLIER: int = 4
    HYBRID_LEXICAL_CANDIDATE_MULTIPLIER: int = 3

    # Legal Documents Path
    LEGAL_DOCS_PATH: str = "data/raw"

    # Live law API + MCP bridge
    LAW_API_OC: str = ""
    LAW_API_BASE_URL: str = "https://www.law.go.kr/DRF"
    LAW_API_TIMEOUT_SECONDS: float = 15.0
    LAW_MCP_ENABLED: bool = True
    LAW_MCP_MAX_RESULTS: int = 3
    
    # MongoDB (OCR 결과 저장소 - Atlas)
    MONGODB_URI: str = "mongodb+srv://mongoadmin:alaw@cluster0.t0cklix.mongodb.net/?appName=Cluster0"
    MONGODB_DB: str = "alaw"
    MONGODB_OCR_COLLECTION: str = "ocr_results"

    # RabbitMQ (Spring Boot 인프라 연결)
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"

    # Redis (Spring Boot 인프라 연결)
    REDIS_URL: str = "redis://localhost:6379/0"

    # Analysis Settings
    ANALYSIS_TIMEOUT: int = 600  # GPT-4o 병렬 호출(요약 + 위험분석) 타임아웃 (초)
    ANALYSIS_EXCHANGE: str = "contract-analysis-ex"
    ANALYSIS_QUEUE: str = "contract-analysis-queue"
    ANALYSIS_ROUTING_KEY: str = "contract.analyze"
    RESULT_EXCHANGE: str = "contract.analysis.result"
    RESULT_QUEUE: str = "contract-analysis-result-queue"
    RESULT_ROUTING_KEY: str = "ai.result"

    # Voice RabbitMQ (Spring Boot 연동)
    VOICE_ANALYSIS_EXCHANGE: str = "voice-analysis-ex"
    VOICE_ANALYSIS_QUEUE: str = "voice-record-queue"
    VOICE_ANALYSIS_ROUTING_KEY: str = "voice.record"
    VOICE_RESULT_EXCHANGE: str = "voice.analysis.result"
    VOICE_RESULT_QUEUE: str = "voice-result-queue"
    VOICE_RESULT_ROUTING_KEY: str = "voice.result"
    VOICE_ANALYSIS_TIMEOUT: int = 300  # 음성 팩트체크 타임아웃 (초)

    # AWS S3 (Spring Boot와 공유)
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "ap-northeast-2"
    AWS_S3_BUCKET: str = "alaw-image-bucket"

    # 음성 증거 분석 설정
    WHISPER_MODEL: str = "whisper-1"  # OpenAI Whisper API 모델명
    EVIDENCE_SIMILARITY_THRESHOLD: float = 0.75  # 조항-발화 연결 유사도 임계값
    MONGODB_VOICE_EVIDENCE_COLLECTION: str = "voice_evidence_meta"  # 원본성 메타데이터 컬렉션
    MONGODB_EVIDENCE_GRAPH_COLLECTION: str = "evidence_graphs"  # 증거 그래프 저장 컬렉션
    MONGODB_VOICE_ANALYSIS_COLLECTION: str = "voice_analysis_results"  # standalone 분석 결과
    MONGODB_VOICE_FACT_CHECK_COLLECTION: str = "voice_fact_check_results"  # 팩트체크 결과

    # PII 마스킹 설정
    ENABLE_MASKING: bool = True   # 마스킹 기능 ON/OFF 토글 (.env에서 ENABLE_MASKING=false로 비활성화)
    MASKING_VERSION: str = "1.1"  # 마스킹 엔진 버전 (MongoDB 메타데이터에 기록)

    # 표 텍스트 후처리 설정
    ENABLE_LLM_TABLE_FIX: bool = False  # LLM 기반 표 줄바꿈 오류 보정 (비용 발생, .env에서 활성화)

    # LLMOps 설정
    ENABLE_LLMOPS_METRICS: bool = True
    LLMOPS_METRIC_SAMPLE_RATE: float = 1.0
    LLMOPS_MIN_LEGAL_CITATION_RATE: float = 0.60
    LLMOPS_MAX_REJECTION_RATE: float = 0.10
    LLMOPS_MAX_EMPTY_CONTEXT_RATE: float = 0.20
    LLMOPS_MIN_AVG_RERANKER_SCORE: float = -3.0
    MLFLOW_TRACKING_URI: str = "file:./mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "a-law-llmops"

    # 일반 설정
    DEBUG: bool = False

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production", "false", "0", "no", "off"}:
                return False
            if normalized in {"debug", "dev", "development", "true", "1", "yes", "on"}:
                return True
        return value

    @field_validator("LLMOPS_METRIC_SAMPLE_RATE")
    @classmethod
    def validate_llmops_sample_rate(cls, value: float) -> float:
        return min(max(float(value), 0.0), 1.0)


settings = Settings()
