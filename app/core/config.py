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

    # Pinecone
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX: str = "alaw-legal"

    # RAG Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_DOCUMENTS: int = 5

    # Legal Documents Path
    LEGAL_DOCS_PATH: str = "data/raw"
    
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


settings = Settings()
