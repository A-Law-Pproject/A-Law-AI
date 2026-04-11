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
    EMBEDDING_MODEL: str

    # Upstage API (OCR용)
    UPSTAGE_API_KEY: str = ""

    # Vector DB 선택: "qdrant" (개발) | "pinecone" (배포)
    VECTOR_DB: str = "qdrant"

    # Qdrant (로컬/개발)
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "legal_documents"

    # Pinecone (배포)
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX: str = "alaw-legal"

    # RAG Settings
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    TOP_K_DOCUMENTS: int

    # Legal Documents Path
    LEGAL_DOCS_PATH: str
    
    # MongoDB (OCR 결과 저장소 - Atlas)
    MONGODB_URI: str = "mongodb+srv://mongoadmin:alaw@cluster0.t0cklix.mongodb.net/?appName=Cluster0"
    MONGODB_DB: str = "alaw"
    MONGODB_OCR_COLLECTION: str = "ocr_results"

    # RabbitMQ (Spring Boot 인프라 연결)
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"

    # Redis (Spring Boot 인프라 연결)
    REDIS_URL: str = "redis://localhost:6379/0"

    # Analysis Settings
    ANALYSIS_TIMEOUT: int = 60  # GPT-4o 호출 타임아웃 (초)
    ANALYSIS_EXCHANGE: str = "contract-analysis-ex"
    ANALYSIS_QUEUE: str = "contract-analysis-queue"
    ANALYSIS_ROUTING_KEY: str = "contract.analyze"
    RESULT_EXCHANGE: str = "contract.analysis.result"
    RESULT_QUEUE: str = "contract-analysis-result-queue"
    RESULT_ROUTING_KEY: str = "ai.result"

    # AWS S3 (Spring Boot와 공유)
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "ap-northeast-2"
    AWS_S3_BUCKET: str = "alaw-image-bucket"

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
