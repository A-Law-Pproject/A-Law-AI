from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 설정 - 모든 값은 .env 파일에서 로드"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # OpenAI API
    OPENAI_API_KEY: str
    MODEL_NAME: str
    EMBEDDING_MODEL: str

    # Upstage API (OCR용)
    UPSTAGE_API_KEY: str = ""

    # Qdrant Vector Database
    QDRANT_URL: str
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str

    # RAG Settings
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    TOP_K_DOCUMENTS: int

    # Legal Documents Path
    LEGAL_DOCS_PATH: str
    
    # RabbitMQ (Spring Boot 인프라 연결)
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"

    # Redis (Spring Boot 인프라 연결)
    REDIS_URL: str = "redis://localhost:6379/0"

    # Analysis Settings
    ANALYSIS_TIMEOUT: int = 60  # GPT-4o 호출 타임아웃 (초)
    ANALYSIS_QUEUE: str = "contract.analysis.queue"
    RESULT_EXCHANGE: str = "contract.analysis.result"

    # PostgreSQL Database (Spring Boot와 공유)
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "alawdb"
    DB_USER: str = "alawuser"
    DB_PASSWORD: str = "alaw"

    # AWS S3 (Spring Boot와 공유)
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "ap-northeast-2"
    AWS_S3_BUCKET: str = "alaw-contracts"


settings = Settings()
