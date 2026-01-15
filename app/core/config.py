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

    # General Settings
    DEBUG: bool


settings = Settings()
