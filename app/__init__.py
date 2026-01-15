import logging
import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO)

LOGGER = logging.getLogger(__name__)


class Settings(BaseModel):
    # 프로젝트 설정
    PROJECT_NAME: str = "A-LAW"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    DESCRIPTION: str = "계약서 검증 서비스 AI 서버"
    APP_BUNDLE_ID: str = os.getenv("APP_BUNDLE_ID", "com.jaesuneo.WatchOut")
    APP_CUSTOM_SCHEME: str = os.getenv("APP_CUSTOM_SCHEME", "watchout")

    # 서버 설정
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8001"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    
    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    SQL_ECHO: bool = os.getenv("SQL_ECHO", "false").lower() == "true"
    

    
    # CORS 설정
    BACKEND_CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8001",
        "http://localhost:8001",
        # Production
    ]

    class Config:
        env_file = ".env"

class Development(Settings):
    # 개발 환경에 맞는 추가 설정
    debug: bool = True


settings = Settings()