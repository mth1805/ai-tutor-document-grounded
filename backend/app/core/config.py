from typing import List, Union
from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]
ROOT_ENV_FILE = PROJECT_ROOT / ".env"
BACKEND_ENV_FILE = BACKEND_ROOT / ".env"
ENV_FILE = ROOT_ENV_FILE if ROOT_ENV_FILE.is_file() else BACKEND_ENV_FILE


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Tutor Assistant API"
    API_V1_STR: str = "/api/v1"
    # Fail closed when deploy-time environment configuration is omitted.
    ENVIRONMENT: str = "production"

    # CORS configuration
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, str) and v.startswith("["):
            return json.loads(v)
        elif isinstance(v, list):
            return v
        return []

    # Supabase credentials (server-side)
    SUPABASE_URL: str | None = None
    SUPABASE_ANON_KEY: str | None = None
    SUPABASE_SERVICE_ROLE_KEY: str | None = None
    SUPABASE_JWT_SECRET: str | None = None

    # PostgreSQL Database URL (Supabase PostgreSQL / asyncpg)
    DATABASE_URL: str | None = None
    # Optional integration-test database; server-side secret, never client exposed.
    RLS_TEST_DATABASE_URL: str | None = None

    # Storage settings
    STORAGE_BUCKET_NAME: str = "documents"
    MAX_FILE_SIZE_BYTES: int = 26214400  # 25 MB

    # Optional Phase 5 system tools. Executables may be absolute paths or PATH names.
    TESSERACT_CMD: str | None = None
    TESSERACT_LANGUAGES: str = "vie+eng"
    LIBREOFFICE_CMD: str = "soffice"
    LIBREOFFICE_TIMEOUT_SECONDS: int = 60

    # Phase 6: BGE-M3 Embedding and pgvector configuration
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 16
    EMBEDDING_DEVICE: str = "auto"  # auto | cpu | cuda
    EMBEDDING_MODEL_CACHE_DIR: Path | None = None
    EMBEDDING_NORMALIZE: bool = True
    AUTO_EMBED_AFTER_INGESTION: bool = True
    PREWARM_MODELS: bool = False


    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
