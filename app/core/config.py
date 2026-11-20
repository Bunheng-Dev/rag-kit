from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    APP_NAME: str = "rag-kit"
    APP_VERSION: str = "1.0.0"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = Field(default=8000, validation_alias=AliasChoices("APP_PORT", "PORT"))
    DEBUG: bool = False

    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-2.5-flash"
    GEMINI_API_KEY: str = ""

    EMBEDDING_PROVIDER: str = "gemini"
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIM: int = 768
    RESPONSE_LANGUAGE: str = "auto"

    VECTOR_DB: str = "qdrant"
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "rag_documents"
    QDRANT_TIMEOUT: float = 120.0

    DATASOURCE: str = "mysql"
    MYSQL_HOST: str = "mysql"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "rag_kit"
    MYSQL_CONNECT_TIMEOUT: int = 5
    MYSQL_DOCUMENT_TABLE: str = "documents"
    MYSQL_DOCUMENT_ID_COLUMN: str = "id"
    MYSQL_DOCUMENT_TEXT_COLUMN: str = "text"
    MYSQL_DOCUMENT_METADATA_COLUMNS: str = ""
    SAMPLE_DATA_PATH: str = "data/sample.json"

    CHAT_INTERNAL_KEY: str = ""
    ENABLE_CARDS: bool = False
    RETRIEVAL_TOP_K: int = 6
    CARD_MAX_ITEMS: int = 5
    PROMPT_TEMPLATE_PATH: str = "prompts/system.txt"
    PROMPT_TEMPLATE: str = ""
    MEDIA_BASE_URL: str = Field(
        default="",
        validation_alias=AliasChoices("MEDIA_BASE_URL", "UPLOAD_BASE_URL"),
    )
    CARD_URL_TEMPLATE: str = "{base_url}/items/{slug}?id={id}"
    CORS_ALLOW_ORIGINS: str = "*"

    @field_validator(
        "LLM_PROVIDER",
        "EMBEDDING_PROVIDER",
        "VECTOR_DB",
        "DATASOURCE",
        mode="before",
    )
    @classmethod
    def normalize_provider_name(cls, value: str) -> str:
        return str(value).strip().lower().replace("-", "_")

    @field_validator("EMBEDDING_DIM")
    @classmethod
    def validate_embedding_dimension(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("EMBEDDING_DIM must be greater than zero")
        return value

    @property
    def prompt_template_file(self) -> Path:
        path = Path(self.PROMPT_TEMPLATE_PATH)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def cors_origins(self) -> list[str]:
        if self.CORS_ALLOW_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ALLOW_ORIGINS.split(",") if origin.strip()]


settings = Settings()
