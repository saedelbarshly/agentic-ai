from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-20b"
    app_env: str = "development"
    log_level: str = "INFO"
    request_timeout_seconds: float = Field(default=45.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_backoff_seconds: float = Field(default=0.5, ge=0)
    max_cv_chars: int = Field(default=50_000, gt=0)
    max_job_description_chars: int = Field(default=30_000, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
