from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
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

    # Optional per-step model overrides. Empty or unset means use groq_model.
    classifier_model: str | None = None
    account_access_model: str | None = None
    troubleshooting_model: str | None = None
    how_to_model: str | None = None

    router_confidence_threshold: float = Field(default=0.7, ge=0, le=1)
    max_message_chars: int = Field(default=5_000, gt=0)

    app_env: str = "development"
    log_level: str = "INFO"
    request_timeout_seconds: float = Field(default=45.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_backoff_seconds: float = Field(default=0.5, ge=0)

    @field_validator(
        "classifier_model",
        "account_access_model",
        "troubleshooting_model",
        "how_to_model",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def model_for(self, step: str) -> str:
        """Model for a step ("classifier" or a route value), falling back to groq_model."""
        return getattr(self, f"{step}_model", None) or self.groq_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
