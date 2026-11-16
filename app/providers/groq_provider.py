import json
import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.errors import (
    InvalidModelOutputError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class GroqJSONProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Groq | None = None

    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__.lower(),
                "strict": True,
                "schema": response_model.model_json_schema(),
            },
        }
        for attempt in range(self._settings.max_retries + 1):
            try:
                response = self._get_client().chat.completions.create(
                    model=self._settings.groq_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": json.dumps(payload, ensure_ascii=False),
                        },
                    ],
                    response_format=response_format,
                    temperature=0.2,
                )
                if not response.choices:
                    raise InvalidModelOutputError("The model returned no choices.")
                content = response.choices[0].message.content
                if not content:
                    raise InvalidModelOutputError("The model returned an empty response.")
                try:
                    return response_model.model_validate_json(content)
                except (ValidationError, ValueError) as exc:
                    raise InvalidModelOutputError(
                        "The model response did not match the required schema."
                    ) from exc
            except AuthenticationError as exc:
                raise ProviderAuthenticationError("Groq authentication failed.") from exc
            except RateLimitError as exc:
                if attempt >= self._settings.max_retries:
                    raise ProviderRateLimitError("Groq rate limit was reached.") from exc
                self._wait_before_retry(attempt, exc)
            except (APITimeoutError, APIConnectionError, InternalServerError) as exc:
                if attempt >= self._settings.max_retries:
                    if isinstance(exc, APITimeoutError):
                        raise ProviderTimeoutError("Groq request timed out.") from exc
                    raise ProviderUnavailableError("Groq is temporarily unavailable.") from exc
                self._wait_before_retry(attempt, exc)
            except APIStatusError as exc:
                raise ProviderError("Groq rejected the request.") from exc

    def _wait_before_retry(self, attempt: int, error: Exception) -> None:
        delay = self._retry_after_seconds(error)
        if delay is None:
            delay = self._settings.retry_backoff_seconds * (2**attempt)
        time.sleep(delay)

    def _get_client(self) -> Groq:
        if self._client is None:
            api_key = self._settings.groq_api_key
            if api_key is None or not api_key.get_secret_value():
                raise ProviderConfigurationError("GROQ_API_KEY is not configured.")
            self._client = Groq(
                api_key=api_key.get_secret_value(),
                timeout=self._settings.request_timeout_seconds,
                max_retries=0,
            )
        return self._client

    @staticmethod
    def _retry_after_seconds(error: Exception) -> float | None:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {})
        retry_after = headers.get("retry-after") if headers else None
        if not retry_after:
            return None
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_time = parsedate_to_datetime(retry_after)
                if retry_time.tzinfo is None:
                    retry_time = retry_time.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_time - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                logger.warning("Ignoring an invalid Retry-After header.")
                return None
