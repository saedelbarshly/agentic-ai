from typing import Any

from pydantic import BaseModel

from app.core.config import Settings


def make_settings(**overrides: Any) -> Settings:
    """Settings for tests. Ignores the local .env so results don't depend on it."""
    defaults: dict[str, Any] = {"groq_api_key": "test-key", "groq_model": "default-model"}
    return Settings(_env_file=None, **(defaults | overrides))


class FakeProvider:
    """Returns queued responses in order and records every call.

    Queue a BaseModel to return it, or an Exception to raise it.
    """

    def __init__(self, *responses: BaseModel | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        response_model: type[BaseModel],
        model: str | None = None,
    ) -> BaseModel:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "payload": payload,
                "response_model": response_model,
                "model": model,
            }
        )
        if not self.responses:
            raise AssertionError("FakeProvider was called more times than expected.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, response_model), (
            f"Queued {type(response).__name__}, but the call expected {response_model.__name__}."
        )
        return response

    @property
    def call_count(self) -> int:
        return len(self.calls)
