from types import SimpleNamespace

import httpx
from groq import RateLimitError

from app.core.config import Settings
from app.providers import groq_provider
from app.schemas import CandidateProfile
from tests.helpers import profile_with_python


def test_provider_uses_configured_model_and_strict_json_schema(monkeypatch):
    profile = profile_with_python()
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=profile.model_dump_json())
                )
            ]
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    monkeypatch.setattr(groq_provider, "Groq", lambda **kwargs: fake_client)
    provider = groq_provider.GroqJSONProvider(
        Settings(
            _env_file=None,
            groq_api_key="test-key",
            groq_model="configured-model",
        )
    )

    result = provider.complete_json(
        system_prompt="Extract facts.",
        payload={"cv_text": "Candidate CV."},
        response_model=CandidateProfile,
    )

    assert result == profile
    assert captured["model"] == "configured-model"
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["response_format"]["json_schema"]["schema"]["additionalProperties"] is False


def test_provider_retries_rate_limit_using_retry_after(monkeypatch):
    profile = profile_with_python()
    attempts = []
    delays = []

    def create(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RateLimitError(
                "limited",
                response=httpx.Response(
                    429,
                    headers={"retry-after": "2"},
                    request=httpx.Request("POST", "https://api.groq.com"),
                ),
                body=None,
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=profile.model_dump_json())
                )
            ]
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    monkeypatch.setattr(groq_provider, "Groq", lambda **kwargs: fake_client)
    monkeypatch.setattr(groq_provider.time, "sleep", delays.append)
    provider = groq_provider.GroqJSONProvider(
        Settings(
            _env_file=None,
            groq_api_key="test-key",
            max_retries=1,
        )
    )

    result = provider.complete_json(
        system_prompt="Extract facts.",
        payload={"cv_text": "Candidate CV."},
        response_model=CandidateProfile,
    )

    assert result == profile
    assert len(attempts) == 2
    assert delays == [2.0]
