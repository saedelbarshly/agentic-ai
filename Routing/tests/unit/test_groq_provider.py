from types import SimpleNamespace

import httpx
import pytest
from groq import APITimeoutError, AuthenticationError, BadRequestError, RateLimitError

from app.core.errors import (
    InvalidModelOutputError,
    ProviderError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderTimeoutError,
)
from app.providers import groq_provider
from app.providers.base import JSONProvider
from app.schemas import Classification, Route
from tests.helpers import FakeProvider, make_settings

CLASSIFICATION = Classification(route=Route.HOW_TO, confidence=0.9, reason="Usage question.")
REQUEST = httpx.Request("POST", "https://api.groq.com")


def completion(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def install_fake_client(monkeypatch, create) -> None:
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(groq_provider, "Groq", lambda **kwargs: client)
    monkeypatch.setattr(groq_provider.time, "sleep", lambda _seconds: None)


def call(provider, **kwargs):
    return provider.complete_json(
        system_prompt="Classify.",
        payload={"message": "How do I export?"},
        response_model=Classification,
        **kwargs,
    )


def test_uses_default_model_and_strict_json_schema(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return completion(CLASSIFICATION.model_dump_json())

    install_fake_client(monkeypatch, create)
    provider = groq_provider.GroqJSONProvider(make_settings())

    assert call(provider) == CLASSIFICATION
    assert captured["model"] == "default-model"
    json_schema = captured["response_format"]["json_schema"]
    assert json_schema["strict"] is True
    assert json_schema["schema"]["additionalProperties"] is False


def test_per_call_model_overrides_default(monkeypatch):
    models = []

    def create(**kwargs):
        models.append(kwargs["model"])
        return completion(CLASSIFICATION.model_dump_json())

    install_fake_client(monkeypatch, create)
    provider = groq_provider.GroqJSONProvider(make_settings())

    call(provider, model="small-model")
    call(provider, model=None)

    assert models == ["small-model", "default-model"]


def test_retries_rate_limit_using_retry_after(monkeypatch):
    attempts, delays = [], []

    def create(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RateLimitError(
                "limited",
                response=httpx.Response(429, headers={"retry-after": "2"}, request=REQUEST),
                body=None,
            )
        return completion(CLASSIFICATION.model_dump_json())

    install_fake_client(monkeypatch, create)
    monkeypatch.setattr(groq_provider.time, "sleep", delays.append)
    provider = groq_provider.GroqJSONProvider(make_settings(max_retries=1))

    assert call(provider) == CLASSIFICATION
    assert len(attempts) == 2
    assert delays == [2.0]


def test_timeout_after_retries_raises_safe_error(monkeypatch):
    attempts = []

    def create(**kwargs):
        attempts.append(kwargs)
        raise APITimeoutError(request=REQUEST)

    install_fake_client(monkeypatch, create)
    provider = groq_provider.GroqJSONProvider(make_settings(max_retries=2))

    with pytest.raises(ProviderTimeoutError):
        call(provider)
    assert len(attempts) == 3  # first try + 2 retries


def test_authentication_error_is_not_retried(monkeypatch):
    attempts = []

    def create(**kwargs):
        attempts.append(kwargs)
        raise AuthenticationError(
            "bad key", response=httpx.Response(401, request=REQUEST), body=None
        )

    install_fake_client(monkeypatch, create)
    provider = groq_provider.GroqJSONProvider(make_settings(max_retries=2))

    with pytest.raises(ProviderAuthenticationError):
        call(provider)
    assert len(attempts) == 1


@pytest.mark.parametrize(
    "content",
    [
        None,
        "not json",
        '{"route": "billing", "confidence": 0.9, "reason": "x"}',  # unknown route
        '{"route": "how_to", "confidence": 1.5, "reason": "x"}',  # out of range
    ],
)
def test_bad_model_output_raises_invalid_output(monkeypatch, content):
    install_fake_client(monkeypatch, lambda **kwargs: completion(content))
    provider = groq_provider.GroqJSONProvider(make_settings())

    with pytest.raises(InvalidModelOutputError):
        call(provider)


def bad_request(code: str, *, wrap_in_error: bool = True) -> BadRequestError:
    detail = {"message": "bad", "code": code}
    return BadRequestError(
        "bad request",
        response=httpx.Response(400, request=REQUEST),
        body={"error": detail} if wrap_in_error else detail,
    )


@pytest.mark.parametrize("wrap_in_error", [True, False])
def test_groq_schema_validation_failure_is_retried(monkeypatch, wrap_in_error):
    attempts = []

    def create(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise bad_request("json_validate_failed", wrap_in_error=wrap_in_error)
        return completion(CLASSIFICATION.model_dump_json())

    install_fake_client(monkeypatch, create)
    provider = groq_provider.GroqJSONProvider(make_settings(max_retries=1))

    assert call(provider) == CLASSIFICATION
    assert len(attempts) == 2


def test_groq_schema_validation_failure_exhausts_into_invalid_output(monkeypatch):
    attempts = []

    def create(**kwargs):
        attempts.append(kwargs)
        raise bad_request("json_validate_failed")

    install_fake_client(monkeypatch, create)
    provider = groq_provider.GroqJSONProvider(make_settings(max_retries=2))

    with pytest.raises(InvalidModelOutputError):
        call(provider)
    assert len(attempts) == 3


def test_other_bad_requests_are_not_retried(monkeypatch):
    attempts = []

    def create(**kwargs):
        attempts.append(kwargs)
        raise bad_request("model_not_found")

    install_fake_client(monkeypatch, create)
    provider = groq_provider.GroqJSONProvider(make_settings(max_retries=2))

    with pytest.raises(ProviderError) as info:
        call(provider)
    assert not isinstance(info.value, InvalidModelOutputError)
    assert len(attempts) == 1


def test_locally_invalid_output_is_retried(monkeypatch):
    outputs = iter(
        [
            '{"route": "how_to", "confidence": 1.5, "reason": "x"}',  # out of range
            CLASSIFICATION.model_dump_json(),
        ]
    )
    install_fake_client(monkeypatch, lambda **kwargs: completion(next(outputs)))
    provider = groq_provider.GroqJSONProvider(make_settings(max_retries=1))

    assert call(provider) == CLASSIFICATION


def test_missing_api_key_raises_configuration_error():
    provider = groq_provider.GroqJSONProvider(make_settings(groq_api_key=None))

    with pytest.raises(ProviderConfigurationError):
        call(provider)


# ------------------------------------------------------------ Task 2.2


def test_fake_provider_returns_in_order_and_records_calls():
    fake: JSONProvider = FakeProvider(CLASSIFICATION, ProviderTimeoutError())

    assert call(fake, model="m1") == CLASSIFICATION
    with pytest.raises(ProviderTimeoutError):
        call(fake)

    assert fake.call_count == 2
    assert fake.calls[0]["model"] == "m1"
    assert fake.calls[0]["payload"] == {"message": "How do I export?"}


def test_fake_provider_fails_on_unexpected_extra_call():
    fake = FakeProvider()

    with pytest.raises(AssertionError, match="more times than expected"):
        call(fake)


def test_every_retry_is_logged_with_its_reason(monkeypatch, caplog):
    attempts = []

    def create(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RateLimitError(
                "limited",
                response=httpx.Response(429, headers={"retry-after": "3"}, request=REQUEST),
                body=None,
            )
        if len(attempts) == 2:
            raise bad_request("json_validate_failed")
        return completion(CLASSIFICATION.model_dump_json())

    install_fake_client(monkeypatch, create)
    provider = groq_provider.GroqJSONProvider(make_settings(max_retries=2))

    with caplog.at_level("WARNING", logger="app.providers.groq_provider"):
        call(provider)

    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("provider_retry")]
    assert lines == [
        "provider_retry reason=rate_limit attempt=1 delay_s=3.00",
        "provider_retry reason=schema_validation_failed attempt=2 delay_s=1.00",
    ]
