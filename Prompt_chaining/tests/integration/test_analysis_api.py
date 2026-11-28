import asyncio

import httpx

from app.api.routes.analysis import get_provider
from app.core.errors import ProviderConfigurationError, ProviderRateLimitError
from app.main import app
from tests.helpers import (
    CV_TEXT,
    JOB_TEXT,
    FakeProvider,
    analysis_with,
    gap_result,
    profile_with_python,
    ready_result,
    requirement,
)

async def make_request(method: str, url: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        return await client.request(method, url, **kwargs)


def test_full_analysis_api_workflow_with_mocked_provider():
    provider = FakeProvider(
        profile_with_python(),
        analysis_with(requirement()),
        ready_result(),
    )
    app.dependency_overrides[get_provider] = lambda: provider
    try:
        response = asyncio.run(
            make_request(
                "POST",
                "/api/v1/analyze",
                json={"cv_text": CV_TEXT, "job_description": JOB_TEXT},
            )
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "ready_to_apply"
    assert response.json()["route"] == "application_ready"
    assert response.headers["x-request-id"]


def test_invalid_request_returns_422():
    response = asyncio.run(
        make_request(
            "POST",
            "/api/v1/analyze",
            json={"cv_text": "   ", "job_description": JOB_TEXT},
        )
    )

    assert response.status_code == 422
    assert "   " not in response.text


def test_provider_rate_limit_returns_safe_error():
    provider = FakeProvider(profile_with_python(), ProviderRateLimitError())
    app.dependency_overrides[get_provider] = lambda: provider
    try:
        response = asyncio.run(
            make_request(
                "POST",
                "/api/v1/analyze",
                json={"cv_text": CV_TEXT, "job_description": JOB_TEXT},
            )
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 429
    assert response.json()["detail"] == "Analysis provider rate limit reached."
    assert "Groq" not in response.text


def test_health_endpoint():
    response = asyncio.run(make_request("GET", "/health"))

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_provider_configuration_failure_returns_safe_error():
    provider = FakeProvider(ProviderConfigurationError())
    app.dependency_overrides[get_provider] = lambda: provider
    try:
        response = asyncio.run(
            make_request(
                "POST",
                "/api/v1/analyze",
                json={"cv_text": CV_TEXT, "job_description": JOB_TEXT},
            )
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["detail"] == "Analysis provider is not configured."
    assert "GROQ_API_KEY" not in response.text
