from fastapi import APIRouter
from fastapi.testclient import TestClient

from app.core.errors import ProviderRateLimitError, ProviderTimeoutError
from app.main import app

_test_router = APIRouter()


@_test_router.get("/_test/rate-limit")
def _raise_rate_limit():
    raise ProviderRateLimitError()


@_test_router.get("/_test/timeout")
def _raise_timeout():
    raise ProviderTimeoutError()


app.include_router(_test_router)
client = TestClient(app)


def test_health_returns_ok_with_request_id():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"]


def test_provider_errors_map_to_safe_responses():
    rate_limited = client.get("/_test/rate-limit")
    timed_out = client.get("/_test/timeout")

    assert rate_limited.status_code == 429
    assert rate_limited.json()["detail"] == "Support provider rate limit reached."
    assert timed_out.status_code == 504
    assert timed_out.json()["request_id"]
