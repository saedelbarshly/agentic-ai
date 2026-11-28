import pytest
from fastapi.testclient import TestClient

from app.api.routes.support import get_provider
from app.core.config import get_settings
from app.core.errors import ProviderConfigurationError, ProviderTimeoutError
from app.main import app
from app.schemas import Classification, LLMAnswer, Route
from tests.helpers import FakeProvider, make_settings

URL = "/api/v1/support/route"
ANSWER = LLMAnswer(summary="Try a smaller file.", steps=["Retry", "Update"], follow_up="Send logs.")


@pytest.fixture
def use_provider():
    """Install a FakeProvider (and test settings) for one test."""

    def install(*responses) -> FakeProvider:
        provider = FakeProvider(*responses)
        app.dependency_overrides[get_provider] = lambda: provider
        # A no-arg lambda: FastAPI would treat make_settings(**overrides) as a query param.
        settings = make_settings()
        app.dependency_overrides[get_settings] = lambda: settings
        return provider

    yield install
    app.dependency_overrides.clear()


client = TestClient(app)


def test_llm_route_returns_answer(use_provider):
    provider = use_provider(
        Classification(route=Route.TROUBLESHOOTING, confidence=0.93, reason="Upload crash."),
        ANSWER,
    )

    response = client.post(URL, json={"message": "The app crashes when I upload a PNG"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    assert response.json() == {
        "route": "troubleshooting",
        "decided_by": "classifier",
        "confidence": 0.93,
        "reason": "Upload crash.",
        "classified_route": "troubleshooting",
        "handled_by": "llm",
        "answer": {
            "summary": "Try a smaller file.",
            "steps": ["Retry", "Update"],
            "follow_up": "Send logs.",
        },
        "handoff": None,
    }
    assert provider.call_count == 2


def test_guard_route_returns_handoff_without_llm_calls(use_provider):
    provider = use_provider()

    response = client.post(URL, json={"message": "Someone hacked my account"})

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "human_escalation"
    assert body["decided_by"] == "guard"
    assert body["handled_by"] == "human"
    assert body["answer"] is None
    assert body["handoff"]["priority"] == "high"
    assert body["handoff"]["summary"] == "Someone hacked my account"
    assert provider.call_count == 0


def test_threshold_route_returns_handoff(use_provider):
    use_provider(Classification(route=Route.HOW_TO, confidence=0.3, reason="Vague."))

    body = client.post(URL, json={"message": "it broke"}).json()

    assert body["decided_by"] == "threshold"
    assert body["classified_route"] == "how_to"
    assert body["handoff"]["priority"] == "low"


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "   "},
        {},
        {"message": 123},
        {"message": "hi", "route": "how_to"},  # a client can't choose its own route
    ],
)
def test_invalid_requests_return_422(use_provider, payload):
    provider = use_provider()

    response = client.post(URL, json=payload)

    assert response.status_code == 422
    assert response.json()["detail"] == "Request validation failed."
    assert response.json()["request_id"]
    assert provider.call_count == 0


def test_too_long_message_returns_422(use_provider):
    use_provider()
    limit = get_settings().max_message_chars  # the request model uses the app's settings

    response = client.post(URL, json={"message": "x" * (limit + 1)})

    assert response.status_code == 422


def test_provider_timeout_returns_safe_504(use_provider):
    use_provider(ProviderTimeoutError("Groq request timed out."))

    response = client.post(URL, json={"message": "SECRET-TICKET-TEXT the app crashes"})

    assert response.status_code == 504
    body = response.json()
    assert body == {"detail": "Support provider timed out.", "request_id": body["request_id"]}
    assert "SECRET-TICKET-TEXT" not in response.text
    assert "Groq" not in response.text


def test_missing_api_key_returns_503(use_provider):
    use_provider(ProviderConfigurationError("GROQ_API_KEY is not configured."))

    response = client.post(URL, json={"message": "How do I export?"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Support provider is not configured."


def test_endpoint_is_in_openapi():
    schema = client.get("/openapi.json").json()

    assert URL in schema["paths"]
    assert "post" in schema["paths"][URL]
