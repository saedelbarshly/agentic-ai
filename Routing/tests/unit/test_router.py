import logging

import pytest

from app.core.errors import InvalidModelOutputError, ProviderTimeoutError
from app.prompts import CLASSIFIER_PROMPT, HOW_TO_PROMPT, TROUBLESHOOTING_PROMPT
from app.schemas import Classification, LLMAnswer, Route
from app.services.handlers import HANDOFF_USER_MESSAGE
from app.services.router import route_ticket
from tests.helpers import FakeProvider, make_settings

ANSWER = LLMAnswer(summary="Likely a file issue.", steps=["Retry", "Update"], follow_up="f")


def classified(route: Route, confidence: float = 0.9, reason: str = "Clear fit.") -> Classification:
    return Classification(route=route, confidence=confidence, reason=reason)


# ------------------------------------------------------------ branch 1: guard


def test_guard_hit_escalates_with_zero_llm_calls():
    provider = FakeProvider()  # nothing queued: any call fails the test

    response = route_ticket(provider, make_settings(), "My account was hacked!")

    assert provider.call_count == 0
    assert response.route is Route.HUMAN_ESCALATION
    assert response.decided_by == "guard"
    assert response.handled_by == "human"
    assert response.confidence is None and response.classified_route is None
    assert response.reason == "guard: security rule matched"
    assert response.handoff.priority == "high"
    assert response.handoff.user_message == HANDOFF_USER_MESSAGE
    assert response.answer is None


# ------------------------------------------------------------ branch 3: threshold


def test_low_confidence_escalates_after_one_call():
    provider = FakeProvider(classified(Route.TROUBLESHOOTING, confidence=0.4, reason="Vague."))

    response = route_ticket(provider, make_settings(), "it doesn't work")

    assert provider.call_count == 1
    assert provider.calls[0]["system_prompt"] == CLASSIFIER_PROMPT
    assert response.route is Route.HUMAN_ESCALATION
    assert response.decided_by == "threshold"
    assert response.confidence == 0.4
    assert response.classified_route is Route.TROUBLESHOOTING  # kept for evals
    assert response.reason.startswith("threshold: confidence 0.40 < 0.70")
    assert "troubleshooting" in response.reason
    assert response.handoff.priority == "low"


def test_low_confidence_human_pick_is_still_decided_by_threshold():
    provider = FakeProvider(classified(Route.HUMAN_ESCALATION, confidence=0.5))

    response = route_ticket(provider, make_settings(), "help")

    assert response.decided_by == "threshold"
    assert response.classified_route is Route.HUMAN_ESCALATION


@pytest.mark.parametrize(
    "confidence, threshold, escalated",
    [
        (0.69, 0.7, True),
        (0.70, 0.7, False),  # equal to the threshold is confident enough
        (0.71, 0.7, False),
        (0.85, 0.9, True),  # a stricter threshold from settings
    ],
)
def test_threshold_boundary(confidence, threshold, escalated):
    queued = [classified(Route.HOW_TO, confidence=confidence)]
    if not escalated:
        queued.append(ANSWER)
    provider = FakeProvider(*queued)

    response = route_ticket(
        provider, make_settings(router_confidence_threshold=threshold), "How do I export?"
    )

    assert (response.decided_by == "threshold") is escalated
    assert provider.call_count == (1 if escalated else 2)


# ------------------------------------------------------------ branch 4: classifier -> human


def test_classifier_choosing_human_escalates_after_one_call():
    provider = FakeProvider(
        classified(Route.HUMAN_ESCALATION, confidence=0.95, reason="Account takeover.")
    )

    response = route_ticket(provider, make_settings(), "Somebody changed my email")

    assert provider.call_count == 1
    assert response.route is Route.HUMAN_ESCALATION
    assert response.decided_by == "classifier"
    assert response.classified_route is Route.HUMAN_ESCALATION
    assert response.reason == "classifier: Account takeover."
    assert response.handoff.priority == "medium"
    assert response.handoff.summary == "Somebody changed my email"


# ------------------------------------------------------------ branch 5: LLM routes


@pytest.mark.parametrize(
    "route, prompt",
    [
        (Route.TROUBLESHOOTING, TROUBLESHOOTING_PROMPT),
        (Route.HOW_TO, HOW_TO_PROMPT),
    ],
)
def test_llm_route_makes_two_calls_and_returns_the_answer(route, prompt):
    provider = FakeProvider(classified(route, confidence=0.92), ANSWER)

    response = route_ticket(provider, make_settings(), "The app crashes on upload")

    assert provider.call_count == 2
    assert provider.calls[0]["system_prompt"] == CLASSIFIER_PROMPT
    assert provider.calls[1]["system_prompt"] == prompt
    assert response.route is route
    assert response.decided_by == "classifier"
    assert response.handled_by == "llm"
    assert response.confidence == 0.92
    assert response.answer == ANSWER
    assert response.handoff is None


def test_each_step_uses_its_own_model():
    provider = FakeProvider(classified(Route.ACCOUNT_ACCESS), ANSWER)
    settings = make_settings(classifier_model="tiny", account_access_model="strong")

    route_ticket(provider, settings, "My 2FA code is invalid")

    assert [call["model"] for call in provider.calls] == ["tiny", "strong"]


# ------------------------------------------------------------ errors


@pytest.mark.parametrize("error", [ProviderTimeoutError(), InvalidModelOutputError()])
def test_classifier_errors_propagate(error):
    provider = FakeProvider(error)

    with pytest.raises(type(error)):
        route_ticket(provider, make_settings(), "The app crashes")


def test_handler_errors_propagate():
    provider = FakeProvider(classified(Route.HOW_TO), ProviderTimeoutError())

    with pytest.raises(ProviderTimeoutError):
        route_ticket(provider, make_settings(), "How do I export?")


# ------------------------------------------------------------ logging


@pytest.mark.parametrize(
    "message, queued",
    [
        ("SECRET-TICKET-TEXT my account was hacked", []),
        ("SECRET-TICKET-TEXT vague", [classified(Route.HOW_TO, confidence=0.3)]),
        ("SECRET-TICKET-TEXT export", [classified(Route.HOW_TO), ANSWER]),
    ],
)
def test_logs_metadata_but_never_the_ticket_or_answer(caplog, message, queued):
    provider = FakeProvider(*queued)

    with caplog.at_level(logging.INFO, logger="app.services.router"):
        route_ticket(provider, make_settings(), message)

    assert "ticket_routed" in caplog.text
    assert "SECRET-TICKET-TEXT" not in caplog.text
    assert ANSWER.summary not in caplog.text


def test_log_line_has_the_routing_fields(caplog):
    provider = FakeProvider(classified(Route.HOW_TO, confidence=0.9), ANSWER)

    with caplog.at_level(logging.INFO, logger="app.services.router"):
        route_ticket(provider, make_settings(), "How do I export?")

    line = caplog.records[-1].getMessage()
    for expected in (
        "route=how_to",
        "decided_by=classifier",
        "classified_route=how_to",
        "confidence=0.90",
        "llm_calls=2",
        "duration_ms=",
    ):
        assert expected in line
