"""The routing workflow: guard -> classify -> threshold -> dispatch.

The LLM only labels the ticket. Every branch is plain Python here, which keeps
this a predictable workflow and not an agent.
"""

import logging
import time

from app.core.config import Settings
from app.providers.base import JSONProvider
from app.schemas import (
    Classification,
    DecidedBy,
    Priority,
    Route,
    SupportResponse,
)
from app.services import guard
from app.services.classifier import classify
from app.services.handlers import build_handoff, handle_llm

logger = logging.getLogger(__name__)

# The classifier gives no priority, so escalations it decides get a default.
# Trade-off: a security ticket the guard missed lands here as "medium".
CLASSIFIER_ESCALATION_PRIORITY: Priority = "medium"
# Low confidence usually means a vague ticket, not an urgent one.
THRESHOLD_ESCALATION_PRIORITY: Priority = "low"


def route_ticket(provider: JSONProvider, settings: Settings, message: str) -> SupportResponse:
    start = time.perf_counter()
    response, llm_calls = _route(provider, settings, message)
    # Metadata only. The ticket text and the model's answer are never logged.
    logger.info(
        "ticket_routed route=%s decided_by=%s classified_route=%s confidence=%s "
        "llm_calls=%d duration_ms=%.0f",
        response.route.value,
        response.decided_by,
        response.classified_route.value if response.classified_route else None,
        f"{response.confidence:.2f}" if response.confidence is not None else None,
        llm_calls,
        (time.perf_counter() - start) * 1000,
    )
    return response


def _route(
    provider: JSONProvider, settings: Settings, message: str
) -> tuple[SupportResponse, int]:
    # 1. Guard: deterministic, 0 LLM calls.
    guard_result = guard.check(message)
    if guard_result.escalate:
        reason = guard_result.reason or "guard: rule matched"
        return _escalate(
            message,
            decided_by="guard",
            reason=reason,
            priority=guard_result.priority or "high",
            classification=None,
        ), 0

    # 2. Classify: 1 LLM call.
    classification = classify(provider, settings, message)

    # 3. Not confident enough: escalate whatever the label was.
    threshold = settings.router_confidence_threshold
    if classification.confidence < threshold:
        reason = (
            f"threshold: confidence {classification.confidence:.2f} < {threshold:.2f} "
            f"(classifier suggested {classification.route.value}: {classification.reason})"
        )
        return _escalate(
            message,
            decided_by="threshold",
            reason=reason,
            priority=THRESHOLD_ESCALATION_PRIORITY,
            classification=classification,
        ), 1

    # 4. Classifier chose a human.
    if classification.route is Route.HUMAN_ESCALATION:
        return _escalate(
            message,
            decided_by="classifier",
            reason=f"classifier: {classification.reason}",
            priority=CLASSIFIER_ESCALATION_PRIORITY,
            classification=classification,
        ), 1

    # 5. One of the three LLM routes: 2 LLM calls in total.
    answer = handle_llm(classification.route, provider, settings, message)
    return SupportResponse(
        route=classification.route,
        decided_by="classifier",
        confidence=classification.confidence,
        reason=classification.reason,
        classified_route=classification.route,
        handled_by="llm",
        answer=answer,
        handoff=None,
    ), 2


def _escalate(
    message: str,
    *,
    decided_by: DecidedBy,
    reason: str,
    priority: Priority,
    classification: Classification | None,
) -> SupportResponse:
    return SupportResponse(
        route=Route.HUMAN_ESCALATION,
        decided_by=decided_by,
        confidence=classification.confidence if classification else None,
        reason=reason,
        classified_route=classification.route if classification else None,
        handled_by="human",
        answer=None,
        handoff=build_handoff(message, reason, priority),
    )
