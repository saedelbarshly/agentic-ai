import re
from types import MappingProxyType

from app.core.config import Settings
from app.prompts import ACCOUNT_ACCESS_PROMPT, HOW_TO_PROMPT, TROUBLESHOOTING_PROMPT
from app.providers.base import JSONProvider
from app.schemas import LLM_ROUTES, Handoff, LLMAnswer, Priority, Route

# ---------------------------------------------------------------- Task 5.2

# Route -> specialized prompt. Adding an LLM route means one new line here.
HANDLER_PROMPTS: MappingProxyType[Route, str] = MappingProxyType(
    {
        Route.ACCOUNT_ACCESS: ACCOUNT_ACCESS_PROMPT,
        Route.TROUBLESHOOTING: TROUBLESHOOTING_PROMPT,
        Route.HOW_TO: HOW_TO_PROMPT,
    }
)
assert set(HANDLER_PROMPTS) == LLM_ROUTES, "Every LLM route needs exactly one prompt."


def handle_llm(
    route: Route, provider: JSONProvider, settings: Settings, message: str
) -> LLMAnswer:
    """Answer the ticket with the route's own prompt and model."""
    prompt = HANDLER_PROMPTS.get(route)
    if prompt is None:
        raise ValueError(f"{route.value} is not an LLM route.")
    return provider.complete_json(
        system_prompt=prompt,
        payload={"ticket": message},
        response_model=LLMAnswer,
        model=settings.model_for(route.value),
    )


# ---------------------------------------------------------------- Task 5.3

HANDOFF_SUMMARY_MAX_CHARS = 300
HANDOFF_USER_MESSAGE = (
    "Thanks for reaching out. Your request has been passed to a support "
    "specialist, who will get back to you shortly."
)


def build_handoff(message: str, reason: str, priority: Priority) -> Handoff:
    """Package a ticket for a human. No LLM call: the customer gets a fixed message.

    The summary is the customer's own text, for the human agent only. Never log it.
    """
    text = re.sub(r"\s+", " ", message).strip()
    if len(text) > HANDOFF_SUMMARY_MAX_CHARS:
        text = text[: HANDOFF_SUMMARY_MAX_CHARS - 1].rstrip() + "…"
    return Handoff(
        priority=priority,
        summary=text,
        escalation_reason=reason,
        user_message=HANDOFF_USER_MESSAGE,
    )
