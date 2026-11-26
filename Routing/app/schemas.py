from enum import Enum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)

from app.core.config import get_settings


class StrictModel(BaseModel):
    # extra="forbid" -> "additionalProperties": false, which Groq strict mode needs.
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- Task 1.1


class Route(str, Enum):
    ACCOUNT_ACCESS = "account_access"
    TROUBLESHOOTING = "troubleshooting"
    HOW_TO = "how_to"
    HUMAN_ESCALATION = "human_escalation"


LLM_ROUTES: frozenset[Route] = frozenset(
    {Route.ACCOUNT_ACCESS, Route.TROUBLESHOOTING, Route.HOW_TO}
)


class SupportRequest(StrictModel):
    message: str

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError("Message must not be empty.")
        # The limit comes from validation context (tests) or from settings (API).
        context = info.context or {}
        max_chars = context.get("max_message_chars") or get_settings().max_message_chars
        if len(value) > max_chars:
            raise ValueError(f"Message must be at most {max_chars} characters.")
        return value


# ---------------------------------------------------------------- Task 1.2
# Constraints such as the 0-1 confidence range are checked in validators, not
# with Field(ge=..., le=...), so the JSON schema sent to the model stays
# simple and works with strict mode.


class Classification(StrictModel):
    """Classifier output (LLM)."""

    route: Route
    confidence: float
    reason: str

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        return value


class LLMAnswer(StrictModel):
    """Output of the three LLM handlers."""

    summary: str
    steps: list[str]
    follow_up: str

    @field_validator("steps")
    @classmethod
    def drop_blank_steps(cls, value: list[str]) -> list[str]:
        # Models sometimes add a trailing "" step. Drop blanks; keep the real ones.
        steps = [step.strip() for step in value if step.strip()]
        if not steps:
            raise ValueError("At least one non-empty step is required.")
        return steps

    @field_validator("summary", "follow_up")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Must not be empty.")
        return value.strip()


Priority = Literal["low", "medium", "high"]


class Handoff(StrictModel):
    """Package for a human agent. Built in code in v1, so no LLM writes it."""

    priority: Priority
    summary: str
    escalation_reason: str
    user_message: str


# ---------------------------------------------------------------- Task 1.3

DecidedBy = Literal["guard", "classifier", "threshold"]
HandledBy = Literal["llm", "human"]


class SupportResponse(StrictModel):
    route: Route
    decided_by: DecidedBy
    # None when the guard decides (no classifier call was made).
    confidence: float | None
    reason: str
    # What the classifier picked, even if the threshold overrode it. Used for evals.
    classified_route: Route | None
    handled_by: HandledBy
    answer: LLMAnswer | None
    handoff: Handoff | None

    @model_validator(mode="after")
    def check_consistency(self) -> "SupportResponse":
        is_human = self.route is Route.HUMAN_ESCALATION

        if (self.handled_by == "human") != is_human:
            raise ValueError("handled_by must be 'human' exactly when route is human_escalation.")
        if is_human and (self.handoff is None or self.answer is not None):
            raise ValueError("A human route needs a handoff and no answer.")
        if not is_human and (self.answer is None or self.handoff is not None):
            raise ValueError("An LLM route needs an answer and no handoff.")

        if self.decided_by in ("guard", "threshold") and not is_human:
            raise ValueError(f"decided_by='{self.decided_by}' always escalates to a human.")
        if self.decided_by == "guard" and (
            self.confidence is not None or self.classified_route is not None
        ):
            raise ValueError("The guard runs before the classifier, so it has no classifier data.")
        if self.decided_by != "guard" and (
            self.confidence is None or self.classified_route is None
        ):
            raise ValueError("Classifier decisions must include confidence and classified_route.")
        return self
