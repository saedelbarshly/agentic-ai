"""Deterministic safety guard, run before any LLM call.

High-risk tickets go straight to a human. It's plain regex: cheap, fast,
predictable, and ticket text can't talk it out of a decision (prompt injection).
Its weakness is brittleness. See the known limitations in the tests and README §5.
"""

import re
from dataclasses import dataclass

from app.schemas import Priority


@dataclass(frozen=True)
class GuardRule:
    group: str
    priority: Priority
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class GuardResult:
    escalate: bool
    group: str | None = None
    priority: Priority | None = None
    # Built only from our own rule names and keywords, never from free user text,
    # so it is safe to log.
    reason: str | None = None


def _rule(group: str, priority: Priority, *alternatives: str) -> GuardRule:
    joined = "|".join(alternatives)
    return GuardRule(group, priority, re.compile(rf"\b(?:{joined})\b", re.IGNORECASE))


# Ordered from highest to lowest priority. The first match wins.
RULES: tuple[GuardRule, ...] = (
    _rule(
        "security",
        "high",
        r"hack(?:ed|ing)?",
        r"breach(?:ed)?",
        r"compromised",
        r"unauthori[sz]ed\s+(?:access|login|charges?)",
        r"phishing",
        r"someone\s+(?:else\s+)?(?:logged|accessed|is\s+using)",
        r"(?:account|password)\s+(?:was\s+)?stolen",
    ),
    _rule(
        "data_loss",
        "high",
        r"data\s+loss",
        # lost my files / lost all my files / lost all of our data / lost the projects
        r"lost\s+(?:all\s+(?:of\s+)?)?(?:my\s+|our\s+|the\s+)?(?:data|files|work|projects?)",
        r"deleted\s+everything",
        r"(?:data|files|projects?)\s+(?:is\s+|are\s+)?(?:gone|disappeared|wiped)",
    ),
    _rule(
        "outage",
        "high",
        r"outage",
        r"down\s+for\s+(?:everyone|everybody|all)",
        r"(?:all|our)\s+users\s+(?:can(?:no|')t|are\s+unable)",
        r"nobody\s+can",
        r"(?:site|service|server|app|system|everything)\s+is\s+down",
    ),
    _rule(
        "legal_billing",
        "medium",
        r"lawyer",
        r"attorney",
        r"lawsuit",
        r"legal\s+action",
        r"sue",
        r"chargeback",
        r"refund",
    ),
    _rule(
        "human_request",
        "medium",
        r"(?:speak|talk)\s+(?:to|with)\s+(?:a\s+|an\s+|your\s+)?"
        r"(?:human|person|agent|someone|manager|supervisor|representative)",
        r"real\s+person",
        r"human\s+(?:agent|support|being)",
    ),
)


def check(message: str) -> GuardResult:
    for rule in RULES:
        if rule.pattern.search(message):
            return GuardResult(
                escalate=True,
                group=rule.group,
                priority=rule.priority,
                reason=f"guard: {rule.group} rule matched",
            )
    return GuardResult(escalate=False)
