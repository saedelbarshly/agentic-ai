import pytest
from pydantic import BaseModel, ValidationError

from app.schemas import (
    LLM_ROUTES,
    Classification,
    Handoff,
    LLMAnswer,
    Route,
    SupportRequest,
    SupportResponse,
)

# ------------------------------------------------------------ Task 1.1


def test_route_has_four_values_three_for_llm():
    assert {r.value for r in Route} == {
        "account_access",
        "troubleshooting",
        "how_to",
        "human_escalation",
    }
    assert Route.HUMAN_ESCALATION not in LLM_ROUTES
    assert len(LLM_ROUTES) == 3


def make_request(message: str, max_chars: int = 50) -> SupportRequest:
    return SupportRequest.model_validate(
        {"message": message}, context={"max_message_chars": max_chars}
    )


def test_request_accepts_normal_message():
    assert make_request("I can't log in").message == "I can't log in"


@pytest.mark.parametrize("message", ["", "   ", "\n\t"])
def test_request_rejects_empty_message(message):
    with pytest.raises(ValidationError, match="must not be empty"):
        make_request(message)


def test_request_rejects_too_long_message():
    make_request("x" * 50)  # exactly at the limit is fine
    with pytest.raises(ValidationError, match="at most 50 characters"):
        make_request("x" * 51)


def test_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        SupportRequest.model_validate(
            {"message": "hi", "route": "how_to"}, context={"max_message_chars": 50}
        )


def test_request_uses_settings_limit_without_context():
    # No context means the limit comes from settings (default 5000, or your .env).
    assert SupportRequest(message="hello").message == "hello"


# ------------------------------------------------------------ Task 1.2


def assert_strict_compatible(schema: dict, path: str = "$") -> None:
    """Every object forbids extra keys and requires all of its properties."""
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            assert schema.get("additionalProperties") is False, f"{path}: extra keys allowed"
            assert set(schema.get("required", [])) == set(schema["properties"]), (
                f"{path}: not all properties are required"
            )
        for key, value in schema.items():
            assert_strict_compatible(value, f"{path}.{key}")
    elif isinstance(schema, list):
        for i, item in enumerate(schema):
            assert_strict_compatible(item, f"{path}[{i}]")


@pytest.mark.parametrize("model", [Classification, LLMAnswer, Handoff])
def test_llm_facing_schemas_are_strict_compatible(model: type[BaseModel]):
    assert_strict_compatible(model.model_json_schema())


def test_classification_parses_model_json():
    result = Classification.model_validate_json(
        '{"route": "how_to", "confidence": 0.9, "reason": "Usage question."}'
    )
    assert result.route is Route.HOW_TO


@pytest.mark.parametrize("confidence", [-0.1, 1.01])
def test_classification_rejects_out_of_range_confidence(confidence):
    with pytest.raises(ValidationError, match="between 0 and 1"):
        Classification(route=Route.HOW_TO, confidence=confidence, reason="x")


def test_classification_rejects_unknown_route():
    with pytest.raises(ValidationError):
        Classification.model_validate(
            {"route": "billing", "confidence": 0.9, "reason": "x"}
        )


def test_handoff_rejects_unknown_priority():
    with pytest.raises(ValidationError):
        Handoff(priority="urgent", summary="s", escalation_reason="r", user_message="u")


# ------------------------------------------------------------ Task 1.3

ANSWER = LLMAnswer(summary="s", steps=["a", "b"], follow_up="f")
HANDOFF = Handoff(priority="high", summary="s", escalation_reason="r", user_message="u")


def llm_response(**overrides) -> SupportResponse:
    data = dict(
        route=Route.HOW_TO,
        decided_by="classifier",
        confidence=0.9,
        reason="r",
        classified_route=Route.HOW_TO,
        handled_by="llm",
        answer=ANSWER,
        handoff=None,
    )
    return SupportResponse(**(data | overrides))


def human_response(**overrides) -> SupportResponse:
    data = dict(
        route=Route.HUMAN_ESCALATION,
        decided_by="guard",
        confidence=None,
        reason="r",
        classified_route=None,
        handled_by="human",
        answer=None,
        handoff=HANDOFF,
    )
    return SupportResponse(**(data | overrides))


def test_valid_llm_response():
    response = llm_response()
    assert response.answer is not None and response.handoff is None


def test_valid_guard_response():
    response = human_response()
    assert response.handoff is not None and response.answer is None


def test_valid_threshold_response_keeps_classifier_pick():
    response = human_response(
        decided_by="threshold", confidence=0.4, classified_route=Route.TROUBLESHOOTING
    )
    assert response.classified_route is Route.TROUBLESHOOTING


@pytest.mark.parametrize(
    "build, overrides",
    [
        (llm_response, {"handoff": HANDOFF}),  # llm with a handoff
        (llm_response, {"answer": None}),  # llm without an answer
        (llm_response, {"handled_by": "human"}),  # llm route marked human
        (human_response, {"answer": ANSWER}),  # human with an answer
        (human_response, {"handoff": None}),  # human without a handoff
        (human_response, {"handled_by": "llm"}),  # human route marked llm
        (human_response, {"confidence": 0.9}),  # guard with confidence
        (human_response, {"decided_by": "classifier"}),  # classifier without data
        (llm_response, {"decided_by": "threshold"}),  # threshold must escalate
        (llm_response, {"decided_by": "guard"}),  # guard must escalate
    ],
)
def test_inconsistent_responses_are_rejected(build, overrides):
    with pytest.raises(ValidationError):
        build(**overrides)


# ------------------------------------------------------------ LLMAnswer cleanup
# Found in a real run: the model returned a trailing empty step.


def test_llm_answer_drops_blank_steps():
    answer = LLMAnswer(summary="s", steps=["one", "  ", "", " two "], follow_up="f")
    assert answer.steps == ["one", "two"]


def test_llm_answer_needs_at_least_one_real_step():
    with pytest.raises(ValidationError, match="non-empty step"):
        LLMAnswer(summary="s", steps=["", "  "], follow_up="f")


@pytest.mark.parametrize("field", ["summary", "follow_up"])
def test_llm_answer_rejects_blank_text(field):
    data = {"summary": "s", "steps": ["one"], "follow_up": "f"} | {field: "   "}
    with pytest.raises(ValidationError, match="Must not be empty"):
        LLMAnswer(**data)
