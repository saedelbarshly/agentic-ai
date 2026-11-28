from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class JSONProvider(Protocol):
    """What services depend on. GroqJSONProvider and the tests' FakeProvider both match it."""

    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        response_model: type[ResponseModel],
        model: str | None = None,
    ) -> ResponseModel: ...
