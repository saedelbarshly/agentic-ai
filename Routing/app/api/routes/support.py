from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.providers.base import JSONProvider
from app.providers.groq_provider import GroqJSONProvider
from app.schemas import SupportRequest, SupportResponse
from app.services.router import route_ticket

router = APIRouter(prefix="/api/v1/support", tags=["support"])


@lru_cache
def get_provider() -> JSONProvider:
    """One shared Groq client. Tests replace it with app.dependency_overrides."""
    return GroqJSONProvider(get_settings())


# A sync def: FastAPI runs it in a thread pool, so the blocking Groq call
# doesn't block the event loop.
@router.post("/route", response_model=SupportResponse)
def route_support_ticket(
    payload: SupportRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    provider: Annotated[JSONProvider, Depends(get_provider)],
) -> SupportResponse:
    """Route a support ticket. Returns an LLM answer or a human handoff, never both."""
    return route_ticket(provider, settings, payload.message)
