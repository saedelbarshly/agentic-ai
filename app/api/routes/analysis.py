from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import Settings, get_settings
from app.providers.groq_provider import GroqJSONProvider
from app.schemas import AnalysisRequest, AnalysisResponse
from app.services.workflow import AnalysisWorkflow, JSONProvider

router = APIRouter()


@lru_cache
def get_provider() -> GroqJSONProvider:
    return GroqJSONProvider(get_settings())


@router.post("/api/v1/analyze", response_model=AnalysisResponse)
def analyze(
    payload: AnalysisRequest,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    provider: Annotated[JSONProvider, Depends(get_provider)],
) -> AnalysisResponse:
    if len(payload.cv_text) > settings.max_cv_chars:
        raise HTTPException(status_code=422, detail="CV exceeds the configured character limit.")
    if len(payload.job_description) > settings.max_job_description_chars:
        raise HTTPException(
            status_code=422,
            detail="Job description exceeds the configured character limit.",
        )
    workflow = AnalysisWorkflow(provider, settings)
    return workflow.run(payload, request.state.request_id)
