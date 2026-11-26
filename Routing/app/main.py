import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes.support import router as support_router
from app.core.config import get_settings
from app.core.errors import (
    InvalidModelOutputError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.core.logging import configure_logging

settings = get_settings()
configure_logging(settings)
logger = logging.getLogger(__name__)

app = FastAPI(title="Technical Support Router", version="0.1.0")
app.include_router(support_router)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.exception_handler(RequestValidationError)
async def request_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    logger.info("request_validation_failed request_id=%s", request_id)
    return JSONResponse(
        status_code=422,
        content={"detail": "Request validation failed.", "request_id": request_id},
    )


# Most specific first; ProviderError is the catch-all.
_PROVIDER_ERRORS: list[tuple[type[ProviderError], int, str]] = [
    (ProviderConfigurationError, 503, "Support provider is not configured."),
    (ProviderAuthenticationError, 502, "Support provider authentication failed."),
    (ProviderRateLimitError, 429, "Support provider rate limit reached."),
    (ProviderUnavailableError, 503, "Support provider is unavailable."),
    (ProviderTimeoutError, 504, "Support provider timed out."),
    (InvalidModelOutputError, 502, "Support provider returned invalid output."),
    (ProviderError, 502, "Support provider request failed."),
]


@app.exception_handler(ProviderError)
async def provider_error(request: Request, exc: ProviderError) -> JSONResponse:
    status_code, detail = next(
        (code, msg) for err, code, msg in _PROVIDER_ERRORS if isinstance(exc, err)
    )
    request_id = getattr(request.state, "request_id", None)
    logger.error(
        "provider_request_failed request_id=%s error_type=%s",
        request_id,
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "request_id": request_id},
    )
