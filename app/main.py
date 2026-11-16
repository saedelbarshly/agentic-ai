import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes.analysis import router
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

app = FastAPI(title="AI CV & Job Analyzer", version="1.0.0")
app.include_router(router)


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


@app.exception_handler(ProviderConfigurationError)
async def provider_configuration_error(
    request: Request, exc: ProviderConfigurationError
) -> JSONResponse:
    return _provider_error_response(request, exc, 503, "Analysis provider is not configured.")


@app.exception_handler(ProviderAuthenticationError)
async def provider_authentication_error(
    request: Request, exc: ProviderAuthenticationError
) -> JSONResponse:
    return _provider_error_response(request, exc, 502, "Analysis provider authentication failed.")


@app.exception_handler(ProviderRateLimitError)
async def provider_rate_limit_error(
    request: Request, exc: ProviderRateLimitError
) -> JSONResponse:
    return _provider_error_response(request, exc, 429, "Analysis provider rate limit reached.")


@app.exception_handler(ProviderUnavailableError)
async def provider_unavailable_error(
    request: Request, exc: ProviderUnavailableError
) -> JSONResponse:
    return _provider_error_response(request, exc, 503, "Analysis provider is unavailable.")


@app.exception_handler(ProviderTimeoutError)
async def provider_timeout_error(
    request: Request, exc: ProviderTimeoutError
) -> JSONResponse:
    return _provider_error_response(request, exc, 504, "Analysis provider timed out.")


@app.exception_handler(InvalidModelOutputError)
async def invalid_model_output_error(
    request: Request, exc: InvalidModelOutputError
) -> JSONResponse:
    return _provider_error_response(request, exc, 502, "Analysis provider returned invalid output.")


@app.exception_handler(ProviderError)
async def provider_error(request: Request, exc: ProviderError) -> JSONResponse:
    return _provider_error_response(request, exc, 502, "Analysis provider request failed.")


def _provider_error_response(
    request: Request, exc: ProviderError, status_code: int, detail: str
) -> JSONResponse:
    logger.error(
        "provider_request_failed request_id=%s error_type=%s",
        getattr(request.state, "request_id", "unknown"),
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": detail,
            "request_id": getattr(request.state, "request_id", None),
        },
    )
