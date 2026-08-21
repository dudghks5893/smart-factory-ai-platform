"""Safe error mapping for the independent RAG service."""

from __future__ import annotations

import logging
from typing import cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger(__name__)


class RagApiError(Exception):
    """Intentional client-facing RAG error with stable code."""

    # ADD 2026-08-21: Public status/code/message를 internal exception과 분리한다.
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


# ADD 2026-08-21: Intentional RAG error를 stable JSON envelope로 변환한다.
async def handle_rag_api_error(request: Request, exc: Exception) -> JSONResponse:
    error = cast(RagApiError, exc)
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"code": error.code, "message": error.message}},
    )


# ADD 2026-08-21: Request validation detail과 untrusted input을 public response에서 숨긴다.
async def handle_request_validation_error(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "invalid_request",
                "message": "Request validation failed.",
            }
        },
    )


# ADD 2026-08-21: Unexpected RAG exception을 server log에 남기고 내부 detail을 숨긴다.
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled RAG request failure", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": "Internal server error."}},
    )


# ADD 2026-08-21: RAG-specific validation, intentional과 fallback handler를 등록한다.
def install_rag_exception_handlers(app: FastAPI) -> None:
    """Install service-local handlers without coupling to Vision persistence errors."""
    app.add_exception_handler(RagApiError, handle_rag_api_error)
    app.add_exception_handler(RequestValidationError, handle_request_validation_error)
    app.add_exception_handler(Exception, handle_unexpected_error)
