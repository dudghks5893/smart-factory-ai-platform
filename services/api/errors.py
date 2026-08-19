"""Centralized HTTP error responses that do not expose internal exceptions."""

from __future__ import annotations

import logging
from typing import cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger(__name__)


class ApiError(Exception):
    """One intentional client-facing error with an HTTP status and stable code."""

    # ADD 2026-08-19: Client-facing status/code/message를 하나의 API exception으로 구성한다.
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


# ADD 2026-08-19: 명시적 API error를 stable JSON envelope로 변환한다.
async def handle_api_error(request: Request, exc: Exception) -> JSONResponse:
    """Return an intentional error without internal exception details."""
    api_error = cast(ApiError, exc)
    return JSONResponse(
        status_code=api_error.status_code,
        content={"error": {"code": api_error.code, "message": api_error.message}},
    )


# ADD 2026-08-19: FastAPI request validation 실패를 공통 public error로 변환한다.
async def handle_request_validation_error(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Hide framework validation internals behind one request error contract."""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "invalid_request",
                "message": "Request validation failed.",
            }
        },
    )


# ADD 2026-08-19: 예상하지 못한 request error를 기록하고 내부 정보를 숨긴다.
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Log an unexpected exception and return a non-sensitive response."""
    LOGGER.exception("Unhandled API request failure", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "Internal server error.",
            }
        },
    )


# ADD 2026-08-19: Application 전역 exception handler를 한 곳에서 등록한다.
def install_exception_handlers(app: FastAPI) -> None:
    """Install reusable API, validation, and fallback exception handlers."""
    app.add_exception_handler(ApiError, handle_api_error)
    app.add_exception_handler(RequestValidationError, handle_request_validation_error)
    app.add_exception_handler(Exception, handle_unexpected_error)
