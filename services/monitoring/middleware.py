"""Low-cardinality ASGI HTTP instrumentation for FastAPI."""

from __future__ import annotations

from time import perf_counter
from typing import Final

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from services.monitoring.metrics import MonitoringMetrics

KNOWN_HTTP_METHODS: Final = frozenset({"DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"})
UNMATCHED_ROUTE: Final = "unmatched"


class HttpMetricsMiddleware:
    """Observe completed HTTP requests without using raw URL paths as labels."""

    # ADD 2026-08-21: ASGI application과 app-local monitoring registry를 연결한다.
    def __init__(self, app: ASGIApp, *, metrics: MonitoringMetrics) -> None:
        self._app = app
        self._metrics = metrics

    # ADD 2026-08-21: Response status와 route template을 request latency와 함께 기록한다.
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        started_at = perf_counter()
        status_code = 500

        # ADD 2026-08-21: ASGI response start에서 최종 HTTP status를 캡처한다.
        async def capture_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        try:
            await self._app(scope, receive, capture_status)
        finally:
            route = _normalized_route(scope)
            if route != "/metrics":
                self._metrics.observe_http(
                    method=_normalized_method(scope),
                    route=route,
                    status_code=status_code,
                    duration_seconds=perf_counter() - started_at,
                )


# ADD 2026-08-21: Known method만 보존하고 arbitrary method token을 bounded label로 변환한다.
def _normalized_method(scope: Scope) -> str:
    method = str(scope.get("method", "")).upper()
    return method if method in KNOWN_HTTP_METHODS else "OTHER"


# ADD 2026-08-21: Router가 결정한 path template 또는 단일 unmatched label을 반환한다.
def _normalized_route(scope: Scope) -> str:
    route = scope.get("route")
    route_path = getattr(route, "path", None)
    return route_path if isinstance(route_path, str) else UNMATCHED_ROUTE
