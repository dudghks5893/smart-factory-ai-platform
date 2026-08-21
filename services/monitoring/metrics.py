"""App-local Prometheus metrics for HTTP, inference, and persistence."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    Info,
    generate_latest,
)

from services.inference.runtime import ModelRuntime, ServingProvenance

HTTP_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
INFERENCE_DURATION_BUCKETS = HTTP_DURATION_BUCKETS
PERSISTENCE_DURATION_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
)


class MonitoringMetrics:
    """Collectors isolated to one FastAPI application instance."""

    # ADD 2026-08-21: App별 registry에 low-cardinality operational metric을 등록한다.
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self._http_requests = Counter(
            "smartfactory_http_requests_total",
            "Completed FastAPI requests excluding the metrics scrape endpoint.",
            ("method", "route", "status_code"),
            registry=self.registry,
        )
        self._http_duration = Histogram(
            "smartfactory_http_request_duration_seconds",
            "FastAPI application request duration excluding the metrics scrape endpoint.",
            ("method", "route"),
            buckets=HTTP_DURATION_BUCKETS,
            registry=self.registry,
        )
        self._predictions = Counter(
            "smartfactory_predictions_total",
            "Successfully persisted model predictions by category and result.",
            ("category", "result"),
            registry=self.registry,
        )
        self._inference_duration = Histogram(
            "smartfactory_inference_duration_seconds",
            "Model runtime predict call duration without HTTP decoding or persistence.",
            ("model_name", "category", "device"),
            buckets=INFERENCE_DURATION_BUCKETS,
            registry=self.registry,
        )
        self._persistence_duration = Histogram(
            "smartfactory_persistence_duration_seconds",
            "Inspection persistence operation duration.",
            ("operation",),
            buckets=PERSISTENCE_DURATION_BUCKETS,
            registry=self.registry,
        )
        self._persistence_errors = Counter(
            "smartfactory_persistence_errors_total",
            "Stable persistence operation failures.",
            ("operation",),
            registry=self.registry,
        )
        self._model = Info(
            "smartfactory_model",
            "Static identity of the process-local serving model.",
            registry=self.registry,
        )

    # ADD 2026-08-21: Template-normalized HTTP request count와 latency를 함께 기록한다.
    def observe_http(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        self._http_requests.labels(
            method=method,
            route=route,
            status_code=str(status_code),
        ).inc()
        self._http_duration.labels(method=method, route=route).observe(duration_seconds)

    # ADD 2026-08-21: Runtime predict 호출 경계의 성공/실패 latency를 관측한다.
    @contextmanager
    def track_inference(
        self,
        *,
        model_name: str,
        category: str,
        device: str,
    ) -> Iterator[None]:
        with self._inference_duration.labels(
            model_name=model_name,
            category=category,
            device=device,
        ).time():
            yield

    # ADD 2026-08-21: Client에 성공 반환 가능한 persisted prediction 결과를 집계한다.
    def record_prediction(self, *, category: str, is_anomaly: bool) -> None:
        result = "anomaly" if is_anomaly else "normal"
        self._predictions.labels(category=category, result=result).inc()

    # ADD 2026-08-21: Stable operation 이름으로 persistence latency를 관측한다.
    @contextmanager
    def track_persistence(self, *, operation: str) -> Iterator[None]:
        with self._persistence_duration.labels(operation=operation).time():
            yield

    # ADD 2026-08-21: Exception detail 없이 stable persistence operation failure를 집계한다.
    def record_persistence_error(self, *, operation: str) -> None:
        self._persistence_errors.labels(operation=operation).inc()

    # ADD 2026-08-21: 배포당 하나인 model identity와 model hash를 Info metric으로 게시한다.
    def set_model_info(self, runtime: ModelRuntime) -> None:
        provenance = getattr(runtime, "provenance", None)
        model_sha256 = (
            provenance.model_sha256 if isinstance(provenance, ServingProvenance) else "unavailable"
        )
        self._model.info(
            {
                "model_name": runtime.model_name,
                "category": runtime.category,
                "device": runtime.device,
                "model_sha256": model_sha256,
            }
        )


# ADD 2026-08-21: App-local registry를 Prometheus text exposition response로 반환한다.
async def metrics_endpoint(request: Request) -> Response:
    """Expose process-local metrics without adding a business OpenAPI operation."""
    metrics = getattr(request.app.state, "monitoring_metrics", None)
    if not isinstance(metrics, MonitoringMetrics):
        raise RuntimeError("Application monitoring registry is unavailable.")
    return Response(
        content=generate_latest(metrics.registry),
        media_type=CONTENT_TYPE_LATEST,
    )
