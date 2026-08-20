"""FastAPI application-boundary benchmark measurement and artifact contracts."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from services.api.tooling import PreparedImageUpload, validate_prediction_payload
from shared.benchmarking import (
    LINEAR_PERCENTILE_METHOD,
    LatencyDistribution,
    summarize_latency_distribution,
)
from shared.hashing import is_sha256_digest

API_BENCHMARK_SCHEMA_VERSION = 2
API_BENCHMARK_NAME = "patchcore_fastapi_http_e2e"
API_LATENCY_DEFINITION = (
    "in-process multipart HTTP request through FastAPI routing, upload read, image decode, "
    "tensor conversion, preprocessing, device transfer, PatchCore inference, strict threshold, "
    "inspection insert/commit, response validation/serialization, and completed ASGI response "
    "delivery"
)
API_TRANSPORT = "in_process_asgi_testclient"


class HttpResponse(Protocol):
    """Minimal response contract needed by benchmark measurement."""

    status_code: int

    # ADD 2026-08-20: Completed HTTP response body를 JSON value로 decode한다.
    def json(self) -> object:
        """Decode the completed HTTP response body."""
        ...


type RequestSender = Callable[[PreparedImageUpload], HttpResponse]


@dataclass(frozen=True)
class HttpBenchmarkMeasurements:
    """Per-attempt HTTP latencies and outcome counts before summarization."""

    latencies_ms: tuple[float, ...]
    successful_request_count: int
    failed_request_count: int


@dataclass(frozen=True)
class HttpBenchmarkMetrics:
    """Finite HTTP latency distribution, throughput, and error statistics."""

    latency: LatencyDistribution
    requests_per_second: float
    successful_request_count: int
    failed_request_count: int
    error_rate: float

    # ADD 2026-08-20: HTTP benchmark metrics를 stable JSON mapping으로 변환한다.
    def to_json_dict(self) -> dict[str, object]:
        return {
            "latency_ms": self.latency.to_json_dict(),
            "requests_per_second": self.requests_per_second,
            "successful_request_count": self.successful_request_count,
            "failed_request_count": self.failed_request_count,
            "error_rate": self.error_rate,
        }


@dataclass(frozen=True)
class PatchCoreApiBenchmarkArtifact:
    """Schema-versioned FastAPI HTTP application benchmark result."""

    model_name: str
    category: str
    device: str
    operating_system: str
    machine: str
    accelerator_name: str
    cuda_version: str | None
    python_version: str
    torch_version: str
    fastapi_version: str
    starlette_version: str
    manifest_sha256: str
    artifact_metadata_sha256: str
    model_sha256: str
    threshold_artifact_sha256: str
    image_payload_bytes: tuple[int, ...]
    warmup_count: int
    measured_count: int
    metrics: HttpBenchmarkMetrics
    created_at: str
    schema_version: int = API_BENCHMARK_SCHEMA_VERSION

    # ADD 2026-08-20: API benchmark schema, provenance와 finite metrics를 검증한다.
    def validate(self) -> None:
        if self.schema_version != API_BENCHMARK_SCHEMA_VERSION:
            raise ValueError(f"Unsupported API benchmark schema_version: {self.schema_version}.")
        for field, value in (
            ("model_name", self.model_name),
            ("category", self.category),
            ("device", self.device),
            ("operating_system", self.operating_system),
            ("machine", self.machine),
            ("accelerator_name", self.accelerator_name),
            ("python_version", self.python_version),
            ("torch_version", self.torch_version),
            ("fastapi_version", self.fastapi_version),
            ("starlette_version", self.starlette_version),
            ("created_at", self.created_at),
        ):
            if not value:
                raise ValueError(f"API benchmark {field} must not be empty.")
        for field, digest in (
            ("manifest_sha256", self.manifest_sha256),
            ("artifact_metadata_sha256", self.artifact_metadata_sha256),
            ("model_sha256", self.model_sha256),
            ("threshold_artifact_sha256", self.threshold_artifact_sha256),
        ):
            if not is_sha256_digest(digest):
                raise ValueError(f"API benchmark {field} must be a SHA-256 hex digest.")
        if self.warmup_count < 0 or self.measured_count <= 0:
            raise ValueError("API benchmark counts must be non-negative with measured_count > 0.")
        if len(self.image_payload_bytes) != self.measured_count:
            raise ValueError("image_payload_bytes must contain one value per measured request.")
        if any(size <= 0 for size in self.image_payload_bytes):
            raise ValueError("All measured image payload sizes must be positive.")
        _validate_metrics(self.metrics, self.measured_count)

    # ADD 2026-08-20: API benchmark result와 명시적 timing boundary를 JSON schema로 변환한다.
    # MODIFY 2026-08-20: Inspection insert/commit 포함 여부를 schema v2에 명시한다.
    def to_json_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "benchmark_name": API_BENCHMARK_NAME,
            "model_name": self.model_name,
            "category": self.category,
            "device": self.device,
            "runtime": {
                "operating_system": self.operating_system,
                "machine": self.machine,
                "accelerator_name": self.accelerator_name,
                "cuda_version": self.cuda_version,
                "python_version": self.python_version,
                "torch_version": self.torch_version,
                "fastapi_version": self.fastapi_version,
                "starlette_version": self.starlette_version,
            },
            "provenance": {
                "manifest_sha256": self.manifest_sha256,
                "artifact_metadata_sha256": self.artifact_metadata_sha256,
                "model_sha256": self.model_sha256,
                "threshold_artifact_sha256": self.threshold_artifact_sha256,
            },
            "conditions": {
                "transport": API_TRANSPORT,
                "request_batch_size": 1,
                "warmup_count": self.warmup_count,
                "measured_count": self.measured_count,
                "image_payload_bytes": {
                    "minimum": min(self.image_payload_bytes),
                    "maximum": max(self.image_payload_bytes),
                    "mean": sum(self.image_payload_bytes) / self.measured_count,
                    "total": sum(self.image_payload_bytes),
                },
            },
            "latency_definition": API_LATENCY_DEFINITION,
            "disk_image_loading_included": False,
            "artifact_restore_included": False,
            "warmup_included": False,
            "external_network_round_trip_included": False,
            "threshold_applied": True,
            "inspection_persistence_included": True,
            "percentile_method": LINEAR_PERCENTILE_METHOD,
            "metrics": self.metrics.to_json_dict(),
            "created_at": self.created_at,
        }


# ADD 2026-08-20: Warmup HTTP request를 timing sample에서 제외하고 schema까지 검증한다.
def run_http_warmup(
    images: Sequence[PreparedImageUpload],
    *,
    warmup_count: int,
    send_request: RequestSender,
) -> int:
    """Run successful HTTP warmup requests outside the measured latency set."""
    if warmup_count < 0:
        raise ValueError("warmup_count must be non-negative.")
    if warmup_count > len(images):
        raise ValueError(f"warmup_count={warmup_count} exceeds available images {len(images)}.")

    for image in images[:warmup_count]:
        response = send_request(image)
        if response.status_code != 200:
            raise RuntimeError(f"Warmup request failed with HTTP {response.status_code}.")
        validate_prediction_payload(response.json())
    return warmup_count


# ADD 2026-08-20: Preloaded image별 completed FastAPI HTTP request latency와 failure를 측정한다.
def measure_http_requests(
    images: Sequence[PreparedImageUpload],
    *,
    measured_count: int,
    send_request: RequestSender,
    clock: Callable[[], float] = time.perf_counter,
) -> HttpBenchmarkMeasurements:
    """Measure one in-process HTTP attempt per image, including failure responses."""
    if measured_count <= 0:
        raise ValueError("measured_count must be positive.")
    if measured_count > len(images):
        raise ValueError(f"measured_count={measured_count} exceeds available images {len(images)}.")

    latencies_ms: list[float] = []
    successful_request_count = 0
    failed_request_count = 0
    for image in images[:measured_count]:
        response: HttpResponse | None = None
        try:
            started_at = clock()
            response = send_request(image)
            finished_at = clock()
        except Exception:
            finished_at = clock()
            failed_request_count += 1
        else:
            try:
                if response.status_code != 200:
                    raise ValueError(f"HTTP {response.status_code}")
                validate_prediction_payload(response.json())
            except (TypeError, ValueError):
                failed_request_count += 1
            else:
                successful_request_count += 1

        elapsed_ms = (finished_at - started_at) * 1000.0
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
            raise ValueError("Measured HTTP latency must be finite and positive.")
        latencies_ms.append(elapsed_ms)

    return HttpBenchmarkMeasurements(
        latencies_ms=tuple(latencies_ms),
        successful_request_count=successful_request_count,
        failed_request_count=failed_request_count,
    )


# ADD 2026-08-20: Per-request latency에서 distribution, request throughput과 error rate를 계산한다.
def summarize_http_measurements(
    measurements: HttpBenchmarkMeasurements,
) -> HttpBenchmarkMetrics:
    """Summarize all measured attempts, including timed error responses."""
    measured_count = len(measurements.latencies_ms)
    if measurements.successful_request_count + measurements.failed_request_count != measured_count:
        raise ValueError("HTTP outcome counts must equal measured latency count.")
    distribution = summarize_latency_distribution(measurements.latencies_ms)
    metrics = HttpBenchmarkMetrics(
        latency=distribution,
        requests_per_second=measured_count / distribution.total_timed_seconds,
        successful_request_count=measurements.successful_request_count,
        failed_request_count=measurements.failed_request_count,
        error_rate=measurements.failed_request_count / measured_count,
    )
    _validate_metrics(metrics, measured_count)
    return metrics


# ADD 2026-08-20: API benchmark JSON을 기존 결과 overwrite 없이 저장한다.
def write_api_benchmark_artifact(
    artifact: PatchCoreApiBenchmarkArtifact,
    output_path: Path,
) -> None:
    """Persist one validated API benchmark artifact without overwriting results."""
    if output_path.exists():
        raise FileExistsError(f"API benchmark output already exists: {output_path}")
    serialized = (
        json.dumps(
            artifact.to_json_dict(),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")


# ADD 2026-08-20: HTTP metric counts, range와 finite invariant를 함께 검증한다.
def _validate_metrics(metrics: HttpBenchmarkMetrics, measured_count: int) -> None:
    latency_values = asdict(metrics.latency).values()
    if any(not math.isfinite(value) or value <= 0.0 for value in latency_values):
        raise ValueError("API benchmark latency values must be finite and positive.")
    if not math.isfinite(metrics.requests_per_second) or metrics.requests_per_second <= 0.0:
        raise ValueError("API benchmark requests_per_second must be finite and positive.")
    if not math.isfinite(metrics.error_rate) or not 0.0 <= metrics.error_rate <= 1.0:
        raise ValueError("API benchmark error_rate must be finite and within [0, 1].")
    if metrics.successful_request_count < 0 or metrics.failed_request_count < 0:
        raise ValueError("API benchmark outcome counts must be non-negative.")
    if metrics.successful_request_count + metrics.failed_request_count != measured_count:
        raise ValueError("API benchmark outcome counts must equal measured_count.")
