"""Unit tests for FastAPI HTTP benchmark measurement and artifact contracts."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from services.api.benchmark import (
    PatchCoreApiBenchmarkArtifact,
    measure_http_requests,
    run_http_warmup,
    summarize_http_measurements,
    write_api_benchmark_artifact,
)
from services.api.tooling import PreparedImageUpload
from shared.benchmarking import LatencyDistribution, summarize_latency_distribution


class _Response:
    """Small completed HTTP response fake for measurement tests."""

    # ADD 2026-08-20: Configured HTTP status와 JSON payload를 보관한다.
    def __init__(self, status_code: int = 200, payload: object | None = None) -> None:
        self.status_code = status_code
        self.payload = _valid_payload() if payload is None else payload

    # ADD 2026-08-20: Configured response JSON payload를 반환한다.
    def json(self) -> object:
        return self.payload


# ADD 2026-08-20: Benchmark request에 사용할 in-memory PNG-like payload를 생성한다.
def _upload(name: str) -> PreparedImageUpload:
    return PreparedImageUpload(name, "image/png", b"png-bytes")


# ADD 2026-08-20: Strict threshold를 만족하는 valid API response payload를 생성한다.
def _valid_payload() -> dict[str, object]:
    return {
        "model_name": "patchcore",
        "category": "metal_nut",
        "is_anomaly": True,
        "anomaly_score": 1.0,
        "threshold": 0.5,
        "comparison_operator": ">",
    }


# ADD 2026-08-20: Deterministic clock 값을 순서대로 반환하는 callback을 생성한다.
def _clock(values: list[float]) -> Callable[[], float]:
    iterator = iter(values)
    return lambda: next(iterator)


# ADD 2026-08-20: Warmup 호출이 measured latency set과 clock에서 제외되는지 검증한다.
def test_warmup_is_excluded_from_measured_requests() -> None:
    uploads = [_upload("one.png"), _upload("two.png")]
    calls: list[str] = []

    # ADD 2026-08-20: Warmup과 measured request 순서를 기록한다.
    def send(upload: PreparedImageUpload) -> _Response:
        calls.append(upload.filename)
        return _Response()

    assert run_http_warmup(uploads, warmup_count=1, send_request=send) == 1
    measurements = measure_http_requests(
        uploads,
        measured_count=2,
        send_request=send,
        clock=_clock([0.0, 0.010, 0.010, 0.030]),
    )

    assert calls == ["one.png", "one.png", "two.png"]
    assert measurements.latencies_ms == pytest.approx((10.0, 20.0))


# ADD 2026-08-20: Linear latency statistics와 request throughput을 known values로 검증한다.
def test_http_latency_statistics_are_finite_and_deterministic() -> None:
    measurements = measure_http_requests(
        [_upload("one.png"), _upload("two.png")],
        measured_count=2,
        send_request=lambda _: _Response(),
        clock=_clock([0.0, 0.010, 0.010, 0.030]),
    )

    metrics = summarize_http_measurements(measurements)

    assert metrics.latency.p50_ms == pytest.approx(15.0)
    assert metrics.latency.p95_ms == pytest.approx(19.5)
    assert metrics.latency.p99_ms == pytest.approx(19.9)
    assert metrics.latency.mean_ms == pytest.approx(15.0)
    assert metrics.latency.total_timed_seconds == pytest.approx(0.03)
    assert metrics.requests_per_second == pytest.approx(2 / 0.03)
    assert metrics.error_rate == 0.0


# ADD 2026-08-20: HTTP error와 malformed success response를 failure/error-rate에 포함한다.
def test_request_failures_are_timed_and_counted() -> None:
    responses = iter([_Response(status_code=500), _Response(payload={"bad": "schema"})])
    measurements = measure_http_requests(
        [_upload("one.png"), _upload("two.png")],
        measured_count=2,
        send_request=lambda _: next(responses),
        clock=_clock([0.0, 0.010, 0.010, 0.030]),
    )

    metrics = summarize_http_measurements(measurements)

    assert metrics.successful_request_count == 0
    assert metrics.failed_request_count == 2
    assert metrics.error_rate == 1.0


# ADD 2026-08-20: NaN/Infinity와 non-positive latency를 distribution 단계에서 거부한다.
@pytest.mark.parametrize("latency", [float("nan"), float("inf"), 0.0, -1.0])
def test_non_finite_or_non_positive_latency_is_rejected(latency: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        summarize_latency_distribution([latency])


# ADD 2026-08-20: Fully populated API benchmark provenance schema fixture를 생성한다.
def _artifact() -> PatchCoreApiBenchmarkArtifact:
    return PatchCoreApiBenchmarkArtifact(
        model_name="patchcore",
        category="metal_nut",
        device="cuda:0",
        operating_system="Linux",
        machine="x86_64",
        accelerator_name="Tesla T4",
        cuda_version="13.0",
        python_version="3.12.13",
        torch_version="2.13.0+cu130",
        fastapi_version="0.141.1",
        starlette_version="1.6.0",
        manifest_sha256="a" * 64,
        artifact_metadata_sha256="b" * 64,
        model_sha256="c" * 64,
        threshold_artifact_sha256="d" * 64,
        image_payload_bytes=(100, 120),
        warmup_count=10,
        measured_count=2,
        metrics=summarize_http_measurements(
            measure_http_requests(
                [_upload("one.png"), _upload("two.png")],
                measured_count=2,
                send_request=lambda _: _Response(),
                clock=_clock([0.0, 0.010, 0.010, 0.030]),
            )
        ),
        created_at="2026-08-20T00:00:00+00:00",
    )


# ADD 2026-08-20: Output provenance와 timing boundary flags가 schema에 보존되는지 검증한다.
def test_api_benchmark_provenance_schema_and_output(tmp_path: Path) -> None:
    output_path = tmp_path / "benchmark.json"
    write_api_benchmark_artifact(_artifact(), output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["benchmark_name"] == "patchcore_fastapi_http_e2e"
    assert payload["provenance"]["threshold_artifact_sha256"] == "d" * 64
    assert payload["conditions"]["request_batch_size"] == 1
    assert payload["external_network_round_trip_included"] is False
    assert payload["disk_image_loading_included"] is False
    assert payload["artifact_restore_included"] is False
    assert payload["threshold_applied"] is True


# ADD 2026-08-20: 기존 API benchmark artifact overwrite를 명시적으로 거부한다.
def test_api_benchmark_output_rejects_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / "benchmark.json"
    write_api_benchmark_artifact(_artifact(), output_path)

    with pytest.raises(FileExistsError, match="already exists"):
        write_api_benchmark_artifact(_artifact(), output_path)


# ADD 2026-08-20: Invalid provenance digest가 artifact validation에서 거부되는지 검증한다.
def test_api_benchmark_rejects_invalid_provenance() -> None:
    with pytest.raises(ValueError, match="manifest_sha256"):
        replace(_artifact(), manifest_sha256="invalid").validate()


# ADD 2026-08-20: Directly constructed non-finite metrics도 artifact validation에서 거부한다.
def test_api_benchmark_rejects_non_finite_metrics() -> None:
    bad_latency = LatencyDistribution(1.0, 1.0, 1.0, float("nan"), 0.001)
    bad_metrics = replace(_artifact().metrics, latency=bad_latency)

    with pytest.raises(ValueError, match="finite and positive"):
        replace(_artifact(), metrics=bad_metrics).validate()
