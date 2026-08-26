"""Tests for lightweight best-effort GPU telemetry."""

from __future__ import annotations

import json
import subprocess
import threading

import pytest

from ml.experiments.gpu_telemetry import (
    GpuTelemetrySample,
    GpuTelemetrySampler,
    aggregate_gpu_samples,
    collect_torch_cuda_metrics,
    query_nvidia_driver_version,
    sample_nvidia_smi,
)


# ADD 2026-08-27: Deterministic device-wide sample fixture를 생성한다.
def _sample(
    utilization: float,
    memory: float,
    *,
    power: float | None,
) -> GpuTelemetrySample:
    return GpuTelemetrySample(
        timestamp="2026-08-27T00:00:00+00:00",
        utilization_percent=utilization,
        memory_used_mib=memory,
        memory_total_mib=16384.0,
        power_draw_watts=power,
    )


# ADD 2026-08-27: Fake nvidia-smi fields와 optional power absence를 parsing한다.
def test_nvidia_smi_sample_handles_optional_power() -> None:
    # ADD 2026-08-27: Power query만 실패하는 fake command runner를 제공한다.
    def runner(command: list[str], timeout: float) -> str:
        assert timeout == 2.0
        if "--query-gpu=power.draw" in command:
            raise subprocess.CalledProcessError(1, command)
        return "75, 4096, 16384"

    sample = sample_nvidia_smi(command_runner=runner, timestamp="timestamp")
    assert sample == GpuTelemetrySample("timestamp", 75.0, 4096.0, 16384.0, None)


# ADD 2026-08-27: Missing/malformed telemetry가 training-fatal이 아님을 검증한다.
def test_nvidia_smi_unavailable_and_malformed_return_none() -> None:
    # ADD 2026-08-27: nvidia-smi 미설치 error를 재현한다.
    def missing(command: list[str], timeout: float) -> str:
        raise FileNotFoundError

    # ADD 2026-08-27: Required numeric field parsing 실패를 재현한다.
    def malformed(command: list[str], timeout: float) -> str:
        return "not,a,number"

    assert sample_nvidia_smi(command_runner=missing) is None
    assert sample_nvidia_smi(command_runner=malformed) is None
    assert query_nvidia_driver_version(command_runner=missing) is None


# ADD 2026-08-27: Utilization percentile, memory max와 supported power만 집계한다.
def test_gpu_sample_aggregation() -> None:
    summary = aggregate_gpu_samples(
        [
            _sample(0.0, 1000.0, power=None),
            _sample(50.0, 5000.0, power=60.0),
            _sample(100.0, 9000.0, power=100.0),
        ],
        attempted_sample_count=4,
    )
    assert summary["valid_sample_count"] == 3
    assert summary["invalid_or_unavailable_sample_count"] == 1
    assert json.loads(json.dumps(summary, allow_nan=False))["valid_sample_count"] == 3
    assert summary["utilization_percent"]["mean"] == 50.0
    assert summary["utilization_percent"]["p50"] == 50.0
    assert summary["utilization_percent"]["p95"] == pytest.approx(95.0)
    assert summary["utilization_percent"]["max"] == 100.0
    assert summary["memory_used_mib"]["max"] == 9000.0
    assert summary["power_draw_watts"] == {
        "sample_count": 2,
        "mean": 80.0,
        "max": 100.0,
    }


# ADD 2026-08-27: Sampler가 invalid provider result를 건너뛰고 thread를 clean stop한다.
def test_sampler_cleanup_and_malformed_sample_handling() -> None:
    called = threading.Event()

    # ADD 2026-08-27: Validation-invalid utilization sample을 반환한다.
    def provider() -> GpuTelemetrySample:
        called.set()
        return _sample(120.0, 1000.0, power=None)

    sampler = GpuTelemetrySampler(sample_interval_seconds=0.01, sample_provider=provider)
    sampler.start()
    assert called.wait(timeout=1.0)
    sampler.stop()
    summary = sampler.summary()
    assert summary["attempted_sample_count"] >= 1
    assert summary["valid_sample_count"] == 0
    assert summary["cleanup_error"] is None
    sampler.start()
    sampler.stop()


# ADD 2026-08-27: With-block cleanup이 training exception을 swallow하지 않는지 검증한다.
def test_sampler_context_propagates_training_error() -> None:
    called = threading.Event()

    # ADD 2026-08-27: Unavailable sample을 반환해 training error와 분리한다.
    def provider() -> GpuTelemetrySample | None:
        called.set()
        return None

    sampler = GpuTelemetrySampler(sample_interval_seconds=0.01, sample_provider=provider)
    with pytest.raises(RuntimeError, match="training failed"):
        with sampler:
            assert called.wait(timeout=1.0)
            raise RuntimeError("training failed")
    sampler.start()
    sampler.stop()


# ADD 2026-08-27: Non-CUDA CI에서도 PyTorch allocator schema가 stable한지 검증한다.
def test_torch_cuda_metrics_have_explicit_availability() -> None:
    metrics = collect_torch_cuda_metrics()
    assert metrics["source"] == "pytorch_cuda_allocator"
    assert isinstance(metrics["available"], bool)
