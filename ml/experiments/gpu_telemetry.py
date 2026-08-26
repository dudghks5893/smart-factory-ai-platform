"""Lightweight CUDA and nvidia-smi telemetry for controlled training experiments."""

from __future__ import annotations

import math
import subprocess
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from statistics import fmean
from typing import Any, Self

import numpy as np
import torch


@dataclass(frozen=True)
class GpuTelemetrySample:
    """One best-effort device-wide nvidia-smi observation."""

    timestamp: str
    utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float
    power_draw_watts: float | None

    # ADD 2026-08-27: Sample numeric bounds와 optional power support를 검증한다.
    def validate(self) -> None:
        required = (
            self.utilization_percent,
            self.memory_used_mib,
            self.memory_total_mib,
        )
        if any(not math.isfinite(value) for value in required):
            raise ValueError("GPU telemetry values must be finite.")
        if not 0.0 <= self.utilization_percent <= 100.0:
            raise ValueError("GPU utilization must be in [0, 100].")
        if not 0.0 <= self.memory_used_mib <= self.memory_total_mib:
            raise ValueError("GPU memory usage must be within total device memory.")
        if self.memory_total_mib <= 0.0:
            raise ValueError("GPU total memory must be positive.")
        if self.power_draw_watts is not None and (
            not math.isfinite(self.power_draw_watts) or self.power_draw_watts < 0.0
        ):
            raise ValueError("GPU power draw must be finite and non-negative.")


type CommandRunner = Callable[[list[str], float], str]
type SampleProvider = Callable[[], GpuTelemetrySample | None]


# ADD 2026-08-27: Bounded subprocess로 nvidia-smi stdout을 가져온다.
def run_nvidia_smi_command(command: list[str], timeout_seconds: float) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return completed.stdout.strip()


# ADD 2026-08-27: nvidia-smi scalar를 finite float로 엄격히 변환한다.
def _parse_float(value: str) -> float:
    parsed = float(value.strip())
    if not math.isfinite(parsed):
        raise ValueError("nvidia-smi value must be finite.")
    return parsed


# ADD 2026-08-27: Required device fields와 optional power를 독립적으로 best-effort sampling한다.
def sample_nvidia_smi(
    *,
    device_index: int = 0,
    timeout_seconds: float = 2.0,
    command_runner: CommandRunner = run_nvidia_smi_command,
    timestamp: str | None = None,
) -> GpuTelemetrySample | None:
    base_command = [
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        values = command_runner(base_command, timeout_seconds).split(",")
        if len(values) != 3:
            return None
        utilization, memory_used, memory_total = (_parse_float(value) for value in values)
    except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError):
        return None

    power: float | None = None
    power_command = [
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw_power = command_runner(power_command, timeout_seconds)
        power = _parse_float(raw_power)
    except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError):
        power = None
    sample = GpuTelemetrySample(
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        utilization_percent=utilization,
        memory_used_mib=memory_used,
        memory_total_mib=memory_total,
        power_draw_watts=power,
    )
    try:
        sample.validate()
    except ValueError:
        return None
    return sample


# ADD 2026-08-27: NVIDIA driver version을 best-effort environment evidence로 조회한다.
def query_nvidia_driver_version(
    *,
    timeout_seconds: float = 2.0,
    command_runner: CommandRunner = run_nvidia_smi_command,
) -> str | None:
    command = [
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        value = command_runner(command, timeout_seconds).splitlines()[0].strip()
    except (FileNotFoundError, IndexError, OSError, subprocess.SubprocessError):
        return None
    return value or None


# ADD 2026-08-27: Sample series를 empty-safe utilization/memory percentile로 집계한다.
def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"sample_count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "sample_count": len(values),
        "mean": float(fmean(values)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


# ADD 2026-08-27: Device-wide samples를 utilization/memory/power distributions로 집계한다.
def aggregate_gpu_samples(
    samples: list[GpuTelemetrySample],
    *,
    attempted_sample_count: int | None = None,
) -> dict[str, Any]:
    for sample in samples:
        sample.validate()
    power_values = [
        sample.power_draw_watts for sample in samples if sample.power_draw_watts is not None
    ]
    power_summary: dict[str, float | int | None] = {
        "sample_count": len(power_values),
        "mean": float(fmean(power_values)) if power_values else None,
        "max": max(power_values) if power_values else None,
    }
    attempted = attempted_sample_count if attempted_sample_count is not None else len(samples)
    return {
        "source": "nvidia-smi_device_wide_sampling",
        "available": bool(samples),
        "attempted_sample_count": attempted,
        "valid_sample_count": len(samples),
        "invalid_or_unavailable_sample_count": attempted - len(samples),
        "utilization_percent": _distribution([sample.utilization_percent for sample in samples]),
        "memory_used_mib": _distribution([sample.memory_used_mib for sample in samples]),
        "memory_total_mib": samples[0].memory_total_mib if samples else None,
        "power_draw_watts": power_summary,
        "samples": [asdict(sample) for sample in samples],
    }


class GpuTelemetrySampler:
    """Lifecycle-managed background sampler that never owns training errors."""

    # ADD 2026-08-27: Bounded interval과 injectable provider로 sampler state를 초기화한다.
    def __init__(
        self,
        *,
        sample_interval_seconds: float,
        sample_provider: SampleProvider,
    ) -> None:
        if sample_interval_seconds <= 0.0:
            raise ValueError("GPU telemetry sample interval must be positive.")
        self._sample_interval_seconds = sample_interval_seconds
        self._sample_provider = sample_provider
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[GpuTelemetrySample] = []
        self._attempted_sample_count = 0
        self._lock = threading.Lock()
        self._cleanup_error: str | None = None

    # ADD 2026-08-27: Exactly one daemon sampling thread를 시작한다.
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU telemetry sampler is already started.")
        self._stop_event.clear()
        self._cleanup_error = None
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="gpu-telemetry-sampler",
            daemon=True,
        )
        self._thread.start()

    # ADD 2026-08-27: Training 성공/실패와 무관하게 sampling thread를 bounded join한다.
    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout=self._sample_interval_seconds + 5.0)
        if thread.is_alive():
            self._cleanup_error = "GPU telemetry sampler did not stop within the timeout."
            return
        self._thread = None

    # ADD 2026-08-27: Provider 오류를 training으로 전파하지 않고 다음 interval까지 대기한다.
    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            sample: GpuTelemetrySample | None = None
            try:
                sample = self._sample_provider()
            except Exception:  # noqa: BLE001 - telemetry must remain best effort
                sample = None
            with self._lock:
                self._attempted_sample_count += 1
                if sample is not None:
                    try:
                        sample.validate()
                    except ValueError:
                        pass
                    else:
                        self._samples.append(sample)
            self._stop_event.wait(self._sample_interval_seconds)

    # ADD 2026-08-27: Current immutable sample snapshot과 invalid attempt count를 집계한다.
    def summary(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
            attempted = self._attempted_sample_count
        summary = aggregate_gpu_samples(samples, attempted_sample_count=attempted)
        summary["cleanup_error"] = self._cleanup_error
        return summary

    # ADD 2026-08-27: With-block 진입 시 sampling을 시작한다.
    def __enter__(self) -> Self:
        self.start()
        return self

    # ADD 2026-08-27: Training exception을 swallow하지 않고 sampler만 정리한다.
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()


# ADD 2026-08-27: Training 직전 framework-owned CUDA peak counters를 reset한다.
def reset_torch_cuda_peaks(device_index: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.reset_peak_memory_stats(device_index)
    except (AssertionError, RuntimeError):
        return False
    return True


# ADD 2026-08-27: PyTorch allocator peak과 device identity를 device-wide samples와 분리해 기록한다.
def collect_torch_cuda_metrics(device_index: int = 0) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "source": "pytorch_cuda_allocator",
            "available": False,
            "device_name": None,
            "total_device_memory_bytes": None,
            "peak_memory_allocated_bytes": None,
            "peak_memory_reserved_bytes": None,
            "collection_error": None,
        }
    try:
        properties = torch.cuda.get_device_properties(device_index)
        return {
            "source": "pytorch_cuda_allocator",
            "available": True,
            "device_name": torch.cuda.get_device_name(device_index),
            "total_device_memory_bytes": int(properties.total_memory),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
            "collection_error": None,
        }
    except (AssertionError, RuntimeError) as exc:
        return {
            "source": "pytorch_cuda_allocator",
            "available": False,
            "device_name": None,
            "total_device_memory_bytes": None,
            "peak_memory_allocated_bytes": None,
            "peak_memory_reserved_bytes": None,
            "collection_error": f"{type(exc).__name__}: {exc}",
        }
