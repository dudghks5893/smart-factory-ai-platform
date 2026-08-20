"""PatchCore inference benchmark measurement and artifact contracts."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from shared.benchmarking import (
    LINEAR_PERCENTILE_METHOD,
    summarize_latency_distribution,
)
from shared.hashing import is_sha256_digest

BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_NAME = "patchcore_inference"
LATENCY_DEFINITION = (
    "after image batch loading: preprocessing -> device transfer -> PatchCore inference "
    "-> prediction materialization and accelerator synchronization"
)
PERCENTILE_METHOD = LINEAR_PERCENTILE_METHOD

type BatchRunner = Callable[[Tensor], object]
type Synchronizer = Callable[[], None]


@dataclass(frozen=True)
class LatencySummary:
    """Finite latency distribution and throughput derived from measured batches."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    total_timed_seconds: float
    throughput_images_per_second: float

    # ADD 2026-08-19: Latency summary를 stable JSON mapping으로 변환한다.
    def to_json_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkMeasurements:
    """Raw measured batch latencies before percentile summarization."""

    latencies_ms: tuple[float, ...]
    measured_sample_count: int
    measured_batch_count: int


@dataclass(frozen=True)
class CudaPeakMemory:
    """CUDA allocator peaks, or an explicit unsupported result for CPU and MPS."""

    supported: bool
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None

    # ADD 2026-08-19: CUDA peak memory를 byte와 MiB 단위 JSON mapping으로 변환한다.
    def to_json_dict(self) -> dict[str, bool | int | float | None]:
        return {
            "supported": self.supported,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_allocated_megabytes": _bytes_to_megabytes(self.peak_allocated_bytes),
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "peak_reserved_megabytes": _bytes_to_megabytes(self.peak_reserved_bytes),
        }


@dataclass(frozen=True)
class PatchCoreBenchmarkArtifact:
    """Schema-versioned offline PatchCore inference benchmark result."""

    category: str
    device: str
    operating_system: str
    machine: str
    accelerator_name: str
    cuda_version: str | None
    python_version: str
    torch_version: str
    torchvision_version: str
    anomalib_version: str
    manifest_sha256: str
    artifact_metadata_sha256: str
    model_sha256: str
    backbone: str
    layers: tuple[str, ...]
    preprocessing: dict[str, object]
    batch_size: int
    warmup_count: int
    measured_sample_count: int
    measured_batch_count: int
    latency: LatencySummary
    model_file_size_bytes: int
    cuda_peak_memory: CudaPeakMemory
    created_at: str
    schema_version: int = BENCHMARK_SCHEMA_VERSION

    # ADD 2026-08-19: Benchmark payload의 schema, provenance와 finite metric을 검증한다.
    def validate(self) -> None:
        if self.schema_version != BENCHMARK_SCHEMA_VERSION:
            raise ValueError(f"Unsupported benchmark schema_version: {self.schema_version}.")
        if not self.category or not self.device or not self.created_at:
            raise ValueError("Benchmark category, device, and created_at must not be empty.")
        for field, value in (
            ("operating_system", self.operating_system),
            ("machine", self.machine),
            ("accelerator_name", self.accelerator_name),
            ("python_version", self.python_version),
            ("torch_version", self.torch_version),
            ("torchvision_version", self.torchvision_version),
            ("anomalib_version", self.anomalib_version),
            ("backbone", self.backbone),
        ):
            if not value:
                raise ValueError(f"Benchmark {field} must not be empty.")
        for field, digest in (
            ("manifest_sha256", self.manifest_sha256),
            ("artifact_metadata_sha256", self.artifact_metadata_sha256),
            ("model_sha256", self.model_sha256),
        ):
            _validate_sha256(digest, field)
        if not self.layers or not all(self.layers):
            raise ValueError("Benchmark layers must contain non-empty names.")
        if not self.preprocessing:
            raise ValueError("Benchmark preprocessing must not be empty.")
        _validate_finite_json_values(self.preprocessing, "preprocessing")
        validate_benchmark_parameters(
            batch_size=self.batch_size,
            warmup_count=self.warmup_count,
            measured_count=self.measured_sample_count,
            num_workers=0,
        )
        if self.measured_batch_count <= 0:
            raise ValueError("measured_batch_count must be positive.")
        if self.model_file_size_bytes <= 0:
            raise ValueError("model_file_size_bytes must be positive.")
        _validate_latency_summary(self.latency)
        _validate_cuda_peak_memory(self.cuda_peak_memory)

    # ADD 2026-08-19: Benchmark result를 inspectable JSON schema로 변환한다.
    def to_json_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "benchmark_name": BENCHMARK_NAME,
            "category": self.category,
            "device": self.device,
            "runtime": {
                "operating_system": self.operating_system,
                "machine": self.machine,
                "accelerator_name": self.accelerator_name,
                "cuda_version": self.cuda_version,
                "python_version": self.python_version,
                "torch_version": self.torch_version,
                "torchvision_version": self.torchvision_version,
                "anomalib_version": self.anomalib_version,
            },
            "provenance": {
                "manifest_sha256": self.manifest_sha256,
                "artifact_metadata_sha256": self.artifact_metadata_sha256,
                "model_sha256": self.model_sha256,
            },
            "model": {
                "backbone": self.backbone,
                "layers": list(self.layers),
                "preprocessing": self.preprocessing,
            },
            "batch_size": self.batch_size,
            "warmup_count": self.warmup_count,
            "warmup_unit": "batch",
            "measured_count": self.measured_sample_count,
            "measured_batch_count": self.measured_batch_count,
            "latency_definition": LATENCY_DEFINITION,
            "disk_image_loading_included": False,
            "warmup_included": False,
            "threshold_applied": False,
            "percentile_method": PERCENTILE_METHOD,
            "latency_ms": self.latency.to_json_dict(),
            "throughput_images_per_second": self.latency.throughput_images_per_second,
            "model_file_size_bytes": self.model_file_size_bytes,
            "model_file_size_megabytes": _bytes_to_megabytes(self.model_file_size_bytes),
            "cuda_peak_memory": self.cuda_peak_memory.to_json_dict(),
            "created_at": self.created_at,
        }


# ADD 2026-08-19: Benchmark batch, warmup, sample count와 worker 설정을 검증한다.
def validate_benchmark_parameters(
    *,
    batch_size: int,
    warmup_count: int,
    measured_count: int | None,
    num_workers: int,
) -> None:
    """Validate benchmark controls before artifact loading or inference."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if warmup_count < 0:
        raise ValueError("warmup_count must be non-negative.")
    if measured_count is not None and measured_count <= 0:
        raise ValueError("measured_count must be positive when provided.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")


# ADD 2026-08-19: Warmup batch를 timing sample에 포함하지 않고 실행한다.
def run_warmup(
    batches: Iterable[Tensor],
    *,
    warmup_count: int,
    run_batch: BatchRunner,
    synchronize: Synchronizer,
) -> int:
    """Run deterministic warmup invocations outside the measured latency set."""
    if warmup_count < 0:
        raise ValueError("warmup_count must be non-negative.")
    if warmup_count == 0:
        return 0

    completed = 0
    for images in batches:
        _validate_image_batch(images)
        run_batch(images)
        synchronize()
        completed += 1
        if completed == warmup_count:
            break

    if completed != warmup_count:
        raise ValueError(
            f"Warmup requested {warmup_count} batches but only {completed} were available."
        )
    return completed


# ADD 2026-08-19: Image loading 이후의 batch inference latency를 accelerator 동기화와 함께 측정한다.
def measure_batches(
    batches: Iterable[Tensor],
    *,
    measured_count: int,
    run_batch: BatchRunner,
    synchronize: Synchronizer,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchmarkMeasurements:
    """Measure preprocessing-through-synchronization latency for a fixed image count."""
    if measured_count <= 0:
        raise ValueError("measured_count must be positive.")

    latencies_ms: list[float] = []
    measured_samples = 0
    for images in batches:
        _validate_image_batch(images)
        remaining = measured_count - measured_samples
        measured_images = images[:remaining]

        # 이전 accelerator 작업을 비운 뒤 현재 batch의 전체 online serving 경계를 측정한다.
        synchronize()
        started_at = clock()
        run_batch(measured_images)
        synchronize()
        finished_at = clock()

        elapsed_ms = (finished_at - started_at) * 1000.0
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
            raise ValueError("Measured latency must be finite and positive.")
        latencies_ms.append(elapsed_ms)
        measured_samples += int(measured_images.shape[0])
        if measured_samples == measured_count:
            break

    if measured_samples != measured_count:
        raise ValueError(
            f"Benchmark requested {measured_count} images but measured {measured_samples}."
        )
    return BenchmarkMeasurements(
        latencies_ms=tuple(latencies_ms),
        measured_sample_count=measured_samples,
        measured_batch_count=len(latencies_ms),
    )


# ADD 2026-08-19: Linear percentile, mean과 image throughput을 measured latency에서 계산한다.
# MODIFY 2026-08-20: Model-local percentile 계산 → shared latency distribution을 재사용한다.
def summarize_latencies(
    latencies_ms: tuple[float, ...] | list[float],
    *,
    measured_sample_count: int,
) -> LatencySummary:
    """Summarize positive finite batch latencies using linear interpolation."""
    if measured_sample_count <= 0:
        raise ValueError("measured_sample_count must be positive.")
    distribution = summarize_latency_distribution(latencies_ms)
    summary = LatencySummary(
        p50_ms=distribution.p50_ms,
        p95_ms=distribution.p95_ms,
        p99_ms=distribution.p99_ms,
        mean_ms=distribution.mean_ms,
        total_timed_seconds=distribution.total_timed_seconds,
        throughput_images_per_second=(measured_sample_count / distribution.total_timed_seconds),
    )
    _validate_latency_summary(summary)
    return summary


# ADD 2026-08-19: CUDA benchmark 구간 직전에 allocator peak 통계를 초기화한다.
def reset_cuda_peak_memory(device: torch.device) -> None:
    """Reset CUDA allocator peaks; CPU and MPS are explicit no-ops."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


# ADD 2026-08-19: CUDA allocator peak 또는 non-CUDA unsupported 값을 반환한다.
def read_cuda_peak_memory(device: torch.device) -> CudaPeakMemory:
    """Read CUDA allocated/reserved peaks without requiring CUDA in unit tests."""
    if device.type != "cuda":
        return CudaPeakMemory(
            supported=False,
            peak_allocated_bytes=None,
            peak_reserved_bytes=None,
        )
    memory = CudaPeakMemory(
        supported=True,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
    )
    _validate_cuda_peak_memory(memory)
    return memory


# ADD 2026-08-19: CPU, MPS와 CUDA의 비동기 작업을 device별로 완료시킨다.
def synchronize_device(device: torch.device) -> None:
    """Synchronize accelerator work while keeping CPU execution portable."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


# ADD 2026-08-19: Benchmark JSON을 기존 파일 overwrite 없이 저장한다.
def write_benchmark_artifact(
    artifact: PatchCoreBenchmarkArtifact,
    output_path: Path,
) -> None:
    """Persist one validated benchmark artifact without overwriting results."""
    if output_path.exists():
        raise FileExistsError(f"Benchmark output already exists: {output_path}")
    payload = artifact.to_json_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output_path.write_text(serialized, encoding="utf-8")


# ADD 2026-08-19: Tensor가 non-empty image batch인지 검증한다.
def _validate_image_batch(images: Tensor) -> None:
    if not isinstance(images, Tensor) or images.ndim < 1 or images.shape[0] <= 0:
        raise ValueError("Benchmark batches must be non-empty tensors.")


# ADD 2026-08-19: Latency summary의 모든 값이 finite positive인지 검증한다.
def _validate_latency_summary(summary: LatencySummary) -> None:
    values = asdict(summary).values()
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Benchmark latency and throughput values must be finite and positive.")


# ADD 2026-08-19: CUDA memory 지원 여부와 byte 값의 일관성을 검증한다.
def _validate_cuda_peak_memory(memory: CudaPeakMemory) -> None:
    values = (memory.peak_allocated_bytes, memory.peak_reserved_bytes)
    if memory.supported:
        if any(value is None or value < 0 for value in values):
            raise ValueError("Supported CUDA peak memory values must be non-negative integers.")
    elif any(value is not None for value in values):
        raise ValueError("Unsupported CUDA peak memory values must be null.")


# ADD 2026-08-19: SHA-256 provenance field의 hex digest 형식을 검증한다.
# MODIFY 2026-08-20: Local hex parsing → shared SHA-256 validator 재사용으로 변경한다.
def _validate_sha256(value: str, field: str) -> None:
    if not is_sha256_digest(value):
        raise ValueError(f"Benchmark {field} must be a SHA-256 hex digest.")


# ADD 2026-08-19: Optional byte 값을 binary megabyte 단위로 변환한다.
def _bytes_to_megabytes(value: int | None) -> float | None:
    if value is None:
        return None
    return value / (1024 * 1024)


# ADD 2026-08-19: Nested JSON metadata에서 NaN과 Infinity를 재귀적으로 거부한다.
def _validate_finite_json_values(value: object, field: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Benchmark {field} must not contain NaN or Infinity.")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json_values(item, f"{field}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_json_values(item, f"{field}[{index}]")
