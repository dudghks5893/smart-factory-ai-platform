"""Benchmark offline PatchCore inference from a portable model artifact."""

from __future__ import annotations

import argparse
import math
import platform
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import anomalib
import torch
import torchvision  # type: ignore[import-untyped]
from torch import Tensor
from torch.utils.data import DataLoader

from ml.datasets.dataset import MVTecManifestDataset, MVTecSample
from ml.evaluation.benchmark import (
    PatchCoreBenchmarkArtifact,
    measure_batches,
    read_cuda_peak_memory,
    reset_cuda_peak_memory,
    run_warmup,
    summarize_latencies,
    synchronize_device,
    validate_benchmark_parameters,
    write_benchmark_artifact,
)
from ml.training.batches import require_batch_tensor
from ml.training.device import SUPPORTED_DEVICES, resolve_device
from ml.training.patchcore import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    PatchCoreAdapter,
    read_artifact_metadata,
)
from ml.training.preprocessing import PatchCorePreprocessor
from shared.hashing import sha256_file

BENCHMARK_FILENAME = "benchmark.json"
DEFAULT_BENCHMARK_ROOT = Path("outputs/benchmarks/patchcore")
DEFAULT_BATCH_SIZE = 1
DEFAULT_WARMUP_COUNT = 10


@dataclass(frozen=True)
class BenchmarkOutputSummary:
    """Summary of one persisted PatchCore inference benchmark."""

    output_dir: Path
    benchmark_path: Path
    device: str
    measured_sample_count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_images_per_second: float


# ADD 2026-08-19: Portable artifact와 실제 MVTec test image로 offline inference를 측정한다.
def benchmark_patchcore(
    *,
    dataset_root: Path,
    manifest_path: Path,
    artifact_dir: Path,
    output_dir: Path,
    requested_device: str = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
    warmup_count: int = DEFAULT_WARMUP_COUNT,
    measured_count: int | None = None,
    num_workers: int = 0,
) -> BenchmarkOutputSummary:
    """Benchmark preprocessing-through-synchronization without disk image loading."""
    validate_benchmark_parameters(
        batch_size=batch_size,
        warmup_count=warmup_count,
        measured_count=measured_count,
        num_workers=num_workers,
    )
    if output_dir.exists():
        raise FileExistsError(f"Benchmark output directory already exists: {output_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Benchmark manifest not found: {manifest_path}")

    # Model restore 전에 device, artifact metadata와 source manifest provenance를 검증한다.
    device = resolve_device(requested_device)
    artifact_metadata = read_artifact_metadata(artifact_dir)
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != artifact_metadata.manifest_sha256:
        raise ValueError("Benchmark manifest SHA-256 does not match the PatchCore artifact.")

    # 실제 MVTec test image를 manifest 순서로 로드하고 benchmark sample 범위를 확정한다.
    dataset = MVTecManifestDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="test",
        load_masks=False,
    )
    if any(record.category != artifact_metadata.category for record in dataset.records):
        raise ValueError("Benchmark manifest category does not match the PatchCore artifact.")
    sample_count = len(dataset) if measured_count is None else measured_count
    if sample_count > len(dataset):
        raise ValueError(f"measured_count={sample_count} exceeds test sample count {len(dataset)}.")
    available_batches = math.ceil(len(dataset) / batch_size)
    if warmup_count > available_batches:
        raise ValueError(
            f"warmup_count={warmup_count} exceeds available test batches {available_batches}."
        )

    # 검증된 metadata로 pretrained download 없이 artifact를 한 번만 복원한다.
    adapter, _ = PatchCoreAdapter.load_artifact(
        artifact_dir,
        device,
        metadata=artifact_metadata,
    )
    preprocessor = PatchCorePreprocessor(artifact_metadata.preprocessing)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    # 같은 adapter를 사용해 warmup을 실행하되 latency와 CUDA peak 통계에서는 제외한다.
    run_batch = _inference_runner(adapter, preprocessor)
    run_warmup(
        _image_batches(loader),
        warmup_count=warmup_count,
        run_batch=run_batch,
        synchronize=lambda: synchronize_device(device),
    )
    synchronize_device(device)
    reset_cuda_peak_memory(device)

    # 각 image batch가 disk에서 반환된 뒤 online serving 경계만 측정한다.
    measurements = measure_batches(
        _image_batches(loader),
        measured_count=sample_count,
        run_batch=run_batch,
        synchronize=lambda: synchronize_device(device),
    )
    latency = summarize_latencies(
        measurements.latencies_ms,
        measured_sample_count=measurements.measured_sample_count,
    )
    cuda_peak_memory = read_cuda_peak_memory(device)

    # Runtime, model/data hash와 측정 정책을 하나의 versioned benchmark artifact로 저장한다.
    model_path = artifact_dir / MODEL_FILENAME
    benchmark = PatchCoreBenchmarkArtifact(
        category=artifact_metadata.category,
        device=str(device),
        operating_system=platform.platform(),
        machine=platform.machine(),
        accelerator_name=_accelerator_name(device),
        cuda_version=torch.version.cuda,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
        anomalib_version=anomalib.__version__,
        manifest_sha256=manifest_sha256,
        artifact_metadata_sha256=sha256_file(artifact_dir / METADATA_FILENAME),
        model_sha256=sha256_file(model_path),
        backbone=artifact_metadata.backbone,
        layers=artifact_metadata.layers,
        preprocessing={
            "resize_size": list(artifact_metadata.preprocessing.resize_size),
            "center_crop_size": list(artifact_metadata.preprocessing.center_crop_size),
            "image_mean": list(artifact_metadata.preprocessing.image_mean),
            "image_std": list(artifact_metadata.preprocessing.image_std),
        },
        batch_size=batch_size,
        warmup_count=warmup_count,
        measured_sample_count=measurements.measured_sample_count,
        measured_batch_count=measurements.measured_batch_count,
        latency=latency,
        model_file_size_bytes=model_path.stat().st_size,
        cuda_peak_memory=cuda_peak_memory,
        created_at=datetime.now(UTC).isoformat(),
    )
    benchmark.validate()
    output_dir.mkdir(parents=True, exist_ok=False)
    benchmark_path = output_dir / BENCHMARK_FILENAME
    write_benchmark_artifact(benchmark, benchmark_path)

    return BenchmarkOutputSummary(
        output_dir=output_dir,
        benchmark_path=benchmark_path,
        device=str(device),
        measured_sample_count=measurements.measured_sample_count,
        p50_ms=latency.p50_ms,
        p95_ms=latency.p95_ms,
        p99_ms=latency.p99_ms,
        mean_ms=latency.mean_ms,
        throughput_images_per_second=latency.throughput_images_per_second,
    )


# ADD 2026-08-19: DataLoader가 image를 반환한 뒤 benchmark timing에 전달한다.
def _image_batches(loader: Iterable[MVTecSample]) -> Iterator[Tensor]:
    for batch in loader:
        if not isinstance(batch, dict):
            raise TypeError("Each benchmark batch must be a mapping.")
        yield require_batch_tensor(cast(dict[object, object], batch), "image")


# ADD 2026-08-19: 동일 adapter의 inference_mode prediction 호출을 benchmark callback으로 구성한다.
def _inference_runner(
    adapter: PatchCoreAdapter,
    preprocessor: PatchCorePreprocessor,
) -> Callable[[Tensor], object]:
    # ADD 2026-08-19: 전달된 batch를 기존 adapter contract로 추론한다.
    def run(images: Tensor) -> object:
        with torch.inference_mode():
            return adapter.predict(images, preprocessor)

    return run


# ADD 2026-08-19: 선택된 accelerator의 inspectable runtime 이름을 반환한다.
def _accelerator_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return platform.processor() or "CPU"


# ADD 2026-08-19: Benchmark CLI 입력 인자를 정의하고 파싱한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PatchCore inference latency.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/mvtec_ad"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/interim/manifests/mvtec_ad_metal_nut.csv"),
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default="auto")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--warmup-count", type=int, default=DEFAULT_WARMUP_COUNT)
    parser.add_argument("--measured-count", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


# ADD 2026-08-19: CLI benchmark 흐름을 실행하고 핵심 latency 결과를 출력한다.
def main() -> int:
    args = _parse_args()
    output_dir = args.output_root / args.output_id

    # CLI 옵션으로 artifact benchmark를 실행하고 inspectable JSON을 저장한다.
    summary = benchmark_patchcore(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        artifact_dir=args.artifact_dir,
        output_dir=output_dir,
        requested_device=args.device,
        batch_size=args.batch_size,
        warmup_count=args.warmup_count,
        measured_count=args.measured_count,
        num_workers=args.num_workers,
    )
    print("PatchCore inference benchmark: PASS")
    print(f"Device: {summary.device}")
    print(f"Measured samples: {summary.measured_sample_count}")
    print(f"Latency p50/p95/p99: {summary.p50_ms:.3f}/{summary.p95_ms:.3f}/{summary.p99_ms:.3f} ms")
    print(f"Mean latency: {summary.mean_ms:.3f} ms")
    print(f"Throughput: {summary.throughput_images_per_second:.3f} images/second")
    print(f"Benchmark: {summary.benchmark_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
