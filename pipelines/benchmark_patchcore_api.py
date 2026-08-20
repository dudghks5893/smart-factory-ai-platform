"""Benchmark real PatchCore inference through the FastAPI HTTP application boundary."""

from __future__ import annotations

import argparse
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from ml.datasets.manifest import ManifestRecord, read_manifest_csv
from ml.training.device import SUPPORTED_DEVICES
from services.api.app import RuntimeLoader, create_app
from services.api.benchmark import (
    HttpResponse,
    PatchCoreApiBenchmarkArtifact,
    measure_http_requests,
    run_http_warmup,
    summarize_http_measurements,
    write_api_benchmark_artifact,
)
from services.api.config import (
    DEFAULT_MAX_UPLOAD_BYTES,
    ServingSettings,
    required_database_url,
)
from services.api.schemas import HealthResponse, ReadinessResponse
from services.api.tooling import PreparedImageUpload, prepare_image_upload
from services.inference.runtime import load_patchcore_runtime, require_serving_provenance
from shared.hashing import sha256_file

BENCHMARK_FILENAME = "benchmark.json"
DEFAULT_API_BENCHMARK_ROOT = Path("outputs/benchmarks/api")
DEFAULT_WARMUP_COUNT = 10


@dataclass(frozen=True)
class ApiBenchmarkOutputSummary:
    """Summary of one persisted PatchCore FastAPI HTTP benchmark."""

    benchmark_path: Path
    device: str
    measured_count: int
    successful_request_count: int
    failed_request_count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    requests_per_second: float


# ADD 2026-08-20: Manifest test image로 FastAPI application HTTP E2E latency를 측정한다.
# MODIFY 2026-08-20: Required inspection persistence를 application benchmark lifecycle에 포함한다.
def benchmark_patchcore_api(
    *,
    dataset_root: Path,
    manifest_path: Path,
    artifact_dir: Path,
    thresholds_path: Path,
    database_url: str,
    output_dir: Path,
    requested_device: str = "auto",
    warmup_count: int = DEFAULT_WARMUP_COUNT,
    measured_count: int | None = None,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    runtime_loader: RuntimeLoader = load_patchcore_runtime,
) -> ApiBenchmarkOutputSummary:
    """Benchmark preloaded test images through one app, runtime, and request at a time."""
    if output_dir.exists():
        raise FileExistsError(f"API benchmark output directory already exists: {output_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"API benchmark manifest not found: {manifest_path}")
    if warmup_count < 0:
        raise ValueError("warmup_count must be non-negative.")
    if measured_count is not None and measured_count <= 0:
        raise ValueError("measured_count must be positive when provided.")

    # Manifest의 official test split과 unique image request 범위를 model startup 전에 확정한다.
    test_records = [record for record in read_manifest_csv(manifest_path) if record.split == "test"]
    if not test_records:
        raise ValueError("No test records found in the API benchmark manifest.")
    _validate_unique_test_records(test_records)
    request_count = len(test_records) if measured_count is None else measured_count
    if request_count > len(test_records):
        raise ValueError(
            f"measured_count={request_count} exceeds test sample count {len(test_records)}."
        )
    if warmup_count > len(test_records):
        raise ValueError(
            f"warmup_count={warmup_count} exceeds test sample count {len(test_records)}."
        )

    settings = ServingSettings(
        artifact_dir=artifact_dir,
        thresholds_path=thresholds_path,
        database_url=database_url,
        model_device=requested_device,
        max_upload_bytes=max_upload_bytes,
    )
    settings.validate()

    # Disk image loading을 latency에서 제외하도록 warmup/measured payload를 한 번만 메모리에 올린다.
    required_record_count = max(warmup_count, request_count)
    uploads = [
        prepare_image_upload(
            dataset_root / record.image_path,
            max_upload_bytes=max_upload_bytes,
        )
        for record in test_records[:required_record_count]
    ]
    manifest_sha256 = sha256_file(manifest_path)
    app = create_app(settings=settings, runtime_loader=runtime_loader)

    # 동일 lifespan/runtime에서 readiness를 확인하고 warmup 뒤 전체 measured HTTP 요청을 수행한다.
    with TestClient(app, raise_server_exceptions=False) as client:
        health = client.get("/health")
        if health.status_code != 200:
            raise RuntimeError(f"API benchmark health check failed with HTTP {health.status_code}.")
        HealthResponse.model_validate(health.json())
        ready = client.get("/ready")
        if ready.status_code != 200:
            raise RuntimeError(f"API benchmark readiness failed with HTTP {ready.status_code}.")
        readiness = ReadinessResponse.model_validate(ready.json())

        runtime = app.state.serving_runtime
        provenance = require_serving_provenance(runtime)
        if provenance.manifest_sha256 != manifest_sha256:
            raise ValueError("API benchmark manifest SHA-256 does not match the serving runtime.")
        if any(record.category != readiness.category for record in test_records):
            raise ValueError("API benchmark manifest category does not match the serving runtime.")

        # Timer 직전에 multipart request를 만들고 completed ASGI response 반환 직후 종료한다.
        # ADD 2026-08-20: Preloaded image를 FastAPI prediction route로 multipart 전송한다.
        def send_request(upload: PreparedImageUpload) -> HttpResponse:
            return client.post(
                "/v1/predictions",
                files={"image": upload.as_multipart_file()},
            )

        run_http_warmup(
            uploads,
            warmup_count=warmup_count,
            send_request=send_request,
        )
        measurements = measure_http_requests(
            uploads,
            measured_count=request_count,
            send_request=send_request,
        )

    metrics = summarize_http_measurements(measurements)
    artifact = PatchCoreApiBenchmarkArtifact(
        model_name=readiness.model_name,
        category=readiness.category,
        device=readiness.device,
        operating_system=platform.platform(),
        machine=platform.machine(),
        accelerator_name=_accelerator_name(readiness.device),
        cuda_version=torch.version.cuda,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        fastapi_version=version("fastapi"),
        starlette_version=version("starlette"),
        manifest_sha256=manifest_sha256,
        artifact_metadata_sha256=provenance.artifact_metadata_sha256,
        model_sha256=provenance.model_sha256,
        threshold_artifact_sha256=provenance.threshold_artifact_sha256,
        image_payload_bytes=tuple(len(upload.content) for upload in uploads[:request_count]),
        warmup_count=warmup_count,
        measured_count=request_count,
        metrics=metrics,
        created_at=datetime.now(UTC).isoformat(),
    )
    benchmark_path = output_dir / BENCHMARK_FILENAME
    write_api_benchmark_artifact(artifact, benchmark_path)

    return ApiBenchmarkOutputSummary(
        benchmark_path=benchmark_path,
        device=readiness.device,
        measured_count=request_count,
        successful_request_count=metrics.successful_request_count,
        failed_request_count=metrics.failed_request_count,
        p50_ms=metrics.latency.p50_ms,
        p95_ms=metrics.latency.p95_ms,
        p99_ms=metrics.latency.p99_ms,
        mean_ms=metrics.latency.mean_ms,
        requests_per_second=metrics.requests_per_second,
    )


# ADD 2026-08-20: Measured request가 distinct manifest image에 일대일 대응하는지 검증한다.
def _validate_unique_test_records(records: list[ManifestRecord]) -> None:
    sample_ids = [record.sample_id for record in records]
    image_paths = [record.image_path for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("API benchmark test records contain duplicate sample_id values.")
    if len(set(image_paths)) != len(image_paths):
        raise ValueError("API benchmark test records contain duplicate image_path values.")


# ADD 2026-08-20: Runtime device의 inspectable accelerator 이름을 반환한다.
def _accelerator_name(device: str) -> str:
    device_type = torch.device(device).type
    if device_type == "cuda":
        return torch.cuda.get_device_name(torch.device(device))
    if device_type == "mps":
        return "Apple Metal Performance Shaders"
    return platform.processor() or "CPU"


# ADD 2026-08-20: FastAPI HTTP benchmark CLI 인자를 정의한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PatchCore FastAPI HTTP inference.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/mvtec_ad"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/interim/manifests/mvtec_ad_metal_nut.csv"),
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_API_BENCHMARK_ROOT)
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default="auto")
    parser.add_argument("--warmup-count", type=int, default=DEFAULT_WARMUP_COUNT)
    parser.add_argument("--measured-count", type=int)
    parser.add_argument("--max-upload-bytes", type=int, default=DEFAULT_MAX_UPLOAD_BYTES)
    return parser.parse_args()


# ADD 2026-08-20: CLI API benchmark를 실행하고 핵심 HTTP metrics를 출력한다.
def main() -> int:
    args = _parse_args()
    output_id = Path(args.output_id)
    if output_id.name != args.output_id or args.output_id in {"", ".", ".."}:
        raise ValueError("output-id must be one non-empty path component.")

    summary = benchmark_patchcore_api(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        artifact_dir=args.artifact_dir,
        thresholds_path=args.thresholds,
        database_url=required_database_url(),
        output_dir=args.output_root / args.output_id,
        requested_device=args.device,
        warmup_count=args.warmup_count,
        measured_count=args.measured_count,
        max_upload_bytes=args.max_upload_bytes,
    )
    print("PatchCore FastAPI HTTP benchmark: PASS")
    print(f"Device: {summary.device}")
    print(f"Measured requests: {summary.measured_count}")
    print(f"Latency p50/p95/p99: {summary.p50_ms:.3f}/{summary.p95_ms:.3f}/{summary.p99_ms:.3f} ms")
    print(f"Mean latency: {summary.mean_ms:.3f} ms")
    print(f"Throughput: {summary.requests_per_second:.3f} requests/second")
    print(
        "Successful/failed requests: "
        f"{summary.successful_request_count}/{summary.failed_request_count}"
    )
    print(f"Benchmark: {summary.benchmark_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
