"""Unit tests for PatchCore inference benchmark contracts."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ml.evaluation.benchmark import (
    BENCHMARK_NAME,
    PERCENTILE_METHOD,
    CudaPeakMemory,
    PatchCoreBenchmarkArtifact,
    measure_batches,
    read_cuda_peak_memory,
    reset_cuda_peak_memory,
    run_warmup,
    summarize_latencies,
    validate_benchmark_parameters,
    write_benchmark_artifact,
)


# ADD 2026-08-19: Known latency sample의 linear percentile, mean과 throughput을 검증한다.
def test_latency_summary_matches_known_distribution() -> None:
    summary = summarize_latencies(
        [10.0, 20.0, 30.0, 40.0],
        measured_sample_count=4,
    )

    assert summary.p50_ms == pytest.approx(25.0)
    assert summary.p95_ms == pytest.approx(38.5)
    assert summary.p99_ms == pytest.approx(39.7)
    assert summary.mean_ms == pytest.approx(25.0)
    assert summary.total_timed_seconds == pytest.approx(0.1)
    assert summary.throughput_images_per_second == pytest.approx(40.0)


# ADD 2026-08-19: Warmup 호출이 measured latency sample과 throughput에서 제외되는지 검증한다.
def test_warmup_is_excluded_from_measured_latency() -> None:
    calls: list[int] = []
    clock_values = iter((1.0, 1.01, 2.0, 2.02))

    # ADD 2026-08-19: Warmup과 measured callback 호출 순서를 기록한다.
    def run_batch(images: torch.Tensor) -> object:
        calls.append(int(images[0, 0].item()))
        return object()

    warmup_completed = run_warmup(
        [torch.tensor([[0.0]])],
        warmup_count=1,
        run_batch=run_batch,
        synchronize=lambda: None,
    )
    measurements = measure_batches(
        [torch.tensor([[1.0]]), torch.tensor([[2.0]])],
        measured_count=2,
        run_batch=run_batch,
        synchronize=lambda: None,
        clock=lambda: next(clock_values),
    )
    summary = summarize_latencies(
        measurements.latencies_ms,
        measured_sample_count=measurements.measured_sample_count,
    )

    assert warmup_completed == 1
    assert calls == [0, 1, 2]
    assert measurements.measured_batch_count == 2
    assert measurements.latencies_ms == pytest.approx((10.0, 20.0))
    assert summary.throughput_images_per_second == pytest.approx(2 / 0.03)


# ADD 2026-08-19: Non-finite latency와 invalid batch 또는 sample count를 거부한다.
@pytest.mark.parametrize("latency", [float("nan"), float("inf"), 0.0, -1.0])
def test_benchmark_rejects_invalid_latency_and_controls(latency: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        summarize_latencies([latency], measured_sample_count=1)
    with pytest.raises(ValueError, match="batch_size"):
        validate_benchmark_parameters(
            batch_size=0,
            warmup_count=0,
            measured_count=1,
            num_workers=0,
        )
    with pytest.raises(ValueError, match="measured_count"):
        validate_benchmark_parameters(
            batch_size=1,
            warmup_count=0,
            measured_count=0,
            num_workers=0,
        )


# ADD 2026-08-19: CUDA allocator API를 mock해 peak allocated/reserved 수집을 검증한다.
def test_cuda_peak_memory_is_mockable(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_devices: list[torch.device] = []
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: reset_devices.append(device),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 2048)
    device = torch.device("cuda")

    reset_cuda_peak_memory(device)
    memory = read_cuda_peak_memory(device)

    assert reset_devices == [device]
    assert memory == CudaPeakMemory(
        supported=True,
        peak_allocated_bytes=1024,
        peak_reserved_bytes=2048,
    )
    assert read_cuda_peak_memory(torch.device("cpu")).to_json_dict() == {
        "supported": False,
        "peak_allocated_bytes": None,
        "peak_allocated_megabytes": None,
        "peak_reserved_bytes": None,
        "peak_reserved_megabytes": None,
    }


# ADD 2026-08-19: Benchmark artifact의 provenance schema와 overwrite 금지를 검증한다.
def test_benchmark_artifact_persists_provenance_without_overwrite(tmp_path: Path) -> None:
    latency = summarize_latencies([10.0], measured_sample_count=1)
    artifact = PatchCoreBenchmarkArtifact(
        category="metal_nut",
        device="cpu",
        operating_system="macOS-15",
        machine="arm64",
        accelerator_name="Apple M4",
        cuda_version=None,
        python_version="3.12.14",
        torch_version="2.13.0",
        torchvision_version="0.28.0",
        anomalib_version="2.5.1",
        manifest_sha256="a" * 64,
        artifact_metadata_sha256="b" * 64,
        model_sha256="c" * 64,
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        preprocessing={"resize_size": [256, 256], "center_crop_size": [224, 224]},
        batch_size=1,
        warmup_count=10,
        measured_sample_count=1,
        measured_batch_count=1,
        latency=latency,
        model_file_size_bytes=1024,
        cuda_peak_memory=CudaPeakMemory(False, None, None),
        created_at="2026-08-19T00:00:00+00:00",
    )
    output_path = tmp_path / "benchmark.json"

    write_benchmark_artifact(artifact, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["benchmark_name"] == BENCHMARK_NAME
    assert payload["percentile_method"] == PERCENTILE_METHOD
    assert payload["provenance"] == {
        "manifest_sha256": "a" * 64,
        "artifact_metadata_sha256": "b" * 64,
        "model_sha256": "c" * 64,
    }
    assert payload["runtime"]["torch_version"] == "2.13.0"
    assert payload["runtime"]["cuda_version"] is None
    assert payload["model"]["preprocessing"]["center_crop_size"] == [224, 224]
    assert payload["measured_count"] == 1
    assert payload["disk_image_loading_included"] is False
    assert payload["model_file_size_bytes"] == 1024

    with pytest.raises(FileExistsError, match="already exists"):
        write_benchmark_artifact(artifact, output_path)

    with pytest.raises(ValueError, match="NaN or Infinity"):
        invalid = replace(
            artifact,
            preprocessing={"image_mean": [float("nan")]},
        )
        invalid.to_json_dict()
