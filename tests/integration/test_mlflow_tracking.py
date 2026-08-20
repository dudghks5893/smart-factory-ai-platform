from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from mlflow import MlflowClient

from ml.datasets.manifest import ManifestRecord, write_manifest_csv
from ml.tracking.patchcore import PatchCoreTrackingInputs, prepare_patchcore_tracking
from pipelines.track_patchcore_run import track_patchcore_run
from services.tracking.mlflow import (
    LoggedRun,
    MlflowTrackingConfig,
    MlflowTrackingError,
    TrackingPayload,
)
from shared.hashing import sha256_file


@dataclass(frozen=True)
class TrackingFixture:
    inputs: PatchCoreTrackingInputs
    model_sha256: str
    metadata_sha256: str
    manifest_sha256: str
    threshold_sha256: str


class FailingTrackingAdapter:
    """Pipeline test double for an unavailable MLflow backend."""

    # ADD 2026-08-20: External logging failure를 caller에 그대로 노출한다.
    def track(self, payload: TrackingPayload) -> LoggedRun:
        raise MlflowTrackingError("backend unavailable")


# ADD 2026-08-20: JSON fixture를 deterministic object artifact로 저장한다.
def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


# ADD 2026-08-20: Small project-native artifact 전체 lineage fixture를 생성한다.
def _tracking_fixture(tmp_path: Path, *, include_api: bool = True) -> TrackingFixture:
    config_path = tmp_path / "patchcore.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "name": "patchcore",
                    "implementation": "anomalib",
                    "backbone": "wide_resnet50_2",
                    "layers": ["layer2", "layer3"],
                    "pretrained": True,
                    "coreset_sampling_ratio": 0.1,
                    "num_neighbors": 9,
                },
                "preprocessing": {
                    "resize_size": [256, 256],
                    "center_crop_size": [224, 224],
                    "image_mean": [0.485, 0.456, 0.406],
                    "image_std": [0.229, 0.224, 0.225],
                },
                "training": {
                    "random_seed": 42,
                    "device": "auto",
                    "batch_size": 4,
                    "num_workers": 0,
                },
                "output": {
                    "artifact_root": "artifacts/models/patchcore",
                    "prediction_root": "outputs/predictions/patchcore",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    records = [
        ManifestRecord("train", "metal_nut", "train", "train", "good", 0, "a.png", "", 8, 8),
        ManifestRecord(
            "validation", "metal_nut", "train", "validation", "good", 0, "b.png", "", 8, 8
        ),
        ManifestRecord("test-good", "metal_nut", "test", "test", "good", 0, "c.png", "", 8, 8),
        ManifestRecord(
            "test-bent", "metal_nut", "test", "test", "bent", 1, "d.png", "d_mask.png", 8, 8
        ),
    ]
    manifest_path = tmp_path / "manifest.csv"
    write_manifest_csv(records, manifest_path)
    manifest_sha256 = sha256_file(manifest_path)

    artifact_dir = tmp_path / "model-artifact"
    artifact_dir.mkdir()
    model_path = artifact_dir / "model.pt"
    model_path.write_bytes(b"small tensor state fixture")
    metadata = {
        "schema_version": 1,
        "model_name": "patchcore",
        "implementation": "anomalib",
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "num_neighbors": 9,
        "coreset_sampling_ratio": 0.1,
        "pretrained_used_during_training": True,
        "preprocessing": {
            "resize_size": [256, 256],
            "center_crop_size": [224, 224],
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        },
        "random_seed": 42,
        "category": "metal_nut",
        "train_sample_count": 1,
        "manifest_sha256": manifest_sha256,
        "anomalib_version": "2.5.1",
        "torch_version": "2.13.0",
        "torchvision_version": "0.28.0",
        "python_version": "3.12.14",
        "created_at": "2026-08-20T00:00:00+00:00",
    }
    metadata_path = artifact_dir / "metadata.json"
    _write_json(metadata_path, metadata)
    model_sha256 = sha256_file(model_path)
    metadata_sha256 = sha256_file(metadata_path)

    summary_path = tmp_path / "manifest_summary.json"
    _write_json(
        summary_path,
        {
            "category": "metal_nut",
            "train_count": 1,
            "validation_count": 1,
            "test_good_count": 1,
            "test_anomaly_count": 1,
            "manifest_count": 4,
        },
    )
    threshold_path = tmp_path / "thresholds.json"
    threshold = {
        "schema_version": 1,
        "model_name": "patchcore",
        "category": "metal_nut",
        "strategy": "max_normal_validation",
        "comparison_operator": ">",
        "image_threshold": 41.0,
        "pixel_threshold": 12.0,
        "validation_sample_count": 1,
        "validation_pixel_count": 50176,
        "manifest_sha256": manifest_sha256,
        "artifact_metadata": metadata,
        "artifact_metadata_sha256": metadata_sha256,
        "model_sha256": model_sha256,
        "validation_predictions_sha256": "a" * 64,
        "validation_anomaly_maps_sha256": "b" * 64,
        "created_at": "2026-08-20T00:01:00+00:00",
    }
    _write_json(threshold_path, threshold)
    threshold_sha256 = sha256_file(threshold_path)

    per_defect = {
        "bent": {"sample_count": 1, "detected_count": 1, "recall": 1.0},
        "good": {"sample_count": 1, "false_positive_count": 0, "false_positive_rate": 0.0},
    }
    metrics_path = tmp_path / "metrics.json"
    _write_json(
        metrics_path,
        {
            "schema_version": 1,
            "category": "metal_nut",
            "threshold_artifact": {"sha256": threshold_sha256},
            "provenance": {
                "manifest_sha256": manifest_sha256,
                "artifact_metadata_sha256": metadata_sha256,
                "model_sha256": model_sha256,
            },
            "sample_counts": {"total": 2, "normal": 1, "anomaly": 1},
            "image_level": {
                "auroc": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "tp": 1,
                "tn": 1,
                "fp": 0,
                "fn": 0,
            },
            "pixel_level": {
                "auroc": 0.98,
                "precision": 0.83,
                "recall": 0.84,
                "f1": 0.835,
                "tp": 10,
                "tn": 20,
                "fp": 2,
                "fn": 2,
            },
            "per_defect": per_defect,
        },
    )
    per_defect_path = tmp_path / "per_defect_metrics.json"
    _write_json(per_defect_path, per_defect)

    model_benchmark_path = tmp_path / "model_benchmark.json"
    _write_json(
        model_benchmark_path,
        {
            "schema_version": 1,
            "benchmark_name": "patchcore_inference",
            "category": "metal_nut",
            "runtime": {
                "accelerator_name": "Tesla T4",
                "torch_version": "2.13.0+cu130",
                "torchvision_version": "0.28.0+cu130",
                "anomalib_version": "2.5.1",
                "python_version": "3.12.13",
                "cuda_version": "13.0",
            },
            "provenance": {
                "manifest_sha256": manifest_sha256,
                "artifact_metadata_sha256": metadata_sha256,
                "model_sha256": model_sha256,
            },
            "batch_size": 1,
            "measured_count": 2,
            "latency_ms": {
                "p50_ms": 21.0,
                "p95_ms": 25.0,
                "p99_ms": 27.0,
                "mean_ms": 22.0,
                "total_timed_seconds": 0.044,
            },
            "throughput_images_per_second": 45.0,
            "model_file_size_megabytes": 0.00002,
            "cuda_peak_memory": {
                "peak_allocated_megabytes": 10.0,
                "peak_reserved_megabytes": 12.0,
            },
        },
    )

    api_benchmark_path: Path | None = None
    if include_api:
        api_benchmark_path = tmp_path / "api_benchmark.json"
        _write_json(
            api_benchmark_path,
            {
                "schema_version": 1,
                "benchmark_name": "patchcore_fastapi_http_e2e",
                "model_name": "patchcore",
                "category": "metal_nut",
                "provenance": {
                    "manifest_sha256": manifest_sha256,
                    "artifact_metadata_sha256": metadata_sha256,
                    "model_sha256": model_sha256,
                    "threshold_artifact_sha256": threshold_sha256,
                },
                "conditions": {"measured_count": 2},
                "metrics": {
                    "latency_ms": {
                        "p50_ms": 44.0,
                        "p95_ms": 48.0,
                        "p99_ms": 53.0,
                        "mean_ms": 45.0,
                        "total_timed_seconds": 0.09,
                    },
                    "requests_per_second": 22.0,
                    "error_rate": 0.0,
                },
            },
        )

    return TrackingFixture(
        inputs=PatchCoreTrackingInputs(
            config_path=config_path,
            manifest_path=manifest_path,
            artifact_dir=artifact_dir,
            manifest_summary_path=summary_path,
            thresholds_path=threshold_path,
            metrics_path=metrics_path,
            per_defect_metrics_path=per_defect_path,
            model_benchmark_path=model_benchmark_path,
            api_benchmark_path=api_benchmark_path,
        ),
        model_sha256=model_sha256,
        metadata_sha256=metadata_sha256,
        manifest_sha256=manifest_sha256,
        threshold_sha256=threshold_sha256,
    )


# ADD 2026-08-20: MLflow artifact tree를 recursive relative path set으로 조회한다.
def _artifact_paths(client: MlflowClient, run_id: str, path: str | None = None) -> set[str]:
    paths: set[str] = set()
    for artifact in client.list_artifacts(run_id, path):
        if artifact.is_dir:
            paths.update(_artifact_paths(client, run_id, artifact.path))
        else:
            paths.add(artifact.path)
    return paths


# ADD 2026-08-20: Temporary SQLite backend에서 complete PatchCore lineage round-trip을 검증한다.
def test_track_patchcore_run_logs_lineage_and_immutable_pointer(tmp_path: Path) -> None:
    fixture = _tracking_fixture(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    config = MlflowTrackingConfig(
        tracking_uri=tracking_uri,
        experiment_name="integration-patchcore",
        run_name="metal_nut_fixture_seed42",
        artifact_location=(tmp_path / "mlflow-artifacts").resolve().as_uri(),
    )

    # 실제 MLflow client로 experiment/run metadata와 allowlist artifact를 기록한다.
    result = track_patchcore_run(
        inputs=fixture.inputs,
        tracking_config=config,
        tracking_id="fixture-run",
        output_root=tmp_path / "tracking-output",
    )

    client = MlflowClient(tracking_uri)
    run = client.get_run(result.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["category"] == "metal_nut"
    assert run.data.params["manifest.row_count"] == "4"
    assert run.data.metrics["image.auroc"] == pytest.approx(1.0)
    assert run.data.metrics["benchmark.model.p50_ms"] == pytest.approx(21.0)
    assert run.data.metrics["api.http.p50_ms"] == pytest.approx(44.0)
    assert run.data.tags["lineage.model_sha256"] == fixture.model_sha256
    assert run.data.tags["api.benchmark_schema_version"] == "1"
    assert run.data.tags["api.inspection_persistence_included"] == "false"

    artifact_paths = _artifact_paths(client, result.run_id)
    assert "model/model.pt" in artifact_paths
    assert "threshold/thresholds.json" in artifact_paths
    assert "dataset/manifest.csv" in artifact_paths
    assert not any(path.endswith((".png", "anomaly_maps.pt", ".env")) for path in artifact_paths)

    pointer = json.loads(result.pointer_path.read_text(encoding="utf-8"))
    assert pointer["run_id"] == result.run_id
    assert pointer["model_sha256"] == fixture.model_sha256
    assert "tracking_uri" not in pointer

    with pytest.raises(FileExistsError, match="already exists"):
        track_patchcore_run(
            inputs=fixture.inputs,
            tracking_config=config,
            tracking_id="fixture-run",
            output_root=tmp_path / "tracking-output",
        )
    assert len(client.search_runs([result.experiment_id])) == 1


# ADD 2026-08-20: API benchmark가 없어도 허위 API metric 없이 payload가 생성되는지 검증한다.
def test_prepare_tracking_accepts_absent_optional_api_benchmark(tmp_path: Path) -> None:
    fixture = _tracking_fixture(tmp_path, include_api=False)

    prepared = prepare_patchcore_tracking(fixture.inputs)

    assert not any(name.startswith("api.http.") for name in prepared.payload.metrics)
    assert not any(
        artifact.artifact_path == "benchmarks/api" for artifact in prepared.payload.artifacts
    )


# ADD 2026-08-20: MLflow logging 실패 시 성공 pointer가 생성되지 않는지 검증한다.
def test_track_patchcore_run_does_not_write_pointer_after_logging_failure(tmp_path: Path) -> None:
    fixture = _tracking_fixture(tmp_path, include_api=False)
    output_root = tmp_path / "tracking-output"

    with pytest.raises(MlflowTrackingError, match="backend unavailable"):
        track_patchcore_run(
            inputs=fixture.inputs,
            tracking_config=MlflowTrackingConfig("sqlite:///:memory:", "test"),
            tracking_id="failed-run",
            output_root=output_root,
            adapter=FailingTrackingAdapter(),
        )

    assert not (output_root / "failed-run").exists()


@pytest.mark.parametrize("source", ["threshold", "evaluation", "model_benchmark"])
# ADD 2026-08-20: Cross-stage SHA mismatch를 MLflow 접근 전에 각각 거부하는지 검증한다.
def test_prepare_tracking_rejects_cross_stage_provenance_mismatch(
    tmp_path: Path,
    source: str,
) -> None:
    fixture = _tracking_fixture(tmp_path)
    if source == "threshold":
        path = fixture.inputs.thresholds_path
        field_parent = None
    elif source == "evaluation":
        path = fixture.inputs.metrics_path
        field_parent = "provenance"
    else:
        path = fixture.inputs.model_benchmark_path
        field_parent = "provenance"
    assert path is not None
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if field_parent is None:
        payload["model_sha256"] = "f" * 64
    else:
        payload[field_parent]["model_sha256"] = "f" * 64
    _write_json(path, payload)

    with pytest.raises(ValueError, match="model|provenance|SHA-256"):
        prepare_patchcore_tracking(fixture.inputs)
