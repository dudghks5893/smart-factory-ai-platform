"""Prepare validated PatchCore lineage for an external tracking adapter."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ml.datasets.manifest import ManifestRecord, read_manifest_csv
from ml.evaluation.metrics import EVALUATION_SCHEMA_VERSION
from ml.evaluation.thresholds import (
    ThresholdArtifact,
    read_threshold_artifact,
    validate_threshold_provenance,
)
from ml.training.config import PatchCoreBaselineConfig, load_patchcore_config
from ml.training.patchcore import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    PatchCoreArtifactMetadata,
    read_artifact_metadata,
)
from services.api.benchmark import API_BENCHMARK_NAME
from services.tracking.mlflow import ParameterValue, TrackedArtifact, TrackingPayload
from shared.hashing import sha256_file

TRACKING_POINTER_SCHEMA_VERSION = 1
DEFAULT_TRACKING_ROOT = Path("outputs/mlflow/patchcore")


@dataclass(frozen=True)
class PatchCoreTrackingInputs:
    """Project-native artifacts accepted by the PatchCore backfill tracker."""

    config_path: Path
    manifest_path: Path
    artifact_dir: Path
    manifest_summary_path: Path | None = None
    thresholds_path: Path | None = None
    metrics_path: Path | None = None
    per_defect_metrics_path: Path | None = None
    model_benchmark_path: Path | None = None
    api_benchmark_path: Path | None = None


@dataclass(frozen=True)
class PatchCoreLineageIdentity:
    """Hashes and artifact identity that uniquely identify one baseline lineage."""

    artifact_id: str
    model_sha256: str
    artifact_metadata_sha256: str
    manifest_sha256: str
    threshold_artifact_sha256: str | None


@dataclass(frozen=True)
class PreparedPatchCoreTracking:
    """Validated generic payload plus identity needed for the project pointer."""

    payload: TrackingPayload
    identity: PatchCoreLineageIdentity


@dataclass(frozen=True)
class TrackingPointer:
    """Project-side immutable pointer to one completed MLflow tracking run."""

    tracking_id: str
    experiment_name: str
    experiment_id: str
    run_id: str
    run_name: str | None
    artifact_id: str
    model_sha256: str
    artifact_metadata_sha256: str
    manifest_sha256: str
    threshold_artifact_sha256: str | None
    created_at: str
    schema_version: int = TRACKING_POINTER_SCHEMA_VERSION


# ADD 2026-08-20: Project-native artifact를 검증해 단일 PatchCore MLflow payload로 준비한다.
def prepare_patchcore_tracking(inputs: PatchCoreTrackingInputs) -> PreparedPatchCoreTracking:
    """Validate all supplied lineage and flatten it without contacting MLflow."""
    _validate_input_combinations(inputs)

    # Config, model artifact와 manifest를 읽고 canonical model/data hash를 계산한다.
    config = load_patchcore_config(inputs.config_path)
    metadata = read_artifact_metadata(inputs.artifact_dir)
    records = read_manifest_csv(inputs.manifest_path)
    model_path = inputs.artifact_dir / MODEL_FILENAME
    metadata_path = inputs.artifact_dir / METADATA_FILENAME
    model_sha256 = sha256_file(model_path)
    metadata_sha256 = sha256_file(metadata_path)
    manifest_sha256 = sha256_file(inputs.manifest_path)
    _validate_core_lineage(config, metadata, records, manifest_sha256)

    parameters = _base_parameters(config, metadata, records, model_path)
    metrics: dict[str, float] = {}
    tags = {
        "project": "smart-factory-ai-platform",
        "lineage.type": "patchcore_baseline",
        "lineage.model_sha256": model_sha256,
        "lineage.artifact_metadata_sha256": metadata_sha256,
        "lineage.manifest_sha256": manifest_sha256,
        "lineage.artifact_id": inputs.artifact_dir.name,
    }
    artifacts = [
        TrackedArtifact(inputs.config_path, "config"),
        TrackedArtifact(inputs.manifest_path, "dataset"),
        TrackedArtifact(model_path, "model"),
        TrackedArtifact(metadata_path, "model"),
    ]

    # Optional manifest summary는 CSV에서 재계산한 count와 일치할 때만 기록한다.
    if inputs.manifest_summary_path is not None:
        summary = _read_json_mapping(inputs.manifest_summary_path, "manifest summary")
        _validate_manifest_summary(summary, metadata, records)
        artifacts.append(TrackedArtifact(inputs.manifest_summary_path, "dataset"))

    # Threshold artifact를 existing domain validator로 model/manifest에 다시 고정한다.
    thresholds: ThresholdArtifact | None = None
    threshold_sha256: str | None = None
    if inputs.thresholds_path is not None:
        thresholds = read_threshold_artifact(inputs.thresholds_path)
        validate_threshold_provenance(
            thresholds,
            artifact_metadata=metadata,
            manifest_sha256=manifest_sha256,
            artifact_metadata_sha256=metadata_sha256,
            model_sha256=model_sha256,
        )
        threshold_sha256 = sha256_file(inputs.thresholds_path)
        parameters.update(
            {
                "threshold.strategy": thresholds.strategy,
                "threshold.comparison_operator": thresholds.comparison_operator,
                "threshold.validation_sample_count": thresholds.validation_sample_count,
            }
        )
        metrics.update(
            {
                "threshold.image": thresholds.image_threshold,
                "threshold.pixel": thresholds.pixel_threshold,
            }
        )
        tags["lineage.threshold_artifact_sha256"] = threshold_sha256
        artifacts.append(TrackedArtifact(inputs.thresholds_path, "threshold"))

    expected_provenance = {
        "manifest_sha256": manifest_sha256,
        "artifact_metadata_sha256": metadata_sha256,
        "model_sha256": model_sha256,
    }

    # Evaluation artifact는 threshold와 core hash가 모두 같은 경우에만 metric으로 변환한다.
    if inputs.metrics_path is not None:
        evaluation = _read_json_mapping(inputs.metrics_path, "evaluation metrics")
        _collect_evaluation(
            evaluation,
            metadata=metadata,
            expected_provenance=expected_provenance,
            threshold_sha256=threshold_sha256,
            metrics=metrics,
            parameters=parameters,
        )
        artifacts.append(TrackedArtifact(inputs.metrics_path, "evaluation"))

        if inputs.per_defect_metrics_path is not None:
            per_defect = _read_json_mapping(
                inputs.per_defect_metrics_path,
                "per-defect metrics",
            )
            if per_defect != _required_mapping(evaluation, "per_defect", "evaluation metrics"):
                raise ValueError("Per-defect metrics do not match metrics.json.")
            artifacts.append(TrackedArtifact(inputs.per_defect_metrics_path, "evaluation"))

    # Offline benchmark runtime과 latency를 artifact JSON에서 읽어 동일 lineage에 연결한다.
    if inputs.model_benchmark_path is not None:
        benchmark = _read_json_mapping(inputs.model_benchmark_path, "model benchmark")
        _collect_model_benchmark(
            benchmark,
            metadata=metadata,
            expected_provenance=expected_provenance,
            metrics=metrics,
            parameters=parameters,
            tags=tags,
        )
        artifacts.append(TrackedArtifact(inputs.model_benchmark_path, "benchmarks/model"))

    # API benchmark schema version으로 persistence boundary를 명시하고 threshold hash를 검증한다.
    if inputs.api_benchmark_path is not None:
        api_benchmark = _read_json_mapping(inputs.api_benchmark_path, "API benchmark")
        _collect_api_benchmark(
            api_benchmark,
            metadata=metadata,
            expected_provenance=expected_provenance,
            threshold_sha256=threshold_sha256,
            metrics=metrics,
            parameters=parameters,
            tags=tags,
        )
        artifacts.append(TrackedArtifact(inputs.api_benchmark_path, "benchmarks/api"))

    identity = PatchCoreLineageIdentity(
        artifact_id=inputs.artifact_dir.name,
        model_sha256=model_sha256,
        artifact_metadata_sha256=metadata_sha256,
        manifest_sha256=manifest_sha256,
        threshold_artifact_sha256=threshold_sha256,
    )
    return PreparedPatchCoreTracking(
        payload=TrackingPayload(
            parameters=parameters,
            metrics=metrics,
            tags=tags,
            artifacts=tuple(artifacts),
        ),
        identity=identity,
    )


# ADD 2026-08-20: 성공한 MLflow run identity를 project output에 overwrite 없이 저장한다.
def write_tracking_pointer(pointer: TrackingPointer, output_dir: Path) -> Path:
    """Persist a credential-free pointer after MLflow reports run success."""
    if pointer.schema_version != TRACKING_POINTER_SCHEMA_VERSION:
        raise ValueError(f"Unsupported tracking pointer schema: {pointer.schema_version}.")
    if output_dir.exists():
        raise FileExistsError(f"MLflow tracking output already exists: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=False)
    pointer_path = output_dir / "tracking.json"
    pointer_path.write_text(
        json.dumps(
            asdict(pointer),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return pointer_path


# ADD 2026-08-20: Completed MLflow identity와 lineage hash로 immutable pointer를 구성한다.
def build_tracking_pointer(
    *,
    tracking_id: str,
    experiment_name: str,
    experiment_id: str,
    run_id: str,
    run_name: str | None,
    identity: PatchCoreLineageIdentity,
) -> TrackingPointer:
    """Build a pointer without persisting credentials or a tracking server URI."""
    validate_tracking_id(tracking_id)
    return TrackingPointer(
        tracking_id=tracking_id,
        experiment_name=experiment_name,
        experiment_id=experiment_id,
        run_id=run_id,
        run_name=run_name,
        artifact_id=identity.artifact_id,
        model_sha256=identity.model_sha256,
        artifact_metadata_sha256=identity.artifact_metadata_sha256,
        manifest_sha256=identity.manifest_sha256,
        threshold_artifact_sha256=identity.threshold_artifact_sha256,
        created_at=datetime.now(UTC).isoformat(),
    )


# ADD 2026-08-20: Tracking output escape를 막도록 ID를 단일 path component로 제한한다.
def validate_tracking_id(tracking_id: str) -> None:
    """Reject empty, nested, and parent-directory tracking identifiers."""
    if not tracking_id or Path(tracking_id).name != tracking_id or tracking_id in {".", ".."}:
        raise ValueError("tracking_id must be one non-empty path component.")


# ADD 2026-08-20: Optional lineage input의 dependency와 file 존재 조건을 검증한다.
def _validate_input_combinations(inputs: PatchCoreTrackingInputs) -> None:
    for required_path in (inputs.config_path, inputs.manifest_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required tracking input not found: {required_path}")
    if inputs.per_defect_metrics_path is not None and inputs.metrics_path is None:
        raise ValueError("per_defect_metrics_path requires metrics_path.")
    if inputs.metrics_path is not None and inputs.thresholds_path is None:
        raise ValueError("metrics_path requires thresholds_path for provenance validation.")
    if inputs.api_benchmark_path is not None and inputs.thresholds_path is None:
        raise ValueError("api_benchmark_path requires thresholds_path for provenance validation.")
    for optional_path in (
        inputs.manifest_summary_path,
        inputs.thresholds_path,
        inputs.metrics_path,
        inputs.per_defect_metrics_path,
        inputs.model_benchmark_path,
        inputs.api_benchmark_path,
    ):
        if optional_path is not None and not optional_path.is_file():
            raise FileNotFoundError(f"Optional tracking input not found: {optional_path}")


# ADD 2026-08-20: Config, artifact metadata와 manifest의 canonical model/data 계약을 검증한다.
def _validate_core_lineage(
    config: PatchCoreBaselineConfig,
    metadata: PatchCoreArtifactMetadata,
    records: list[ManifestRecord],
    manifest_sha256: str,
) -> None:
    if not records:
        raise ValueError("Tracking manifest must not be empty.")
    if any(record.category != metadata.category for record in records):
        raise ValueError("Manifest category does not match model artifact metadata.")
    if metadata.manifest_sha256 != manifest_sha256:
        raise ValueError("Manifest SHA-256 does not match model artifact metadata.")
    if (
        config.model.name != metadata.model_name
        or config.model.implementation != metadata.implementation
        or config.model.backbone != metadata.backbone
        or config.model.layers != metadata.layers
        or config.model.pretrained != metadata.pretrained_used_during_training
        or config.model.coreset_sampling_ratio != metadata.coreset_sampling_ratio
        or config.model.num_neighbors != metadata.num_neighbors
        or config.preprocessing != metadata.preprocessing
        or config.training.random_seed != metadata.random_seed
    ):
        raise ValueError("PatchCore config does not match model artifact metadata.")
    train_count = sum(record.split == "train" for record in records)
    if train_count != metadata.train_sample_count:
        raise ValueError("Manifest train count does not match model artifact metadata.")


# ADD 2026-08-20: Model config와 dataset count를 stable scalar MLflow parameter로 변환한다.
def _base_parameters(
    config: PatchCoreBaselineConfig,
    metadata: PatchCoreArtifactMetadata,
    records: list[ManifestRecord],
    model_path: Path,
) -> dict[str, ParameterValue]:
    return {
        "model_name": metadata.model_name,
        "category": metadata.category,
        "implementation": metadata.implementation,
        "backbone": metadata.backbone,
        "layers": ",".join(metadata.layers),
        "pretrained_used_during_training": metadata.pretrained_used_during_training,
        "resize_size": "x".join(map(str, metadata.preprocessing.resize_size)),
        "center_crop_size": "x".join(map(str, metadata.preprocessing.center_crop_size)),
        "coreset_sampling_ratio": metadata.coreset_sampling_ratio,
        "num_neighbors": metadata.num_neighbors,
        "random_seed": config.training.random_seed,
        "train_sample_count": metadata.train_sample_count,
        "manifest.row_count": len(records),
        "manifest.train_count": sum(record.split == "train" for record in records),
        "manifest.validation_count": sum(record.split == "validation" for record in records),
        "manifest.test_normal_count": sum(
            record.split == "test" and record.label == 0 for record in records
        ),
        "manifest.test_anomaly_count": sum(
            record.split == "test" and record.label == 1 for record in records
        ),
        "model.artifact_id": model_path.parent.name,
        "model.file_size_bytes": model_path.stat().st_size,
    }


# ADD 2026-08-20: Optional manifest summary가 canonical manifest count를 재현하는지 검증한다.
def _validate_manifest_summary(
    summary: dict[str, Any],
    metadata: PatchCoreArtifactMetadata,
    records: list[ManifestRecord],
) -> None:
    expected = {
        "category": metadata.category,
        "train_count": sum(record.split == "train" for record in records),
        "validation_count": sum(record.split == "validation" for record in records),
        "test_good_count": sum(record.split == "test" and record.label == 0 for record in records),
        "test_anomaly_count": sum(
            record.split == "test" and record.label == 1 for record in records
        ),
        "manifest_count": len(records),
    }
    for field, expected_value in expected.items():
        if summary.get(field) != expected_value:
            raise ValueError(f"Manifest summary field '{field}' does not match manifest CSV.")


# ADD 2026-08-20: Evaluation JSON의 provenance와 metric schema를 MLflow naming으로 변환한다.
def _collect_evaluation(
    evaluation: dict[str, Any],
    *,
    metadata: PatchCoreArtifactMetadata,
    expected_provenance: dict[str, str],
    threshold_sha256: str | None,
    metrics: dict[str, float],
    parameters: dict[str, ParameterValue],
) -> None:
    if evaluation.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError("Evaluation metrics have an unsupported schema_version.")
    if evaluation.get("category") != metadata.category:
        raise ValueError("Evaluation category does not match model artifact metadata.")
    _validate_provenance(evaluation, expected_provenance, "evaluation metrics")
    threshold = _required_mapping(evaluation, "threshold_artifact", "evaluation metrics")
    if threshold_sha256 is None or threshold.get("sha256") != threshold_sha256:
        raise ValueError("Evaluation threshold SHA-256 does not match thresholds.json.")

    for level in ("image", "pixel"):
        section = _required_mapping(evaluation, f"{level}_level", "evaluation metrics")
        for name in ("auroc", "precision", "recall", "f1", "tp", "tn", "fp", "fn"):
            metrics[f"{level}.{name}"] = _required_finite_number(
                section.get(name),
                f"{level}_level.{name}",
            )
    per_defect = _required_mapping(evaluation, "per_defect", "evaluation metrics")
    for defect_type, raw_values in per_defect.items():
        values = _mapping_value(raw_values, f"per_defect.{defect_type}")
        metric_name = "false_positive_rate" if defect_type == "good" else "recall"
        metrics[f"defect.{defect_type}.{metric_name}"] = _required_finite_number(
            values.get(metric_name),
            f"per_defect.{defect_type}.{metric_name}",
        )
    sample_counts = _required_mapping(evaluation, "sample_counts", "evaluation metrics")
    parameters["evaluation.sample_count"] = _required_integer(
        sample_counts.get("total"), "sample_counts.total"
    )


# ADD 2026-08-20: Offline model benchmark의 hash, runtime tag와 latency metric을 수집한다.
def _collect_model_benchmark(
    benchmark: dict[str, Any],
    *,
    metadata: PatchCoreArtifactMetadata,
    expected_provenance: dict[str, str],
    metrics: dict[str, float],
    parameters: dict[str, ParameterValue],
    tags: dict[str, str],
) -> None:
    if (
        benchmark.get("schema_version") != 1
        or benchmark.get("benchmark_name") != "patchcore_inference"
    ):
        raise ValueError("Model benchmark has an unsupported schema or benchmark_name.")
    if benchmark.get("category") != metadata.category:
        raise ValueError("Model benchmark category does not match model artifact metadata.")
    _validate_provenance(benchmark, expected_provenance, "model benchmark")
    latency = _required_mapping(benchmark, "latency_ms", "model benchmark")
    for name in ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "total_timed_seconds"):
        metrics[f"benchmark.model.{name}"] = _required_finite_number(
            latency.get(name), f"latency_ms.{name}"
        )
    for source, target in (
        ("throughput_images_per_second", "benchmark.model.throughput_images_per_second"),
        ("model_file_size_megabytes", "benchmark.model.model_file_size_megabytes"),
    ):
        metrics[target] = _required_finite_number(benchmark.get(source), source)
    cuda_memory = _required_mapping(benchmark, "cuda_peak_memory", "model benchmark")
    for source, target in (
        ("peak_allocated_megabytes", "benchmark.model.cuda_peak_allocated_megabytes"),
        ("peak_reserved_megabytes", "benchmark.model.cuda_peak_reserved_megabytes"),
    ):
        value = cuda_memory.get(source)
        if value is not None:
            metrics[target] = _required_finite_number(value, f"cuda_peak_memory.{source}")
    parameters["benchmark.model.batch_size"] = _required_integer(
        benchmark.get("batch_size"), "batch_size"
    )
    parameters["benchmark.model.measured_count"] = _required_integer(
        benchmark.get("measured_count"), "measured_count"
    )
    _collect_runtime_tags(_required_mapping(benchmark, "runtime", "model benchmark"), tags)


# ADD 2026-08-20: FastAPI benchmark provenance와 schema별 persistence boundary를 수집한다.
def _collect_api_benchmark(
    benchmark: dict[str, Any],
    *,
    metadata: PatchCoreArtifactMetadata,
    expected_provenance: dict[str, str],
    threshold_sha256: str | None,
    metrics: dict[str, float],
    parameters: dict[str, ParameterValue],
    tags: dict[str, str],
) -> None:
    schema_version = _required_integer(benchmark.get("schema_version"), "schema_version")
    if schema_version not in {1, 2} or benchmark.get("benchmark_name") != API_BENCHMARK_NAME:
        raise ValueError("API benchmark has an unsupported schema or benchmark_name.")
    if benchmark.get("model_name") != metadata.model_name:
        raise ValueError("API benchmark model_name does not match model artifact metadata.")
    if benchmark.get("category") != metadata.category:
        raise ValueError("API benchmark category does not match model artifact metadata.")
    _validate_provenance(benchmark, expected_provenance, "API benchmark")
    provenance = _required_mapping(benchmark, "provenance", "API benchmark")
    if threshold_sha256 is None or provenance.get("threshold_artifact_sha256") != threshold_sha256:
        raise ValueError("API benchmark threshold SHA-256 does not match thresholds.json.")

    result = _required_mapping(benchmark, "metrics", "API benchmark")
    latency = _required_mapping(result, "latency_ms", "API benchmark metrics")
    for name in ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "total_timed_seconds"):
        metrics[f"api.http.{name}"] = _required_finite_number(
            latency.get(name), f"metrics.latency_ms.{name}"
        )
    for source in ("requests_per_second", "error_rate"):
        metrics[f"api.http.{source}"] = _required_finite_number(
            result.get(source), f"metrics.{source}"
        )
    conditions = _required_mapping(benchmark, "conditions", "API benchmark")
    parameters["benchmark.api.measured_count"] = _required_integer(
        conditions.get("measured_count"), "conditions.measured_count"
    )
    persistence_included = benchmark.get("inspection_persistence_included", False)
    if not isinstance(persistence_included, bool):
        raise ValueError("API benchmark inspection_persistence_included must be boolean.")
    if persistence_included != (schema_version == 2):
        raise ValueError("API benchmark schema and inspection persistence boundary do not match.")
    tags["api.benchmark_schema_version"] = str(schema_version)
    tags["api.inspection_persistence_included"] = str(persistence_included).lower()


# ADD 2026-08-20: Model benchmark runtime metadata를 stable MLflow tag namespace로 변환한다.
def _collect_runtime_tags(runtime: dict[str, Any], tags: dict[str, str]) -> None:
    for field in (
        "accelerator_name",
        "torch_version",
        "torchvision_version",
        "anomalib_version",
        "python_version",
        "cuda_version",
    ):
        value = runtime.get(field)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"Model benchmark runtime.{field} must be a non-empty string.")
            tags[f"runtime.{field}"] = value


# ADD 2026-08-20: Artifact JSON의 core provenance가 canonical hash와 일치하는지 검증한다.
def _validate_provenance(
    payload: dict[str, Any],
    expected: dict[str, str],
    source: str,
) -> None:
    provenance = _required_mapping(payload, "provenance", source)
    for field, digest in expected.items():
        if provenance.get(field) != digest:
            raise ValueError(f"{source} {field} does not match canonical lineage.")


# ADD 2026-08-20: JSON file을 object root로 읽고 parse/access 오류를 명확히 보고한다.
def _read_json_mapping(path: Path, source: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {source}: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{source} root must be a JSON object.")
    return cast(dict[str, Any], raw)


# ADD 2026-08-20: Parent JSON object에서 required mapping field를 반환한다.
def _required_mapping(parent: dict[str, Any], field: str, source: str) -> dict[str, Any]:
    if field not in parent:
        raise ValueError(f"{source} is missing field '{field}'.")
    return _mapping_value(parent[field], f"{source}.{field}")


# ADD 2026-08-20: JSON value가 string-keyed mapping인지 검증한다.
def _mapping_value(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object.")
    return cast(dict[str, Any], value)


# ADD 2026-08-20: JSON metric value를 finite float로 검증해 변환한다.
def _required_finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be a finite number.")
    return converted


# ADD 2026-08-20: JSON count value가 bool이 아닌 integer인지 검증한다.
def _required_integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer.")
    return value
