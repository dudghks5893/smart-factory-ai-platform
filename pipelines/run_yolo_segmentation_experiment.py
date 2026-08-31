"""Run one validation-only controlled YOLO experiment with CUDA telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torchvision  # type: ignore[import-untyped]

from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.evaluation.final_benchmark import RepositoryProvenance, resolve_repository_provenance
from ml.evaluation.yolo_segmentation import serialize_ultralytics_metrics
from ml.evaluation.yolo_segmentation_visualization import (
    render_baseline_candidate_comparison,
)
from ml.experiments.gpu_telemetry import (
    GpuTelemetrySampler,
    collect_torch_cuda_metrics,
    query_nvidia_driver_version,
    reset_torch_cuda_peaks,
    sample_nvidia_smi,
)
from ml.experiments.yolo_crop_sampling import (
    CropTrainViewArtifact,
    RuntimeCropTrainViewArtifact,
    build_component_aware_crop_train_view,
    build_runtime_crop_train_view_adapter,
)
from ml.experiments.yolo_sampling import (
    RuntimeTrainViewArtifact,
    TrainViewArtifact,
    build_component_aware_train_view,
    build_runtime_train_view_adapter,
)
from ml.experiments.yolo_segmentation import (
    BASELINE_METADATA_SHA256,
    BASELINE_MODEL_SHA256,
    CROP_CONFIRMATION_INTERVENTION,
    ExperimentRecommendation,
    YoloExperimentConfig,
    build_experiment_metadata,
    confirm_c4_2c_candidate,
    load_yolo_experiment_config,
    recommend_experiment,
    validate_experiment_result,
)
from ml.experiments.yolo_workbench_visualization import (
    render_epoch_curves,
    render_gpu_telemetry,
    write_visualization_manifest,
)
from ml.training.device import resolve_device
from ml.training.yolo_segmentation import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    YoloSegmentationBaselineConfig,
    YoloTrainerOverrides,
    load_yolo_segmentation_config,
    validate_experiment_dataset,
    validate_yolo_artifact,
)
from pipelines.analyze_yolo_segmentation_errors import (
    ErrorAnalysisArtifacts,
    analyze_yolo_segmentation_errors,
)
from pipelines.train_yolo_segmentation import (
    DEFAULT_DATASET_ROOT,
    TrainingRunner,
    YoloTrainingResult,
    run_ultralytics_training,
    train_yolo_segmentation,
    write_runtime_dataset_yaml,
)
from shared.hashing import sha256_file

DEFAULT_EXPERIMENT_CONFIG = Path(
    "configs/experiments/yolo_segmentation/c4_2a_yolo11n_seg_imgsz1024_seed42.yaml"
)
DEFAULT_BASELINE_ARTIFACT_DIR = Path(
    "artifacts/runtime/yolo_segmentation/smartfactory_yolo11n_seg_metal_nut_seed42_t4"
)


@dataclass(frozen=True)
class ValidationMetricsResult:
    """Ultralytics validation box/mask metrics from one fixed model."""

    overall: dict[str, dict[str, float]]
    per_class: dict[str, dict[str, dict[str, float]]]
    framework_version: str
    device: str
    parameter_count: int
    model_size_bytes: int


@dataclass(frozen=True)
class ExperimentRunArtifacts:
    """Final evidence and candidate artifact paths from one successful experiment."""

    experiment_dir: Path
    experiment_result_path: Path
    comparison_path: Path
    telemetry_path: Path
    candidate_artifact_dir: Path
    package_path: Path
    package_metadata_path: Path


@dataclass(frozen=True)
class PreparedExperimentDataset:
    """Validated split records and optional C4-2B runtime training adapters."""

    validated_records: tuple[DerivedManifestRecord, ...]
    train_view: TrainViewArtifact | None
    runtime_train_view: RuntimeTrainViewArtifact | None
    runtime_dataset_yaml: Path
    crop_train_view: CropTrainViewArtifact | None = None
    runtime_crop_train_view: RuntimeCropTrainViewArtifact | None = None


type ValidationRunner = Callable[
    [YoloSegmentationBaselineConfig, Path, Path, Path, str], ValidationMetricsResult
]
type ErrorAnalysisRunner = Callable[..., ErrorAnalysisArtifacts]


# ADD 2026-08-27: Experiment evidence를 deterministic strict JSON으로 저장한다.
def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


# ADD 2026-08-27: Framework CUDA device argument를 project device policy에서 변환한다.
def _framework_device(requested_device: str) -> str | int | None:
    resolved = resolve_device(requested_device)
    if resolved.type == "cuda":
        return 0
    if requested_device != "auto":
        return resolved.type
    return None


# ADD 2026-08-27: Val metrics를 계산한다. → MODIFY 2026-08-28: Runtime YAML에서 test를 제외한다.
def run_ultralytics_validation_metrics(
    config: YoloSegmentationBaselineConfig,
    model_artifact_dir: Path,
    dataset_root: Path,
    output_dir: Path,
    requested_device: str,
) -> ValidationMetricsResult:
    metadata = validate_yolo_artifact(
        model_artifact_dir,
        expected_contract=config.dataset_contract,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_yaml = write_runtime_dataset_yaml(
        dataset_root=dataset_root,
        destination=output_dir / "dataset.runtime.yaml",
        classes=config.dataset_contract.classes,
        include_test=False,
    )
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version

    model_path = model_artifact_dir / MODEL_FILENAME
    model = YOLO(str(model_path), task=config.model.task)
    if ultralytics_version != metadata.framework_version:
        raise ValueError("Validation framework version does not match artifact metadata.")

    # Best checkpoint를 derived val split에서만 평가하고 test metric path를 호출하지 않는다.
    metrics = model.val(
        data=str(dataset_yaml),
        split="val",
        imgsz=config.training.imgsz,
        batch=config.training.batch,
        workers=config.training.workers,
        device=_framework_device(requested_device),
        project=str(output_dir),
        name="ultralytics-validation",
        exist_ok=True,
        plots=False,
        save_json=False,
        verbose=True,
    )
    overall, per_class = serialize_ultralytics_metrics(
        metrics,
        classes=config.dataset_contract.classes,
    )
    torch_model = getattr(model, "model", None)
    parameters = getattr(torch_model, "parameters", None)
    if not callable(parameters):
        raise ValueError("Loaded Ultralytics model does not expose parameter tensors.")
    parameter_count = sum(int(parameter.numel()) for parameter in parameters())
    if parameter_count <= 0:
        raise ValueError("YOLO parameter count must be positive.")
    return ValidationMetricsResult(
        overall=overall,
        per_class=per_class,
        framework_version=ultralytics_version,
        device=str(resolve_device(requested_device)),
        parameter_count=parameter_count,
        model_size_bytes=model_path.stat().st_size,
    )


# ADD 2026-08-27: Flat training artifact를 C4-1 runtime bundle view로 byte-identical하게 복사한다.
def create_diagnostic_runtime_bundle(
    model_artifact_dir: Path,
    destination: Path,
) -> Path:
    if destination.exists():
        raise FileExistsError(f"Diagnostic runtime bundle already exists: {destination}")
    model_dir = destination / "model"
    model_dir.mkdir(parents=True)
    for filename in (MODEL_FILENAME, METADATA_FILENAME):
        source = model_artifact_dir / filename
        target = model_dir / filename
        shutil.copy2(source, target)
        if sha256_file(source) != sha256_file(target):
            raise RuntimeError("Diagnostic runtime bundle copy changed artifact bytes.")
    return destination


# ADD 2026-08-27: Ultralytics results.csv에서 completed epoch와 early-stop evidence를 읽는다.
def read_training_progress(
    runtime_dir: Path,
    *,
    configured_epochs: int,
) -> dict[str, int | bool]:
    results_path = runtime_dir / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError("Ultralytics training results.csv is missing.")
    with results_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Ultralytics training results.csv contains no completed epochs.")
    return {
        "epochs_completed": len(rows),
        "early_stopping": len(rows) < configured_epochs,
    }


# ADD 2026-08-27: C4-1 outputs를 validation-only summary/sample payload로 복원한다.
def _load_analysis_payload(
    artifacts: ErrorAnalysisArtifacts,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    samples = [
        json.loads(line)
        for line in artifacts.sample_analysis_path.read_text(encoding="utf-8").splitlines()
    ]
    if summary.get("split") != "val" or summary.get("test_split_used") is not False:
        raise ValueError("Error-analysis payload violated the validation-only protocol.")
    return summary, samples


# ADD 2026-08-27: Quality payload를 만든다. → MODIFY 2026-08-31: Region evidence를 보존한다.
def build_quality_payload(
    validation: ValidationMetricsResult,
    analysis_artifacts: ErrorAnalysisArtifacts,
) -> dict[str, Any]:
    summary, samples = _load_analysis_payload(analysis_artifacts)
    aggregate = summary["aggregate"]
    negative = aggregate["negative_analysis"]
    failure_modes = {
        "small_recall": aggregate["size_analysis"]["small"]["recall"],
        "medium_recall": aggregate["size_analysis"]["medium"]["recall"],
        "large_recall": aggregate["size_analysis"]["large"]["recall"],
        "single_component_recall": aggregate["component_analysis"]["single_component"]["recall"],
        "multi_component_recall": aggregate["component_analysis"]["multi_component"]["recall"],
        "good_negative_fp_image_count": negative["false_positive_image_count"],
        "good_negative_fp_image_rate": negative["false_positive_image_rate"],
        "good_negative_fp_instance_count": negative["false_positive_instance_count"],
        "positive_unmatched_prediction_count": (
            aggregate["fp"] - negative["false_positive_instance_count"]
        ),
        "wrong_class_sample_count": aggregate["error_taxonomy"]["secondary"].get("WRONG_CLASS", 0),
        "low_iou_sample_count": aggregate["error_taxonomy"]["secondary"].get(
            "LOW_IOU_LOCALIZATION", 0
        ),
        "complete_miss_sample_count": sum(
            sample["ground_truth_instance_count"] > 0 and sample["predicted_instance_count"] == 0
            for sample in samples
        ),
    }
    result = {
        "split": "val",
        "test_split_used": False,
        "ultralytics": validation.overall,
        "ultralytics_per_class": validation.per_class,
        "diagnostic": {
            "protocol": "confidence_0.25_class_aware_mask_iou_0.5",
            "tp": aggregate["tp"],
            "fp": aggregate["fp"],
            "fn": aggregate["fn"],
            "precision": aggregate["precision"],
            "recall": aggregate["recall"],
            "f1": aggregate["f1"],
        },
        "diagnostic_per_class": aggregate["per_class"],
        "failure_modes": failure_modes,
        "error_taxonomy": aggregate["error_taxonomy"],
    }
    if analysis_artifacts.region_coverage_path is not None:
        result["secondary_region_coverage"] = json.loads(
            analysis_artifacts.region_coverage_path.read_text(encoding="utf-8")
        )
    return result


# ADD 2026-08-27: Runtime environment를 기록한다. → MODIFY 2026-08-31: GPU name을 포함한다.
def collect_software_environment(
    *,
    framework_version: str,
    requested_device: str,
) -> dict[str, Any]:
    resolved = resolve_device(requested_device)
    gpu_name = torch.cuda.get_device_name(0) if resolved.type == "cuda" else None
    return {
        "benchmark_domain": "kaggle_nvidia_cuda_training",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "cuda_runtime_version": torch.version.cuda,
        "nvidia_driver_version": query_nvidia_driver_version(),
        "ultralytics_version": framework_version,
        "requested_device": requested_device,
        "actual_device": str(resolved),
        "gpu_name": gpu_name,
    }


# ADD 2026-08-28: Dataset을 준비한다. → MODIFY 2026-08-31: Sealed C4-2C crop view를 연결한다.
def prepare_experiment_training_dataset(
    *,
    experiment_config: YoloExperimentConfig,
    baseline_config: YoloSegmentationBaselineConfig,
    dataset_root: Path,
    repository_root: Path,
    experiment_dir: Path,
) -> PreparedExperimentDataset:
    # Sealed test row는 split gating 이후 materialize하지 않고 train/val content만 검증한다.
    validated_records = validate_experiment_dataset(
        dataset_root,
        baseline_config.dataset_contract,
    )
    train_records = tuple(record for record in validated_records if record.derived_split == "train")
    if len(train_records) != baseline_config.dataset_contract.sample_counts["train"]:
        raise ValueError("Controlled experiment validated train record count changed.")

    if experiment_config.intervention_type == CROP_CONFIRMATION_INTERVENTION:
        policy = experiment_config.crop_sampling_policy
        expected_crop = experiment_config.expected_crop_train_view
        if policy is None or expected_crop is None:
            raise ValueError("C4-2C crop train-view config is incomplete.")
        crop_view = build_component_aware_crop_train_view(
            dataset_root=dataset_root,
            train_records=train_records,
            contract=baseline_config.dataset_contract,
            experiment_id=experiment_config.experiment_id,
            crop_size=policy.crop_size,
            output_dir=experiment_dir / "train_view",
        )
        expected_crop.validate_evidence(crop_view.evidence)
        crop_runtime = build_runtime_crop_train_view_adapter(
            dataset_root=dataset_root,
            crop_train_view=crop_view,
            destination=experiment_dir / "runtime" / "train.runtime.txt",
        )
        runtime_yaml = write_runtime_dataset_yaml(
            dataset_root=dataset_root,
            destination=experiment_dir / "runtime" / "dataset.runtime.yaml",
            classes=baseline_config.dataset_contract.classes,
            include_test=False,
            train_source=crop_runtime.train_list_path,
        )
        return PreparedExperimentDataset(
            validated_records=validated_records,
            train_view=None,
            runtime_train_view=None,
            runtime_dataset_yaml=runtime_yaml,
            crop_train_view=crop_view,
            runtime_crop_train_view=crop_runtime,
        )
    if experiment_config.sampling_policy is None:
        runtime_dataset_yaml = write_runtime_dataset_yaml(
            dataset_root=dataset_root,
            destination=experiment_dir / "runtime" / "dataset.runtime.yaml",
            classes=baseline_config.dataset_contract.classes,
            include_test=False,
        )
        return PreparedExperimentDataset(
            validated_records,
            None,
            None,
            runtime_dataset_yaml,
        )

    # Validated train records로 portable evidence를 만든 뒤 machine-local absolute list로 변환한다.
    train_view = build_component_aware_train_view(
        repository_root=repository_root,
        dataset_root=dataset_root,
        train_records=train_records,
        contract=baseline_config.dataset_contract,
        experiment_id=experiment_config.experiment_id,
    )
    if train_view.output_dir != experiment_dir / "train_view":
        raise ValueError("C4-2B train-view output does not match the experiment namespace.")
    expected = experiment_config.expected_train_view
    if expected is None:
        raise ValueError("C4-2B expected train-view snapshot is missing.")
    expected.validate_evidence(train_view.evidence)
    runtime_adapter = build_runtime_train_view_adapter(
        repository_root=repository_root,
        dataset_root=dataset_root,
        portable_train_view=train_view,
        destination=experiment_dir / "runtime" / "train.runtime.txt",
    )
    runtime_dataset_yaml = write_runtime_dataset_yaml(
        dataset_root=dataset_root,
        destination=experiment_dir / "runtime" / "dataset.runtime.yaml",
        classes=baseline_config.dataset_contract.classes,
        include_test=False,
        train_source=runtime_adapter.train_list_path,
    )
    return PreparedExperimentDataset(
        validated_records=validated_records,
        train_view=train_view,
        runtime_train_view=runtime_adapter,
        runtime_dataset_yaml=runtime_dataset_yaml,
    )


# ADD 2026-08-27: Evidence ZIP을 만든다. → MODIFY 2026-08-31: Portable crop provenance도 포함한다.
def build_experiment_package(
    *,
    experiment_dir: Path,
    candidate_artifact_dir: Path,
    experiment_config_path: Path,
    package_path: Path,
    train_view: TrainViewArtifact | None = None,
    crop_train_view: CropTrainViewArtifact | None = None,
) -> Path:
    if package_path.exists():
        raise FileExistsError(f"Experiment package already exists: {package_path}")
    package_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_names: tuple[str, ...] = (
        "experiment_metadata.json",
        "training_metrics.json",
        "validation_metrics.json",
        "error_analysis_summary.json",
        "resource_telemetry.json",
        "comparison_to_baseline.json",
        "experiment_result.json",
        "environment.json",
        "epoch_metrics.jsonl",
        "visualization_manifest.json",
        "region_coverage.json",
    )
    if not (experiment_dir / "region_coverage.json").is_file():
        evidence_names = tuple(name for name in evidence_names if name != "region_coverage.json")
    files: list[tuple[Path, str]] = [
        (experiment_dir / name, f"evidence/{name}") for name in evidence_names
    ]
    if train_view is not None:
        train_view.evidence.validate()
        if sha256_file(train_view.train_list_path) != train_view.evidence.train_list_sha256:
            raise ValueError("Packaged train-view list SHA does not match its evidence.")
        files.extend(
            (
                (train_view.metadata_path, "evidence/train_view_metadata.json"),
                (train_view.train_list_path, "evidence/train_view.txt"),
            )
        )
    if crop_train_view is not None:
        crop_train_view.evidence.validate()
        if sha256_file(crop_train_view.train_list_path) != (
            crop_train_view.evidence.portable_train_list_sha256
        ):
            raise ValueError("Packaged C4-2C train-view SHA does not match its evidence.")
        files.extend(
            (
                (crop_train_view.metadata_path, "evidence/train_view_metadata.json"),
                (crop_train_view.train_list_path, "evidence/train_view.txt"),
            )
        )
        for crop in crop_train_view.evidence.crops:
            files.extend(
                (
                    (
                        crop_train_view.output_dir / crop.generated_image_path,
                        f"evidence/train_view/{crop.generated_image_path}",
                    ),
                    (
                        crop_train_view.output_dir / crop.generated_label_path,
                        f"evidence/train_view/{crop.generated_label_path}",
                    ),
                )
            )
    files.extend(
        (
            (candidate_artifact_dir / MODEL_FILENAME, f"model/{MODEL_FILENAME}"),
            (candidate_artifact_dir / METADATA_FILENAME, f"model/{METADATA_FILENAME}"),
            (experiment_config_path, f"config/{experiment_config_path.name}"),
        )
    )
    hashes: list[str] = []
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, archive_name in files:
            if not source.is_file():
                raise FileNotFoundError(f"Experiment package source is missing: {source}")
            archive.write(source, archive_name)
            hashes.append(f"{sha256_file(source)}  {archive_name}")
        archive.writestr("SHA256SUMS.txt", "\n".join(sorted(hashes)) + "\n")
    return package_path


# ADD 2026-08-27: Experiment lifecycle 전후 approved baseline SHA가 동일한지 확인한다.
def _assert_baseline_immutable(
    baseline_artifact_dir: Path,
    *,
    model_sha256: str,
    metadata_sha256: str,
) -> None:
    if (
        sha256_file(baseline_artifact_dir / "model" / MODEL_FILENAME) != model_sha256
        or sha256_file(baseline_artifact_dir / "model" / METADATA_FILENAME) != metadata_sha256
    ):
        raise RuntimeError("Baseline runtime artifact changed during the experiment.")


# ADD 2026-08-27: Lifecycle을 조율한다. → MODIFY 2026-08-31: C4-2C confirmation을 연결한다.
def run_yolo_segmentation_experiment(
    *,
    experiment_config: YoloExperimentConfig,
    dataset_root: Path,
    baseline_artifact_dir: Path,
    requested_device: str,
    repository_root: Path,
    training_runner: TrainingRunner = run_ultralytics_training,
    validation_runner: ValidationRunner = run_ultralytics_validation_metrics,
    error_analysis_runner: ErrorAnalysisRunner = analyze_yolo_segmentation_errors,
    repository_provenance: RepositoryProvenance | None = None,
) -> ExperimentRunArtifacts:
    baseline_config = load_yolo_segmentation_config(experiment_config.baseline_config_path)
    candidate_config = experiment_config.training_config(baseline_config)
    is_c4_2c = experiment_config.intervention_type == CROP_CONFIRMATION_INTERVENTION
    if requested_device != "cuda" or str(resolve_device(requested_device)) != "cuda":
        raise ValueError("Controlled YOLO experiment requires explicit available CUDA.")
    experiment_dir = experiment_config.output.experiment_root / experiment_config.experiment_id
    if experiment_dir.exists():
        raise FileExistsError(f"Experiment output already exists: {experiment_dir}")
    experiment_dir.mkdir(parents=True)
    prepared_dataset = prepare_experiment_training_dataset(
        experiment_config=experiment_config,
        baseline_config=baseline_config,
        dataset_root=dataset_root,
        repository_root=repository_root,
        experiment_dir=experiment_dir,
    )
    validated_analysis_records = prepared_dataset.validated_records

    baseline_model_path = baseline_artifact_dir / "model" / MODEL_FILENAME
    baseline_metadata_path = baseline_artifact_dir / "model" / METADATA_FILENAME
    baseline_model_sha = sha256_file(baseline_model_path)
    baseline_metadata_sha = sha256_file(baseline_metadata_path)
    if (
        baseline_model_sha != BASELINE_MODEL_SHA256
        or baseline_metadata_sha != BASELINE_METADATA_SHA256
    ):
        raise ValueError("Baseline runtime artifact SHA does not match the approved C4 reference.")
    provenance = repository_provenance or resolve_repository_provenance(repository_root)
    manifest_sha = sha256_file(dataset_root / "manifest.csv")
    metadata_payload = build_experiment_metadata(
        experiment_config,
        git_commit=provenance.git_commit,
        manifest_sha256=manifest_sha,
    )
    metadata_payload["working_tree_dirty"] = provenance.working_tree_dirty
    metadata_payload["status"] = "RUNNING"
    metadata_payload["dataset"] = {
        "manifest_sha256": manifest_sha,
        "semantic_fingerprint_sha256": baseline_config.dataset_contract.semantic_fingerprint_sha256,
        "category": baseline_config.dataset_contract.category,
        "classes": baseline_config.dataset_contract.classes,
        "test_used": False,
    }
    if prepared_dataset.train_view is not None:
        metadata_payload["train_view"] = prepared_dataset.train_view.evidence.to_json_dict()
        runtime_view = prepared_dataset.runtime_train_view
        if runtime_view is None:
            raise RuntimeError("C4-2B runtime train-view adapter is missing.")
        metadata_payload["runtime_train_view_adapter"] = {
            "source_train_list_sha256": runtime_view.source_train_list_sha256,
            "entry_count": runtime_view.entry_count,
            "order_and_multiplicity_verified": True,
            "dataset_root_containment_verified": True,
            "portable_package_evidence": False,
        }
    if prepared_dataset.crop_train_view is not None:
        metadata_payload["train_view"] = prepared_dataset.crop_train_view.evidence.to_json_dict()
        crop_runtime = prepared_dataset.runtime_crop_train_view
        if crop_runtime is None:
            raise RuntimeError("C4-2C runtime crop train-view adapter is missing.")
        metadata_payload["runtime_train_view_adapter"] = {
            "source_train_list_sha256": crop_runtime.source_train_list_sha256,
            "entry_count": crop_runtime.entry_count,
            "order_and_multiplicity_verified": True,
            "dataset_and_generated_root_containment_verified": True,
            "portable_package_evidence": False,
        }
    _write_json(experiment_dir / "experiment_metadata.json", metadata_payload)

    # Candidate 결과를 보기 전에 동일 val protocol로 baseline quality reference를 복원한다.
    baseline_validation = validation_runner(
        baseline_config,
        baseline_artifact_dir / "model",
        dataset_root,
        experiment_dir / "baseline_framework_validation",
        requested_device,
    )
    baseline_analysis = error_analysis_runner(
        config=baseline_config,
        dataset_root=dataset_root,
        artifact_dir=baseline_artifact_dir,
        output_dir=experiment_dir / "baseline_error_analysis",
        requested_device=requested_device,
        size_policy_override=experiment_config.validation_protocol.size_policy(),
        validated_records=validated_analysis_records,
        c4_2c_confirmation_protocol=is_c4_2c,
    )
    quality_before = build_quality_payload(baseline_validation, baseline_analysis)
    environment = collect_software_environment(
        framework_version=baseline_validation.framework_version,
        requested_device=requested_device,
    )
    _write_json(experiment_dir / "environment.json", environment)

    sampler = GpuTelemetrySampler(
        sample_interval_seconds=experiment_config.telemetry.sample_interval_seconds,
        sample_provider=lambda: sample_nvidia_smi(
            timeout_seconds=experiment_config.telemetry.nvidia_smi_timeout_seconds
        ),
    )
    cuda_peak_reset_succeeded = reset_torch_cuda_peaks()
    training_started = datetime.now(UTC)
    started = perf_counter()
    training_result: YoloTrainingResult | None = None
    training_error: Exception | None = None
    try:
        with sampler:
            experiment_overrides = (
                YoloTrainerOverrides(**asdict(experiment_config.trainer_overrides))
                if experiment_config.trainer_overrides is not None
                else None
            )
            training_result = train_yolo_segmentation(
                config=candidate_config,
                dataset_root=dataset_root,
                artifact_id=experiment_config.experiment_id,
                requested_device=requested_device,
                training_runner=training_runner,
                prepared_dataset_yaml=prepared_dataset.runtime_dataset_yaml,
                experiment_overrides=experiment_overrides,
            )
    except Exception as exc:
        training_error = exc
    training_duration_seconds = perf_counter() - started
    training_ended = datetime.now(UTC)
    resource_metrics = {
        "training_started_at": training_started.isoformat(),
        "training_ended_at": training_ended.isoformat(),
        "training_wall_clock_seconds": training_duration_seconds,
        "pytorch_cuda_peak_reset_succeeded": cuda_peak_reset_succeeded,
        "pytorch_cuda": collect_torch_cuda_metrics(),
        "nvidia_smi": sampler.summary(),
    }
    if prepared_dataset.train_view is not None or prepared_dataset.crop_train_view is not None:
        batch = candidate_config.training.batch
        if prepared_dataset.train_view is not None:
            sampling_evidence = prepared_dataset.train_view.evidence
            canonical_count = sampling_evidence.unique_train_count
            expanded_count = sampling_evidence.expanded_entry_count
            canonical_positive = sampling_evidence.unique_positive_count
            expanded_positive = sampling_evidence.expanded_positive_count
            canonical_negative = sampling_evidence.unique_good_negative_count
            expanded_negative = sampling_evidence.expanded_good_negative_count
        else:
            if prepared_dataset.crop_train_view is None:
                raise RuntimeError("Prepared train-view evidence is missing.")
            crop_evidence = prepared_dataset.crop_train_view.evidence
            canonical_count = crop_evidence.canonical_entry_count
            expanded_count = crop_evidence.total_entry_count
            canonical_positive = crop_evidence.canonical_positive_count
            expanded_positive = crop_evidence.positive_exposure
            canonical_negative = crop_evidence.canonical_negative_count
            expanded_negative = crop_evidence.negative_exposure
        resource_metrics["train_view_exposure"] = {
            "canonical_train_entries": canonical_count,
            "expanded_train_entries": expanded_count,
            "canonical_positive_exposure": canonical_positive,
            "expanded_positive_exposure": expanded_positive,
            "canonical_good_negative_exposure": canonical_negative,
            "expanded_good_negative_exposure": expanded_negative,
            "nominal_batches_per_epoch_before": (canonical_count + batch - 1) // batch,
            "nominal_batches_per_epoch_after": (expanded_count + batch - 1) // batch,
        }
    _write_json(experiment_dir / "resource_telemetry.json", resource_metrics)
    epoch_log_source = (
        candidate_config.output.training_runtime_root
        / experiment_config.experiment_id
        / "epoch_metrics.jsonl"
    )
    if training_error is not None:
        if epoch_log_source.is_file():
            shutil.copy2(epoch_log_source, experiment_dir / "epoch_metrics.partial.jsonl")
        metadata_payload["status"] = "REJECTED"
        metadata_payload["decision"] = "REJECT"
        _write_json(experiment_dir / "experiment_metadata.json", metadata_payload)
        failure_payload = {
            "experiment_id": experiment_config.experiment_id,
            "status": "REJECTED",
            "split": "val",
            "test_split_used": False,
            "decision": "REJECT",
            "decision_reason": "Training failed; batch was not automatically changed.",
            "failure": {
                "type": type(training_error).__name__,
                "message": str(training_error),
                "batch": candidate_config.training.batch,
                "imgsz": candidate_config.training.imgsz,
            },
            "resource_metrics": resource_metrics,
            "environment": environment,
        }
        _write_json(experiment_dir / "experiment_result.json", failure_payload)
        _assert_baseline_immutable(
            baseline_artifact_dir,
            model_sha256=baseline_model_sha,
            metadata_sha256=baseline_metadata_sha,
        )
        raise training_error
    if training_result is None:
        raise RuntimeError("Training completed without a project artifact result.")

    # Completed epoch evidence를 experiment namespace에 byte-identical하게 보존한다.
    epoch_log_source = training_result.runtime_dir / "epoch_metrics.jsonl"
    epoch_log_path = experiment_dir / "epoch_metrics.jsonl"
    if not epoch_log_source.is_file():
        raise FileNotFoundError("Project-owned epoch_metrics.jsonl is missing.")
    shutil.copy2(epoch_log_source, epoch_log_path)
    if sha256_file(epoch_log_source) != sha256_file(epoch_log_path):
        raise RuntimeError("Epoch metrics evidence copy changed bytes.")

    progress = read_training_progress(
        training_result.runtime_dir,
        configured_epochs=candidate_config.training.epochs,
    )
    training_metrics = {
        **progress,
        "best_epoch": training_result.metadata.best_epoch,
        "best_checkpoint": training_result.metadata.source_checkpoint,
        "training_wall_clock_seconds": training_duration_seconds,
        "average_seconds_per_completed_epoch": (
            training_duration_seconds / int(progress["epochs_completed"])
        ),
        "model_size_bytes": (training_result.artifact_dir / MODEL_FILENAME).stat().st_size,
        "model_size_mib": (
            (training_result.artifact_dir / MODEL_FILENAME).stat().st_size / 1024**2
        ),
    }
    _write_json(experiment_dir / "training_metrics.json", training_metrics)
    resource_metrics["training"] = training_metrics
    _write_json(experiment_dir / "resource_telemetry.json", resource_metrics)
    training_visualization_dir = experiment_dir / "visualizations" / "training"
    epoch_curve_path = render_epoch_curves(
        epoch_log_path,
        training_visualization_dir / "training_curves.png",
    )
    gpu_visualization_path = render_gpu_telemetry(
        experiment_dir / "resource_telemetry.json",
        training_visualization_dir / "gpu_telemetry.png",
    )

    candidate_validation = validation_runner(
        candidate_config,
        training_result.artifact_dir,
        dataset_root,
        experiment_dir / "candidate_framework_validation",
        requested_device,
    )
    candidate_runtime_bundle = create_diagnostic_runtime_bundle(
        training_result.artifact_dir,
        experiment_dir / "candidate_runtime_bundle",
    )
    candidate_analysis = error_analysis_runner(
        config=candidate_config,
        dataset_root=dataset_root,
        artifact_dir=candidate_runtime_bundle,
        output_dir=experiment_dir / "candidate_error_analysis",
        requested_device=requested_device,
        size_policy_override=experiment_config.validation_protocol.size_policy(),
        validated_records=validated_analysis_records,
        c4_2c_confirmation_protocol=is_c4_2c,
    )
    if candidate_analysis.region_coverage_path is not None:
        shutil.copy2(
            candidate_analysis.region_coverage_path,
            experiment_dir / "region_coverage.json",
        )
    quality_after = build_quality_payload(candidate_validation, candidate_analysis)
    validation_metrics_payload = {
        "split": "val",
        "test_split_used": False,
        "quality_before": quality_before,
        "quality_after": quality_after,
    }
    _write_json(experiment_dir / "validation_metrics.json", validation_metrics_payload)
    candidate_summary = json.loads(candidate_analysis.summary_path.read_text(encoding="utf-8"))
    _write_json(experiment_dir / "error_analysis_summary.json", candidate_summary)

    # 동일 validation sample의 Baseline/Candidate evidence를 regression-first로 비교한다.
    comparison_gallery_path = render_baseline_candidate_comparison(
        baseline_sample_analysis_path=baseline_analysis.sample_analysis_path,
        candidate_sample_analysis_path=candidate_analysis.sample_analysis_path,
        baseline_cards_dir=(
            baseline_analysis.output_dir / "visualizations" / "validation_failures" / "all_samples"
        ),
        candidate_cards_dir=(
            candidate_analysis.output_dir / "visualizations" / "validation_failures" / "all_samples"
        ),
        output_dir=experiment_dir / "visualizations" / "baseline_vs_candidate",
    )
    write_visualization_manifest(
        output_path=experiment_dir / "visualization_manifest.json",
        experiment_id=experiment_config.experiment_id,
        manifest_sha256=manifest_sha,
        repository=provenance.to_json_dict(),
        entries=[
            {
                "visualization_type": "training_curves",
                "source_split": "val",
                "generated_path": epoch_curve_path.as_posix(),
            },
            {
                "visualization_type": "gpu_telemetry",
                "source_split": "none",
                "generated_path": gpu_visualization_path.as_posix(),
            },
            {
                "visualization_type": "baseline_vs_candidate_failures",
                "source_split": "val",
                "generated_path": comparison_gallery_path.as_posix(),
            },
        ],
    )

    recommendation: ExperimentRecommendation = (
        confirm_c4_2c_candidate(
            quality_after=quality_after,
            protocol=experiment_config.confirmation_protocol,
        )
        if is_c4_2c and experiment_config.confirmation_protocol is not None
        else recommend_experiment(
            quality_before=quality_before,
            quality_after=quality_after,
            policy=experiment_config.decision_policy,
        )
    )
    comparison = {
        "experiment_id": experiment_config.experiment_id,
        "split": "val",
        "test_split_used": False,
        "historical_baseline_reference": experiment_config.baseline_evidence,
        "quality_before": quality_before,
        "quality_after": quality_after,
        "resource_cost_after": resource_metrics,
        "recommendation": asdict(recommendation),
    }
    if is_c4_2c:
        comparison["primary_confirmation"] = asdict(recommendation)
        comparison["secondary_evidence"] = {
            "blocking": False,
            "strict_recall": quality_after["diagnostic"]["recall"],
            "strict_f1": quality_after["diagnostic"]["f1"],
            "framework_mask_recall": quality_after["ultralytics"]["mask"]["recall"],
            "region_coverage": quality_after.get("secondary_region_coverage"),
        }
    _write_json(experiment_dir / "comparison_to_baseline.json", comparison)
    if candidate_validation.framework_version != baseline_validation.framework_version:
        raise ValueError("Baseline and candidate validation framework versions differ.")

    model_path = training_result.artifact_dir / MODEL_FILENAME
    metadata_path = training_result.artifact_dir / METADATA_FILENAME
    result_payload = {
        "experiment_id": experiment_config.experiment_id,
        "hypothesis": experiment_config.hypothesis,
        "controlled_change": asdict(experiment_config.controlled_change),
        "intervention_type": experiment_config.intervention_type,
        "constants": experiment_config.baseline_identity,
        "split": "val",
        "test_split_used": False,
        "test_used": False,
        "quality_before": quality_before,
        "quality_after": quality_after,
        "resource_metrics": resource_metrics,
        "failure_mode_metrics": quality_after["failure_modes"],
        "model_sha256": sha256_file(model_path),
        "metadata_sha256": sha256_file(metadata_path),
        "manifest_sha256": manifest_sha,
        "experiment_config_sha256": sha256_file(experiment_config.config_path),
        "artifact_path": training_result.artifact_dir.as_posix(),
        "model_size_bytes": candidate_validation.model_size_bytes,
        "parameter_count": candidate_validation.parameter_count,
        "decision": recommendation.decision,
        "decision_reason": recommendation.decision_reason,
        "status": (
            "ACCEPTED"
            if recommendation.decision == "ACCEPT"
            else (
                "REJECTED"
                if recommendation.decision == "REJECT"
                else recommendation.decision
                if recommendation.decision in {"CONFIRMED_CANDIDATE", "CONFIRMATION_FAILED"}
                else "COMPLETED"
            )
        ),
        "repository": provenance.to_json_dict(),
    }
    if prepared_dataset.train_view is not None:
        result_payload["train_view"] = prepared_dataset.train_view.evidence.to_json_dict()
    if prepared_dataset.crop_train_view is not None:
        result_payload["train_view"] = prepared_dataset.crop_train_view.evidence.to_json_dict()
        result_payload["primary_confirmation"] = asdict(recommendation)
        result_payload["secondary_evidence"] = comparison["secondary_evidence"]
        if experiment_config.trainer_overrides is None:
            raise RuntimeError("C4-2C applied trainer args are missing.")
        result_payload["applied_trainer_args"] = asdict(experiment_config.trainer_overrides)
    validate_experiment_result(result_payload)
    result_path = experiment_dir / "experiment_result.json"
    _write_json(result_path, result_payload)
    metadata_payload["status"] = result_payload["status"]
    metadata_payload["decision"] = recommendation.decision
    _write_json(experiment_dir / "experiment_metadata.json", metadata_payload)

    package_path = experiment_config.output.package_root / f"{experiment_config.experiment_id}.zip"
    build_experiment_package(
        experiment_dir=experiment_dir,
        candidate_artifact_dir=training_result.artifact_dir,
        experiment_config_path=experiment_config.config_path,
        package_path=package_path,
        train_view=prepared_dataset.train_view,
        crop_train_view=prepared_dataset.crop_train_view,
    )
    package_metadata_path = experiment_dir / "package_metadata.json"
    _write_json(
        package_metadata_path,
        {
            "package_path": package_path.as_posix(),
            "package_sha256": sha256_file(package_path),
            "model_sha256": sha256_file(model_path),
            "metadata_sha256": sha256_file(metadata_path),
            "experiment_config_sha256": sha256_file(experiment_config.config_path),
        },
    )
    _assert_baseline_immutable(
        baseline_artifact_dir,
        model_sha256=baseline_model_sha,
        metadata_sha256=baseline_metadata_sha,
    )
    return ExperimentRunArtifacts(
        experiment_dir=experiment_dir,
        experiment_result_path=result_path,
        comparison_path=experiment_dir / "comparison_to_baseline.json",
        telemetry_path=experiment_dir / "resource_telemetry.json",
        candidate_artifact_dir=training_result.artifact_dir,
        package_path=package_path,
        package_metadata_path=package_metadata_path,
    )


# ADD 2026-08-27: Kaggle T4 experiment의 explicit paths와 CUDA device arguments를 정의한다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--baseline-artifact-dir",
        type=Path,
        default=DEFAULT_BASELINE_ARTIFACT_DIR,
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser.parse_args()


# ADD 2026-08-27: Repository-owned Kaggle runner를 실행하고 evidence/package 위치를 출력한다.
def main() -> int:
    args = parse_args()
    experiment_config = load_yolo_experiment_config(args.experiment_config)
    artifacts = run_yolo_segmentation_experiment(
        experiment_config=experiment_config,
        dataset_root=args.dataset,
        baseline_artifact_dir=args.baseline_artifact_dir,
        requested_device=args.device,
        repository_root=args.repository_root,
    )
    print("YOLO controlled experiment: PASS")
    print(f"Experiment result: {artifacts.experiment_result_path}")
    print(f"Candidate artifact: {artifacts.candidate_artifact_dir}")
    print(f"Package: {artifacts.package_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
