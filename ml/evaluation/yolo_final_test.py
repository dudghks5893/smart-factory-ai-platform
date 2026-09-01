"""Guarded one-time final-test evaluation for the frozen YOLO candidate."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml

from ml.datasets.segmentation_annotations import rasterize_segmentation_label_instances
from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.evaluation.final_benchmark import RepositoryProvenance, resolve_repository_provenance
from ml.evaluation.yolo_confirmation_prediction import (
    ConfirmationModel,
    predict_c4_2c_instances,
)
from ml.evaluation.yolo_segmentation import serialize_ultralytics_metrics
from ml.evaluation.yolo_segmentation_error_analysis import (
    GroundTruthInstance,
    PredictedInstance,
    SizeBucketPolicy,
    aggregate_analysis,
    analyze_sample,
    filter_predictions,
    mask_box,
    mask_overlap,
    match_instances,
)
from ml.experiments.yolo_final_candidate import (
    FINAL_CANDIDATE_SELECTION_BASIS,
    FINAL_CANDIDATE_STATE,
    FINAL_TEST_STATE,
    FinalCandidateManifest,
    OfficialCandidateEvidence,
    load_final_candidate_manifest,
    load_official_candidate_evidence,
    materialize_official_candidate_artifact,
    verify_official_candidate_identity,
)
from ml.experiments.yolo_segmentation import YoloExperimentConfig, load_yolo_experiment_config
from ml.training.device import resolve_device
from ml.training.yolo_segmentation import (
    YoloSegmentationBaselineConfig,
    load_yolo_segmentation_config,
    validate_final_test_dataset,
    validate_yolo_artifact,
)
from shared.hashing import is_sha256_digest, sha256_bytes, sha256_file

EXPECTED_FINAL_CANDIDATE_MANIFEST_SHA256 = (
    "2a26b1bc03a1876f828e12a625c69c76af5e8c5713e3f64be699feffe2e8aa09"
)
C4_3_FREEZE_COMMIT = "9c6916c74beed01875421e2faf1f8113232f2d15"
FINAL_TEST_SCHEMA_VERSION = 1
FINAL_TEST_COMPLETED_STATE = "FINAL_TEST_COMPLETED"
FINAL_TEST_OUTPUT_ROOT = Path("outputs/final_test/yolo_segmentation")
FINAL_TEST_RESULT_FILENAME = "final_test_result.json"
FINAL_TEST_PACKAGE_FILENAME = "final_test_evidence.zip"
FINAL_TEST_PACKAGE_CHECKSUM_FILENAME = "SHA256SUMS.txt"
FINAL_TEST_RUN_STATE_FILENAME = "run_state.json"
FINAL_TEST_READY_STATE = "READY"
FINAL_TEST_RUNNING_STATE = "RUNNING"
FINAL_TEST_PREPARATION_FAILED_STATE = "PREPARATION_FAILED"
FINAL_TEST_FAILED_STATE = "FINAL_TEST_FAILED"
FINAL_TEST_DIAGNOSTIC_CONFIDENCE = 0.25
FINAL_TEST_MATCH_IOU = 0.5
FINAL_TEST_REGION_COVERAGE = 0.5
FINAL_TEST_NEAR_MISS_MIN_IOU = 0.3


@dataclass(frozen=True)
class FinalTestProtocol:
    """Immutable framework and diagnostic protocol fixed before test access."""

    framework_split: str
    framework_confidence_policy: str
    diagnostic_confidence: float
    matching_method: str
    mask_iou_threshold: float
    prediction_initial_confidence: float
    prediction_iou: float
    max_detections: int
    retina_masks: bool
    mask_threshold: float
    mask_resize: str
    size_bucket_policy: SizeBucketPolicy

    # ADD 2026-09-01: C4-2C에서 고정된 final-test protocol 값을 변경 없이 검증한다.
    def validate(self) -> None:
        if (
            self.framework_split != "test"
            or self.framework_confidence_policy != "ultralytics_framework_default"
            or self.diagnostic_confidence != FINAL_TEST_DIAGNOSTIC_CONFIDENCE
            or self.matching_method != "class_aware_greedy_max_mask_iou"
            or self.mask_iou_threshold != FINAL_TEST_MATCH_IOU
            or self.prediction_initial_confidence != 0.001
            or self.prediction_iou != 0.7
            or self.max_detections != 300
            or self.retina_masks is not False
            or self.mask_threshold != 0.5
            or self.mask_resize != "opencv_inter_nearest"
        ):
            raise ValueError("C4-4 final-test evaluation protocol changed from the frozen recipe.")
        if self.size_bucket_policy.method != "validation_gt_mask_area_ratio_tertiles":
            raise ValueError("C4-4 size buckets must remain frozen from validation evidence.")

    # ADD 2026-09-01: Final-test protocol을 stable evidence mapping으로 변환한다.
    def to_json_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class FinalTestMeasurements:
    """Framework, diagnostic, region, environment, and runtime observations."""

    evaluation_split: str
    framework_metric_source: str
    sample_count: int
    framework_metrics: dict[str, dict[str, float]]
    framework_per_class_metrics: dict[str, dict[str, dict[str, float]]]
    diagnostic: dict[str, Any]
    failure_analysis: dict[str, Any]
    region_coverage: dict[str, float | int]
    sample_analysis: tuple[dict[str, Any], ...]
    environment: dict[str, Any]
    resource_metrics: dict[str, Any]
    started_at: str
    completed_at: str

    # ADD 2026-09-01: Completed measurement가 finite, test-sized evidence인지 검증한다.
    def validate(self) -> None:
        if (
            self.evaluation_split != "test"
            or self.framework_metric_source != "ultralytics_model.val(split=test)"
        ):
            raise ValueError("Final-test measurements must identify the actual test metric source.")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("Final-test sample count must be a positive integer.")
        if set(self.framework_metrics) != {"box", "mask"}:
            raise ValueError("Final-test framework metrics require box and mask sections.")
        metric_names = {"precision", "recall", "map50", "map50_95"}
        for component in self.framework_metrics.values():
            if set(component) != metric_names or any(
                type(value) not in {int, float} or not math.isfinite(float(value))
                for value in component.values()
            ):
                raise ValueError("Final-test framework metrics are incomplete or non-finite.")
        if set(self.framework_per_class_metrics) != {"bent", "color", "scratch"}:
            raise ValueError("Final-test per-class framework metrics are incomplete.")
        for class_metrics in self.framework_per_class_metrics.values():
            if set(class_metrics) != {"box", "mask"}:
                raise ValueError("Final-test per-class metrics require box and mask sections.")
            for component in class_metrics.values():
                if set(component) != metric_names or any(
                    type(value) not in {int, float} or not math.isfinite(float(value))
                    for value in component.values()
                ):
                    raise ValueError("Final-test per-class metrics are incomplete or non-finite.")
        required_diagnostic = {"tp", "fp", "fn", "precision", "recall", "f1", "per_class"}
        if not required_diagnostic.issubset(self.diagnostic):
            raise ValueError("Final-test strict diagnostic evidence is incomplete.")
        if any(
            type(self.diagnostic[name]) is not int or self.diagnostic[name] < 0
            for name in ("tp", "fp", "fn")
        ):
            raise ValueError("Final-test diagnostic counts must be non-negative integers.")
        required_failures = {
            "size_analysis",
            "component_analysis",
            "negative_analysis",
            "error_taxonomy",
            "complete_miss_sample_count",
            "wrong_class_sample_count",
        }
        if not required_failures.issubset(self.failure_analysis):
            raise ValueError("Final-test failure-focused evidence is incomplete.")
        required_region = {
            "strict_instance_gt_recall",
            "gt_component_coverage_recall_at_50",
            "small_gt_coverage_recall_at_50",
            "class_aware_union_iou",
            "class_aware_union_gt_coverage",
            "class_aware_union_prediction_precision",
            "near_miss_iou_030_to_050",
            "covered50_but_strict_instance_fail",
            "test_gt_total",
            "small_gt_total",
        }
        if set(self.region_coverage) != required_region:
            raise ValueError("Final-test Region Coverage evidence is incomplete.")
        required_environment = {"framework", "framework_version", "torch_version", "device"}
        if not required_environment.issubset(self.environment):
            raise ValueError("Final-test runtime environment evidence is incomplete.")
        wall_time = self.resource_metrics.get("evaluation_wall_time_seconds")
        if isinstance(wall_time, bool) or not isinstance(wall_time, (int, float)):
            raise ValueError("Final-test runtime evidence requires finite non-negative wall time.")
        if not math.isfinite(float(wall_time)) or float(wall_time) < 0.0:
            raise ValueError("Final-test runtime evidence requires finite non-negative wall time.")
        if len(self.sample_analysis) != self.sample_count:
            raise ValueError("Final-test sample diagnostics do not match the evaluated split.")
        _validate_timestamp(self.started_at, field="started_at")
        _validate_timestamp(self.completed_at, field="completed_at")
        try:
            json.dumps(asdict(self), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Final-test measurements must be strict JSON data.") from exc


@dataclass(frozen=True)
class FinalTestRunState:
    """Practical one-time namespace lifecycle independent of completed metric evidence."""

    schema_version: int
    lifecycle_state: str
    frozen_candidate_manifest_sha256: str
    execution_commit: str
    test_access_started: bool
    completed_evidence: bool
    cleanup_permitted: bool
    message: str

    # ADD 2026-09-01: Namespace state가 test-access와 cleanup 의미를 보존하는지 검증한다.
    def validate(self) -> None:
        expected_flags = {
            FINAL_TEST_READY_STATE: (False, False, True),
            FINAL_TEST_PREPARATION_FAILED_STATE: (False, False, True),
            FINAL_TEST_RUNNING_STATE: (True, False, False),
            FINAL_TEST_FAILED_STATE: (True, False, False),
            FINAL_TEST_COMPLETED_STATE: (True, True, False),
        }
        if self.schema_version != FINAL_TEST_SCHEMA_VERSION:
            raise ValueError("Unsupported C4-4 run-state schema version.")
        if expected_flags.get(self.lifecycle_state) != (
            self.test_access_started,
            self.completed_evidence,
            self.cleanup_permitted,
        ):
            raise ValueError("C4-4 run-state lifecycle flags are inconsistent.")
        if not is_sha256_digest(self.frozen_candidate_manifest_sha256):
            raise ValueError("C4-4 run-state frozen Manifest SHA is invalid.")
        if len(self.execution_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.execution_commit
        ):
            raise ValueError("C4-4 run-state execution commit is invalid.")
        if not self.message:
            raise ValueError("C4-4 run-state message must not be blank.")

    # ADD 2026-09-01: Namespace state를 deterministic strict JSON bytes로 직렬화한다.
    def to_json_bytes(self) -> bytes:
        self.validate()
        return (json.dumps(asdict(self), indent=2, sort_keys=True) + "\n").encode()


@dataclass(frozen=True)
class FinalTestEvidence:
    """Authoritative report-only evidence emitted after one guarded test run."""

    schema_version: int
    task: str
    model_family: str
    final_test_state: str
    evaluation_split: str
    frozen_candidate_manifest_sha256: str
    c4_3_freeze_commit: str
    c4_4_execution_commit: str
    selected_experiment_id: str
    selected_model_name: str
    official_package_sha256: str
    model_sha256: str
    metadata_sha256: str
    experiment_config_sha256: str
    packaged_experiment_result_sha256: str
    dataset_manifest_sha256: str
    candidate_was_frozen_before_test_access: bool
    candidate_selection_changed: bool
    threshold_tuned_on_test: bool
    test_used: bool
    test_split_used: bool
    protocol: dict[str, Any]
    measurements: dict[str, Any]

    # ADD 2026-09-01: Completed evidence의 frozen identity와 report-only lifecycle을 검증한다.
    def validate(self) -> None:
        if (
            self.schema_version != FINAL_TEST_SCHEMA_VERSION
            or self.final_test_state != FINAL_TEST_COMPLETED_STATE
            or self.evaluation_split != "test"
        ):
            raise ValueError("C4-4 final-test result lifecycle is invalid.")
        if (
            self.candidate_was_frozen_before_test_access is not True
            or self.candidate_selection_changed is not False
            or self.threshold_tuned_on_test is not False
            or self.test_used is not True
            or self.test_split_used is not True
        ):
            raise ValueError("C4-4 final-test result violates the frozen report-only contract.")
        for digest in (
            self.frozen_candidate_manifest_sha256,
            self.official_package_sha256,
            self.model_sha256,
            self.metadata_sha256,
            self.experiment_config_sha256,
            self.packaged_experiment_result_sha256,
            self.dataset_manifest_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C4-4 final-test provenance contains an invalid SHA-256.")
        for commit in (self.c4_3_freeze_commit, self.c4_4_execution_commit):
            if len(commit) != 40 or any(
                character not in "0123456789abcdef" for character in commit
            ):
                raise ValueError("C4-4 final-test Git provenance is invalid.")
        if not self.task or not self.model_family or not self.selected_experiment_id:
            raise ValueError("C4-4 final-test candidate identity is incomplete.")
        try:
            json.dumps(asdict(self), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("C4-4 final-test result must be strict JSON data.") from exc

    # ADD 2026-09-01: Completed final-test evidence를 deterministic pretty JSON bytes로 직렬화한다.
    def to_json_bytes(self) -> bytes:
        self.validate()
        return (json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


@dataclass(frozen=True)
class FinalTestPreflight:
    """Fully verified resources that still have not resolved the derived test split."""

    repository_root: Path
    frozen_manifest_path: Path
    frozen_manifest_sha256: str
    official_package_path: Path
    dataset_root: Path
    dataset_manifest_path: Path
    output_dir: Path
    candidate: FinalCandidateManifest
    official_evidence: OfficialCandidateEvidence
    experiment_config: YoloExperimentConfig
    baseline_config: YoloSegmentationBaselineConfig
    protocol: FinalTestProtocol
    repository_provenance: RepositoryProvenance
    requested_device: str
    provenance_resolver: ProvenanceResolver


@dataclass(frozen=True)
class FinalTestRunArtifacts:
    """Completed evidence and deterministic evidence-package paths."""

    output_dir: Path
    result_path: Path
    package_path: Path
    package_sha256: str
    evidence: FinalTestEvidence


type FinalTestEvaluator = Callable[[FinalTestPreflight, Path], FinalTestMeasurements]
type ArtifactMaterializer = Callable[[FinalTestPreflight], Path]
type ArtifactVerifier = Callable[[FinalTestPreflight, Path], None]
type ProvenanceResolver = Callable[[Path], RepositoryProvenance]


# ADD 2026-09-01: Evidence timestamps가 timezone-aware ISO-8601인지 검증한다.
def _validate_timestamp(value: str, *, field: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Final-test {field} must be ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Final-test {field} must include a timezone offset.")


# ADD 2026-09-01: Empty denominator를 zero rate로 표현하는 diagnostic ratio를 계산한다.
def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


# ADD 2026-09-01: Repository-owned path가 root 밖으로 벗어나지 않게 resolve한다.
def _repository_path(repository_root: Path, declared_path: Path, *, field: str) -> Path:
    resolved_root = repository_root.resolve()
    resolved = (
        declared_path.resolve()
        if declared_path.is_absolute()
        else (resolved_root / declared_path).resolve()
    )
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"C4-4 {field} must remain inside repository_root.") from exc
    return resolved


# ADD 2026-09-01: External runtime input path를 cwd-independent absolute path로 resolve한다.
def _runtime_path(repository_root: Path, declared_path: Path) -> Path:
    return (
        declared_path.resolve()
        if declared_path.is_absolute()
        else (repository_root.resolve() / declared_path).resolve()
    )


# ADD 2026-09-01: Final-test namespace를 frozen identity에서 한 가지 canonical path로 계산한다.
def _expected_output_dir(
    repository_root: Path,
    candidate: FinalCandidateManifest,
    manifest_sha256: str,
) -> Path:
    namespace = f"{candidate.selected_experiment_id}-{manifest_sha256[:12]}"
    return (repository_root / FINAL_TEST_OUTPUT_ROOT / namespace).resolve()


# ADD 2026-09-01: Canonical namespace의 atomic reservation staging path를 계산한다.
def _starting_output_dir(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.starting"


# ADD 2026-09-01: Final result/state file을 same-filesystem atomic replace로 기록한다.
def _write_atomic(path: Path, content: bytes, *, replace_existing: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace_existing:
        raise FileExistsError(f"C4-4 output already exists: {path}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        if path.exists() and not replace_existing:
            raise FileExistsError(f"C4-4 output already exists: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


# ADD 2026-09-01: One-time namespace lifecycle state를 atomic하게 전환한다.
def _write_run_state(
    preflight: FinalTestPreflight,
    *,
    lifecycle_state: str,
    message: str,
) -> FinalTestRunState:
    flags = {
        FINAL_TEST_READY_STATE: (False, False, True),
        FINAL_TEST_PREPARATION_FAILED_STATE: (False, False, True),
        FINAL_TEST_RUNNING_STATE: (True, False, False),
        FINAL_TEST_FAILED_STATE: (True, False, False),
        FINAL_TEST_COMPLETED_STATE: (True, True, False),
    }
    try:
        test_access_started, completed_evidence, cleanup_permitted = flags[lifecycle_state]
    except KeyError as exc:
        raise ValueError(f"Unsupported C4-4 run state: {lifecycle_state}") from exc
    state = FinalTestRunState(
        schema_version=FINAL_TEST_SCHEMA_VERSION,
        lifecycle_state=lifecycle_state,
        frozen_candidate_manifest_sha256=preflight.frozen_manifest_sha256,
        execution_commit=preflight.repository_provenance.git_commit,
        test_access_started=test_access_started,
        completed_evidence=completed_evidence,
        cleanup_permitted=cleanup_permitted,
        message=message,
    )
    _write_atomic(
        preflight.output_dir / FINAL_TEST_RUN_STATE_FILENAME,
        state.to_json_bytes(),
        replace_existing=True,
    )
    return state


# ADD 2026-09-01: Evaluator 호출 전에 READY state가 포함된 namespace를 atomic rename으로 예약한다.
def _reserve_output_namespace(preflight: FinalTestPreflight) -> None:
    if preflight.output_dir != _expected_output_dir(
        preflight.repository_root,
        preflight.candidate,
        preflight.frozen_manifest_sha256,
    ):
        raise ValueError("C4-4 preflight output namespace is not canonical.")
    if preflight.output_dir.exists():
        raise FileExistsError(f"C4-4 final-test namespace already exists: {preflight.output_dir}")
    preflight.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = _starting_output_dir(preflight.output_dir)
    if staging.exists():
        raise FileExistsError(f"C4-4 namespace reservation already exists: {staging}")
    staging.mkdir(exist_ok=False)
    staging_preflight = replace(preflight, output_dir=staging)
    _write_run_state(
        staging_preflight,
        lifecycle_state=FINAL_TEST_READY_STATE,
        message="Trust anchors verified; test evaluator has not been called.",
    )
    try:
        staging.rename(preflight.output_dir)
    except OSError:
        if staging.exists():
            shutil.rmtree(staging)
        raise


# ADD 2026-09-01: C4-2C config의 frozen model/checkpoint와 evaluation recipe를 검증한다.
def _build_final_test_protocol(
    candidate: FinalCandidateManifest,
    experiment: YoloExperimentConfig,
    baseline: YoloSegmentationBaselineConfig,
) -> FinalTestProtocol:
    if experiment.experiment_id != candidate.selected_experiment_id:
        raise ValueError("Experiment config identity does not match the frozen candidate.")
    controlled = experiment.training_config(baseline)
    if (
        controlled.model.architecture != candidate.model_family
        or controlled.model.weights != candidate.selected_model_name
        or controlled.model.task != candidate.task
        or controlled.training.seed != candidate.seed
    ):
        raise ValueError(
            "Experiment model/checkpoint identity does not match the frozen candidate."
        )
    confirmation = experiment.confirmation_protocol
    if confirmation is None:
        raise ValueError("Frozen C4-2C candidate requires its explicit confirmation protocol.")
    protocol = FinalTestProtocol(
        framework_split="test",
        framework_confidence_policy="ultralytics_framework_default",
        diagnostic_confidence=FINAL_TEST_DIAGNOSTIC_CONFIDENCE,
        matching_method=experiment.validation_protocol.matching_method,
        mask_iou_threshold=experiment.validation_protocol.mask_iou_threshold,
        prediction_initial_confidence=confirmation.initial_confidence,
        prediction_iou=confirmation.prediction_iou,
        max_detections=confirmation.max_det,
        retina_masks=confirmation.retina_masks,
        mask_threshold=confirmation.mask_threshold,
        mask_resize=confirmation.mask_resize,
        size_bucket_policy=experiment.validation_protocol.size_policy(),
    )
    protocol.validate()
    return protocol


# ADD 2026-09-01: Test row를 해석하지 않고 frozen/package/config/Manifest/Git/output을 선검증한다.
def prepare_yolo_final_test(
    *,
    frozen_manifest_path: Path,
    official_package_path: Path,
    dataset_root: Path,
    repository_root: Path,
    output_root: Path = FINAL_TEST_OUTPUT_ROOT,
    requested_device: str = "auto",
    provenance_resolver: ProvenanceResolver = resolve_repository_provenance,
) -> FinalTestPreflight:
    resolved_repository = repository_root.resolve()
    manifest_path = _repository_path(
        resolved_repository,
        frozen_manifest_path,
        field="frozen candidate manifest",
    )

    # Strict frozen lifecycle과 repository-owned manifest byte identity를 먼저 확인한다.
    candidate = load_final_candidate_manifest(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != EXPECTED_FINAL_CANDIDATE_MANIFEST_SHA256:
        raise ValueError("Frozen candidate manifest SHA-256 does not match C4-3.")
    if (
        candidate.selection_state != FINAL_CANDIDATE_STATE
        or candidate.selection_basis != FINAL_CANDIDATE_SELECTION_BASIS
        or candidate.final_test_state != FINAL_TEST_STATE
        or candidate.test_used is not False
        or candidate.test_split_used is not False
    ):
        raise ValueError("Frozen candidate is not eligible for C4-4 final-test access.")

    # External Official package를 C4-3 검증기로 확인하고 frozen pointer와 교차 검증한다.
    package_path = _runtime_path(resolved_repository, official_package_path)
    evidence = load_official_candidate_evidence(
        package_path,
        expected_package_sha256=candidate.official_package_sha256,
    )
    verify_official_candidate_identity(candidate, evidence)

    # Repository config bytes와 typed model/protocol identity를 frozen evidence에 고정한다.
    experiment_config_path = _repository_path(
        resolved_repository,
        Path("configs/experiments/yolo_segmentation") / f"{candidate.selected_experiment_id}.yaml",
        field="experiment config",
    )
    if sha256_file(experiment_config_path) != candidate.experiment_config_sha256:
        raise ValueError("Repository experiment config SHA-256 does not match the candidate.")
    experiment = load_yolo_experiment_config(experiment_config_path)
    baseline = load_yolo_segmentation_config(experiment.baseline_config_path)
    protocol = _build_final_test_protocol(candidate, experiment, baseline)
    if baseline.dataset_contract.manifest_sha256 != candidate.dataset_manifest_sha256:
        raise ValueError("Baseline dataset contract does not match the frozen candidate.")

    # Full Manifest bytes만 provenance로 hash하며 CSV rows/image/label은 아직 열지 않는다.
    resolved_dataset = _runtime_path(resolved_repository, dataset_root)
    dataset_manifest_path = resolved_dataset / "manifest.csv"
    if not dataset_manifest_path.is_file():
        raise FileNotFoundError(f"Final-test Dataset Manifest is missing: {dataset_manifest_path}")
    if sha256_file(dataset_manifest_path) != candidate.dataset_manifest_sha256:
        raise ValueError("Final-test Dataset Manifest SHA-256 does not match the candidate.")

    # Actual execution은 clean committed C4-4 revision과 unused namespace에서만 허용한다.
    repository_provenance = provenance_resolver(resolved_repository)
    repository_provenance.validate()
    if repository_provenance.working_tree_dirty:
        raise ValueError("C4-4 final test requires a clean committed repository state.")
    if requested_device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("C4-4 device must be auto, cpu, mps, or cuda.")
    resolved_output_root = _repository_path(
        resolved_repository,
        output_root,
        field="final-test output root",
    )
    required_root = (resolved_repository / FINAL_TEST_OUTPUT_ROOT).resolve()
    if resolved_output_root != required_root:
        raise ValueError("C4-4 output root must be outputs/final_test/yolo_segmentation exactly.")
    output_dir = _expected_output_dir(resolved_repository, candidate, manifest_sha256)
    if output_dir.exists():
        raise FileExistsError(f"C4-4 final-test namespace already exists: {output_dir}")
    if _starting_output_dir(output_dir).exists():
        raise FileExistsError("C4-4 final-test namespace reservation already exists.")
    return FinalTestPreflight(
        repository_root=resolved_repository,
        frozen_manifest_path=manifest_path,
        frozen_manifest_sha256=manifest_sha256,
        official_package_path=package_path,
        dataset_root=resolved_dataset,
        dataset_manifest_path=dataset_manifest_path,
        output_dir=output_dir,
        candidate=candidate,
        official_evidence=evidence,
        experiment_config=experiment,
        baseline_config=baseline,
        protocol=protocol,
        repository_provenance=repository_provenance,
        requested_device=requested_device,
        provenance_resolver=provenance_resolver,
    )


# ADD 2026-09-01: Unlock 직전 모든 non-test trust anchor와 clean Git state를 다시 확인한다.
def _revalidate_preflight(preflight: FinalTestPreflight) -> None:
    current_candidate = load_final_candidate_manifest(preflight.frozen_manifest_path)
    if current_candidate != preflight.candidate or (
        sha256_file(preflight.frozen_manifest_path) != preflight.frozen_manifest_sha256
    ):
        raise RuntimeError("Frozen candidate changed after C4-4 preflight.")
    if sha256_file(preflight.dataset_manifest_path) != preflight.candidate.dataset_manifest_sha256:
        raise RuntimeError("Dataset Manifest changed after C4-4 preflight.")
    if sha256_file(preflight.experiment_config.config_path) != (
        preflight.candidate.experiment_config_sha256
    ):
        raise RuntimeError("Experiment config changed after C4-4 preflight.")
    current_protocol = _build_final_test_protocol(
        preflight.candidate,
        preflight.experiment_config,
        preflight.baseline_config,
    )
    if current_protocol != preflight.protocol:
        raise RuntimeError("Frozen evaluation protocol changed after C4-4 preflight.")
    current_evidence = load_official_candidate_evidence(
        preflight.official_package_path,
        expected_package_sha256=preflight.candidate.official_package_sha256,
    )
    verify_official_candidate_identity(preflight.candidate, current_evidence)
    current_provenance = preflight.provenance_resolver(preflight.repository_root)
    current_provenance.validate()
    if (
        current_provenance != preflight.repository_provenance
        or current_provenance.working_tree_dirty
    ):
        raise RuntimeError("Repository state changed after C4-4 preflight.")


# ADD 2026-09-01: ZIP entry를 복원한다. → MODIFY 2026-09-02: 공유 materializer를 재사용한다.
def _materialize_verified_artifact(preflight: FinalTestPreflight) -> Path:
    evidence = load_official_candidate_evidence(
        preflight.official_package_path,
        expected_package_sha256=preflight.candidate.official_package_sha256,
    )
    verify_official_candidate_identity(preflight.candidate, evidence)
    artifact_dir = preflight.output_dir / "runtime_artifact"
    materialize_official_candidate_artifact(
        package_path=preflight.official_package_path,
        candidate=preflight.candidate,
        evidence=evidence,
        artifact_dir=artifact_dir,
    )
    validate_yolo_artifact(
        artifact_dir / "model",
        expected_contract=preflight.baseline_config.dataset_contract,
    )
    return artifact_dir


# ADD 2026-09-01: Evaluator가 사용하는 materialized model/metadata bytes를 frozen SHA로 검증한다.
def _verify_materialized_artifact(preflight: FinalTestPreflight, artifact_dir: Path) -> None:
    model_dir = artifact_dir / "model"
    if (
        sha256_file(model_dir / "model.pt") != preflight.candidate.model_sha256
        or sha256_file(model_dir / "metadata.json") != preflight.candidate.metadata_sha256
    ):
        raise RuntimeError("C4-4 materialized artifact identity changed.")
    validate_yolo_artifact(
        model_dir,
        expected_contract=preflight.baseline_config.dataset_contract,
    )


# ADD 2026-09-01: Ultralytics required keys와 explicit test split을 runtime YAML에 기록한다.
def _write_final_test_dataset_yaml(preflight: FinalTestPreflight) -> Path:
    path = preflight.output_dir / "dataset.test.runtime.yaml"
    payload = {
        "path": str(preflight.dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": preflight.baseline_config.dataset_contract.classes,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


# ADD 2026-09-01: One test annotation을 source-resolution diagnostic GT instances로 복원한다.
def _load_test_ground_truth(
    record: DerivedManifestRecord,
    *,
    dataset_root: Path,
    valid_class_ids: set[int],
) -> tuple[GroundTruthInstance, ...]:
    if record.derived_split != "test":
        raise ValueError("C4-4 ground truth accepts only derived test records.")
    label_text = (dataset_root / record.label_path).read_text(encoding="utf-8")
    instances = rasterize_segmentation_label_instances(
        label_text,
        image_width=record.image_width,
        image_height=record.image_height,
        valid_class_ids=valid_class_ids,
    )
    if record.is_negative:
        if instances or record.component_count:
            raise ValueError("C4-4 good-negative row contains segmentation ground truth.")
        return ()
    if len(instances) != record.component_count:
        raise ValueError(f"C4-4 GT component count mismatch: {record.sample_id}")
    return tuple(
        GroundTruthInstance(
            class_id=instance.class_id,
            mask=instance.mask,
            box_xyxy=mask_box(instance.mask),
            area_ratio=instance.area_ratio,
        )
        for instance in instances
    )


# ADD 2026-09-01: Test records와 prediction pool에서 frozen Region Coverage 지표를 계산한다.
def _calculate_test_region_coverage(
    *,
    records: tuple[DerivedManifestRecord, ...],
    ground_truth_by_sample: dict[str, tuple[GroundTruthInstance, ...]],
    predictions_by_sample: dict[str, tuple[PredictedInstance, ...]],
    classes: dict[int, str],
    size_policy: SizeBucketPolicy,
) -> dict[str, float | int]:
    strict_total = 0
    strict_matched = 0
    coverage_total = 0
    coverage_matched = 0
    small_total = 0
    small_coverage_matched = 0
    near_miss = 0
    covered_strict_fail = 0
    class_intersection = 0
    class_union = 0
    class_gt_pixels = 0
    class_prediction_pixels = 0
    for record in records:
        ground_truth = ground_truth_by_sample[record.sample_id]
        predictions = predictions_by_sample[record.sample_id]
        strict_total += len(ground_truth)
        strict_matched += len(match_instances(ground_truth, predictions, iou_threshold=0.5))
        for class_id in classes:
            gt_union = np.zeros((record.image_height, record.image_width), dtype=np.bool_)
            prediction_union = np.zeros_like(gt_union)
            for ground_truth_item in ground_truth:
                if ground_truth_item.class_id == class_id:
                    gt_union |= ground_truth_item.mask
            for prediction_item in predictions:
                if prediction_item.class_id == class_id:
                    prediction_union |= prediction_item.mask
            class_intersection += int(np.count_nonzero(gt_union & prediction_union))
            class_union += int(np.count_nonzero(gt_union | prediction_union))
            class_gt_pixels += int(np.count_nonzero(gt_union))
            class_prediction_pixels += int(np.count_nonzero(prediction_union))
        for ground_truth_instance in ground_truth:
            same_class = tuple(
                prediction
                for prediction in predictions
                if prediction.class_id == ground_truth_instance.class_id
            )
            best_iou = max(
                (
                    mask_overlap(ground_truth_instance.mask, prediction.mask)[0]
                    for prediction in same_class
                ),
                default=0.0,
            )
            prediction_union = np.zeros_like(ground_truth_instance.mask)
            for prediction in same_class:
                prediction_union |= prediction.mask
            ground_truth_pixels = int(np.count_nonzero(ground_truth_instance.mask))
            coverage = (
                int(np.count_nonzero(ground_truth_instance.mask & prediction_union))
                / ground_truth_pixels
            )
            strict_pass = best_iou >= FINAL_TEST_MATCH_IOU
            coverage_pass = coverage >= FINAL_TEST_REGION_COVERAGE
            coverage_total += 1
            coverage_matched += int(coverage_pass)
            if size_policy.classify(ground_truth_instance.area_ratio) == "small":
                small_total += 1
                small_coverage_matched += int(coverage_pass)
            near_miss += int(
                not strict_pass and FINAL_TEST_NEAR_MISS_MIN_IOU <= best_iou < FINAL_TEST_MATCH_IOU
            )
            covered_strict_fail += int(not strict_pass and coverage_pass)

    # Empty denominators are represented as zero rates without inventing positive evidence.
    return {
        "strict_instance_gt_recall": _safe_ratio(strict_matched, strict_total),
        "gt_component_coverage_recall_at_50": _safe_ratio(coverage_matched, coverage_total),
        "small_gt_coverage_recall_at_50": _safe_ratio(small_coverage_matched, small_total),
        "class_aware_union_iou": _safe_ratio(class_intersection, class_union),
        "class_aware_union_gt_coverage": _safe_ratio(class_intersection, class_gt_pixels),
        "class_aware_union_prediction_precision": _safe_ratio(
            class_intersection, class_prediction_pixels
        ),
        "near_miss_iou_030_to_050": near_miss,
        "covered50_but_strict_instance_fail": covered_strict_fail,
        "test_gt_total": strict_total,
        "small_gt_total": small_total,
    }


# ADD 2026-09-01: Explicit unlock 이후 verified artifact로 framework/frozen diagnostics를 실행한다.
def run_verified_final_test_evaluator(
    preflight: FinalTestPreflight,
    artifact_dir: Path,
) -> FinalTestMeasurements:
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()

    # Verified artifact path가 전달된 뒤에만 test records와 image/label content를 연다.
    test_records = validate_final_test_dataset(
        preflight.dataset_root,
        preflight.baseline_config.dataset_contract,
    )
    dataset_yaml = _write_final_test_dataset_yaml(preflight)

    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version

    if ultralytics_version != preflight.candidate.framework_version:
        raise ValueError("C4-4 Ultralytics version does not match the frozen artifact.")
    device = resolve_device(preflight.requested_device)
    requested_device = str(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = YOLO(str(artifact_dir / "model" / "model.pt"), task=preflight.candidate.task)

    # Ultralytics standard test metrics는 fixed checkpoint와 test split에서만 계산한다.
    metrics = model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=preflight.experiment_config.training_config(preflight.baseline_config).training.imgsz,
        batch=preflight.baseline_config.evaluation.batch,
        workers=preflight.baseline_config.evaluation.workers,
        device=(0 if device.type == "cuda" else device.type),
        project=str(preflight.output_dir / "framework"),
        name="ultralytics-test",
        exist_ok=False,
        plots=False,
        save_json=False,
        verbose=True,
    )
    framework_metrics, per_class_metrics = serialize_ultralytics_metrics(
        metrics,
        classes=preflight.baseline_config.dataset_contract.classes,
    )

    # C4-2C에서 고정한 prediction/matching/size policy로 report-only diagnostics를 만든다.
    classes = preflight.baseline_config.dataset_contract.classes
    ground_truth_by_sample: dict[str, tuple[GroundTruthInstance, ...]] = {}
    raw_predictions: dict[str, tuple[PredictedInstance, ...]] = {}
    for record in test_records:
        ground_truth_by_sample[record.sample_id] = _load_test_ground_truth(
            record,
            dataset_root=preflight.dataset_root,
            valid_class_ids=set(classes),
        )
        raw_predictions[record.sample_id] = predict_c4_2c_instances(
            model=cast(ConfirmationModel, model),
            source_image_path=preflight.dataset_root / record.image_path,
            image_width=record.image_width,
            image_height=record.image_height,
            imgsz=preflight.experiment_config.training_config(
                preflight.baseline_config
            ).training.imgsz,
            device=requested_device,
            valid_class_ids=set(classes),
        )
    predictions = filter_predictions(raw_predictions, FINAL_TEST_DIAGNOSTIC_CONFIDENCE)
    analyses = tuple(
        analyze_sample(
            record=record,
            ground_truth=ground_truth_by_sample[record.sample_id],
            predictions=predictions[record.sample_id],
            classes=classes,
            size_policy=preflight.protocol.size_bucket_policy,
            expected_split="test",
        )
        for record in test_records
    )
    aggregate = aggregate_analysis(list(analyses), classes=classes)
    diagnostic = {
        key: aggregate[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1", "per_class")
    }
    failure_analysis = {
        "size_analysis": aggregate["size_analysis"],
        "component_analysis": aggregate["component_analysis"],
        "negative_analysis": aggregate["negative_analysis"],
        "error_taxonomy": aggregate["error_taxonomy"],
        "complete_miss_sample_count": sum(
            analysis.ground_truth_instance_count > 0 and analysis.predicted_instance_count == 0
            for analysis in analyses
        ),
        "wrong_class_sample_count": sum(
            "WRONG_CLASS" in analysis.secondary_tags for analysis in analyses
        ),
    }
    region_coverage = _calculate_test_region_coverage(
        records=test_records,
        ground_truth_by_sample=ground_truth_by_sample,
        predictions_by_sample=predictions,
        classes=classes,
        size_policy=preflight.protocol.size_bucket_policy,
    )
    elapsed_seconds = time.perf_counter() - started
    resource_metrics: dict[str, Any] = {"evaluation_wall_time_seconds": elapsed_seconds}
    if device.type == "cuda":
        resource_metrics.update(
            {
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    measurements = FinalTestMeasurements(
        evaluation_split="test",
        framework_metric_source="ultralytics_model.val(split=test)",
        sample_count=len(test_records),
        framework_metrics=framework_metrics,
        framework_per_class_metrics=per_class_metrics,
        diagnostic=diagnostic,
        failure_analysis=failure_analysis,
        region_coverage=region_coverage,
        sample_analysis=tuple(analysis.to_dict() for analysis in analyses),
        environment={
            "framework": "ultralytics",
            "framework_version": ultralytics_version,
            "torch_version": str(torch.__version__),
            "device": requested_device,
        },
        resource_metrics=resource_metrics,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
    )
    measurements.validate()
    return measurements


# ADD 2026-09-01: Frozen preflight와 completed measurements를 report-only result로 결합한다.
def _build_final_test_evidence(
    preflight: FinalTestPreflight,
    measurements: FinalTestMeasurements,
) -> FinalTestEvidence:
    measurements.validate()
    evidence = FinalTestEvidence(
        schema_version=FINAL_TEST_SCHEMA_VERSION,
        task=preflight.candidate.task,
        model_family=preflight.candidate.model_family,
        final_test_state=FINAL_TEST_COMPLETED_STATE,
        evaluation_split="test",
        frozen_candidate_manifest_sha256=preflight.frozen_manifest_sha256,
        c4_3_freeze_commit=C4_3_FREEZE_COMMIT,
        c4_4_execution_commit=preflight.repository_provenance.git_commit,
        selected_experiment_id=preflight.candidate.selected_experiment_id,
        selected_model_name=preflight.candidate.selected_model_name,
        official_package_sha256=preflight.candidate.official_package_sha256,
        model_sha256=preflight.candidate.model_sha256,
        metadata_sha256=preflight.candidate.metadata_sha256,
        experiment_config_sha256=preflight.candidate.experiment_config_sha256,
        packaged_experiment_result_sha256=(preflight.candidate.packaged_experiment_result_sha256),
        dataset_manifest_sha256=preflight.candidate.dataset_manifest_sha256,
        candidate_was_frozen_before_test_access=True,
        candidate_selection_changed=False,
        threshold_tuned_on_test=False,
        test_used=True,
        test_split_used=True,
        protocol=preflight.protocol.to_json_dict(),
        measurements=asdict(measurements),
    )
    evidence.validate()
    return evidence


# ADD 2026-09-01: Final result JSON을 deterministic single-entry evidence ZIP으로 묶는다.
def _write_evidence_package(result_bytes: bytes, package_path: Path) -> str:
    checksum = sha256_bytes(result_bytes)
    checksum_bytes = f"{checksum}  {FINAL_TEST_RESULT_FILENAME}\n".encode()
    with zipfile.ZipFile(package_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in (
            (FINAL_TEST_RESULT_FILENAME, result_bytes),
            (FINAL_TEST_PACKAGE_CHECKSUM_FILENAME, checksum_bytes),
        ):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    return sha256_file(package_path)


# ADD 2026-09-01: Explicit unlock 뒤 unused namespace에 completed evidence를 기록한다.
def execute_yolo_final_test(
    preflight: FinalTestPreflight,
    *,
    confirm_final_test: bool,
    evaluator: FinalTestEvaluator = run_verified_final_test_evaluator,
    artifact_materializer: ArtifactMaterializer = _materialize_verified_artifact,
    artifact_verifier: ArtifactVerifier = _verify_materialized_artifact,
) -> FinalTestRunArtifacts | None:
    if confirm_final_test is not True:
        return None
    _revalidate_preflight(preflight)
    _reserve_output_namespace(preflight)

    # READY에서는 package materialization만 허용하며 test resolver는 아직 호출하지 않는다.
    try:
        artifact_dir = artifact_materializer(preflight)
        artifact_verifier(preflight, artifact_dir)
        _revalidate_preflight(preflight)
    except Exception as exc:
        _write_run_state(
            preflight,
            lifecycle_state=FINAL_TEST_PREPARATION_FAILED_STATE,
            message=f"Preparation failed before test access: {type(exc).__name__}.",
        )
        raise

    _write_run_state(
        preflight,
        lifecycle_state=FINAL_TEST_RUNNING_STATE,
        message=(
            "Test evaluator was authorized; test access may have occurred and automatic rerun "
            "is forbidden."
        ),
    )
    completed_dir = preflight.output_dir / "completed"
    completion_staging = preflight.output_dir / ".completion-staging"
    try:
        measurements = evaluator(preflight, artifact_dir)
        artifact_verifier(preflight, artifact_dir)
        _revalidate_preflight(preflight)
        evidence = _build_final_test_evidence(preflight, measurements)
        result_bytes = evidence.to_json_bytes()
        completion_staging.mkdir(exist_ok=False)
        result_path = completion_staging / FINAL_TEST_RESULT_FILENAME
        result_path.write_bytes(result_bytes)
        package_path = completion_staging / FINAL_TEST_PACKAGE_FILENAME
        package_sha256 = _write_evidence_package(result_bytes, package_path)
        completion_staging.rename(completed_dir)
        result_path = completed_dir / FINAL_TEST_RESULT_FILENAME
        package_path = completed_dir / FINAL_TEST_PACKAGE_FILENAME
        _write_run_state(
            preflight,
            lifecycle_state=FINAL_TEST_COMPLETED_STATE,
            message="Final-test evidence completed; rerun and overwrite are forbidden.",
        )
    except Exception:
        if completed_dir.exists():
            _write_run_state(
                preflight,
                lifecycle_state=FINAL_TEST_COMPLETED_STATE,
                message=(
                    "Completed evidence was atomically published; rerun and overwrite are "
                    "forbidden."
                ),
            )
        else:
            if completion_staging.exists():
                shutil.rmtree(completion_staging)
            _write_run_state(
                preflight,
                lifecycle_state=FINAL_TEST_FAILED_STATE,
                message=(
                    "Evaluator started and test access may have occurred; automatic rerun and "
                    "cleanup are forbidden."
                ),
            )
            raise
    return FinalTestRunArtifacts(
        output_dir=preflight.output_dir,
        result_path=result_path,
        package_path=package_path,
        package_sha256=package_sha256,
        evidence=evidence,
    )
