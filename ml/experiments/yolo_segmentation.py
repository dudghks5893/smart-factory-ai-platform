"""Configuration and comparison contracts for controlled YOLO experiments."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Protocol, cast

import yaml

from ml.evaluation.yolo_segmentation_error_analysis import (
    MATCH_IOU_THRESHOLD,
    SizeBucketPolicy,
)
from ml.experiments.yolo_sampling import (
    ELIGIBLE_MULTIPLICITY,
    SAMPLING_RULE_VERSION,
    SMALL_FRACTION_RULE,
    TrainViewEvidence,
)
from ml.training.yolo_segmentation import (
    YoloOutputConfig,
    YoloSegmentationBaselineConfig,
    load_yolo_segmentation_config,
    validate_artifact_id,
)
from shared.hashing import sha256_bytes, sha256_file

EXPERIMENT_SCHEMA_VERSION = 1
EXPERIMENT_STATUSES = {
    "PLANNED",
    "RUNNING",
    "COMPLETED",
    "ACCEPTED",
    "REJECTED",
    "CONFIRMED_CANDIDATE",
    "CONFIRMATION_FAILED",
}
EXPERIMENT_DECISIONS = {
    "PENDING",
    "ACCEPT",
    "REJECT",
    "CONFIRMED_CANDIDATE",
    "CONFIRMATION_FAILED",
}
BASELINE_MODEL_SHA256 = "594003121b0e071c47d68c3e53c10f438dcec18b5b56b4e5d8831d64001192bd"
BASELINE_METADATA_SHA256 = "9f3e3878141e831a6721c5136d67057da906485b9825262bd4e0897b2879fc6b"
BASELINE_VALIDATION_METRICS = {
    "box": {"precision": 0.55955, "recall": 0.59848, "map50": 0.56157, "map50_95": 0.32398},
    "mask": {"precision": 0.55955, "recall": 0.59848, "map50": 0.59929, "map50_95": 0.34359},
}
EXPECTED_PRIORITIES = {
    "primary": ("validation_mask_map50_95", "diagnostic_instance_recall"),
    "failure_focused": ("small_defect_recall", "multi_component_recall"),
    "guardrail": ("good_negative_fp_image_rate",),
}
RESOLUTION_INTERVENTION = "resolution"
TRAIN_SAMPLING_INTERVENTION = "train_sampling_multiplicity"
CROP_CONFIRMATION_INTERVENTION = "component_aware_crop_confirmation"
SAMPLING_CONTROLLED_FIELD = "training.dataset_index_multiplicity"
SAMPLING_CONTROLLED_BEFORE = "canonical_x1"
SAMPLING_CONTROLLED_AFTER = "component_aware_eligible_x2"
CROP_CONTROLLED_FIELD = "training.recipe"
CROP_CONTROLLED_BEFORE = "c4_2b_component_aware_x2"
CROP_CONTROLLED_AFTER = "crop350_nomosaic_maskratio2"


@dataclass(frozen=True)
class CropSamplingPolicy:
    """Exact component-aware duplicate plus small-centered crop policy for C4-2C."""

    sampling_mode: str
    sampling_multiplicity: int
    crop_size: int

    # ADD 2026-08-31: C4-2C R17-derived train-view recipe를 고정한다.
    def validate(self) -> None:
        if (
            self.sampling_mode != "component_aware_crop"
            or self.sampling_multiplicity != 2
            or self.crop_size != 350
        ):
            raise ValueError("C4-2C crop sampling recipe changed from the approved design.")


@dataclass(frozen=True)
class TrainerOverrides:
    """Explicit Ultralytics augmentation arguments applied only to C4-2C."""

    mosaic: float
    mask_ratio: int
    overlap_mask: bool
    scale: float

    # ADD 2026-08-31: Every C4-2C augmentation argument is explicit and exact.
    def validate(self) -> None:
        if asdict(self) != {
            "mosaic": 0.0,
            "mask_ratio": 2,
            "overlap_mask": True,
            "scale": 0.5,
        }:
            raise ValueError("C4-2C trainer overrides changed from the approved recipe.")


@dataclass(frozen=True)
class ExpectedCropTrainView:
    """Exact approved train-view snapshot asserted for C4-2C preparation."""

    canonical_entries: int
    canonical_positives: int
    canonical_negatives: int
    component_duplicate_entries: int
    small_centered_crop_entries: int
    total_entries: int
    positive_exposure: int
    negative_exposure: int
    small_aware_count: int
    multi_component_count: int
    eligible_overlap_count: int
    eligible_union_count: int
    observed_train_small_cutoff: float

    # ADD 2026-09-01: Actual train-view mapping을 approved snapshot과 비교한다.
    def _validate_actual(self, actual: dict[str, int | float]) -> None:
        expected = asdict(self)
        if actual != expected:
            raise ValueError(
                f"C4-2C train-view snapshot mismatch: expected={expected}, actual={actual}"
            )

    # ADD 2026-08-31: Crop evidence를 검증한다. → MODIFY 2026-09-01: Actual count만 사용한다.
    def validate_evidence(self, evidence: CropEvidenceContract) -> None:
        evidence.validate()
        actual = {
            "canonical_entries": evidence.canonical_entry_count,
            "canonical_positives": evidence.canonical_positive_count,
            "canonical_negatives": evidence.canonical_negative_count,
            "component_duplicate_entries": evidence.component_duplicate_count,
            "small_centered_crop_entries": evidence.crop_entry_count,
            "total_entries": evidence.total_entry_count,
            "positive_exposure": evidence.positive_exposure,
            "negative_exposure": evidence.negative_exposure,
            "small_aware_count": evidence.small_aware_count,
            "multi_component_count": evidence.multi_component_count,
            "eligible_overlap_count": evidence.eligible_overlap_count,
            "eligible_union_count": evidence.eligible_union_count,
            "observed_train_small_cutoff": evidence.observed_train_small_cutoff,
        }
        self._validate_actual(actual)

    # ADD 2026-09-01: Shared planner의 actual projected crop view를 Official preflight에서 검증한다.
    def validate_planned_evidence(
        self,
        evidence: TrainViewEvidence,
    ) -> dict[str, int | float]:
        evidence.validate()
        actual: dict[str, int | float] = {
            "canonical_entries": evidence.unique_train_count,
            "canonical_positives": evidence.unique_positive_count,
            "canonical_negatives": evidence.unique_good_negative_count,
            "component_duplicate_entries": evidence.eligible_union_count,
            "small_centered_crop_entries": evidence.small_aware_count,
            "total_entries": (
                evidence.unique_train_count
                + evidence.eligible_union_count
                + evidence.small_aware_count
            ),
            "positive_exposure": (
                evidence.unique_positive_count
                + evidence.eligible_union_count
                + evidence.small_aware_count
            ),
            "negative_exposure": evidence.unique_good_negative_count,
            "small_aware_count": evidence.small_aware_count,
            "multi_component_count": evidence.multi_component_count,
            "eligible_overlap_count": evidence.eligible_overlap_count,
            "eligible_union_count": evidence.eligible_union_count,
            "observed_train_small_cutoff": evidence.observed_train_small_cutoff,
        }
        self._validate_actual(actual)
        return actual


class CropEvidenceContract(Protocol):
    """Structural train-view fields consumed without introducing an import cycle."""

    def validate(self) -> None: ...

    @property
    def canonical_entry_count(self) -> int: ...

    @property
    def canonical_positive_count(self) -> int: ...

    @property
    def canonical_negative_count(self) -> int: ...

    @property
    def component_duplicate_count(self) -> int: ...

    @property
    def crop_entry_count(self) -> int: ...

    @property
    def total_entry_count(self) -> int: ...

    @property
    def positive_exposure(self) -> int: ...

    @property
    def negative_exposure(self) -> int: ...

    @property
    def small_aware_count(self) -> int: ...

    @property
    def multi_component_count(self) -> int: ...

    @property
    def eligible_overlap_count(self) -> int: ...

    @property
    def eligible_union_count(self) -> int: ...

    @property
    def observed_train_small_cutoff(self) -> float: ...


@dataclass(frozen=True)
class ConfirmationProtocol:
    """Fast-compatible validation prediction and absolute confirmation gates."""

    initial_confidence: float
    final_confidence: float
    prediction_iou: float
    max_det: int
    retina_masks: bool
    mask_threshold: float
    mask_resize: str
    matching_iou: float
    small_recall_floor_exclusive: float
    mask_map50_95_floor: float
    multi_recall_floor: float
    good_negative_fp_rate: float

    # ADD 2026-08-31: C4-2C prediction normalization과 Primary gate를 exact하게 고정한다.
    def validate(self) -> None:
        if asdict(self) != {
            "initial_confidence": 0.001,
            "final_confidence": 0.25,
            "prediction_iou": 0.7,
            "max_det": 300,
            "retina_masks": False,
            "mask_threshold": 0.5,
            "mask_resize": "opencv_inter_nearest",
            "matching_iou": 0.5,
            "small_recall_floor_exclusive": 0.25,
            "mask_map50_95_floor": 0.4,
            "multi_recall_floor": 0.5,
            "good_negative_fp_rate": 0.0,
        }:
            raise ValueError("C4-2C confirmation protocol changed from the approved design.")


@dataclass(frozen=True)
class ControlledChange:
    """One intentionally varied model-quality field."""

    field: str
    before: int | str
    after: int | str


@dataclass(frozen=True)
class SamplingPolicy:
    """Train-only component-aware multiplicity policy for C4-2B."""

    policy_type: str
    sampling_rule_version: str
    small_fraction_rule: str
    multi_component: bool
    eligible_multiplicity: int
    validation_used_for_sampling: bool
    test_split_used: bool

    # ADD 2026-08-28: C4-2B sampling policy가 predeclared train-only x2 규칙인지 검증한다.
    def validate(self) -> None:
        if (
            self.policy_type != "component_aware_train_multiplicity"
            or self.sampling_rule_version != SAMPLING_RULE_VERSION
            or self.small_fraction_rule != SMALL_FRACTION_RULE
            or self.multi_component is not True
            or self.eligible_multiplicity != ELIGIBLE_MULTIPLICITY
            or self.validation_used_for_sampling is not False
            or self.test_split_used is not False
        ):
            raise ValueError("C4-2B sampling policy changed from the predeclared train-only rule.")


@dataclass(frozen=True)
class ExpectedTrainView:
    """Official dataset snapshot asserted without defining the sampling behavior."""

    unique_train_count: int
    unique_positive_count: int
    unique_good_negative_count: int
    small_aware_count: int
    multi_component_count: int
    eligible_overlap_count: int
    eligible_union_count: int
    expanded_entry_count: int
    expanded_positive_count: int
    expanded_good_negative_count: int
    expanded_good_negative_ratio: float
    observed_train_small_cutoff: float

    # ADD 2026-08-28: Generated train-view가 approved canonical snapshot과 일치하는지 검증한다.
    def validate_evidence(self, evidence: TrainViewEvidence) -> None:
        expected = asdict(self)
        actual = {key: getattr(evidence, key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"C4-2B train-view snapshot mismatch: expected={expected}, actual={actual}"
            )


@dataclass(frozen=True)
class ValidationProtocol:
    """Sealed validation diagnostics shared with C4-1."""

    split: str
    test_split_used: bool
    diagnostic_confidence: float
    matching_method: str
    mask_iou_threshold: float
    small_max_area_ratio: float
    medium_max_area_ratio: float

    # ADD 2026-08-27: C4-1 metric identity와 sealed-test invariants를 검증한다.
    def validate(self) -> None:
        if self.split != "val" or self.test_split_used:
            raise ValueError("Controlled YOLO experiments must use validation only.")
        if self.diagnostic_confidence != 0.25:
            raise ValueError("C4-2A diagnostic confidence must remain 0.25.")
        if (
            self.matching_method != "class_aware_greedy_max_mask_iou"
            or self.mask_iou_threshold != MATCH_IOU_THRESHOLD
        ):
            raise ValueError("C4-2A matching protocol must remain identical to C4-1.")
        expected = SizeBucketPolicy(
            method="validation_gt_mask_area_ratio_tertiles",
            small_max=0.015947619047619047,
            medium_max=0.02447142857142857,
        )
        if (
            self.small_max_area_ratio != expected.small_max
            or self.medium_max_area_ratio != expected.medium_max
        ):
            raise ValueError("C4-2A size bucket boundaries must remain fixed from C4-1.")

    # ADD 2026-08-27: Experiment diagnostics에 주입할 immutable C4-1 size policy를 반환한다.
    def size_policy(self) -> SizeBucketPolicy:
        return SizeBucketPolicy(
            method="validation_gt_mask_area_ratio_tertiles",
            small_max=self.small_max_area_ratio,
            medium_max=self.medium_max_area_ratio,
        )


@dataclass(frozen=True)
class TelemetryConfig:
    """Best-effort device sampler lifecycle settings."""

    sample_interval_seconds: float
    nvidia_smi_timeout_seconds: float

    # ADD 2026-08-27: Sampling/command timeout bounds를 검증한다.
    def validate(self) -> None:
        if self.sample_interval_seconds <= 0.0 or self.nvidia_smi_timeout_seconds <= 0.0:
            raise ValueError("Telemetry intervals must be positive.")


@dataclass(frozen=True)
class ExperimentOutputConfig:
    """Ignored evidence, artifact, runtime and package roots."""

    experiment_root: Path
    artifact_root: Path
    training_runtime_root: Path
    package_root: Path


@dataclass(frozen=True)
class DecisionPolicy:
    """Predeclared multi-metric candidate recommendation policy."""

    require_mask_map50_95_non_regression: bool
    require_instance_recall_non_regression: bool
    require_small_recall_improvement: bool
    require_multi_component_recall_non_regression: bool
    allow_good_negative_fp_rate_increase: bool

    # ADD 2026-08-27: Decision gate flag가 YAML truthiness로 왜곡되지 않도록 검증한다.
    def validate(self) -> None:
        if any(type(value) is not bool for value in asdict(self).values()):
            raise ValueError("Experiment decision policy values must be booleans.")


@dataclass(frozen=True)
class YoloExperimentConfig:
    """Reusable controlled-experiment identity and policy."""

    experiment_id: str
    experiment_date: str
    status: str
    hypothesis: str
    target_failure_mode: str
    baseline_config_path: Path
    baseline_identity: dict[str, Any]
    baseline_evidence: dict[str, Any]
    candidate_identity: dict[str, Any]
    intervention_type: str
    controlled_change: ControlledChange
    sampling_policy: SamplingPolicy | None
    expected_train_view: ExpectedTrainView | None
    crop_sampling_policy: CropSamplingPolicy | None
    expected_crop_train_view: ExpectedCropTrainView | None
    trainer_overrides: TrainerOverrides | None
    confirmation_protocol: ConfirmationProtocol | None
    validation_protocol: ValidationProtocol
    telemetry: TelemetryConfig
    output: ExperimentOutputConfig
    evaluation_priorities: dict[str, tuple[str, ...]]
    decision_policy: DecisionPolicy
    config_path: Path

    # ADD 2026-08-27: Imgsz를 검증한다. → MODIFY 2026-08-31: C4-2C confirmation recipe도 검증한다.
    def validate(self, baseline: YoloSegmentationBaselineConfig) -> None:
        validate_artifact_id(self.experiment_id)
        try:
            date.fromisoformat(self.experiment_date)
        except ValueError as exc:
            raise ValueError("Experiment date must be ISO YYYY-MM-DD.") from exc
        if self.status not in EXPERIMENT_STATUSES or not self.hypothesis:
            raise ValueError("Experiment status/hypothesis is invalid.")
        legacy_identity = {
            "model": baseline.model.architecture,
            "pretrained_checkpoint": baseline.model.weights,
            "imgsz": baseline.training.imgsz,
            "seed": baseline.training.seed,
            "batch": baseline.training.batch,
            "epochs": baseline.training.epochs,
            "patience": baseline.training.patience,
        }
        fixed_training_identity = {
            **legacy_identity,
            "workers": baseline.training.workers,
            "optimizer": baseline.training.optimizer,
            "deterministic": baseline.training.deterministic,
            "amp": baseline.training.amp,
        }
        expected_identity = (
            legacy_identity
            if self.intervention_type == RESOLUTION_INTERVENTION
            else fixed_training_identity
        )
        if self.baseline_identity != expected_identity:
            raise ValueError("Experiment baseline identity does not match the baseline config.")
        _validate_baseline_evidence(self.baseline_evidence)
        if self.intervention_type == RESOLUTION_INTERVENTION:
            if self.controlled_change != ControlledChange("training.imgsz", 640, 1024):
                raise ValueError("C4-2A must contain only the imgsz 640 -> 1024 change.")
            expected_candidate_identity = {
                **expected_identity,
                "imgsz": self.controlled_change.after,
            }
            if self.sampling_policy is not None or self.expected_train_view is not None:
                raise ValueError("C4-2A must not declare a train sampling policy.")
        elif self.intervention_type == TRAIN_SAMPLING_INTERVENTION:
            expected_change = ControlledChange(
                SAMPLING_CONTROLLED_FIELD,
                SAMPLING_CONTROLLED_BEFORE,
                SAMPLING_CONTROLLED_AFTER,
            )
            if self.controlled_change != expected_change:
                raise ValueError("C4-2B must contain only the train multiplicity policy change.")
            if self.sampling_policy is None or self.expected_train_view is None:
                raise ValueError("C4-2B requires sampling policy and expected train-view evidence.")
            self.sampling_policy.validate()
            expected_candidate_identity = expected_identity
            if any(
                value is not None
                for value in (
                    self.crop_sampling_policy,
                    self.expected_crop_train_view,
                    self.trainer_overrides,
                    self.confirmation_protocol,
                )
            ):
                raise ValueError("C4-2B must not declare C4-2C confirmation fields.")
        elif self.intervention_type == CROP_CONFIRMATION_INTERVENTION:
            if self.controlled_change != ControlledChange(
                CROP_CONTROLLED_FIELD,
                CROP_CONTROLLED_BEFORE,
                CROP_CONTROLLED_AFTER,
            ):
                raise ValueError("C4-2C must declare only the approved combined recipe.")
            if self.sampling_policy is not None or self.expected_train_view is not None:
                raise ValueError("C4-2C must use its typed crop train-view policy.")
            required = (
                self.crop_sampling_policy,
                self.expected_crop_train_view,
                self.trainer_overrides,
                self.confirmation_protocol,
            )
            if any(value is None for value in required):
                raise ValueError("C4-2C typed recipe sections are incomplete.")
            assert self.crop_sampling_policy is not None
            assert self.trainer_overrides is not None
            assert self.confirmation_protocol is not None
            self.crop_sampling_policy.validate()
            self.trainer_overrides.validate()
            self.confirmation_protocol.validate()
            expected_candidate_identity = expected_identity
        else:
            raise ValueError("Unsupported YOLO experiment intervention type.")
        if self.candidate_identity != expected_candidate_identity:
            raise ValueError("Experiment candidate identity changed outside its intervention.")
        if baseline.training.batch != 16:
            raise ValueError("Controlled YOLO experiments require the baseline batch size 16.")
        self.validation_protocol.validate()
        self.telemetry.validate()
        self.decision_policy.validate()
        for key, expected in EXPECTED_PRIORITIES.items():
            if self.evaluation_priorities.get(key) != expected:
                raise ValueError(f"Experiment priority changed or is incomplete: {key}")

        if self.intervention_type in {
            RESOLUTION_INTERVENTION,
            TRAIN_SAMPLING_INTERVENTION,
        } and any(
            value is not None
            for value in (
                self.crop_sampling_policy,
                self.expected_crop_train_view,
                self.trainer_overrides,
                self.confirmation_protocol,
            )
        ):
            raise ValueError("Legacy controlled experiments must not declare C4-2C fields.")

    # ADD 2026-08-27: Training config를 만든다. → MODIFY 2026-08-28: Intervention을 분기한다.
    def training_config(
        self, baseline: YoloSegmentationBaselineConfig
    ) -> YoloSegmentationBaselineConfig:
        self.validate(baseline)
        controlled_training = baseline.training
        if self.intervention_type == RESOLUTION_INTERVENTION:
            controlled_training = replace(
                baseline.training,
                imgsz=cast(int, self.controlled_change.after),
            )
        controlled_output = YoloOutputConfig(
            artifact_root=self.output.artifact_root,
            training_runtime_root=self.output.training_runtime_root,
            evaluation_root=self.output.experiment_root / "framework_evaluation",
        )
        result = replace(
            baseline,
            training=controlled_training,
            output=controlled_output,
        )
        result.validate()
        return result


@dataclass(frozen=True)
class ExperimentRecommendation:
    """Transparent predeclared decision result for one validation comparison."""

    decision: str
    decision_reason: str
    checks: dict[str, bool]


# ADD 2026-08-27: YAML section을 typed mapping boundary에서 검증한다.
def _mapping(raw: object, *, name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment config section must be a mapping: {name}")
    return cast(dict[str, Any], raw)


# ADD 2026-08-27: YAML boolean field를 implicit truthiness 없이 읽는다.
def _boolean(raw: object, *, name: str) -> bool:
    if type(raw) is not bool:
        raise ValueError(f"Experiment config field must be a boolean: {name}")
    return cast(bool, raw)


# ADD 2026-08-31: New typed C4-2C sections reject silently ignored keys.
def _require_keys(section: dict[str, Any], expected: set[str], *, name: str) -> None:
    if set(section) != expected:
        raise ValueError(
            f"Experiment config section keys changed: {name}; "
            f"expected={sorted(expected)}, actual={sorted(section)}"
        )


# ADD 2026-08-28: Controlled change scalar를 integer 또는 non-empty string으로 제한한다.
def _controlled_value(raw: object, *, name: str) -> int | str:
    if isinstance(raw, bool) or not isinstance(raw, int | str) or raw == "":
        raise ValueError(f"Experiment controlled-change value is invalid: {name}")
    return raw


# ADD 2026-08-27: Trusted Baseline checkpoint evidence와 sealed-test exclusion을 검증한다.
def _validate_baseline_evidence(evidence: dict[str, Any]) -> None:
    sources = _mapping(evidence.get("sources"), name="baseline_evidence.sources")
    environment = _mapping(
        evidence.get("environment"),
        name="baseline_evidence.environment",
    )
    training = _mapping(evidence.get("training"), name="baseline_evidence.training")
    validation = _mapping(
        evidence.get("validation_framework"),
        name="baseline_evidence.validation_framework",
    )
    if sources.get("checkpoint_sha256") != BASELINE_MODEL_SHA256:
        raise ValueError("Baseline evidence checkpoint SHA is not approved.")
    if sources.get("metadata_sha256") != BASELINE_METADATA_SHA256:
        raise ValueError("Baseline evidence metadata SHA is not approved.")
    expected_environment = {
        "gpu_model": "Tesla T4",
        "torch_version": "2.13.0+cu130",
        "torchvision_version": "0.28.0+cu130",
        "torchvision_evidence_scope": "linux_x86_64_uv_lock_resolution",
        "ultralytics_version": "8.4.128",
    }
    if environment != expected_environment:
        raise ValueError("Baseline environment evidence changed.")
    if (
        training.get("configured_epochs") != 100
        or training.get("completed_epochs") != 80
        or training.get("early_stopping") is not True
        or training.get("best_epoch") != 60
        or training.get("checkpoint_cumulative_epoch_time_seconds") != 222.485
        or training.get("exact_end_to_end_wall_clock_seconds") is not None
    ):
        raise ValueError("Baseline checkpoint training history is invalid.")
    if validation.get("split") != "val" or validation.get("test_split_used") is not False:
        raise ValueError("Baseline framework evidence must be validation-only.")
    actual_metrics = {
        metric_type: _mapping(validation.get(metric_type), name=f"validation.{metric_type}")
        for metric_type in ("box", "mask")
    }
    if actual_metrics != BASELINE_VALIDATION_METRICS:
        raise ValueError("Baseline best-checkpoint validation metrics changed.")
    resource_metrics = _mapping(
        evidence.get("resource_metrics"),
        name="baseline_evidence.resource_metrics",
    )
    if any(value is not None for value in resource_metrics.values()):
        raise ValueError("Uncaptured Baseline resource metrics must remain null.")
    if evidence.get("derived_test_metrics_used_for_selection") is not False:
        raise ValueError("Derived-test metrics must be excluded from experiment selection.")


# ADD 2026-08-27: Final resolved project path가 repository boundary 안인지 검증한다.
def _require_repository_path(
    repository_root: Path,
    resolved_path: Path,
    *,
    field: str,
) -> Path:
    resolved = resolved_path.resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(
            f"{field} resolves outside repository root: "
            f"repository={repository_root}, resolved={resolved}"
        ) from exc
    return resolved


# ADD 2026-08-27: Config provenance root 안에서 relative/absolute Baseline config를 결정한다.
def _resolve_experiment_repository_root(
    config_path: Path,
    baseline_config_path: Path,
) -> tuple[Path, Path]:
    repository_root: Path | None = None
    for candidate in config_path.parents:
        if (candidate / "pyproject.toml").is_file():
            repository_root = candidate.resolve()
            break
    if repository_root is None:
        raise FileNotFoundError(
            f"Repository root not found from experiment config provenance: {config_path}"
        )

    declared_baseline = (
        baseline_config_path.resolve()
        if baseline_config_path.is_absolute()
        else (repository_root / baseline_config_path).resolve()
    )
    resolved_baseline = _require_repository_path(
        repository_root,
        declared_baseline,
        field="Experiment Baseline config",
    )
    if not resolved_baseline.is_file():
        raise FileNotFoundError(
            "Experiment Baseline config was not found from repository provenance: "
            f"config={config_path}, declared={baseline_config_path}, "
            f"resolved={resolved_baseline}"
        )
    return repository_root, resolved_baseline


# ADD 2026-08-27: Relative/absolute experiment output을 repository 안에서 resolve한다.
def _resolve_experiment_output_path(
    repository_root: Path,
    declared_path: object,
    *,
    field: str,
) -> Path:
    path = Path(str(declared_path))
    declared_output = path.resolve() if path.is_absolute() else (repository_root / path).resolve()
    return _require_repository_path(
        repository_root,
        declared_output,
        field=f"Experiment output path {field}",
    )


# ADD 2026-08-28: Optional C4-2B train-only sampling policy를 typed contract로 복원한다.
def _load_sampling_policy(raw: object) -> SamplingPolicy | None:
    if raw is None:
        return None
    section = _mapping(raw, name="sampling_policy")
    return SamplingPolicy(
        policy_type=str(section["type"]),
        sampling_rule_version=str(section["sampling_rule_version"]),
        small_fraction_rule=str(section["small_fraction_rule"]),
        multi_component=_boolean(
            section["multi_component"],
            name="sampling_policy.multi_component",
        ),
        eligible_multiplicity=int(section["eligible_multiplicity"]),
        validation_used_for_sampling=_boolean(
            section["validation_used_for_sampling"],
            name="sampling_policy.validation_used_for_sampling",
        ),
        test_split_used=_boolean(
            section["test_split_used"],
            name="sampling_policy.test_split_used",
        ),
    )


# ADD 2026-08-28: Optional official train-view snapshot을 behavior rule과 분리해 복원한다.
def _load_expected_train_view(raw: object) -> ExpectedTrainView | None:
    if raw is None:
        return None
    section = _mapping(raw, name="expected_train_view")
    return ExpectedTrainView(
        unique_train_count=int(section["unique_train_count"]),
        unique_positive_count=int(section["unique_positive_count"]),
        unique_good_negative_count=int(section["unique_good_negative_count"]),
        small_aware_count=int(section["small_aware_count"]),
        multi_component_count=int(section["multi_component_count"]),
        eligible_overlap_count=int(section["eligible_overlap_count"]),
        eligible_union_count=int(section["eligible_union_count"]),
        expanded_entry_count=int(section["expanded_entry_count"]),
        expanded_positive_count=int(section["expanded_positive_count"]),
        expanded_good_negative_count=int(section["expanded_good_negative_count"]),
        expanded_good_negative_ratio=float(section["expanded_good_negative_ratio"]),
        observed_train_small_cutoff=float(section["observed_train_small_cutoff"]),
    )


# ADD 2026-08-31: Optional C4-2C crop policy를 strict typed contract로 복원한다.
def _load_crop_sampling_policy(raw: object) -> CropSamplingPolicy | None:
    if raw is None:
        return None
    section = _mapping(raw, name="crop_sampling_policy")
    _require_keys(
        section,
        {"sampling_mode", "sampling_multiplicity", "crop_size"},
        name="crop_sampling_policy",
    )
    return CropSamplingPolicy(
        sampling_mode=str(section["sampling_mode"]),
        sampling_multiplicity=int(section["sampling_multiplicity"]),
        crop_size=int(section["crop_size"]),
    )


# ADD 2026-08-31: C4-2C expected train-view snapshot을 strict mapping으로 복원한다.
def _load_expected_crop_train_view(raw: object) -> ExpectedCropTrainView | None:
    if raw is None:
        return None
    section = _mapping(raw, name="expected_crop_train_view")
    names = {
        "canonical_entries",
        "canonical_positives",
        "canonical_negatives",
        "component_duplicate_entries",
        "small_centered_crop_entries",
        "total_entries",
        "positive_exposure",
        "negative_exposure",
        "small_aware_count",
        "multi_component_count",
        "eligible_overlap_count",
        "eligible_union_count",
        "observed_train_small_cutoff",
    }
    _require_keys(section, names, name="expected_crop_train_view")
    return ExpectedCropTrainView(
        canonical_entries=int(section["canonical_entries"]),
        canonical_positives=int(section["canonical_positives"]),
        canonical_negatives=int(section["canonical_negatives"]),
        component_duplicate_entries=int(section["component_duplicate_entries"]),
        small_centered_crop_entries=int(section["small_centered_crop_entries"]),
        total_entries=int(section["total_entries"]),
        positive_exposure=int(section["positive_exposure"]),
        negative_exposure=int(section["negative_exposure"]),
        small_aware_count=int(section["small_aware_count"]),
        multi_component_count=int(section["multi_component_count"]),
        eligible_overlap_count=int(section["eligible_overlap_count"]),
        eligible_union_count=int(section["eligible_union_count"]),
        observed_train_small_cutoff=float(section["observed_train_small_cutoff"]),
    )


# ADD 2026-08-31: Explicit C4-2C Ultralytics arguments를 typed mapping으로 복원한다.
def _load_trainer_overrides(raw: object) -> TrainerOverrides | None:
    if raw is None:
        return None
    section = _mapping(raw, name="trainer_overrides")
    _require_keys(
        section,
        {"mosaic", "mask_ratio", "overlap_mask", "scale"},
        name="trainer_overrides",
    )
    return TrainerOverrides(
        mosaic=float(section["mosaic"]),
        mask_ratio=int(section["mask_ratio"]),
        overlap_mask=_boolean(section["overlap_mask"], name="trainer_overrides.overlap_mask"),
        scale=float(section["scale"]),
    )


# ADD 2026-08-31: Fast-compatible C4-2C prediction/gate protocol을 strict하게 복원한다.
def _load_confirmation_protocol(raw: object) -> ConfirmationProtocol | None:
    if raw is None:
        return None
    section = _mapping(raw, name="confirmation_protocol")
    names = {
        "initial_confidence",
        "final_confidence",
        "prediction_iou",
        "max_det",
        "retina_masks",
        "mask_threshold",
        "mask_resize",
        "matching_iou",
        "small_recall_floor_exclusive",
        "mask_map50_95_floor",
        "multi_recall_floor",
        "good_negative_fp_rate",
    }
    _require_keys(section, names, name="confirmation_protocol")
    return ConfirmationProtocol(
        initial_confidence=float(section["initial_confidence"]),
        final_confidence=float(section["final_confidence"]),
        prediction_iou=float(section["prediction_iou"]),
        max_det=int(section["max_det"]),
        retina_masks=_boolean(section["retina_masks"], name="confirmation_protocol.retina_masks"),
        mask_threshold=float(section["mask_threshold"]),
        mask_resize=str(section["mask_resize"]),
        matching_iou=float(section["matching_iou"]),
        small_recall_floor_exclusive=float(section["small_recall_floor_exclusive"]),
        mask_map50_95_floor=float(section["mask_map50_95_floor"]),
        multi_recall_floor=float(section["multi_recall_floor"]),
        good_negative_fp_rate=float(section["good_negative_fp_rate"]),
    )


# ADD 2026-08-27: Typed YAML을 읽는다. → MODIFY 2026-08-31: C4-2C strict recipe도 복원한다.
def load_yolo_experiment_config(path: Path) -> YoloExperimentConfig:
    resolved_config_path = path.resolve()
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"YOLO experiment config not found: {resolved_config_path}")
    raw = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, name="root")
    allowed_root_sections = {
        "experiment",
        "baseline",
        "baseline_evidence",
        "candidate",
        "controlled_change",
        "sampling_policy",
        "expected_train_view",
        "crop_sampling_policy",
        "expected_crop_train_view",
        "trainer_overrides",
        "confirmation_protocol",
        "validation_protocol",
        "telemetry",
        "output",
        "evaluation_priorities",
        "decision_policy",
    }
    unknown_sections = set(root) - allowed_root_sections
    if unknown_sections:
        raise ValueError(f"Unknown experiment config sections: {sorted(unknown_sections)}")
    experiment = _mapping(root.get("experiment"), name="experiment")
    baseline = _mapping(root.get("baseline"), name="baseline")
    baseline_evidence = _mapping(root.get("baseline_evidence"), name="baseline_evidence")
    candidate = _mapping(root.get("candidate"), name="candidate")
    controlled = _mapping(root.get("controlled_change"), name="controlled_change")
    validation = _mapping(root.get("validation_protocol"), name="validation_protocol")
    telemetry = _mapping(root.get("telemetry"), name="telemetry")
    output = _mapping(root.get("output"), name="output")
    priorities = _mapping(root.get("evaluation_priorities"), name="evaluation_priorities")
    decision = _mapping(root.get("decision_policy"), name="decision_policy")
    sampling_raw = root.get("sampling_policy")
    expected_train_view_raw = root.get("expected_train_view")
    crop_sampling_raw = root.get("crop_sampling_policy")
    expected_crop_raw = root.get("expected_crop_train_view")
    trainer_overrides_raw = root.get("trainer_overrides")
    confirmation_raw = root.get("confirmation_protocol")
    try:
        declared_baseline_path = Path(str(experiment["baseline_config"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Experiment baseline config path is missing or malformed.") from exc
    repository_root, baseline_config_path = _resolve_experiment_repository_root(
        resolved_config_path,
        declared_baseline_path,
    )
    try:
        config = YoloExperimentConfig(
            experiment_id=str(experiment["experiment_id"]),
            experiment_date=str(experiment["date"]),
            status=str(experiment["status"]),
            hypothesis=str(experiment["hypothesis"]),
            target_failure_mode=str(experiment["target_failure_mode"]),
            baseline_config_path=baseline_config_path,
            baseline_identity=dict(baseline),
            baseline_evidence=dict(baseline_evidence),
            candidate_identity=dict(candidate),
            intervention_type=str(experiment.get("intervention_type", RESOLUTION_INTERVENTION)),
            controlled_change=ControlledChange(
                field=str(controlled["field"]),
                before=_controlled_value(controlled["before"], name="controlled_change.before"),
                after=_controlled_value(controlled["after"], name="controlled_change.after"),
            ),
            sampling_policy=_load_sampling_policy(sampling_raw),
            expected_train_view=_load_expected_train_view(expected_train_view_raw),
            crop_sampling_policy=_load_crop_sampling_policy(crop_sampling_raw),
            expected_crop_train_view=_load_expected_crop_train_view(expected_crop_raw),
            trainer_overrides=_load_trainer_overrides(trainer_overrides_raw),
            confirmation_protocol=_load_confirmation_protocol(confirmation_raw),
            validation_protocol=ValidationProtocol(
                split=str(validation["split"]),
                test_split_used=_boolean(
                    validation["test_split_used"],
                    name="validation_protocol.test_split_used",
                ),
                diagnostic_confidence=float(validation["diagnostic_confidence"]),
                matching_method=str(validation["matching_method"]),
                mask_iou_threshold=float(validation["mask_iou_threshold"]),
                small_max_area_ratio=float(validation["small_max_area_ratio"]),
                medium_max_area_ratio=float(validation["medium_max_area_ratio"]),
            ),
            telemetry=TelemetryConfig(
                sample_interval_seconds=float(telemetry["sample_interval_seconds"]),
                nvidia_smi_timeout_seconds=float(telemetry["nvidia_smi_timeout_seconds"]),
            ),
            output=ExperimentOutputConfig(
                experiment_root=_resolve_experiment_output_path(
                    repository_root,
                    output["experiment_root"],
                    field="output.experiment_root",
                ),
                artifact_root=_resolve_experiment_output_path(
                    repository_root,
                    output["artifact_root"],
                    field="output.artifact_root",
                ),
                training_runtime_root=_resolve_experiment_output_path(
                    repository_root,
                    output["training_runtime_root"],
                    field="output.training_runtime_root",
                ),
                package_root=_resolve_experiment_output_path(
                    repository_root,
                    output["package_root"],
                    field="output.package_root",
                ),
            ),
            evaluation_priorities={
                key: tuple(str(item) for item in cast(list[object], value))
                for key, value in priorities.items()
            },
            decision_policy=DecisionPolicy(
                require_mask_map50_95_non_regression=_boolean(
                    decision["require_mask_map50_95_non_regression"],
                    name="decision_policy.require_mask_map50_95_non_regression",
                ),
                require_instance_recall_non_regression=_boolean(
                    decision["require_instance_recall_non_regression"],
                    name="decision_policy.require_instance_recall_non_regression",
                ),
                require_small_recall_improvement=_boolean(
                    decision["require_small_recall_improvement"],
                    name="decision_policy.require_small_recall_improvement",
                ),
                require_multi_component_recall_non_regression=_boolean(
                    decision["require_multi_component_recall_non_regression"],
                    name=("decision_policy.require_multi_component_recall_non_regression"),
                ),
                allow_good_negative_fp_rate_increase=_boolean(
                    decision["allow_good_negative_fp_rate_increase"],
                    name="decision_policy.allow_good_negative_fp_rate_increase",
                ),
            ),
            config_path=resolved_config_path,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Experiment config is missing or malformed: {exc}") from exc
    baseline_config = load_yolo_segmentation_config(config.baseline_config_path)
    config.validate(baseline_config)
    return config


# ADD 2026-08-27: Nested comparison metric을 finite numeric value로 읽는다.
def _metric(payload: dict[str, Any], *path: str) -> float:
    value: object = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Comparison metric is missing: {'.'.join(path)}")
        value = value[key]
    if not isinstance(value, int | float):
        raise ValueError(f"Comparison metric must be numeric: {'.'.join(path)}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Comparison metric must be finite: {'.'.join(path)}")
    return numeric


# ADD 2026-08-27: Predeclared primary/failure/guardrail metrics로 candidate recommendation을 만든다.
def recommend_experiment(
    *,
    quality_before: dict[str, Any],
    quality_after: dict[str, Any],
    policy: DecisionPolicy,
) -> ExperimentRecommendation:
    checks = {
        "mask_map50_95_non_regression": _metric(quality_after, "ultralytics", "mask", "map50_95")
        >= _metric(quality_before, "ultralytics", "mask", "map50_95"),
        "instance_recall_non_regression": _metric(quality_after, "diagnostic", "recall")
        >= _metric(quality_before, "diagnostic", "recall"),
        "small_recall_improvement": _metric(quality_after, "failure_modes", "small_recall")
        > _metric(quality_before, "failure_modes", "small_recall"),
        "multi_component_recall_non_regression": _metric(
            quality_after, "failure_modes", "multi_component_recall"
        )
        >= _metric(quality_before, "failure_modes", "multi_component_recall"),
        "good_negative_fp_guardrail": _metric(
            quality_after, "failure_modes", "good_negative_fp_image_rate"
        )
        <= _metric(quality_before, "failure_modes", "good_negative_fp_image_rate"),
    }
    required = {
        "mask_map50_95_non_regression": policy.require_mask_map50_95_non_regression,
        "instance_recall_non_regression": policy.require_instance_recall_non_regression,
        "small_recall_improvement": policy.require_small_recall_improvement,
        "multi_component_recall_non_regression": (
            policy.require_multi_component_recall_non_regression
        ),
        "good_negative_fp_guardrail": not policy.allow_good_negative_fp_rate_increase,
    }
    failed = [name for name, enabled in required.items() if enabled and not checks[name]]
    if not failed:
        return ExperimentRecommendation(
            decision="ACCEPT",
            decision_reason=(
                "All predeclared validation primary, failure-focused, and negative guardrail "
                "checks passed; this accepts a candidate only, not a runtime replacement."
            ),
            checks=checks,
        )
    primary_or_guardrail = {
        "mask_map50_95_non_regression",
        "instance_recall_non_regression",
        "good_negative_fp_guardrail",
    }
    if primary_or_guardrail.intersection(failed):
        return ExperimentRecommendation(
            decision="REJECT",
            decision_reason="Predeclared primary or good-negative guardrail failed: "
            + ", ".join(failed),
            checks=checks,
        )
    return ExperimentRecommendation(
        decision="PENDING",
        decision_reason="Failure-focused improvement is incomplete: " + ", ".join(failed),
        checks=checks,
    )


# ADD 2026-08-31: C4-2C absolute Primary gates를 legacy recommendation과 분리해 판정한다.
def confirm_c4_2c_candidate(
    *,
    quality_after: dict[str, Any],
    protocol: ConfirmationProtocol,
) -> ExperimentRecommendation:
    protocol.validate()
    checks = {
        "small_recall_above_floor": _metric(quality_after, "failure_modes", "small_recall")
        > protocol.small_recall_floor_exclusive,
        "mask_map50_95_floor": _metric(quality_after, "ultralytics", "mask", "map50_95")
        >= protocol.mask_map50_95_floor,
        "multi_recall_floor": _metric(quality_after, "failure_modes", "multi_component_recall")
        >= protocol.multi_recall_floor,
        "good_negative_fp_guardrail": _metric(
            quality_after, "failure_modes", "good_negative_fp_image_rate"
        )
        == protocol.good_negative_fp_rate,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if not failed:
        return ExperimentRecommendation(
            decision="CONFIRMED_CANDIDATE",
            decision_reason=(
                "All absolute C4-2C validation gates passed; this confirms a candidate only, "
                "not a final model promotion."
            ),
            checks=checks,
        )
    return ExperimentRecommendation(
        decision="CONFIRMATION_FAILED",
        decision_reason="C4-2C absolute confirmation gates failed: " + ", ".join(failed),
        checks=checks,
    )


# ADD 2026-08-27: Pre-run metadata를 만든다. → MODIFY 2026-08-31: Resolved C4-2C recipe를 포함한다.
def build_experiment_metadata(
    config: YoloExperimentConfig,
    *,
    git_commit: str | None,
    manifest_sha256: str,
) -> dict[str, Any]:
    recipe = {
        "crop_sampling_policy": (
            None if config.crop_sampling_policy is None else asdict(config.crop_sampling_policy)
        ),
        "trainer_overrides": (
            None if config.trainer_overrides is None else asdict(config.trainer_overrides)
        ),
        "confirmation_protocol": (
            None if config.confirmation_protocol is None else asdict(config.confirmation_protocol)
        ),
    }
    recipe_fingerprint = sha256_bytes(
        json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "date": config.experiment_date,
        "status": config.status,
        "hypothesis": config.hypothesis,
        "target_failure_mode": config.target_failure_mode,
        "intervention_type": config.intervention_type,
        "controlled_change": asdict(config.controlled_change),
        "sampling_policy": (
            None if config.sampling_policy is None else asdict(config.sampling_policy)
        ),
        "expected_train_view": (
            None if config.expected_train_view is None else asdict(config.expected_train_view)
        ),
        "expected_crop_train_view": (
            None
            if config.expected_crop_train_view is None
            else asdict(config.expected_crop_train_view)
        ),
        "resolved_recipe": recipe,
        "resolved_recipe_fingerprint_sha256": recipe_fingerprint,
        "constants": config.baseline_identity,
        "historical_baseline_evidence": config.baseline_evidence,
        "validation_protocol": asdict(config.validation_protocol),
        "evaluation_priorities": config.evaluation_priorities,
        "dataset_manifest_sha256": manifest_sha256,
        "experiment_config_path": config.config_path.as_posix(),
        "experiment_config_sha256": sha256_file(config.config_path),
        "git_commit": git_commit,
        "decision": "PENDING",
        "test_used": False,
    }


# ADD 2026-08-27: Final result schema를 검증한다. → MODIFY 2026-08-31: C4-2C decision을 허용한다.
def validate_experiment_result(payload: dict[str, Any]) -> None:
    required = {
        "experiment_id",
        "hypothesis",
        "controlled_change",
        "constants",
        "split",
        "test_split_used",
        "quality_before",
        "quality_after",
        "resource_metrics",
        "failure_mode_metrics",
        "model_sha256",
        "metadata_sha256",
        "manifest_sha256",
        "decision",
        "decision_reason",
    }
    if set(payload) < required:
        raise ValueError("Experiment result is missing required fields.")
    if payload["split"] != "val" or payload["test_split_used"] is not False:
        raise ValueError("Experiment result must remain validation-only.")
    if payload.get("test_used", False) is not False:
        raise ValueError("Experiment result must keep the derived test split sealed.")
    if payload["decision"] not in EXPERIMENT_DECISIONS:
        raise ValueError("Experiment result decision is invalid.")
    json.dumps(payload, allow_nan=False)
