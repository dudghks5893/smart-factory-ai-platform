"""Configuration and comparison contracts for controlled YOLO experiments."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, cast

import yaml

from ml.evaluation.yolo_segmentation_error_analysis import (
    MATCH_IOU_THRESHOLD,
    SizeBucketPolicy,
)
from ml.training.yolo_segmentation import (
    YoloOutputConfig,
    YoloSegmentationBaselineConfig,
    load_yolo_segmentation_config,
    validate_artifact_id,
)
from shared.hashing import sha256_file

EXPERIMENT_SCHEMA_VERSION = 1
EXPERIMENT_STATUSES = {"PLANNED", "RUNNING", "COMPLETED", "ACCEPTED", "REJECTED"}
EXPERIMENT_DECISIONS = {"PENDING", "ACCEPT", "REJECT"}
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


@dataclass(frozen=True)
class ControlledChange:
    """One intentionally varied model-quality field."""

    field: str
    before: int
    after: int


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
    controlled_change: ControlledChange
    validation_protocol: ValidationProtocol
    telemetry: TelemetryConfig
    output: ExperimentOutputConfig
    evaluation_priorities: dict[str, tuple[str, ...]]
    decision_policy: DecisionPolicy
    config_path: Path

    # ADD 2026-08-27: C4-2A가 imgsz 외 model-quality 변수를 바꾸지 않는지 검증한다.
    def validate(self, baseline: YoloSegmentationBaselineConfig) -> None:
        validate_artifact_id(self.experiment_id)
        try:
            date.fromisoformat(self.experiment_date)
        except ValueError as exc:
            raise ValueError("Experiment date must be ISO YYYY-MM-DD.") from exc
        if self.status not in EXPERIMENT_STATUSES or not self.hypothesis:
            raise ValueError("Experiment status/hypothesis is invalid.")
        expected_identity = {
            "model": baseline.model.architecture,
            "pretrained_checkpoint": baseline.model.weights,
            "imgsz": baseline.training.imgsz,
            "seed": baseline.training.seed,
            "batch": baseline.training.batch,
            "epochs": baseline.training.epochs,
            "patience": baseline.training.patience,
        }
        if self.baseline_identity != expected_identity:
            raise ValueError("Experiment baseline identity does not match the baseline config.")
        _validate_baseline_evidence(self.baseline_evidence)
        if self.controlled_change != ControlledChange("training.imgsz", 640, 1024):
            raise ValueError("C4-2A must contain only the imgsz 640 -> 1024 controlled change.")
        expected_candidate_identity = {
            **expected_identity,
            "imgsz": self.controlled_change.after,
        }
        if self.candidate_identity != expected_candidate_identity:
            raise ValueError("Experiment candidate identity must change only imgsz.")
        if self.controlled_change.before != baseline.training.imgsz:
            raise ValueError("Controlled-change before value must match the baseline.")
        if baseline.training.batch != 16:
            raise ValueError("C4-2A first attempt requires the baseline batch size 16.")
        self.validation_protocol.validate()
        self.telemetry.validate()
        self.decision_policy.validate()
        for key, expected in EXPECTED_PRIORITIES.items():
            if self.evaluation_priorities.get(key) != expected:
                raise ValueError(f"Experiment priority changed or is incomplete: {key}")

    # ADD 2026-08-27: Baseline에서 imgsz와 isolated output root만 바꾼 config를 만든다.
    def training_config(
        self, baseline: YoloSegmentationBaselineConfig
    ) -> YoloSegmentationBaselineConfig:
        self.validate(baseline)
        controlled_training = replace(
            baseline.training,
            imgsz=self.controlled_change.after,
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


# ADD 2026-08-27: Dedicated YAML을 typed controlled-experiment contract로 복원한다.
def load_yolo_experiment_config(path: Path) -> YoloExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, name="root")
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
    try:
        config = YoloExperimentConfig(
            experiment_id=str(experiment["experiment_id"]),
            experiment_date=str(experiment["date"]),
            status=str(experiment["status"]),
            hypothesis=str(experiment["hypothesis"]),
            target_failure_mode=str(experiment["target_failure_mode"]),
            baseline_config_path=Path(str(experiment["baseline_config"])),
            baseline_identity=dict(baseline),
            baseline_evidence=dict(baseline_evidence),
            candidate_identity=dict(candidate),
            controlled_change=ControlledChange(
                field=str(controlled["field"]),
                before=int(controlled["before"]),
                after=int(controlled["after"]),
            ),
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
                experiment_root=Path(str(output["experiment_root"])),
                artifact_root=Path(str(output["artifact_root"])),
                training_runtime_root=Path(str(output["training_runtime_root"])),
                package_root=Path(str(output["package_root"])),
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
            config_path=path,
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


# ADD 2026-08-27: Pre-run identity, priorities와 lineage를 deterministic metadata로 만든다.
def build_experiment_metadata(
    config: YoloExperimentConfig,
    *,
    git_commit: str | None,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "date": config.experiment_date,
        "status": config.status,
        "hypothesis": config.hypothesis,
        "target_failure_mode": config.target_failure_mode,
        "controlled_change": asdict(config.controlled_change),
        "constants": config.baseline_identity,
        "historical_baseline_evidence": config.baseline_evidence,
        "validation_protocol": asdict(config.validation_protocol),
        "evaluation_priorities": config.evaluation_priorities,
        "dataset_manifest_sha256": manifest_sha256,
        "experiment_config_path": config.config_path.as_posix(),
        "experiment_config_sha256": sha256_file(config.config_path),
        "git_commit": git_commit,
        "decision": "PENDING",
    }


# ADD 2026-08-27: Final machine result가 sealed validation과 decision schema를 보존하는지 검증한다.
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
    if payload["decision"] not in EXPERIMENT_DECISIONS:
        raise ValueError("Experiment result decision is invalid.")
    json.dumps(payload, allow_nan=False)
