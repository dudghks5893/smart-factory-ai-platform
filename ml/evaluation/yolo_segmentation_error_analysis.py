"""Validation-only diagnostics for YOLO known-defect segmentation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord

MATCH_IOU_THRESHOLD = 0.5
LOW_IOU_THRESHOLD = 0.65
COVERAGE_THRESHOLD = 0.75
CONFIDENCE_LEVELS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
ERROR_PRIORITY = {
    "MISSED_DEFECT": 0,
    "WRONG_CLASS": 1,
    "MULTI_COMPONENT_MISS": 2,
    "FALSE_POSITIVE": 3,
    "LOW_IOU_LOCALIZATION": 4,
    "MASK_UNDER_SEGMENTATION": 5,
    "MASK_OVER_SEGMENTATION": 6,
    "TRUE_POSITIVE": 7,
    "TRUE_NEGATIVE": 8,
}


@dataclass(frozen=True)
class GroundTruthInstance:
    """One validation annotation represented at source-image resolution."""

    class_id: int
    mask: NDArray[np.bool_]
    box_xyxy: tuple[float, float, float, float]
    area_ratio: float


@dataclass(frozen=True)
class PredictedInstance:
    """One confidence-bearing model prediction at source-image resolution."""

    class_id: int
    confidence: float
    mask: NDArray[np.bool_]
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class InstanceMatch:
    """One deterministic class-aware GT/prediction match."""

    ground_truth_index: int
    prediction_index: int
    mask_iou: float
    box_iou: float
    ground_truth_coverage: float
    prediction_precision: float


@dataclass(frozen=True)
class GroundTruthOutcome:
    """Size-aware outcome for one ground-truth validation instance."""

    class_id: int
    area_ratio: float
    size_bucket: str
    matched: bool
    mask_iou: float | None


@dataclass(frozen=True)
class SampleAnalysis:
    """Serializable validation diagnostics for one image."""

    sample_id: str
    ground_truth_class: str
    is_negative: bool
    ground_truth_instance_count: int
    predicted_instance_count: int
    expected_class_hit: bool
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    best_mask_iou: float | None
    best_box_iou: float | None
    predicted_confidence: float | None
    ground_truth_mask_area_ratio: float
    predicted_mask_area_ratio: float
    ground_truth_component_count: int
    size_bucket: str | None
    main_error: str
    secondary_tags: tuple[str, ...]
    predicted_classes: tuple[str, ...]
    predicted_instance_confidences: tuple[float, ...]
    matches: tuple[InstanceMatch, ...]
    ground_truth_outcomes: tuple[GroundTruthOutcome, ...]
    confusion_pairs: tuple[tuple[str, str], ...]

    # ADD 2026-08-26: Nested tuple/dataclass fields를 stable JSON object로 변환한다.
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SizeBucketPolicy:
    """Validation-distribution tertile boundaries for GT mask area ratio."""

    method: str
    small_max: float
    medium_max: float

    # ADD 2026-08-26: Validation area ratio를 descriptive tertile bucket으로 분류한다.
    def classify(self, area_ratio: float) -> str:
        if area_ratio <= self.small_max:
            return "small"
        if area_ratio <= self.medium_max:
            return "medium"
        return "large"


# ADD 2026-08-26: Analysis boundary에서 derived validation row 외 입력을 fail-fast한다.
def require_validation_records(
    records: list[DerivedManifestRecord],
) -> list[DerivedManifestRecord]:
    """Return deterministic validation rows while rejecting any train/test leakage."""
    if not records:
        raise ValueError("Validation error analysis requires manifest records.")
    invalid = sorted({record.derived_split for record in records if record.derived_split != "val"})
    if invalid:
        raise ValueError(f"Validation error analysis rejects non-val split rows: {invalid}")
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Validation error analysis sample IDs must be unique.")
    return sorted(records, key=lambda record: record.sample_id)


# ADD 2026-08-26: Binary mask의 tight pixel-edge bounding box를 계산한다.
def mask_box(mask: NDArray[np.bool_]) -> tuple[float, float, float, float]:
    if mask.dtype != np.bool_ or mask.ndim != 2 or not mask.any():
        raise ValueError("Segmentation instance mask must be a non-empty 2D boolean array.")
    ys, xs = np.nonzero(mask)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


# ADD 2026-08-26: Empty-safe binary mask IoU와 directional coverage를 계산한다.
def mask_overlap(
    ground_truth: NDArray[np.bool_], prediction: NDArray[np.bool_]
) -> tuple[float, float, float]:
    if ground_truth.shape != prediction.shape or ground_truth.dtype != np.bool_:
        raise ValueError("GT and prediction masks must be aligned boolean arrays.")
    if prediction.dtype != np.bool_ or not ground_truth.any() or not prediction.any():
        raise ValueError("GT and prediction masks must both be non-empty.")
    intersection = int(np.count_nonzero(ground_truth & prediction))
    union = int(np.count_nonzero(ground_truth | prediction))
    return (
        intersection / union,
        intersection / int(np.count_nonzero(ground_truth)),
        intersection / int(np.count_nonzero(prediction)),
    )


# ADD 2026-08-26: Pixel-edge xyxy box의 geometric IoU를 계산한다.
def box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


# ADD 2026-08-26: Validation GT mask area distribution에서 descriptive tertile을 고정한다.
def derive_size_bucket_policy(instances: list[GroundTruthInstance]) -> SizeBucketPolicy:
    if not instances:
        raise ValueError("Size bucket policy requires positive validation instances.")
    values = np.asarray([instance.area_ratio for instance in instances], dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0.0).any() or (values > 1.0).any():
        raise ValueError("GT mask area ratios must be finite and in (0, 1].")
    return SizeBucketPolicy(
        method="validation_gt_mask_area_ratio_tertiles",
        small_max=float(np.quantile(values, 1.0 / 3.0)),
        medium_max=float(np.quantile(values, 2.0 / 3.0)),
    )


# ADD 2026-08-26: Highest mask-IoU first class-aware greedy matching을 deterministic하게 수행한다.
def match_instances(
    ground_truth: tuple[GroundTruthInstance, ...],
    predictions: tuple[PredictedInstance, ...],
    *,
    iou_threshold: float = MATCH_IOU_THRESHOLD,
) -> tuple[InstanceMatch, ...]:
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("Matching IoU threshold must be in (0, 1].")
    candidates: list[InstanceMatch] = []
    for gt_index, gt in enumerate(ground_truth):
        for prediction_index, prediction in enumerate(predictions):
            if gt.class_id != prediction.class_id:
                continue
            overlap, gt_coverage, prediction_precision = mask_overlap(gt.mask, prediction.mask)
            if overlap >= iou_threshold:
                candidates.append(
                    InstanceMatch(
                        ground_truth_index=gt_index,
                        prediction_index=prediction_index,
                        mask_iou=overlap,
                        box_iou=box_iou(gt.box_xyxy, prediction.box_xyxy),
                        ground_truth_coverage=gt_coverage,
                        prediction_precision=prediction_precision,
                    )
                )
    matched_gt: set[int] = set()
    matched_predictions: set[int] = set()
    selected: list[InstanceMatch] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-item.mask_iou, item.ground_truth_index, item.prediction_index),
    ):
        if (
            candidate.ground_truth_index in matched_gt
            or candidate.prediction_index in matched_predictions
        ):
            continue
        selected.append(candidate)
        matched_gt.add(candidate.ground_truth_index)
        matched_predictions.add(candidate.prediction_index)
    return tuple(sorted(selected, key=lambda item: item.ground_truth_index))


# ADD 2026-08-26: IoU/confidence/area 값의 empty-safe descriptive distribution을 만든다.
def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


# ADD 2026-08-26: 남은 spatial overlap을 wrong-class evidence로 deterministic하게 대응한다.
def _wrong_class_pairs(
    ground_truth: tuple[GroundTruthInstance, ...],
    predictions: tuple[PredictedInstance, ...],
    matches: tuple[InstanceMatch, ...],
) -> tuple[tuple[int, int], ...]:
    matched_gt = {match.ground_truth_index for match in matches}
    matched_predictions = {match.prediction_index for match in matches}
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt in enumerate(ground_truth):
        if gt_index in matched_gt:
            continue
        for prediction_index, prediction in enumerate(predictions):
            if prediction_index in matched_predictions or gt.class_id == prediction.class_id:
                continue
            overlap, _, _ = mask_overlap(gt.mask, prediction.mask)
            if overlap >= MATCH_IOU_THRESHOLD:
                candidates.append((overlap, gt_index, prediction_index))
    selected: list[tuple[int, int]] = []
    for _, gt_index, prediction_index in sorted(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    ):
        if any(gt_index == pair[0] or prediction_index == pair[1] for pair in selected):
            continue
        selected.append((gt_index, prediction_index))
    return tuple(selected)


# ADD 2026-08-26: Validation error를 분류한다. → MODIFY 2026-09-01: Test split을 opt-in 허용한다.
def analyze_sample(
    *,
    record: DerivedManifestRecord,
    ground_truth: tuple[GroundTruthInstance, ...],
    predictions: tuple[PredictedInstance, ...],
    classes: dict[int, str],
    size_policy: SizeBucketPolicy,
    expected_split: Literal["val", "test"] = "val",
) -> SampleAnalysis:
    if record.derived_split != expected_split:
        raise ValueError(f"Sample analysis accepts only derived {expected_split} rows.")
    matches = match_instances(ground_truth, predictions)
    matched_gt = {match.ground_truth_index for match in matches}
    matched_predictions = {match.prediction_index for match in matches}
    wrong_pairs = _wrong_class_pairs(ground_truth, predictions, matches)
    false_negative_count = len(ground_truth) - len(matches)
    false_positive_count = len(predictions) - len(matches)

    tags: set[str] = set()
    if false_negative_count:
        tags.add("MISSED_DEFECT")
    if false_positive_count:
        tags.add("FALSE_POSITIVE")
    if wrong_pairs:
        tags.add("WRONG_CLASS")
    if record.component_count > 1 and false_negative_count:
        tags.add("MULTI_COMPONENT_MISS")
    if any(match.mask_iou < LOW_IOU_THRESHOLD for match in matches):
        tags.add("LOW_IOU_LOCALIZATION")
    unmatched_gt = [index for index in range(len(ground_truth)) if index not in matched_gt]
    unmatched_predictions = {
        index for index in range(len(predictions)) if index not in matched_predictions
    }
    if any(
        0.0
        < mask_overlap(ground_truth[gt_index].mask, predictions[prediction_index].mask)[0]
        < MATCH_IOU_THRESHOLD
        for gt_index in unmatched_gt
        for prediction_index in unmatched_predictions
        if ground_truth[gt_index].class_id == predictions[prediction_index].class_id
    ):
        tags.add("LOW_IOU_LOCALIZATION")
    if any(match.ground_truth_coverage < COVERAGE_THRESHOLD for match in matches):
        tags.add("MASK_UNDER_SEGMENTATION")
    if any(match.prediction_precision < COVERAGE_THRESHOLD for match in matches):
        tags.add("MASK_OVER_SEGMENTATION")

    if record.is_negative and not predictions:
        main_error = "TRUE_NEGATIVE"
    elif wrong_pairs:
        main_error = "WRONG_CLASS"
    elif not tags:
        main_error = "TRUE_POSITIVE"
    else:
        main_error = min(tags, key=lambda tag: (ERROR_PRIORITY[tag], tag))

    confusion_pairs: list[tuple[str, str]] = []
    match_by_gt = {match.ground_truth_index: match for match in matches}
    wrong_by_gt = dict(wrong_pairs)
    for gt_index, gt in enumerate(ground_truth):
        gt_name = classes[gt.class_id]
        if gt_index in match_by_gt:
            outcome = gt_name
        elif gt_index in wrong_by_gt:
            outcome = classes[predictions[wrong_by_gt[gt_index]].class_id]
        else:
            outcome = "NO_PREDICTION"
        confusion_pairs.append((gt_name, outcome))

    gt_outcomes = tuple(
        GroundTruthOutcome(
            class_id=instance.class_id,
            area_ratio=instance.area_ratio,
            size_bucket=size_policy.classify(instance.area_ratio),
            matched=index in matched_gt,
            mask_iou=match_by_gt[index].mask_iou if index in match_by_gt else None,
        )
        for index, instance in enumerate(ground_truth)
    )
    gt_union = np.logical_or.reduce([item.mask for item in ground_truth]) if ground_truth else None
    prediction_union = (
        np.logical_or.reduce([item.mask for item in predictions]) if predictions else None
    )
    target_class_id = int(record.target_class_id) if record.target_class_id else None
    return SampleAnalysis(
        sample_id=record.sample_id,
        ground_truth_class=record.defect_type,
        is_negative=record.is_negative,
        ground_truth_instance_count=len(ground_truth),
        predicted_instance_count=len(predictions),
        expected_class_hit=(
            target_class_id is not None
            and any(prediction.class_id == target_class_id for prediction in predictions)
        ),
        true_positive_count=len(matches),
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        best_mask_iou=max((match.mask_iou for match in matches), default=None),
        best_box_iou=max((match.box_iou for match in matches), default=None),
        predicted_confidence=max((item.confidence for item in predictions), default=None),
        ground_truth_mask_area_ratio=(
            float(np.count_nonzero(gt_union) / gt_union.size) if gt_union is not None else 0.0
        ),
        predicted_mask_area_ratio=(
            float(np.count_nonzero(prediction_union) / prediction_union.size)
            if prediction_union is not None
            else 0.0
        ),
        ground_truth_component_count=record.component_count,
        size_bucket=(
            size_policy.classify(float(np.count_nonzero(gt_union) / gt_union.size))
            if gt_union is not None
            else None
        ),
        main_error=main_error,
        secondary_tags=tuple(sorted(tags, key=lambda tag: (ERROR_PRIORITY[tag], tag))),
        predicted_classes=tuple(classes[item.class_id] for item in predictions),
        predicted_instance_confidences=tuple(item.confidence for item in predictions),
        matches=matches,
        ground_truth_outcomes=gt_outcomes,
        confusion_pairs=tuple(confusion_pairs),
    )


# ADD 2026-08-26: Sample diagnostics를 overall/class/size/component/negative evidence로 집계한다.
def aggregate_analysis(
    analyses: list[SampleAnalysis], *, classes: dict[int, str]
) -> dict[str, Any]:
    if not analyses:
        raise ValueError("Validation analysis cannot be empty.")
    positives = [item for item in analyses if not item.is_negative]
    negatives = [item for item in analyses if item.is_negative]
    tp = sum(item.true_positive_count for item in analyses)
    fp = sum(item.false_positive_count for item in analyses)
    fn = sum(item.false_negative_count for item in analyses)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    per_class: dict[str, Any] = {}
    for class_id, class_name in sorted(classes.items()):
        class_samples = [item for item in positives if item.ground_truth_class == class_name]
        class_outcomes = [
            outcome
            for item in analyses
            for outcome in item.ground_truth_outcomes
            if outcome.class_id == class_id
        ]
        class_tp = sum(outcome.matched for outcome in class_outcomes)
        class_fn = len(class_outcomes) - class_tp
        class_fp = (
            sum(
                1
                for item in analyses
                for predicted_class in item.predicted_classes
                if predicted_class == class_name
            )
            - class_tp
        )
        per_class[class_name] = {
            "sample_count": len(class_samples),
            "instance_count": len(class_outcomes),
            "tp": class_tp,
            "fp": class_fp,
            "fn": class_fn,
            "precision": class_tp / (class_tp + class_fp) if class_tp + class_fp else 0.0,
            "recall": class_tp / len(class_outcomes) if class_outcomes else 0.0,
            "mask_iou": _distribution(
                [outcome.mask_iou for outcome in class_outcomes if outcome.mask_iou is not None]
            ),
            "box_iou": _distribution(
                [match.box_iou for item in class_samples for match in item.matches]
            ),
            "confidence": _distribution(
                [
                    confidence
                    for item in analyses
                    for predicted_class, confidence in zip(
                        item.predicted_classes,
                        item.predicted_instance_confidences,
                        strict=True,
                    )
                    if predicted_class == class_name
                ]
            ),
            "ground_truth_area_ratio": _distribution(
                [outcome.area_ratio for outcome in class_outcomes]
            ),
        }

    size_summary: dict[str, Any] = {}
    all_outcomes = [outcome for item in positives for outcome in item.ground_truth_outcomes]
    for bucket in ("small", "medium", "large"):
        outcomes = [item for item in all_outcomes if item.size_bucket == bucket]
        matched = sum(item.matched for item in outcomes)
        size_summary[bucket] = {
            "instance_count": len(outcomes),
            "tp": matched,
            "fn": len(outcomes) - matched,
            "recall": matched / len(outcomes) if outcomes else 0.0,
            "mask_iou": _distribution(
                [item.mask_iou for item in outcomes if item.mask_iou is not None]
            ),
        }

    component_summary: dict[str, Any] = {}
    for group_name, predicate in (
        ("single_component", lambda item: item.ground_truth_component_count == 1),
        ("multi_component", lambda item: item.ground_truth_component_count > 1),
    ):
        samples = [item for item in positives if predicate(item)]
        group_tp = sum(item.true_positive_count for item in samples)
        group_fn = sum(item.false_negative_count for item in samples)
        component_summary[group_name] = {
            "sample_count": len(samples),
            "instance_count": group_tp + group_fn,
            "tp": group_tp,
            "fn": group_fn,
            "recall": group_tp / (group_tp + group_fn) if group_tp + group_fn else 0.0,
            "mask_iou": _distribution(
                [match.mask_iou for item in samples for match in item.matches]
            ),
        }

    confusion: dict[str, dict[str, int]] = {
        class_name: {**{name: 0 for name in classes.values()}, "NO_PREDICTION": 0}
        for class_name in classes.values()
    }
    for item in analyses:
        for ground_truth_class, outcome in item.confusion_pairs:
            confusion[ground_truth_class][outcome] += 1

    taxonomy = Counter(item.main_error for item in analyses)
    secondary = Counter(tag for item in analyses for tag in item.secondary_tags)
    negative_predictions = [
        confidence for item in negatives for confidence in item.predicted_instance_confidences
    ]
    return {
        "sample_count": len(analyses),
        "positive_sample_count": len(positives),
        "negative_sample_count": len(negatives),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "per_class": per_class,
        "size_analysis": size_summary,
        "component_analysis": component_summary,
        "confusion": confusion,
        "negative_analysis": {
            "sample_count": len(negatives),
            "false_positive_image_count": sum(
                item.predicted_instance_count > 0 for item in negatives
            ),
            "false_positive_instance_count": sum(
                item.predicted_instance_count for item in negatives
            ),
            "false_positive_image_rate": (
                sum(item.predicted_instance_count > 0 for item in negatives) / len(negatives)
                if negatives
                else 0.0
            ),
            "predicted_classes": dict(
                sorted(
                    Counter(
                        class_name for item in negatives for class_name in item.predicted_classes
                    ).items()
                )
            ),
            "confidence": _distribution(negative_predictions),
        },
        "error_taxonomy": {
            "main": dict(sorted(taxonomy.items())),
            "secondary": dict(sorted(secondary.items())),
        },
    }


# ADD 2026-08-26: Lowest-confidence prediction pool을 validation-only operating points로 재평가한다.
def build_confidence_sweep(
    *,
    records: list[DerivedManifestRecord],
    ground_truth_by_sample: dict[str, tuple[GroundTruthInstance, ...]],
    predictions_by_sample: dict[str, tuple[PredictedInstance, ...]],
    classes: dict[int, str],
    size_policy: SizeBucketPolicy,
    confidence_levels: tuple[float, ...] = CONFIDENCE_LEVELS,
) -> list[dict[str, Any]]:
    validation_records = require_validation_records(records)
    sweep: list[dict[str, Any]] = []
    for confidence in confidence_levels:
        analyses = [
            analyze_sample(
                record=record,
                ground_truth=ground_truth_by_sample[record.sample_id],
                predictions=tuple(
                    item
                    for item in predictions_by_sample[record.sample_id]
                    if item.confidence >= confidence
                ),
                classes=classes,
                size_policy=size_policy,
            )
            for record in validation_records
        ]
        summary = aggregate_analysis(analyses, classes=classes)
        sweep.append(
            {
                "confidence": confidence,
                "precision": summary["precision"],
                "recall": summary["recall"],
                "f1": summary["f1"],
                "false_positive_image_rate": summary["negative_analysis"][
                    "false_positive_image_rate"
                ],
                "mean_predicted_instances": fmean(
                    item.predicted_instance_count for item in analyses
                ),
                "class_recall": {
                    class_name: summary["per_class"][class_name]["recall"]
                    for class_name in classes.values()
                },
            }
        )
    return sweep


# ADD 2026-08-26: Error severity, mask IoU와 sample identity로 worst samples를 stable하게 정렬한다.
def rank_worst_samples(analyses: list[SampleAnalysis]) -> list[SampleAnalysis]:
    return sorted(
        analyses,
        key=lambda item: (
            0 if item.false_negative_count else 1,
            ERROR_PRIORITY[item.main_error],
            item.best_mask_iou if item.best_mask_iou is not None else -1.0,
            item.sample_id,
        ),
    )


# ADD 2026-08-26: Fixed prediction pool에서 confidence별 immutable sample view를 만든다.
def filter_predictions(
    predictions: dict[str, tuple[PredictedInstance, ...]], confidence: float
) -> dict[str, tuple[PredictedInstance, ...]]:
    return {
        sample_id: tuple(item for item in items if item.confidence >= confidence)
        for sample_id, items in predictions.items()
    }


# ADD 2026-08-26: Validation evidence가 지지하는 C4-2 candidate와 unsupported 가설을 분리한다.
def derive_improvement_hypotheses(
    summary: dict[str, Any], confidence_sweep: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    size = summary["size_analysis"]
    if size["large"]["recall"] - size["small"]["recall"] >= 0.15:
        supported.append(
            {
                "evidence": (
                    "Small-instance validation recall trails large-instance recall "
                    "by at least 0.15."
                ),
                "candidate": (
                    "Compare a higher-imgsz run while keeping the validation protocol fixed."
                ),
            }
        )
    else:
        unsupported.append(
            {
                "hypothesis": "Higher imgsz is required for small defects.",
                "reason": "Validation size-bucket recall does not show the required gap.",
            }
        )

    baseline = next(item for item in confidence_sweep if item["confidence"] == 0.25)
    low = confidence_sweep[0]
    if low["recall"] - baseline["recall"] >= 0.10:
        supported.append(
            {
                "evidence": (
                    "Validation recall at confidence 0.10 exceeds the 0.25 baseline "
                    "by at least 0.10."
                ),
                "candidate": "Run a dedicated validation confidence-calibration experiment.",
            }
        )
    else:
        unsupported.append(
            {
                "hypothesis": "Lowering confidence materially recovers missed validation defects.",
                "reason": "The validation recall gain from 0.25 to 0.10 is below 0.10.",
            }
        )

    wrong_class_count = summary["error_taxonomy"]["secondary"].get("WRONG_CLASS", 0)
    if wrong_class_count:
        supported.append(
            {
                "evidence": f"Validation contains {wrong_class_count} wrong-class sample(s).",
                "candidate": (
                    "Audit confused samples before targeted data or augmentation experiments."
                ),
            }
        )
    else:
        unsupported.append(
            {
                "hypothesis": "Class confusion requires targeted augmentation.",
                "reason": "No validation sample meets the wrong-class IoU diagnostic criterion.",
            }
        )

    low_iou_count = summary["error_taxonomy"]["secondary"].get("LOW_IOU_LOCALIZATION", 0)
    if low_iou_count:
        supported.append(
            {
                "evidence": f"Validation contains {low_iou_count} low-IoU localization sample(s).",
                "candidate": (
                    "Compare higher imgsz or a larger segmentation model as isolated experiments."
                ),
            }
        )
    else:
        unsupported.append(
            {
                "hypothesis": "A larger model is required for localization.",
                "reason": (
                    "No matched validation sample falls below the low-IoU diagnostic boundary."
                ),
            }
        )

    multi = summary["component_analysis"]["multi_component"]
    single = summary["component_analysis"]["single_component"]
    if single["recall"] - multi["recall"] >= 0.15:
        supported.append(
            {
                "evidence": (
                    "Multi-component validation recall trails single-component recall "
                    "by at least 0.15."
                ),
                "candidate": "Test component-preserving crop or sampling strategies on validation.",
            }
        )
    else:
        unsupported.append(
            {
                "hypothesis": "Multi-component defects require a dedicated sampling strategy.",
                "reason": "Validation component groups do not show the required recall gap.",
            }
        )
    unsupported.extend(
        (
            {
                "hypothesis": "The existing evidence requires more augmentation by default.",
                "reason": (
                    "Validation diagnostics identify failure modes but contain no controlled "
                    "augmentation comparison."
                ),
            },
            {
                "hypothesis": "One swept confidence is production-calibrated.",
                "reason": (
                    "The validation sweep has no approved production objective "
                    "or error-cost policy."
                ),
            },
        )
    )
    return {"supported": supported, "unsupported": unsupported}
