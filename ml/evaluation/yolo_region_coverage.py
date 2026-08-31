"""Class-aware validation Region Coverage diagnostics for YOLO segmentation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.evaluation.yolo_segmentation_error_analysis import (
    GroundTruthInstance,
    PredictedInstance,
    SizeBucketPolicy,
    mask_overlap,
    match_instances,
    require_validation_records,
)

REGION_COVERAGE_THRESHOLD = 0.5
NEAR_MISS_MIN_IOU = 0.30
STRICT_INSTANCE_IOU_THRESHOLD = 0.5


@dataclass(frozen=True)
class RegionCoverageMetrics:
    """Dataset-level secondary Region Coverage evidence from one validation prediction pool."""

    strict_instance_gt_recall: float
    gt_component_coverage_recall_at_50: float
    small_gt_coverage_recall_at_50: float
    class_aware_union_iou: float
    class_aware_union_gt_coverage: float
    class_aware_union_pred_precision: float
    near_miss_iou_030_to_050: int
    covered50_but_strict_instance_fail: int
    strict_iou_threshold: float
    coverage_threshold: float
    val_gt_total: int
    small_gt_total: int
    test_used: bool

    # ADD 2026-08-31: Region Coverage를 finite validation-only JSON evidence로 검증한다.
    def to_json_dict(self) -> dict[str, float | int | bool]:
        payload = asdict(self)
        numeric_rates = (
            self.strict_instance_gt_recall,
            self.gt_component_coverage_recall_at_50,
            self.small_gt_coverage_recall_at_50,
            self.class_aware_union_iou,
            self.class_aware_union_gt_coverage,
            self.class_aware_union_pred_precision,
        )
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in numeric_rates):
            raise ValueError("Region Coverage rates must be finite values in [0, 1].")
        if self.val_gt_total <= 0 or self.small_gt_total <= 0 or self.test_used:
            raise ValueError("Region Coverage must contain positive validation-only GT counts.")
        json.dumps(payload, allow_nan=False)
        return payload


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _union_masks(
    masks: list[NDArray[np.bool_]],
    *,
    height: int,
    width: int,
) -> NDArray[np.bool_]:
    union = np.zeros((height, width), dtype=np.bool_)
    for mask in masks:
        if mask.dtype != np.bool_ or mask.shape != union.shape:
            raise ValueError("Region Coverage masks must share source resolution.")
        union |= mask
    return union


# ADD 2026-08-31: Region 지표를 계산한다. → MODIFY 2026-09-01: Fast failure 경계를 맞춘다.
def calculate_region_coverage(
    *,
    records: list[DerivedManifestRecord],
    ground_truth_by_sample: dict[str, tuple[GroundTruthInstance, ...]],
    predictions_by_sample: dict[str, tuple[PredictedInstance, ...]],
    classes: dict[int, str],
    size_policy: SizeBucketPolicy,
) -> RegionCoverageMetrics:
    validation_records = require_validation_records(records)
    if not classes:
        raise ValueError("Region Coverage classes must not be empty.")
    expected_ids = {record.sample_id for record in validation_records}
    if set(ground_truth_by_sample) != expected_ids or set(predictions_by_sample) != expected_ids:
        raise ValueError("Region Coverage sample prediction/GT identities are incomplete.")

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

    for record in validation_records:
        ground_truth = ground_truth_by_sample[record.sample_id]
        predictions = predictions_by_sample[record.sample_id]
        matches = match_instances(
            ground_truth,
            predictions,
            iou_threshold=STRICT_INSTANCE_IOU_THRESHOLD,
        )
        strict_total += len(ground_truth)
        strict_matched += len(matches)

        for class_id in classes:
            gt_union = _union_masks(
                [instance.mask for instance in ground_truth if instance.class_id == class_id],
                height=record.image_height,
                width=record.image_width,
            )
            prediction_union = _union_masks(
                [instance.mask for instance in predictions if instance.class_id == class_id],
                height=record.image_height,
                width=record.image_width,
            )
            class_intersection += int(np.count_nonzero(gt_union & prediction_union))
            class_union += int(np.count_nonzero(gt_union | prediction_union))
            class_gt_pixels += int(np.count_nonzero(gt_union))
            class_prediction_pixels += int(np.count_nonzero(prediction_union))

        for gt in ground_truth:
            same_class = tuple(
                prediction for prediction in predictions if prediction.class_id == gt.class_id
            )
            best_single_iou = max(
                (mask_overlap(gt.mask, prediction.mask)[0] for prediction in same_class),
                default=0.0,
            )
            same_class_union = _union_masks(
                [prediction.mask for prediction in same_class],
                height=record.image_height,
                width=record.image_width,
            )
            gt_pixels = int(np.count_nonzero(gt.mask))
            coverage = _safe_ratio(int(np.count_nonzero(gt.mask & same_class_union)), gt_pixels)
            strict_pass = best_single_iou >= STRICT_INSTANCE_IOU_THRESHOLD
            coverage_pass = coverage >= REGION_COVERAGE_THRESHOLD
            coverage_total += 1
            coverage_matched += int(coverage_pass)
            if size_policy.classify(gt.area_ratio) == "small":
                small_total += 1
                small_coverage_matched += int(coverage_pass)
            if not strict_pass and NEAR_MISS_MIN_IOU <= best_single_iou < (
                STRICT_INSTANCE_IOU_THRESHOLD
            ):
                near_miss += 1
            if not strict_pass and coverage_pass:
                covered_strict_fail += 1

    metrics = RegionCoverageMetrics(
        strict_instance_gt_recall=_safe_ratio(strict_matched, strict_total),
        gt_component_coverage_recall_at_50=_safe_ratio(coverage_matched, coverage_total),
        small_gt_coverage_recall_at_50=_safe_ratio(
            small_coverage_matched,
            small_total,
        ),
        class_aware_union_iou=_safe_ratio(class_intersection, class_union),
        class_aware_union_gt_coverage=_safe_ratio(class_intersection, class_gt_pixels),
        class_aware_union_pred_precision=_safe_ratio(
            class_intersection,
            class_prediction_pixels,
        ),
        near_miss_iou_030_to_050=near_miss,
        covered50_but_strict_instance_fail=covered_strict_fail,
        strict_iou_threshold=STRICT_INSTANCE_IOU_THRESHOLD,
        coverage_threshold=REGION_COVERAGE_THRESHOLD,
        val_gt_total=strict_total,
        small_gt_total=small_total,
        test_used=False,
    )
    metrics.to_json_dict()
    return metrics
