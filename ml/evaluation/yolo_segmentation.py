"""Metric and prediction diagnostics for YOLO segmentation evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean, median
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PredictionObservation:
    """Framework-neutral prediction summary for one derived test image."""

    sample_id: str
    defect_type: str
    is_negative: bool
    predicted_class_ids: tuple[int, ...]
    confidences: tuple[float, ...]
    segmentation_instance_count: int

    # ADD 2026-08-25: Prediction class/confidence/mask alignment과 numeric bounds를 검증한다.
    def validate(self, *, valid_class_ids: set[int]) -> None:
        if not self.sample_id or not self.defect_type:
            raise ValueError("Prediction observation identity must not be blank.")
        if len(self.predicted_class_ids) != len(self.confidences):
            raise ValueError("Prediction class and confidence counts do not match.")
        if self.segmentation_instance_count != len(self.predicted_class_ids):
            raise ValueError("Every predicted YOLO instance must include a segmentation mask.")
        if any(class_id not in valid_class_ids for class_id in self.predicted_class_ids):
            raise ValueError("Prediction contains an unknown class ID.")
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.confidences):
            raise ValueError("Prediction confidence must be finite and in [0, 1].")


# ADD 2026-08-25: Ultralytics box/seg metric component를 stable overall/per-class JSON으로 변환한다.
def serialize_metric_component(
    component: object,
    *,
    classes: dict[int, str],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Parse only documented Metric mean_results and class_result values."""
    mean_results = getattr(component, "mean_results", None)
    class_result = getattr(component, "class_result", None)
    if not callable(mean_results) or not callable(class_result):
        raise ValueError("Malformed Ultralytics metric component.")
    try:
        overall_values = [float(value) for value in mean_results()]
    except (TypeError, ValueError) as exc:
        raise ValueError("Ultralytics mean metric values are malformed.") from exc
    if len(overall_values) != 4 or not np.isfinite(overall_values).all():
        raise ValueError("Ultralytics mean metrics must contain four finite values.")
    overall = dict(
        zip(
            ("precision", "recall", "map50", "map50_95"),
            overall_values,
            strict=True,
        )
    )
    per_class: dict[str, dict[str, float]] = {}
    for class_id, class_name in sorted(classes.items()):
        try:
            values = [float(value) for value in class_result(class_id)]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Ultralytics class metrics are malformed: {class_name}") from exc
        if len(values) != 4 or not np.isfinite(values).all():
            raise ValueError(f"Class metrics must contain four finite values: {class_name}")
        per_class[class_name] = dict(
            zip(
                ("precision", "recall", "map50", "map50_95"),
                values,
                strict=True,
            )
        )
    return overall, per_class


# ADD 2026-08-25: Ultralytics segmentation result에서 box/mask metric view를 분리한다.
def serialize_ultralytics_metrics(
    metrics: object,
    *,
    classes: dict[int, str],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    """Return overall and per-class box/mask metrics from a fixed test evaluation."""
    box = getattr(metrics, "box", None)
    segmentation = getattr(metrics, "seg", None)
    if box is None or segmentation is None:
        raise ValueError("Ultralytics segmentation metrics require both box and seg components.")
    box_overall, box_per_class = serialize_metric_component(box, classes=classes)
    mask_overall, mask_per_class = serialize_metric_component(segmentation, classes=classes)
    return (
        {"box": box_overall, "mask": mask_overall},
        {
            class_name: {
                "box": box_per_class[class_name],
                "mask": mask_per_class[class_name],
            }
            for class_name in classes.values()
        },
    )


# ADD 2026-08-25: Confidence list를 empty-safe distribution으로 집계한다.
def summarize_confidences(values: list[float]) -> dict[str, float | int | None]:
    """Summarize prediction confidence without inventing values for empty predictions."""
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or (array < 0.0).any() or (array > 1.0).any():
        raise ValueError("Confidence summary values must be finite and in [0, 1].")
    return {
        "count": len(values),
        "min": float(array.min()),
        "median": float(median(values)),
        "mean": float(fmean(values)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


# ADD 2026-08-25: Good negative false-positive와 positive class behavior를 별도 집계한다.
def aggregate_prediction_diagnostics(
    observations: list[PredictionObservation],
    *,
    classes: dict[int, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute transparent image/instance diagnostics without defining a new quality metric."""
    if not observations:
        raise ValueError("Prediction diagnostics require test observations.")
    valid_class_ids = set(classes)
    for observation in observations:
        observation.validate(valid_class_ids=valid_class_ids)

    negatives = [observation for observation in observations if observation.is_negative]
    if not negatives:
        raise ValueError("Good-negative diagnostics require at least one test negative.")
    negative_fp_images = sum(bool(observation.predicted_class_ids) for observation in negatives)
    negative_confidences = [value for observation in negatives for value in observation.confidences]
    negative_analysis = {
        "sample_count": len(negatives),
        "false_positive_image_count": negative_fp_images,
        "false_positive_image_rate": negative_fp_images / len(negatives),
        "predicted_instance_count": sum(
            len(observation.predicted_class_ids) for observation in negatives
        ),
        "confidence": summarize_confidences(negative_confidences),
    }

    class_ids_by_name = {name: class_id for class_id, name in classes.items()}
    positives_by_class: dict[str, list[PredictionObservation]] = defaultdict(list)
    for observation in observations:
        if not observation.is_negative:
            positives_by_class[observation.defect_type].append(observation)
    positive_analysis: dict[str, Any] = {}
    for class_name in classes.values():
        class_observations = positives_by_class[class_name]
        if not class_observations:
            raise ValueError(f"Positive diagnostic class is missing: {class_name}")
        class_id = class_ids_by_name[class_name]
        target_success_count = sum(
            class_id in observation.predicted_class_ids for observation in class_observations
        )
        confidences = [
            confidence
            for observation in class_observations
            for predicted_class, confidence in zip(
                observation.predicted_class_ids,
                observation.confidences,
                strict=True,
            )
            if predicted_class == class_id
        ]
        positive_analysis[class_name] = {
            "sample_count": len(class_observations),
            "any_prediction_image_count": sum(
                bool(observation.predicted_class_ids) for observation in class_observations
            ),
            "target_class_segmentation_success_count": target_success_count,
            "target_class_segmentation_success_rate": target_success_count
            / len(class_observations),
            "predicted_instance_count": sum(
                len(observation.predicted_class_ids) for observation in class_observations
            ),
            "target_class_instance_count": len(confidences),
            "target_class_confidence": summarize_confidences(confidences),
        }
    return negative_analysis, positive_analysis
