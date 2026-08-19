"""Binary image and pixel metrics for fixed PatchCore thresholds."""

from dataclasses import asdict, dataclass
from typing import Any

import torch
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]
from torch import Tensor

from ml.datasets.constants import GOOD_DIR_NAME
from ml.evaluation.predictions import RawPredictionRecord

EVALUATION_SCHEMA_VERSION = 1
COMPARISON_OPERATOR = ">"


@dataclass(frozen=True)
class BinaryMetrics:
    """Confusion counts and zero-division-safe binary metrics."""

    precision: float
    recall: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int
    normal_support: int
    anomaly_support: int

    # ADD 2026-08-19: Binary metric을 JSON-compatible mapping으로 변환한다.
    def to_json_dict(self) -> dict[str, Any]:
        """Convert this metric result to JSON-compatible values."""
        return asdict(self)


# ADD 2026-08-19: 고정 threshold에 strict greater-than operator를 적용한다.
def apply_strict_threshold(scores: Tensor, threshold: float) -> Tensor:
    """Classify scores with the required score > threshold contract."""
    if not torch.isfinite(scores).all():
        raise ValueError("Metric scores must contain only finite values.")
    if not torch.isfinite(torch.tensor(threshold, dtype=torch.float64)):
        raise ValueError("Metric threshold must be finite.")
    return scores > threshold


# ADD 2026-08-19: Binary label과 fixed threshold prediction에서 confusion metric을 계산한다.
def calculate_binary_metrics(labels: Tensor, scores: Tensor, threshold: float) -> BinaryMetrics:
    """Calculate binary metrics with an explicit zero-division value of 0.0."""
    _validate_metric_vectors(labels, scores)
    predicted = apply_strict_threshold(scores, threshold)
    positive = labels == 1
    negative = labels == 0

    tp = int(torch.logical_and(predicted, positive).sum().item())
    tn = int(torch.logical_and(~predicted, negative).sum().item())
    fp = int(torch.logical_and(predicted, negative).sum().item())
    fn = int(torch.logical_and(~predicted, positive).sum().item())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        normal_support=int(negative.sum().item()),
        anomaly_support=int(positive.sum().item()),
    )


# ADD 2026-08-19: Standard scikit-learn 구현으로 binary AUROC를 계산한다.
def calculate_auroc(labels: Tensor, scores: Tensor) -> float:
    """Calculate AUROC and reject inputs that contain only one class."""
    _validate_metric_vectors(labels, scores)
    if torch.unique(labels).numel() != 2:
        raise ValueError("AUROC requires both normal and anomaly classes.")

    value = float(roc_auc_score(labels.numpy(), scores.numpy()))
    if not torch.isfinite(torch.tensor(value, dtype=torch.float64)):
        raise ValueError("AUROC calculation produced a non-finite value.")
    return value


# ADD 2026-08-19: 동일 image threshold로 defect별 detection diagnostics를 집계한다.
def calculate_per_defect_diagnostics(
    records: tuple[RawPredictionRecord, ...],
    scores: Tensor,
    threshold: float,
) -> dict[str, dict[str, int | float]]:
    """Calculate per-defect diagnostics without selecting new thresholds."""
    if len(records) != scores.numel():
        raise ValueError("Per-defect records and scores must have the same length.")
    predicted = apply_strict_threshold(scores, threshold)
    grouped_indices: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        grouped_indices.setdefault(record.defect_type, []).append(index)

    diagnostics: dict[str, dict[str, int | float]] = {}
    for defect_type in sorted(grouped_indices):
        indices = torch.tensor(grouped_indices[defect_type], dtype=torch.int64)
        detected_count = int(predicted[indices].sum().item())
        sample_count = len(grouped_indices[defect_type])
        if defect_type == GOOD_DIR_NAME:
            diagnostics[defect_type] = {
                "sample_count": sample_count,
                "false_positive_count": detected_count,
                "false_positive_rate": detected_count / sample_count,
            }
        else:
            diagnostics[defect_type] = {
                "sample_count": sample_count,
                "detected_count": detected_count,
                "recall": detected_count / sample_count,
            }
    return diagnostics


# ADD 2026-08-19: Metric vector의 shape, binary label, finite score 계약을 검증한다.
def _validate_metric_vectors(labels: Tensor, scores: Tensor) -> None:
    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("Metric labels and scores must be one-dimensional.")
    if labels.shape != scores.shape or labels.numel() == 0:
        raise ValueError("Metric labels and scores must be non-empty and have the same shape.")
    if not torch.all(torch.logical_or(labels == 0, labels == 1)):
        raise ValueError("Metric labels must contain only binary values 0 and 1.")
    if not torch.isfinite(scores).all():
        raise ValueError("Metric scores must contain only finite values.")
