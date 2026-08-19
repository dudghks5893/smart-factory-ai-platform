"""Unit tests for fixed-threshold PatchCore metrics."""

import pytest
import torch

from ml.evaluation.metrics import (
    apply_strict_threshold,
    calculate_auroc,
    calculate_binary_metrics,
)


# ADD 2026-08-19: Threshold와 같은 score를 normal로 유지하는 strict operator를 검증한다.
def test_strict_threshold_does_not_classify_equal_score_as_anomaly() -> None:
    predicted = apply_strict_threshold(torch.tensor([0.4, 0.5, 0.6]), 0.5)

    assert predicted.tolist() == [False, False, True]


# ADD 2026-08-19: Confusion count와 precision recall F1 계산을 검증한다.
def test_binary_metrics_calculate_expected_confusion_counts() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    scores = torch.tensor([0.1, 0.8, 0.9, 0.2])

    metrics = calculate_binary_metrics(labels, scores, threshold=0.5)

    assert (metrics.tp, metrics.tn, metrics.fp, metrics.fn) == (1, 1, 1, 1)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.normal_support == 2
    assert metrics.anomaly_support == 2


# ADD 2026-08-19: Undefined precision recall F1의 explicit zero-division 정책을 검증한다.
def test_binary_metrics_use_zero_for_zero_division() -> None:
    metrics = calculate_binary_metrics(
        torch.tensor([0, 0]),
        torch.tensor([0.1, 0.2]),
        threshold=0.5,
    )

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


# ADD 2026-08-19: Known synthetic ranking의 standard AUROC 값을 검증한다.
def test_auroc_matches_known_synthetic_example() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    scores = torch.tensor([0.1, 0.4, 0.35, 0.8])

    assert calculate_auroc(labels, scores) == pytest.approx(0.75)


# ADD 2026-08-19: Single-class AUROC와 non-finite metric input을 명확히 거부한다.
def test_metrics_reject_single_class_and_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="both normal and anomaly"):
        calculate_auroc(torch.tensor([0, 0]), torch.tensor([0.1, 0.2]))
    with pytest.raises(ValueError, match="finite"):
        calculate_binary_metrics(
            torch.tensor([0, 1]),
            torch.tensor([0.1, float("nan")]),
            threshold=0.5,
        )
