"""Unit tests for YOLO metric parsing and test-image diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ml.evaluation.yolo_segmentation import (
    PredictionObservation,
    aggregate_prediction_diagnostics,
    serialize_ultralytics_metrics,
)
from pipelines.evaluate_yolo_segmentation import parse_prediction_result

CLASSES = {0: "bent", 1: "color", 2: "scratch"}


class FakeMetric:
    """Documented Ultralytics Metric method surface for serialization tests."""

    # ADD 2026-08-25: Stable overall box/mask metric tuple을 반환한다.
    def mean_results(self) -> list[float]:
        return [0.8, 0.7, 0.75, 0.55]

    # ADD 2026-08-25: Class ID별 stable precision/recall/AP tuple을 반환한다.
    def class_result(self, class_id: int) -> tuple[float, ...]:
        base = 0.1 * class_id
        return 0.8 - base, 0.7 - base, 0.75 - base, 0.55 - base


@dataclass
class FakeMetrics:
    """Box and segmentation metric views returned by a fake val call."""

    box: object
    seg: object


class FakeTensor:
    """Minimal tensor materialization surface used by result parsing."""

    # ADD 2026-08-25: Fake tensor가 materialize할 fixed value를 보관한다.
    def __init__(self, values: list[float]) -> None:
        self.values = values

    # ADD 2026-08-25: Fake tensor detach call을 chainable하게 반환한다.
    def detach(self) -> FakeTensor:
        return self

    # ADD 2026-08-25: Fake tensor CPU call을 chainable하게 반환한다.
    def cpu(self) -> FakeTensor:
        return self

    # ADD 2026-08-25: Fake tensor value를 Python list로 materialize한다.
    def tolist(self) -> list[float]:
        return self.values


@dataclass
class FakeBoxes:
    """Class and confidence tensor pair from one fake result."""

    cls: object
    conf: object


@dataclass
class FakeMasks:
    """Mask collection from one fake result."""

    data: list[object]


@dataclass
class FakeResult:
    """Boxes and masks required by prediction parsing."""

    boxes: object
    masks: object


# ADD 2026-08-25: Overall/per-class box와 mask metric serialization을 검증한다.
def test_metric_serialization_includes_box_mask_and_per_class() -> None:
    overall, per_class = serialize_ultralytics_metrics(
        FakeMetrics(FakeMetric(), FakeMetric()),
        classes=CLASSES,
    )
    assert overall["mask"]["map50_95"] == 0.55
    assert overall["box"]["precision"] == 0.8
    assert per_class["scratch"]["mask"]["map50_95"] == pytest.approx(0.35)


# ADD 2026-08-25: Missing segmentation metric과 malformed class result를 명시적으로 거부한다.
def test_metric_serialization_rejects_malformed_result() -> None:
    with pytest.raises(ValueError, match="both box and seg"):
        serialize_ultralytics_metrics(FakeMetrics(FakeMetric(), None), classes=CLASSES)

    class BrokenMetric(FakeMetric):
        # ADD 2026-08-25: Malformed per-class result length을 반환한다.
        def class_result(self, class_id: int) -> tuple[float, ...]:
            return (0.5,)

    with pytest.raises(ValueError, match="four finite"):
        serialize_ultralytics_metrics(
            FakeMetrics(BrokenMetric(), FakeMetric()),
            classes=CLASSES,
        )


# ADD 2026-08-25: Good FP와 positive target-class success aggregation을 검증한다.
def test_prediction_diagnostics_separate_negative_and_positive_behavior() -> None:
    observations = [
        PredictionObservation("good-1", "good", True, (1,), (0.6,), 1),
        PredictionObservation("good-2", "good", True, (), (), 0),
        PredictionObservation("bent-1", "bent", False, (0,), (0.9,), 1),
        PredictionObservation("color-1", "color", False, (2,), (0.7,), 1),
        PredictionObservation("scratch-1", "scratch", False, (2, 2), (0.8, 0.5), 2),
    ]
    negative, positive = aggregate_prediction_diagnostics(observations, classes=CLASSES)
    assert negative["false_positive_image_count"] == 1
    assert negative["false_positive_image_rate"] == 0.5
    assert positive["bent"]["target_class_segmentation_success_rate"] == 1.0
    assert positive["color"]["target_class_segmentation_success_rate"] == 0.0
    assert positive["scratch"]["target_class_instance_count"] == 2


# ADD 2026-08-25: Fake Ultralytics result의 class/confidence/mask alignment을 검증한다.
def test_parse_prediction_result_and_malformed_mask() -> None:
    from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord

    record = DerivedManifestRecord(
        dataset_name="dataset",
        dataset_version="v1",
        derived_task="yolo_segmentation",
        source_manifest_sha256="a" * 64,
        source_split="test",
        source_manifest_split="test",
        source_image_path="source.png",
        source_mask_path="mask.png",
        category="metal_nut",
        sample_id="sample",
        defect_type="bent",
        target_class="bent",
        target_class_id="0",
        derived_split="test",
        is_negative=False,
        image_width=16,
        image_height=16,
        image_path="images/test/sample.png",
        label_path="labels/test/sample.txt",
        image_sha256="b" * 64,
        mask_sha256="c" * 64,
        polygon_count=1,
        component_count=1,
        hole_count=0,
        polygon_vertex_count=4,
        round_trip_iou="1.0",
        pixel_precision="1.0",
        pixel_recall="1.0",
    )
    result = FakeResult(
        boxes=FakeBoxes(FakeTensor([0.0]), FakeTensor([0.95])),
        masks=FakeMasks([object()]),
    )
    observation = parse_prediction_result(result, record=record)
    observation.validate(valid_class_ids=set(CLASSES))
    assert observation.predicted_class_ids == (0,)

    malformed = FakeResult(
        boxes=FakeBoxes(FakeTensor([0.0]), FakeTensor([0.95])),
        masks=None,
    )
    with pytest.raises(ValueError, match="segmentation mask"):
        parse_prediction_result(malformed, record=record).validate(valid_class_ids=set(CLASSES))
