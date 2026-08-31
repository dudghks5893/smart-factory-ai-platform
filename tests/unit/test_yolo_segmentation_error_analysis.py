"""Deterministic tests for validation-only YOLO segmentation diagnostics."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.evaluation.yolo_segmentation_error_analysis import (
    GroundTruthInstance,
    PredictedInstance,
    SizeBucketPolicy,
    aggregate_analysis,
    analyze_sample,
    box_iou,
    build_confidence_sweep,
    derive_improvement_hypotheses,
    derive_size_bucket_policy,
    mask_box,
    mask_overlap,
    match_instances,
    rank_worst_samples,
    require_validation_records,
)

CLASSES = {0: "bent", 1: "color", 2: "scratch"}
SIZE_POLICY = SizeBucketPolicy("validation_gt_mask_area_ratio_tertiles", 0.10, 0.25)


# ADD 2026-08-26: Rectangular binary mask fixture를 source-resolution geometry로 생성한다.
def _mask(y1: int, y2: int, x1: int, x2: int) -> NDArray[np.bool_]:
    mask = np.zeros((10, 10), dtype=np.bool_)
    mask[y1:y2, x1:x2] = True
    mask.setflags(write=False)
    return mask


# ADD 2026-08-26: Positive/negative validation sample의 full manifest schema를 생성한다.
def _record(
    sample_id: str,
    *,
    class_id: int | None = 0,
    component_count: int = 1,
    split: str = "val",
) -> DerivedManifestRecord:
    is_negative = class_id is None
    defect_type = "good" if class_id is None else CLASSES[class_id]
    return DerivedManifestRecord(
        dataset_name="synthetic",
        dataset_version="v1",
        derived_task="yolo_segmentation",
        source_manifest_sha256="a" * 64,
        source_split="test",
        source_manifest_split="test",
        source_image_path=f"source/{sample_id}.png",
        source_mask_path="" if is_negative else f"source/{sample_id}_mask.png",
        category="metal_nut",
        sample_id=sample_id,
        defect_type=defect_type,
        target_class="" if is_negative else defect_type,
        target_class_id="" if is_negative else str(class_id),
        derived_split=split,
        is_negative=is_negative,
        image_width=10,
        image_height=10,
        image_path=f"images/{split}/{sample_id}.png",
        label_path=f"labels/{split}/{sample_id}.txt",
        image_sha256="b" * 64,
        mask_sha256="" if is_negative else "c" * 64,
        polygon_count=0 if is_negative else component_count,
        component_count=0 if is_negative else component_count,
        hole_count=0,
        polygon_vertex_count=0 if is_negative else 4 * component_count,
        round_trip_iou="" if is_negative else "1.0",
        pixel_precision="" if is_negative else "1.0",
        pixel_recall="" if is_negative else "1.0",
    )


# ADD 2026-08-26: Mask fixture를 GT instance geometry로 감싼다.
def _gt(class_id: int, mask: NDArray[np.bool_]) -> GroundTruthInstance:
    return GroundTruthInstance(
        class_id=class_id,
        mask=mask,
        box_xyxy=mask_box(mask),
        area_ratio=float(np.count_nonzero(mask) / mask.size),
    )


# ADD 2026-08-26: Mask fixture를 confidence-bearing prediction으로 감싼다.
def _prediction(
    class_id: int, mask: NDArray[np.bool_], confidence: float = 0.9
) -> PredictedInstance:
    return PredictedInstance(
        class_id=class_id,
        confidence=confidence,
        mask=mask,
        box_xyxy=mask_box(mask),
    )


# ADD 2026-08-26: Validation-only gate가 test leakage와 duplicate identity를 거부하는지 검증한다.
def test_validation_only_gate_rejects_test_leakage_and_duplicate_ids() -> None:
    records = [_record("b"), _record("a")]
    assert [record.sample_id for record in require_validation_records(records)] == ["a", "b"]

    with pytest.raises(ValueError, match="non-val"):
        require_validation_records([replace(records[0], derived_split="test")])
    with pytest.raises(ValueError, match="unique"):
        require_validation_records([records[0], records[0]])


# ADD 2026-08-26: Mask/box IoU와 class-aware deterministic greedy matching을 검증한다.
def test_iou_and_matching_are_class_aware_and_deterministic() -> None:
    large = _mask(1, 5, 1, 5)
    overlap = _mask(2, 6, 2, 6)
    mask_iou, gt_coverage, prediction_precision = mask_overlap(large, overlap)
    assert mask_iou == pytest.approx(9 / 23)
    assert gt_coverage == pytest.approx(9 / 16)
    assert prediction_precision == pytest.approx(9 / 16)
    assert box_iou(mask_box(large), mask_box(overlap)) == pytest.approx(9 / 23)

    ground_truth = (_gt(0, large), _gt(0, _mask(6, 9, 6, 9)))
    predictions = (
        _prediction(0, large),
        _prediction(0, _mask(6, 9, 6, 9)),
        _prediction(1, large),
    )
    matches = match_instances(ground_truth, predictions)
    assert [(item.ground_truth_index, item.prediction_index) for item in matches] == [
        (0, 0),
        (1, 1),
    ]


# ADD 2026-08-26: TP/FP/FN, wrong class, empty prediction과 multi-instance miss를 검증한다.
def test_sample_taxonomy_covers_detection_and_class_errors() -> None:
    first = _gt(0, _mask(1, 5, 1, 5))
    second = _gt(0, _mask(6, 9, 6, 9))
    exact = _prediction(0, first.mask)

    partial = analyze_sample(
        record=_record("multi", component_count=2),
        ground_truth=(first, second),
        predictions=(exact,),
        classes=CLASSES,
        size_policy=SIZE_POLICY,
    )
    assert (
        partial.true_positive_count,
        partial.false_positive_count,
        partial.false_negative_count,
    ) == (
        1,
        0,
        1,
    )
    assert partial.main_error == "MISSED_DEFECT"
    assert "MULTI_COMPONENT_MISS" in partial.secondary_tags

    missed = analyze_sample(
        record=_record("empty"),
        ground_truth=(first,),
        predictions=(),
        classes=CLASSES,
        size_policy=SIZE_POLICY,
    )
    assert missed.expected_class_hit is False
    assert missed.main_error == "MISSED_DEFECT"

    wrong = analyze_sample(
        record=_record("wrong"),
        ground_truth=(first,),
        predictions=(_prediction(2, first.mask),),
        classes=CLASSES,
        size_policy=SIZE_POLICY,
    )
    assert "WRONG_CLASS" in wrong.secondary_tags
    assert wrong.confusion_pairs == (("bent", "scratch"),)

    negative = analyze_sample(
        record=_record("negative", class_id=None, component_count=0),
        ground_truth=(),
        predictions=(_prediction(1, _mask(2, 4, 2, 4)),),
        classes=CLASSES,
        size_policy=SIZE_POLICY,
    )
    assert negative.main_error == "FALSE_POSITIVE"
    assert negative.false_positive_count == 1


# ADD 2026-09-01: Analyzer는 validation default를 유지하며 final-test split은 명시적으로만 허용한다.
def test_sample_analysis_requires_explicit_final_test_split() -> None:
    record = _record("final-test", split="test")
    ground_truth = (_gt(0, _mask(1, 5, 1, 5)),)
    prediction = (_prediction(0, ground_truth[0].mask),)

    with pytest.raises(ValueError, match="derived val"):
        analyze_sample(
            record=record,
            ground_truth=ground_truth,
            predictions=prediction,
            classes=CLASSES,
            size_policy=SIZE_POLICY,
        )

    analysis = analyze_sample(
        record=record,
        ground_truth=ground_truth,
        predictions=prediction,
        classes=CLASSES,
        size_policy=SIZE_POLICY,
        expected_split="test",
    )
    assert analysis.sample_id == "final-test"
    assert analysis.true_positive_count == 1


# ADD 2026-08-26: Low IoU와 directional under/over segmentation tag를 검증한다.
def test_localization_under_and_over_segmentation_tags() -> None:
    ground_truth_mask = _mask(1, 6, 1, 6)
    ground_truth = (_gt(0, ground_truth_mask),)
    under = analyze_sample(
        record=_record("under"),
        ground_truth=ground_truth,
        predictions=(_prediction(0, _mask(1, 6, 1, 4)),),
        classes=CLASSES,
        size_policy=SIZE_POLICY,
    )
    assert "LOW_IOU_LOCALIZATION" in under.secondary_tags
    assert "MASK_UNDER_SEGMENTATION" in under.secondary_tags

    over = analyze_sample(
        record=_record("over"),
        ground_truth=(_gt(0, _mask(2, 7, 2, 5)),),
        predictions=(_prediction(0, _mask(2, 7, 1, 6)),),
        classes=CLASSES,
        size_policy=SIZE_POLICY,
    )
    assert "LOW_IOU_LOCALIZATION" in over.secondary_tags
    assert "MASK_OVER_SEGMENTATION" in over.secondary_tags


# ADD 2026-08-26: Validation tertile size policy와 class/negative/component summary를 검증한다.
def test_size_class_component_and_negative_aggregation() -> None:
    masks = (_mask(1, 3, 1, 3), _mask(1, 4, 1, 4), _mask(1, 6, 1, 6))
    ground_truth = [_gt(index, mask) for index, mask in enumerate(masks)]
    policy = derive_size_bucket_policy(ground_truth)
    assert [policy.classify(item.area_ratio) for item in ground_truth] == [
        "small",
        "medium",
        "large",
    ]

    analyses = [
        analyze_sample(
            record=_record(f"class-{class_id}", class_id=class_id),
            ground_truth=(instance,),
            predictions=(_prediction(class_id, instance.mask),),
            classes=CLASSES,
            size_policy=policy,
        )
        for class_id, instance in enumerate(ground_truth)
    ]
    analyses.append(
        analyze_sample(
            record=_record("good", class_id=None, component_count=0),
            ground_truth=(),
            predictions=(_prediction(2, _mask(7, 9, 7, 9), 0.6),),
            classes=CLASSES,
            size_policy=policy,
        )
    )
    summary = aggregate_analysis(analyses, classes=CLASSES)
    assert summary["tp"] == 3
    assert summary["fp"] == 1
    assert summary["per_class"]["scratch"]["fp"] == 1
    assert summary["negative_analysis"]["false_positive_image_count"] == 1
    assert summary["negative_analysis"]["predicted_classes"] == {"scratch": 1}
    assert summary["component_analysis"]["single_component"]["recall"] == 1.0


# ADD 2026-08-26: Confidence sweep, ranking, hypotheses와 machine-readable output을 검증한다.
def test_confidence_sweep_ranking_hypotheses_and_serialization() -> None:
    records = [_record("bent-low"), _record("good", class_id=None, component_count=0)]
    gt_mask = _mask(1, 5, 1, 5)
    ground_truth: dict[str, tuple[GroundTruthInstance, ...]] = {
        "bent-low": (_gt(0, gt_mask),),
        "good": (),
    }
    predictions: dict[str, tuple[PredictedInstance, ...]] = {
        "bent-low": (_prediction(0, gt_mask, 0.20),),
        "good": (_prediction(1, _mask(6, 8, 6, 8), 0.12),),
    }
    sweep = build_confidence_sweep(
        records=records,
        ground_truth_by_sample=ground_truth,
        predictions_by_sample=predictions,
        classes=CLASSES,
        size_policy=SIZE_POLICY,
        confidence_levels=(0.10, 0.25),
    )
    assert sweep[0]["recall"] == 1.0
    assert sweep[0]["false_positive_image_rate"] == 1.0
    assert sweep[1]["recall"] == 0.0
    assert sweep[1]["false_positive_image_rate"] == 0.0

    baseline = [
        analyze_sample(
            record=record,
            ground_truth=ground_truth[record.sample_id],
            predictions=(),
            classes=CLASSES,
            size_policy=SIZE_POLICY,
        )
        for record in records
    ]
    ranked = rank_worst_samples(list(reversed(baseline)))
    assert [item.sample_id for item in ranked] == ["bent-low", "good"]
    assert json.loads(json.dumps(ranked[0].to_dict()))["sample_id"] == "bent-low"

    full_sweep = [
        {**sweep[0], "confidence": 0.10},
        {**sweep[0], "confidence": 0.15},
        {**sweep[0], "confidence": 0.20},
        {**sweep[1], "confidence": 0.25},
        {**sweep[1], "confidence": 0.30},
        {**sweep[1], "confidence": 0.40},
        {**sweep[1], "confidence": 0.50},
    ]
    summary = aggregate_analysis(baseline, classes=CLASSES)
    hypotheses = derive_improvement_hypotheses(summary, full_sweep)
    assert any("confidence" in item["candidate"] for item in hypotheses["supported"])
