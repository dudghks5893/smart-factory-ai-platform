"""Focused C4-2C crop, prediction, Region and confirmation contracts."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.evaluation.yolo_confirmation_prediction import predict_c4_2c_instances
from ml.evaluation.yolo_region_coverage import RegionCoverageMetrics, calculate_region_coverage
from ml.evaluation.yolo_segmentation_error_analysis import (
    GroundTruthInstance,
    PredictedInstance,
    SizeBucketPolicy,
    mask_box,
    match_instances,
)
from ml.experiments.yolo_crop_sampling import (
    CROP_ORDERING_POLICY,
    CropProvenance,
    CropTrainViewEntry,
    CropTrainViewEvidence,
    create_small_center_crop,
    crop_square_bounds,
)
from ml.experiments.yolo_segmentation import (
    confirm_c4_2c_candidate,
    load_yolo_experiment_config,
)
from ml.training.yolo_segmentation import (
    YoloTrainerOverrides,
    build_ultralytics_training_overrides,
    load_yolo_segmentation_config,
)
from pipelines.train_yolo_segmentation import run_ultralytics_training

C4_2C_CONFIG = Path(
    "configs/experiments/yolo_segmentation/"
    "c4_2c_yolo11n_seg_crop350_nomosaic_maskratio2_seed42.yaml"
)
BASELINE_CONFIG = Path("configs/model/yolo_segmentation_baseline.yaml")


def _record(sample_id: str, *, width: int = 500, height: int = 500) -> DerivedManifestRecord:
    return DerivedManifestRecord(
        dataset_name="fixture",
        dataset_version="v1",
        derived_task="yolo_segmentation",
        source_manifest_sha256="a" * 64,
        source_split="test",
        source_manifest_split="test",
        source_image_path=f"source/{sample_id}.png",
        source_mask_path=f"source/{sample_id}_mask.png",
        category="metal_nut",
        sample_id=sample_id,
        defect_type="bent",
        target_class="bent",
        target_class_id="0",
        derived_split="train",
        is_negative=False,
        image_width=width,
        image_height=height,
        image_path=f"images/train/{sample_id}.png",
        label_path=f"labels/train/{sample_id}.txt",
        image_sha256="b" * 64,
        mask_sha256="c" * 64,
        polygon_count=2,
        component_count=2,
        hole_count=0,
        polygon_vertex_count=8,
        round_trip_iou="1.0",
        pixel_precision="1.0",
        pixel_recall="1.0",
    )


def _write_crop_source(root: Path, record: DerivedManifestRecord, label: str) -> None:
    image_path = root / record.image_path
    label_path = root / record.label_path
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    image = np.full((record.image_height, record.image_width, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    label_path.write_text(label, encoding="utf-8")


# ADD 2026-08-31: C4-2C config, exact trainer args와 absolute Primary decision을 검증한다.
def test_c4_2c_typed_recipe_and_primary_confirmation() -> None:
    experiment = load_yolo_experiment_config(C4_2C_CONFIG)
    baseline = load_yolo_segmentation_config(BASELINE_CONFIG)
    assert experiment.crop_sampling_policy is not None
    assert experiment.expected_crop_train_view is not None
    assert experiment.trainer_overrides is not None
    assert experiment.confirmation_protocol is not None
    assert experiment.training_config(baseline).training.imgsz == 640
    assert asdict(experiment.crop_sampling_policy) == {
        "sampling_mode": "component_aware_crop",
        "sampling_multiplicity": 2,
        "crop_size": 350,
    }
    assert asdict(experiment.expected_crop_train_view) == {
        "canonical_entries": 84,
        "canonical_positives": 42,
        "canonical_negatives": 42,
        "component_duplicate_entries": 19,
        "small_centered_crop_entries": 14,
        "total_entries": 117,
        "positive_exposure": 75,
        "negative_exposure": 42,
        "small_aware_count": 14,
        "multi_component_count": 14,
        "eligible_overlap_count": 9,
        "eligible_union_count": 19,
        "observed_train_small_cutoff": 0.011273469387755102,
    }
    overrides = YoloTrainerOverrides(**asdict(experiment.trainer_overrides))
    applied = build_ultralytics_training_overrides(baseline, overrides)
    assert {key: applied[key] for key in ("mosaic", "mask_ratio", "overlap_mask", "scale")} == {
        "mosaic": 0.0,
        "mask_ratio": 2,
        "overlap_mask": True,
        "scale": 0.5,
    }
    quality: dict[str, Any] = {
        "ultralytics": {"mask": {"map50_95": 0.40}},
        "diagnostic": {"recall": 0.6, "f1": 0.6},
        "failure_modes": {
            "small_recall": 0.2501,
            "multi_component_recall": 0.50,
            "good_negative_fp_image_rate": 0.0,
        },
    }
    recommendation = confirm_c4_2c_candidate(
        quality_after=quality,
        protocol=experiment.confirmation_protocol,
    )
    assert recommendation.decision == "CONFIRMED_CANDIDATE"
    assert all(recommendation.checks.values())
    quality["failure_modes"]["small_recall"] = 0.25
    assert (
        confirm_c4_2c_candidate(
            quality_after=quality,
            protocol=experiment.confirmation_protocol,
        ).decision
        == "CONFIRMATION_FAILED"
    )


class _FakeYolo:
    last_train_kwargs: dict[str, object] = {}

    def __init__(self, weights: str, *, task: str) -> None:
        self.callbacks: dict[str, list[Callable[[Any], None]]] = {}
        self.trainer = SimpleNamespace(best=Path(), device="cuda:0")

    def add_callback(self, name: str, callback: Callable[[Any], None]) -> None:
        self.callbacks.setdefault(name, []).append(callback)

    def train(self, **kwargs: object) -> None:
        _FakeYolo.last_train_kwargs = kwargs
        runtime = Path(str(kwargs["project"])) / str(kwargs["name"])
        best = runtime / "weights/best.pt"
        best.parent.mkdir(parents=True)
        best.write_bytes(b"checkpoint")
        trainer = SimpleNamespace(
            epoch=0,
            best_fitness=1.0,
            fitness=1.0,
            best=best,
            device="cuda:0",
            metrics={},
            tloss=np.array([1.0, 1.0, 1.0, 1.0]),
            lr={"lr/pg0": 0.01},
            label_loss_items=lambda values, prefix: {
                "train/box_loss": 1.0,
                "train/seg_loss": 1.0,
                "train/cls_loss": 1.0,
                "train/dfl_loss": 1.0,
            },
        )
        for callback in self.callbacks["on_train_epoch_start"]:
            callback(trainer)
        for callback in self.callbacks["on_fit_epoch_end"]:
            callback(trainer)
        self.trainer = trainer


# ADD 2026-08-31: Explicit C4-2C augmentation values가 model.train boundary에 도달하는지 검증한다.
def test_c4_2c_overrides_reach_model_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_yolo_experiment_config(C4_2C_CONFIG)
    baseline = load_yolo_segmentation_config(BASELINE_CONFIG)
    assert experiment.trainer_overrides is not None
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("train: images/train\nval: images/val\n", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(YOLO=_FakeYolo, __version__="8.4.128"),
    )
    monkeypatch.setattr(
        "pipelines.train_yolo_segmentation.resolve_device",
        lambda _: SimpleNamespace(type="cuda"),
    )
    overrides = YoloTrainerOverrides(**asdict(experiment.trainer_overrides))
    result = run_ultralytics_training(
        baseline,
        dataset_yaml,
        tmp_path / "runtime",
        experiment.experiment_id,
        "cuda",
        overrides,
    )
    assert result.best_checkpoint.is_file()
    assert {
        key: _FakeYolo.last_train_kwargs[key]
        for key in ("mosaic", "mask_ratio", "overlap_mask", "scale")
    } == {"mosaic": 0.0, "mask_ratio": 2, "overlap_mask": True, "scale": 0.5}


# ADD 2026-08-31: Vertex mean, edge clamp, tie line order와 all-intersection labels를 검증한다.
def test_crop350_preserves_fast_geometry_and_provenance(tmp_path: Path) -> None:
    record = _record("tie")
    label = "0 0.02 0.02 0.08 0.02 0.08 0.08 0.02 0.08\n0 0.04 0.04 0.10 0.04 0.10 0.10 0.04 0.10\n"
    _write_crop_source(tmp_path, record, label)
    image_path = tmp_path / "view/images/train_crops/tie__small_center_crop350.png"
    label_path = tmp_path / "view/labels/train_crops/tie__small_center_crop350.txt"
    provenance = create_small_center_crop(
        record=record,
        dataset_root=tmp_path,
        crop_size=350,
        crop_image_path=image_path,
        crop_label_path=label_path,
        valid_class_ids={0, 1, 2},
    )
    assert provenance.target_polygon_index == 0
    assert provenance.crop_box_xyxy == (0, 0, 350, 350)
    assert provenance.retained_instance_count == 2
    assert provenance.generated_image_path.endswith("tie__small_center_crop350.png")
    assert provenance.generated_label_path.endswith("tie__small_center_crop350.txt")
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        assert values[0] == "0"
        assert all(0.0 <= float(value) <= 1.0 for value in values[1:])
        assert all(len(value.partition(".")[2]) == 6 for value in values[1:])


# ADD 2026-08-31: Exact exclusive square bounds와 undersized source failure를 검증한다.
def test_crop_square_bounds_clamps_and_rejects_small_source() -> None:
    assert crop_square_bounds(
        center_x=490.0,
        center_y=490.0,
        crop_size=350,
        width=500,
        height=500,
    ) == (150, 150, 500, 500)
    with pytest.raises(ValueError, match="fit inside"):
        crop_square_bounds(
            center_x=10,
            center_y=10,
            crop_size=350,
            width=349,
            height=500,
        )


class _PredictionModel:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def predict(self, **kwargs: object) -> list[object]:
        self.kwargs = kwargs
        boxes = SimpleNamespace(
            cls=np.array([0.0]),
            conf=np.array([0.25]),
        )
        masks = SimpleNamespace(data=np.array([[[0.5, 0.6], [0.0, 1.0]]], dtype=np.float32))
        return [SimpleNamespace(boxes=boxes, masks=masks)]


# ADD 2026-08-31: Fast predict args와 strict greater-than/nearest mask normalization을 검증한다.
def test_c4_2c_prediction_protocol_is_explicit(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"identity-only")
    model = _PredictionModel()
    predictions = predict_c4_2c_instances(
        model=model,
        source_image_path=source,
        image_width=4,
        image_height=4,
        imgsz=640,
        device="cuda",
        valid_class_ids={0, 1, 2},
    )
    assert model.kwargs == {
        "source": str(source),
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "retina_masks": False,
        "imgsz": 640,
        "device": "cuda",
        "save": False,
        "stream": False,
        "verbose": False,
    }
    assert len(predictions) == 1
    assert predictions[0].confidence == 0.25
    assert predictions[0].mask.shape == (4, 4)
    assert not predictions[0].mask[0, 0]
    assert predictions[0].mask[0, 2]


def _mask(value: list[list[int]]) -> np.ndarray:
    return np.asarray(value, dtype=np.bool_)


def _region_metrics(
    *,
    ground_truth_masks: tuple[np.ndarray, ...],
    prediction_masks: tuple[np.ndarray, ...],
) -> RegionCoverageMetrics:
    height, width = ground_truth_masks[0].shape
    record = _record("region-boundary", width=width, height=height)
    record = DerivedManifestRecord(**{**asdict(record), "derived_split": "val"})
    ground_truth = tuple(
        GroundTruthInstance(0, mask, mask_box(mask), 0.1) for mask in ground_truth_masks
    )
    predictions = tuple(
        PredictedInstance(0, 0.9, mask, mask_box(mask)) for mask in prediction_masks
    )
    return calculate_region_coverage(
        records=[record],
        ground_truth_by_sample={record.sample_id: ground_truth},
        predictions_by_sample={record.sample_id: predictions},
        classes={0: "bent", 1: "color", 2: "scratch"},
        size_policy=SizeBucketPolicy("fixture", small_max=0.3, medium_max=0.6),
    )


# ADD 2026-08-31: Equal-IoU에서 Fast와 Official tie ordering 동등성을 검증한다.
def test_strict_matching_equal_iou_is_fast_stable_equivalent() -> None:
    mask = _mask([[1, 1], [1, 1]])
    ground_truth = tuple(GroundTruthInstance(0, mask, mask_box(mask), 1.0) for _ in range(2))
    predictions = tuple(PredictedInstance(0, 0.9, mask, mask_box(mask)) for _ in range(2))
    official = match_instances(ground_truth, predictions, iou_threshold=0.5)
    fast_candidates = [
        (1.0, gt_index, prediction_index) for gt_index in range(2) for prediction_index in range(2)
    ]
    matched_gt: set[int] = set()
    matched_prediction: set[int] = set()
    fast: list[tuple[int, int]] = []
    for _, gt_index, prediction_index in sorted(
        fast_candidates,
        key=lambda item: item[0],
        reverse=True,
    ):
        if gt_index in matched_gt or prediction_index in matched_prediction:
            continue
        fast.append((gt_index, prediction_index))
        matched_gt.add(gt_index)
        matched_prediction.add(prediction_index)
    assert [(item.ground_truth_index, item.prediction_index) for item in official] == fast


# ADD 2026-08-31: Region formulas, class isolation과 secondary-only test seal을 검증한다.
def test_region_coverage_uses_class_aware_validation_pool() -> None:
    record = _record("region", width=4, height=4)
    record = DerivedManifestRecord(**{**asdict(record), "derived_split": "val"})
    gt_mask = _mask([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pred_mask = _mask([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    wrong_class = gt_mask.copy()
    gt = GroundTruthInstance(0, gt_mask, mask_box(gt_mask), 0.25)
    predictions = (
        PredictedInstance(0, 0.9, pred_mask, mask_box(pred_mask)),
        PredictedInstance(1, 0.9, wrong_class, mask_box(wrong_class)),
    )
    metrics = calculate_region_coverage(
        records=[record],
        ground_truth_by_sample={record.sample_id: (gt,)},
        predictions_by_sample={record.sample_id: predictions},
        classes={0: "bent", 1: "color", 2: "scratch"},
        size_policy=SizeBucketPolicy("fixture", small_max=0.3, medium_max=0.6),
    )
    assert metrics.strict_instance_gt_recall == 1.0
    assert metrics.gt_component_coverage_recall_at_50 == 1.0
    assert metrics.small_gt_coverage_recall_at_50 == 1.0
    assert metrics.class_aware_union_iou == pytest.approx(3 / 8)
    assert metrics.class_aware_union_gt_coverage == pytest.approx(3 / 4)
    assert metrics.class_aware_union_pred_precision == pytest.approx(3 / 7)
    assert metrics.test_used is False


# ADD 2026-09-01: Greedy competition과 Fast best-IoU strict semantics를 분리해 검증한다.
def test_region_failure_counters_use_fast_best_iou_during_greedy_competition() -> None:
    top = _mask([[1, 1], [0, 0]])
    bottom = _mask([[0, 0], [1, 1]])
    combined_prediction = _mask([[1, 1], [1, 1]])

    metrics = _region_metrics(
        ground_truth_masks=(top, bottom),
        prediction_masks=(combined_prediction,),
    )

    assert metrics.strict_instance_gt_recall == 0.5
    assert metrics.near_miss_iou_030_to_050 == 0
    assert metrics.covered50_but_strict_instance_fail == 0


# ADD 2026-09-01: Fast failure counter의 0.30/0.50 IoU 및 union coverage 경계를 검증한다.
@pytest.mark.parametrize(
    ("ground_truth", "predictions", "expected_near_miss", "expected_covered"),
    [
        (
            np.ones((1, 5), dtype=np.bool_),
            (_mask([[1, 1, 0, 0, 0]]),),
            1,
            0,
        ),
        (
            np.ones((10, 10), dtype=np.bool_),
            (
                np.arange(100).reshape(10, 10) < 49,
                np.arange(100).reshape(10, 10) == 49,
            ),
            1,
            1,
        ),
        (
            np.ones((1, 2), dtype=np.bool_),
            (_mask([[1, 0]]),),
            0,
            0,
        ),
    ],
)
def test_region_failure_counter_iou_boundaries(
    ground_truth: np.ndarray,
    predictions: tuple[np.ndarray, ...],
    expected_near_miss: int,
    expected_covered: int,
) -> None:
    metrics = _region_metrics(
        ground_truth_masks=(ground_truth,),
        prediction_masks=predictions,
    )

    assert metrics.near_miss_iou_030_to_050 == expected_near_miss
    assert metrics.covered50_but_strict_instance_fail == expected_covered


def _crop_train_view_evidence() -> CropTrainViewEvidence:
    small_ids = tuple(f"small-{index:02d}" for index in range(14))
    multi_ids = tuple(sorted((*small_ids[5:], *(f"multi-{index:02d}" for index in range(5)))))
    eligible_ids = tuple(sorted(set(small_ids).union(multi_ids)))
    canonical_entries = tuple(
        CropTrainViewEntry(
            kind="canonical",
            source_sample_id=f"positive-{index:02d}",
            is_negative=False,
            portable_image_path=f"canonical/images/train/positive-{index:02d}.png",
            source_relative_image_path=f"images/train/positive-{index:02d}.png",
            generated_relative_path=None,
        )
        for index in range(42)
    ) + tuple(
        CropTrainViewEntry(
            kind="canonical",
            source_sample_id=f"negative-{index:02d}",
            is_negative=True,
            portable_image_path=f"canonical/images/train/negative-{index:02d}.png",
            source_relative_image_path=f"images/train/negative-{index:02d}.png",
            generated_relative_path=None,
        )
        for index in range(42)
    )
    duplicate_entries = tuple(
        CropTrainViewEntry(
            kind="component_aware_duplicate",
            source_sample_id=sample_id,
            is_negative=False,
            portable_image_path=f"canonical/images/train/{sample_id}.png",
            source_relative_image_path=f"images/train/{sample_id}.png",
            generated_relative_path=None,
        )
        for sample_id in eligible_ids
    )
    crop_entries = tuple(
        CropTrainViewEntry(
            kind="small_center_crop",
            source_sample_id=sample_id,
            is_negative=False,
            portable_image_path=f"generated/images/train_crops/{sample_id}.png",
            source_relative_image_path=f"images/train/{sample_id}.png",
            generated_relative_path=f"images/train_crops/{sample_id}.png",
        )
        for sample_id in small_ids
    )
    crops = tuple(
        CropProvenance(
            source_sample_id=sample_id,
            source_relative_image_path=f"images/train/{sample_id}.png",
            generated_image_path=f"images/train_crops/{sample_id}.png",
            generated_label_path=f"labels/train_crops/{sample_id}.txt",
            crop_box_xyxy=(0, 0, 350, 350),
            source_width=500,
            source_height=500,
            crop_size=350,
            retained_instance_count=1,
            target_polygon_index=0,
            crop_image_sha256="c" * 64,
            crop_label_sha256="d" * 64,
        )
        for sample_id in small_ids
    )
    entries = (*canonical_entries, *duplicate_entries, *crop_entries)
    return CropTrainViewEvidence(
        schema_version=1,
        experiment_id="c4_2c_yolo11n_seg_crop350_nomosaic_maskratio2_seed42",
        sampling_mode="component_aware_crop",
        sampling_rule_version="component_aware_bottom_third_union_multi_x2_v1",
        canonical_manifest_sha256="a" * 64,
        crop_size=350,
        canonical_entry_count=84,
        canonical_positive_count=42,
        canonical_negative_count=42,
        component_duplicate_count=19,
        crop_entry_count=14,
        total_entry_count=117,
        positive_exposure=75,
        negative_exposure=42,
        small_aware_count=14,
        multi_component_count=14,
        eligible_overlap_count=9,
        eligible_union_count=19,
        small_aware_sample_ids=small_ids,
        multi_component_sample_ids=multi_ids,
        component_aware_sample_ids=eligible_ids,
        observed_train_small_cutoff=0.011273469387755102,
        ordering_policy=CROP_ORDERING_POLICY,
        entries=entries,
        crops=crops,
        portable_train_list_sha256="e" * 64,
        train_view_fingerprint_sha256="f" * 64,
        validation_used_for_sampling=False,
        test_used=False,
    )


# ADD 2026-09-01: Expected snapshot을 actual ID/entry-derived C4-2C evidence와 대조한다.
def test_expected_crop_train_view_validates_actual_evidence_and_rejects_drift() -> None:
    experiment = load_yolo_experiment_config(C4_2C_CONFIG)
    expected = experiment.expected_crop_train_view
    assert expected is not None
    evidence = _crop_train_view_evidence()

    expected.validate_evidence(evidence)
    assert (
        evidence.small_aware_count,
        evidence.multi_component_count,
        evidence.eligible_overlap_count,
        evidence.eligible_union_count,
        evidence.canonical_entry_count,
        evidence.component_duplicate_count,
        evidence.crop_entry_count,
        evidence.total_entry_count,
        evidence.positive_exposure,
        evidence.negative_exposure,
    ) == (14, 14, 9, 19, 84, 19, 14, 117, 75, 42)

    for drifted in (
        replace(evidence, multi_component_count=13),
        replace(evidence, eligible_overlap_count=8),
        replace(evidence, total_entry_count=116),
        replace(evidence, positive_exposure=74),
    ):
        with pytest.raises(ValueError):
            expected.validate_evidence(drifted)
