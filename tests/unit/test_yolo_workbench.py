"""Tests for validation-safe YOLO workbench controls and EDA."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.datasets.yolo_segmentation_manifest import (
    DerivedManifestRecord,
    write_derived_manifest,
)
from ml.experiments.yolo_sampling import (
    PlannedTrainView,
    SamplingEligibility,
    TrainViewEvidence,
)
from ml.experiments.yolo_segmentation import load_yolo_experiment_config
from ml.experiments.yolo_workbench import (
    WorkbenchSample,
    build_eda_summary,
    build_research_config,
    build_sampling_workbench_summary,
    load_workbench_records,
    select_representative_samples,
    select_small_validation_sample,
    validate_workbench_controls,
)
from ml.training.yolo_segmentation import load_yolo_segmentation_config


# ADD 2026-08-27: Small Manifest row fixture를 명시적 split으로 만든다.
def _record(sample_id: str, split: str) -> DerivedManifestRecord:
    return DerivedManifestRecord(
        dataset_name="fixture",
        dataset_version="v1",
        derived_task="yolo_segmentation",
        source_manifest_sha256="a" * 64,
        source_split="test",
        source_manifest_split="test",
        source_image_path=f"source/{sample_id}.png",
        source_mask_path="",
        category="metal_nut",
        sample_id=sample_id,
        defect_type="good",
        target_class="",
        target_class_id="",
        derived_split=split,
        is_negative=True,
        image_width=32,
        image_height=32,
        image_path=f"images/{split}/{sample_id}.png",
        label_path=f"labels/{split}/{sample_id}.txt",
        image_sha256="b" * 64,
        mask_sha256="",
        polygon_count=0,
        component_count=0,
        hole_count=0,
        polygon_vertex_count=0,
        round_trip_iou="",
        pixel_precision="",
        pixel_recall="",
    )


# ADD 2026-08-27: Reader가 sealed test row를 workbench object로 materialize하지 않는지 검증한다.
def test_load_workbench_records_filters_sealed_test(tmp_path: Path) -> None:
    write_derived_manifest(
        [_record("train-good", "train"), _record("val-good", "val"), _record("secret", "test")],
        tmp_path / "manifest.csv",
    )
    records = load_workbench_records(tmp_path)
    assert [record.sample_id for record in records] == ["train-good", "val-good"]
    assert all(record.derived_split != "test" for record in records)


# ADD 2026-08-27: EDA class/component/size/good counts와 deterministic selection을 검증한다.
def test_eda_summary_and_representative_selection_are_deterministic() -> None:
    samples = [
        WorkbenchSample(
            "bent-small", "train", "bent", False, 1, (0.01,), ("small",), 32, 32, "a", "a"
        ),
        WorkbenchSample(
            "scratch-multi",
            "val",
            "scratch",
            False,
            2,
            (0.02, 0.03),
            ("medium", "large"),
            32,
            32,
            "b",
            "b",
        ),
        WorkbenchSample("good-train", "train", "good", True, 0, (), (), 32, 32, "c", "c"),
        WorkbenchSample("good-val", "val", "good", True, 0, (), (), 32, 32, "d", "d"),
    ]
    summary = build_eda_summary(
        samples,
        manifest_sha256="f" * 64,
        dataset_name="fixture",
        dataset_version="v1",
    )
    assert summary["image_distribution"]["train"] == {
        "total": 2,
        "positive": 1,
        "good_negative": 1,
    }
    assert summary["class_component_count"] == {"bent": 1, "scratch": 2}
    assert summary["size_component_count"] == {"small": 1, "medium": 1, "large": 1}
    assert summary["component_type"] == {
        "single_component_samples": 1,
        "multi_component_samples": 1,
    }
    first = select_representative_samples(samples, seed=42)
    second = select_representative_samples(list(reversed(samples)), seed=42)
    assert [item.sample_id for item in first] == [item.sample_id for item in second]


# ADD 2026-08-27: Full sample set의 val-small deterministic selection을 검증한다.
def test_small_validation_selection_uses_full_allowed_sample_set() -> None:
    samples = [
        WorkbenchSample(
            "train-small", "train", "bent", False, 1, (0.01,), ("small",), 32, 32, "a", "a"
        ),
        WorkbenchSample(
            "val-small-b", "val", "bent", False, 1, (0.01,), ("small",), 32, 32, "b", "b"
        ),
        WorkbenchSample(
            "val-small-a", "val", "color", False, 1, (0.01,), ("small",), 32, 32, "c", "c"
        ),
        WorkbenchSample("val-good", "val", "good", True, 0, (), (), 32, 32, "d", "d"),
    ]
    gallery_subset = [samples[0], samples[3]]
    assert not any(
        sample.split == "val" and not sample.is_negative and "small" in sample.size_buckets
        for sample in gallery_subset
    )

    first = select_small_validation_sample(samples, seed=42)
    second = select_small_validation_sample(list(reversed(samples)), seed=42)
    assert first == second
    assert first.split == "val"
    assert not first.is_negative
    assert "small" in first.size_buckets


# ADD 2026-08-27: Missing val-small과 sealed-test input을 명확한 error로 거부한다.
def test_small_validation_selection_fails_clearly() -> None:
    good_val = WorkbenchSample("good", "val", "good", True, 0, (), (), 32, 32, "a", "a")
    with pytest.raises(ValueError, match="No positive validation sample"):
        select_small_validation_sample([good_val], seed=42)
    sealed_test = WorkbenchSample(
        "sealed", "test", "bent", False, 1, (0.01,), ("small",), 32, 32, "b", "b"
    )
    with pytest.raises(ValueError, match="outside train/validation"):
        select_small_validation_sample([sealed_test], seed=42)


# ADD 2026-08-27: Research outputs/overrides 격리와 official override 거부를 검증한다.
def test_research_overrides_do_not_mutate_official_config(tmp_path: Path) -> None:
    baseline = load_yolo_segmentation_config(Path("configs/model/yolo_segmentation_baseline.yaml"))
    research = build_research_config(
        baseline,
        overrides={"imgsz": 768, "batch": 8, "epochs": 3},
        output_root=tmp_path / "research",
    )
    assert research.training.imgsz == 768
    assert research.output.artifact_root == tmp_path / "research" / "artifacts"
    assert baseline.training.imgsz == 640
    with pytest.raises(ValueError, match="rejects"):
        validate_workbench_controls("official", overrides={"imgsz": 768})


# ADD 2026-08-28: C4-2B Workbench summary가 approved exposure와 eligibility groups를 표시한다.
def test_c4_2b_sampling_workbench_summary_is_compact_and_train_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_yolo_experiment_config(
        Path(
            "configs/experiments/yolo_segmentation/"
            "c4_2b_yolo11n_seg_component_aware_sampling_x2_seed42.yaml"
        )
    )
    baseline = load_yolo_segmentation_config(experiment.baseline_config_path)
    small_ids = tuple(f"train-{index:03d}" for index in range(14))
    multi_ids = tuple(f"train-{index:03d}" for index in range(5, 19))
    eligible_ids = tuple(f"train-{index:03d}" for index in range(19))
    evidence = TrainViewEvidence(
        schema_version=1,
        experiment_id=experiment.experiment_id,
        sampling_rule_version="component_aware_bottom_third_union_multi_x2_v1",
        canonical_manifest_sha256="a" * 64,
        unique_train_count=84,
        unique_positive_count=42,
        unique_good_negative_count=42,
        small_aware_count=14,
        multi_component_count=14,
        eligible_overlap_count=9,
        eligible_union_count=19,
        expanded_entry_count=103,
        expanded_positive_count=61,
        expanded_good_negative_count=42,
        expanded_good_negative_ratio=0.4077669902912621,
        small_fraction_rule="bottom_third",
        eligible_multiplicity=2,
        observed_train_small_cutoff=0.011273469387755102,
        eligible_sample_ids=eligible_ids,
        sample_multiplicity={f"train-{index:03d}": 2 if index < 19 else 1 for index in range(84)},
        train_list_sha256="b" * 64,
        train_list_path_base="canonical_dataset_root",
        ordering_policy=("canonical_sample_id_order_then_eligible_second_copy_in_sample_id_order"),
        validation_used_for_sampling=False,
        test_split_used=False,
    )
    plan = PlannedTrainView(
        entries=(),
        profiles=(),
        eligibility=SamplingEligibility(
            small_aware_sample_ids=small_ids,
            multi_component_sample_ids=multi_ids,
            eligible_sample_ids=eligible_ids,
            observed_train_small_cutoff=0.011273469387755102,
        ),
        evidence=evidence,
    )
    monkeypatch.setattr(
        "ml.experiments.yolo_workbench.plan_component_aware_train_view",
        lambda **kwargs: plan,
    )
    records = [_record(f"train-{index:03d}", "train") for index in range(84)]

    summary = build_sampling_workbench_summary(
        experiment=experiment,
        baseline=baseline,
        dataset_root=Path("dataset"),
        records=records,
    )

    assert summary is not None
    assert summary["sampling_validation_source"] == "TRAIN_ONLY"
    assert summary["test_split"] == "SEALED_NOT_USED"
    assert summary["canonical_train_entries"] == 84
    assert summary["expanded_train_entries"] == 103
    assert summary["positive_exposure"] == [42, 61]
    assert summary["good_negative_exposure"] == [42, 42]
    assert len(summary["eligible_samples"]) == 19
    assert sum(row["overlap"] for row in summary["eligible_samples"]) == 9
