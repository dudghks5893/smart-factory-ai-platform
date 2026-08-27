"""Tests for validation-safe YOLO workbench controls and EDA."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.datasets.yolo_segmentation_manifest import (
    DerivedManifestRecord,
    write_derived_manifest,
)
from ml.experiments.yolo_workbench import (
    WorkbenchSample,
    build_eda_summary,
    build_research_config,
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
