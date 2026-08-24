"""Unit contracts for binary-mask known-defect annotation analysis."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ml.datasets.defect_annotations import (
    analyze_manifest_annotation,
    extract_mask_components,
    load_binary_mask,
    resolve_dataset_path,
    summarize_defect_metrics,
)
from ml.datasets.manifest import ManifestRecord


# ADD 2026-08-25: Edge component와 compact internal component를 포함한 binary mask를 생성한다.
def _binary_mask() -> np.ndarray:
    mask = np.zeros((6, 8), dtype=np.bool_)
    mask[0, 0:2] = True
    mask[2:4, 4:7] = True
    return mask


# ADD 2026-08-25: Synthetic annotation test용 image/mask와 official anomaly record를 생성한다.
def _write_annotation_fixture(tmp_path: Path) -> tuple[Path, ManifestRecord]:
    dataset_root = tmp_path / "mvtec_ad"
    image_path = dataset_root / "metal_nut/test/scratch/000.png"
    mask_path = dataset_root / "metal_nut/ground_truth/scratch/000_mask.png"
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 6), color=(120, 100, 80)).save(image_path)
    Image.fromarray(_binary_mask().astype(np.uint8) * 255).save(mask_path)
    record = ManifestRecord(
        sample_id="metal_nut_test_scratch_000",
        category="metal_nut",
        source_split="test",
        split="test",
        defect_type="scratch",
        label=1,
        image_path="metal_nut/test/scratch/000.png",
        mask_path="metal_nut/ground_truth/scratch/000_mask.png",
        width=8,
        height=6,
    )
    return dataset_root, record


# ADD 2026-08-25: 8-connected component bbox, fill, edge-touch와 YOLO normalization을 검증한다.
def test_component_extraction_bbox_and_yolo_coordinates() -> None:
    components = extract_mask_components(_binary_mask())

    assert len(components) == 2
    first, second = components
    assert (first.x_min, first.y_min, first.x_max, first.y_max) == (0, 0, 1, 0)
    assert first.positive_pixel_count == 2
    assert first.bbox_area == 2
    assert first.mask_bbox_fill_ratio == 1.0
    assert first.touches_edge is True
    assert (second.x_min, second.y_min, second.x_max, second.y_max) == (4, 2, 6, 3)
    assert second.positive_pixel_count == 6
    assert second.touches_edge is False
    assert second.to_yolo_xywh(image_width=8, image_height=6) == pytest.approx(
        (0.6875, 0.5, 0.375, 1 / 3)
    )


# ADD 2026-08-25: Mask loader가 size, binary value, nonempty와 missing file을 검증한다.
def test_binary_mask_loading_validation(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.png"
    invalid_path = tmp_path / "invalid.png"
    empty_path = tmp_path / "empty.png"
    Image.fromarray(_binary_mask().astype(np.uint8) * 255).save(valid_path)
    Image.fromarray(np.full((6, 8), 127, dtype=np.uint8)).save(invalid_path)
    Image.fromarray(np.zeros((6, 8), dtype=np.uint8)).save(empty_path)

    loaded = load_binary_mask(valid_path, expected_size=(8, 6))

    assert loaded.dtype == np.bool_
    assert int(loaded.sum()) == 8
    with pytest.raises(ValueError, match="Mask size mismatch"):
        load_binary_mask(valid_path, expected_size=(7, 6))
    with pytest.raises(ValueError, match="only 0/255"):
        load_binary_mask(invalid_path, expected_size=(8, 6))
    with pytest.raises(ValueError, match="no positive pixels"):
        load_binary_mask(empty_path, expected_size=(8, 6))
    with pytest.raises(FileNotFoundError, match="Ground-truth mask not found"):
        load_binary_mask(tmp_path / "missing.png", expected_size=(8, 6))


# ADD 2026-08-25: Manifest adapter가 sample union geometry와 source lineage를 보존하는지 검증한다.
def test_manifest_annotation_metrics_and_summary(tmp_path: Path) -> None:
    dataset_root, record = _write_annotation_fixture(tmp_path)

    metric = analyze_manifest_annotation(
        dataset_name="mvtec_ad",
        dataset_root=dataset_root,
        record=record,
    )
    summary = summarize_defect_metrics([metric, replace(metric, sample_id="sample-2")])

    assert metric.image_path == record.image_path
    assert metric.mask_path == record.mask_path
    assert metric.positive_pixel_count == 8
    assert metric.positive_area_ratio == pytest.approx(1 / 6)
    assert metric.component_count == 2
    assert metric.largest_component_area == 6
    assert metric.largest_component_to_mask_ratio == 0.75
    assert (metric.bbox_x_min, metric.bbox_y_min, metric.bbox_x_max, metric.bbox_y_max) == (
        0,
        0,
        6,
        3,
    )
    assert metric.bbox_area == 28
    assert metric.mask_bbox_fill_ratio == pytest.approx(2 / 7)
    assert metric.touches_edge is True
    assert summary["sample_count"] == 2
    assert summary["component_count_total"] == 4
    assert summary["multi_component_sample_ratio"] == 1.0
    assert summary["edge_touch_sample_count"] == 2


# ADD 2026-08-25: Malformed mask type/shape과 dataset root escape를 fail-fast하는지 검증한다.
def test_annotation_rejects_malformed_mask_and_unsafe_paths(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="boolean dtype"):
        extract_mask_components(_binary_mask().astype(np.uint8))
    with pytest.raises(ValueError, match="two dimensions"):
        extract_mask_components(np.zeros((1, 2, 3), dtype=np.bool_))
    with pytest.raises(ValueError, match="at least one positive"):
        extract_mask_components(np.zeros((2, 2), dtype=np.bool_))
    with pytest.raises(ValueError, match="relative path"):
        resolve_dataset_path(tmp_path, "/absolute/mask.png")
    with pytest.raises(ValueError, match="escapes configured root"):
        resolve_dataset_path(tmp_path, "../outside.png")


# ADD 2026-08-25: C1 adapter가 derived split record를 official anomaly로 사용하지 않는지 검증한다.
def test_manifest_annotation_requires_official_test_anomaly(tmp_path: Path) -> None:
    dataset_root, record = _write_annotation_fixture(tmp_path)

    with pytest.raises(ValueError, match="official test anomaly"):
        analyze_manifest_annotation(
            dataset_name="mvtec_ad",
            dataset_root=dataset_root,
            record=replace(record, split="train"),
        )
