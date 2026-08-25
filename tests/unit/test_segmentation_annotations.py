"""Unit tests for framework-independent YOLO segmentation annotations."""

from __future__ import annotations

import numpy as np
import pytest

from ml.datasets.segmentation_annotations import (
    SegmentationPolygon,
    allocate_class_split_counts,
    mask_to_yolo_polygons,
    parse_yolo_segmentation_label,
    rasterize_polygons,
    sample_negative_ids,
    stratified_split_sample_ids,
    summarize_fidelity,
)


# ADD 2026-08-25: Multi-component polygon과 normalized round-trip 보존을 검증한다.
def test_mask_conversion_preserves_components_coordinates_and_round_trip() -> None:
    mask = np.zeros((12, 14), dtype=np.bool_)
    mask[1:5, 2:7] = True
    mask[7:11, 9:13] = True

    # 두 disconnected component를 union하지 않고 독립 polygon으로 변환한다.
    conversion = mask_to_yolo_polygons(mask, class_id=2)
    assert len(conversion.polygons) == 2
    assert conversion.hole_count == 0
    assert conversion.iou == 1.0
    assert conversion.precision == 1.0
    assert conversion.recall == 1.0
    assert all(
        0.0 <= coordinate <= 1.0
        for polygon in conversion.polygons
        for point in polygon.points
        for coordinate in point
    )

    label = "\n".join(polygon.to_yolo_line() for polygon in conversion.polygons)
    parsed = parse_yolo_segmentation_label(label, valid_class_ids={0, 1, 2})
    reconstructed = rasterize_polygons(parsed, image_width=14, image_height=12)
    assert np.array_equal(reconstructed, mask)


# ADD 2026-08-25: Thin region, unsupported hole와 degenerate contour 처리를 검증한다.
def test_thin_region_hole_and_degenerate_contour_are_explicit() -> None:
    thin_mask = np.zeros((12, 12), dtype=np.bool_)
    thin_mask[2:10, 5:7] = True
    thin_conversion = mask_to_yolo_polygons(thin_mask, class_id=0)
    assert thin_conversion.iou == 1.0

    hole_mask = np.zeros((12, 12), dtype=np.bool_)
    hole_mask[1:11, 1:11] = True
    hole_mask[4:8, 4:8] = False
    hole_conversion = mask_to_yolo_polygons(hole_mask, class_id=1)
    assert hole_conversion.hole_count == 1
    assert hole_conversion.iou < 1.0

    single_pixel = np.zeros((5, 5), dtype=np.bool_)
    single_pixel[2, 2] = True
    with pytest.raises(ValueError, match="at least three unique vertices"):
        mask_to_yolo_polygons(single_pixel, class_id=0)


# ADD 2026-08-25: YOLO label schema, class와 coordinate bounds 검증 실패를 확인한다.
def test_polygon_parser_rejects_invalid_schema_and_bounds() -> None:
    with pytest.raises(ValueError, match="Invalid YOLO"):
        parse_yolo_segmentation_label("0 0.1 0.1 0.2 0.2", valid_class_ids={0})
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        parse_yolo_segmentation_label(
            "0 0.0 0.0 1.1 0.0 0.5 0.5",
            valid_class_ids={0},
        )
    with pytest.raises(ValueError, match="Invalid segmentation class"):
        parse_yolo_segmentation_label(
            "3 0.0 0.0 0.5 0.0 0.5 0.5",
            valid_class_ids={0, 1, 2},
        )


# ADD 2026-08-25: Class-stratified split과 negative sampling 재현성을 검증한다.
def test_stratified_split_and_negative_sampling_are_deterministic() -> None:
    samples_by_class = {
        "bent": [f"bent-{index:02d}" for index in range(25)],
        "color": [f"color-{index:02d}" for index in range(22)],
        "scratch": [f"scratch-{index:02d}" for index in range(23)],
    }
    assignments = stratified_split_sample_ids(
        samples_by_class,
        seed=42,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
    )
    repeated = stratified_split_sample_ids(
        dict(reversed(list(samples_by_class.items()))),
        seed=42,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
    )
    assert assignments == repeated
    by_id = {assignment.sample_id: assignment.derived_split for assignment in assignments}
    assert len(by_id) == 70
    assert sum(split == "train" for split in by_id.values()) == 42
    assert sum(split == "val" for split in by_id.values()) == 14
    assert sum(split == "test" for split in by_id.values()) == 14
    assert allocate_class_split_counts(
        23,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
    ) == {"train": 13, "val": 5, "test": 5}

    negatives = sample_negative_ids(
        [f"good-{index:03d}" for index in range(100)],
        split_counts={"train": 42, "val": 14, "test": 14},
        seed=42,
    )
    assert len(negatives) == 70
    assert len({assignment.sample_id for assignment in negatives}) == 70
    assert negatives == sample_negative_ids(
        list(reversed([f"good-{index:03d}" for index in range(100)])),
        split_counts={"train": 42, "val": 14, "test": 14},
        seed=42,
    )


# ADD 2026-08-25: Fidelity summary가 topology와 metric distribution을 집계하는지 검증한다.
def test_fidelity_summary_reports_topology_and_distribution() -> None:
    mask = np.zeros((8, 8), dtype=np.bool_)
    mask[1:7, 2:6] = True
    conversion = mask_to_yolo_polygons(mask, class_id=0)
    summary = summarize_fidelity([conversion, conversion])
    assert summary["sample_count"] == 2
    assert summary["polygon_count"] == 2
    assert summary["hole_count"] == 0
    assert summary["iou"]["min"] == 1.0


# ADD 2026-08-25: Zero-area polygon이 rasterization 전에 거부되는지 검증한다.
def test_rasterization_rejects_degenerate_polygon() -> None:
    polygon = SegmentationPolygon(class_id=0, points=((0.1, 0.1), (0.2, 0.2), (0.3, 0.3)))
    with pytest.raises(ValueError, match="zero area"):
        rasterize_polygons((polygon,), image_width=10, image_height=10)
