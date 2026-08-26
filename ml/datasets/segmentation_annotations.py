"""Framework-independent polygon conversion for supervised-derived segmentation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

DERIVED_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SegmentationPolygon:
    """One normalized YOLO polygon derived from one external mask contour."""

    class_id: int
    points: tuple[tuple[float, float], ...]

    # ADD 2026-08-25: Polygon을 deterministic precision의 YOLO label line으로 직렬화한다.
    def to_yolo_line(self) -> str:
        """Serialize the class and normalized vertices without scientific notation."""
        coordinates = " ".join(f"{value:.8f}" for point in self.points for value in point)
        return f"{self.class_id} {coordinates}"


@dataclass(frozen=True)
class PolygonConversion:
    """Polygon representation and loss measured against its source binary mask."""

    polygons: tuple[SegmentationPolygon, ...]
    hole_count: int
    vertex_count: int
    source_positive_pixels: int
    reconstructed_positive_pixels: int
    intersection_pixels: int
    iou: float
    precision: float
    recall: float


@dataclass(frozen=True)
class SplitAssignment:
    """One unique source sample assigned to a derived supervised split."""

    sample_id: str
    derived_split: str


@dataclass(frozen=True)
class RasterizedSegmentationInstance:
    """One validated YOLO polygon represented as a source-resolution mask."""

    class_id: int
    mask: NDArray[np.bool_]
    area_ratio: float


# ADD 2026-08-25: Seed와 sample identity를 stable SHA-256 ranking으로 결합한다.
def deterministic_rank(sample_id: str, *, seed: int, namespace: str) -> str:
    """Return a cross-process stable rank without relying on runtime hash state."""
    if not sample_id or not namespace:
        raise ValueError("Sample ID and ranking namespace must not be blank.")
    value = f"{namespace}\0{seed}\0{sample_id}".encode()
    return hashlib.sha256(value).hexdigest()


# ADD 2026-08-25: Small-data evaluation 수를 반올림하고 remainder를 train에 보존한다.
def allocate_class_split_counts(
    sample_count: int,
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> dict[str, int]:
    """Allocate one class while keeping validation and test independently meaningful."""
    if sample_count < 3:
        raise ValueError("Each positive class requires at least three samples.")
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0.0 for ratio in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("Derived split ratios must be positive and sum to one.")

    validation_count = max(1, round(sample_count * validation_ratio))
    test_count = max(1, round(sample_count * test_ratio))
    train_count = sample_count - validation_count - test_count
    if train_count < 1:
        raise ValueError("Split ratios leave no training samples for a positive class.")
    return {"train": train_count, "val": validation_count, "test": test_count}


# ADD 2026-08-25: Class별 source sample을 deterministic image-level split에 배정한다.
def stratified_split_sample_ids(
    sample_ids_by_class: dict[str, list[str]],
    *,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[SplitAssignment, ...]:
    """Split each class independently without separating its masks or polygons."""
    if not sample_ids_by_class:
        raise ValueError("Positive class samples must not be empty.")
    assignments: list[SplitAssignment] = []
    observed_ids: set[str] = set()
    for class_name in sorted(sample_ids_by_class):
        sample_ids = sample_ids_by_class[class_name]
        if len(sample_ids) != len(set(sample_ids)) or observed_ids.intersection(sample_ids):
            raise ValueError("Source sample IDs must be globally unique.")
        observed_ids.update(sample_ids)
        counts = allocate_class_split_counts(
            len(sample_ids),
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )
        ordered = sorted(
            sample_ids,
            key=lambda sample_id: (
                deterministic_rank(sample_id, seed=seed, namespace=f"positive:{class_name}"),
                sample_id,
            ),
        )
        offset = 0
        for split_name in DERIVED_SPLITS:
            split_ids = ordered[offset : offset + counts[split_name]]
            assignments.extend(SplitAssignment(sample_id, split_name) for sample_id in split_ids)
            offset += counts[split_name]
    return tuple(sorted(assignments, key=lambda item: item.sample_id))


# ADD 2026-08-25: Good pool에서 필요한 수만 stable하게 선택하고 target split 수에 맞춘다.
def sample_negative_ids(
    sample_ids: list[str],
    *,
    split_counts: dict[str, int],
    seed: int,
) -> tuple[SplitAssignment, ...]:
    """Select unique background samples at an explicit split-level ratio."""
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Negative source sample IDs must be unique.")
    if set(split_counts) != set(DERIVED_SPLITS) or any(
        count < 0 for count in split_counts.values()
    ):
        raise ValueError("Negative split counts must cover train, val, and test.")
    required_count = sum(split_counts.values())
    if required_count > len(sample_ids):
        raise ValueError("Not enough unique good images for the negative sampling policy.")
    ordered = sorted(
        sample_ids,
        key=lambda sample_id: (
            deterministic_rank(sample_id, seed=seed, namespace="negative"),
            sample_id,
        ),
    )[:required_count]
    assignments: list[SplitAssignment] = []
    offset = 0
    for split_name in DERIVED_SPLITS:
        split_ids = ordered[offset : offset + split_counts[split_name]]
        assignments.extend(SplitAssignment(sample_id, split_name) for sample_id in split_ids)
        offset += split_counts[split_name]
    return tuple(sorted(assignments, key=lambda item: item.sample_id))


# ADD 2026-08-25: Normalized polygon 좌표와 최소 면적 계약을 검증한다.
def validate_polygon(polygon: SegmentationPolygon, *, valid_class_ids: set[int]) -> None:
    """Reject unsupported classes, degenerate vertices, and out-of-bounds coordinates."""
    if polygon.class_id not in valid_class_ids:
        raise ValueError(f"Invalid segmentation class ID: {polygon.class_id}")
    if len(polygon.points) < 3 or len(set(polygon.points)) < 3:
        raise ValueError("Segmentation polygon requires at least three unique vertices.")
    if any(not (0.0 <= coordinate <= 1.0) for point in polygon.points for coordinate in point):
        raise ValueError("Segmentation polygon coordinate is outside [0, 1].")
    points = np.asarray(polygon.points, dtype=np.float32)
    if abs(float(cv2.contourArea(points))) <= 0.0:
        raise ValueError("Segmentation polygon has zero area.")


# ADD 2026-08-25: YOLO polygon을 source resolution의 binary mask로 rasterize한다.
def rasterize_polygons(
    polygons: tuple[SegmentationPolygon, ...],
    *,
    image_width: int,
    image_height: int,
) -> NDArray[np.bool_]:
    """Reconstruct an external-contour mask using the exporter coordinate contract."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Raster dimensions must be positive.")
    canvas = np.zeros((image_height, image_width), dtype=np.uint8)
    for polygon in polygons:
        validate_polygon(polygon, valid_class_ids={polygon.class_id})
        pixel_points = np.asarray(
            [
                (
                    min(image_width - 1, round(x * image_width)),
                    min(image_height - 1, round(y * image_height)),
                )
                for x, y in polygon.points
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [pixel_points], color=1)
    return canvas.astype(np.bool_)


# ADD 2026-08-27: YOLO label의 polygon/component를 shared source-resolution instances로 복원한다.
def rasterize_segmentation_label_instances(
    label_text: str,
    *,
    image_width: int,
    image_height: int,
    valid_class_ids: set[int],
) -> tuple[RasterizedSegmentationInstance, ...]:
    polygons = parse_yolo_segmentation_label(label_text, valid_class_ids=valid_class_ids)
    instances: list[RasterizedSegmentationInstance] = []
    for polygon in polygons:
        mask = np.asarray(
            rasterize_polygons(
                (polygon,),
                image_width=image_width,
                image_height=image_height,
            ),
            dtype=np.bool_,
        )
        mask.setflags(write=False)
        instances.append(
            RasterizedSegmentationInstance(
                class_id=polygon.class_id,
                mask=mask,
                area_ratio=float(np.count_nonzero(mask) / mask.size),
            )
        )
    return tuple(instances)


# ADD 2026-08-25: Binary mask를 lossless chain-compressed external YOLO polygon으로 변환한다.
def mask_to_yolo_polygons(
    mask: NDArray[np.bool_],
    *,
    class_id: int,
) -> PolygonConversion:
    """Extract disconnected contours, expose holes, and measure raster round-trip loss."""
    if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("Polygon conversion requires a non-empty two-dimensional boolean mask.")
    if class_id < 0:
        raise ValueError("Segmentation class ID must be non-negative.")

    image_height, image_width = mask.shape
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None:
        raise ValueError("Mask contour extraction returned no hierarchy.")

    polygons: list[SegmentationPolygon] = []
    hole_count = 0
    for contour_index, contour in enumerate(contours):
        parent_index = int(hierarchy[0][contour_index][3])
        if parent_index >= 0:
            hole_count += 1
            continue
        pixel_points = contour.reshape(-1, 2)
        polygon = SegmentationPolygon(
            class_id=class_id,
            points=tuple(
                (float(x) / image_width, float(y) / image_height) for x, y in pixel_points.tolist()
            ),
        )
        validate_polygon(polygon, valid_class_ids={class_id})
        polygons.append(polygon)

    polygons_tuple = tuple(polygons)
    reconstructed = rasterize_polygons(
        polygons_tuple,
        image_width=image_width,
        image_height=image_height,
    )
    intersection = int(np.logical_and(mask, reconstructed).sum())
    union = int(np.logical_or(mask, reconstructed).sum())
    source_positive = int(mask.sum())
    reconstructed_positive = int(reconstructed.sum())
    return PolygonConversion(
        polygons=polygons_tuple,
        hole_count=hole_count,
        vertex_count=sum(len(polygon.points) for polygon in polygons_tuple),
        source_positive_pixels=source_positive,
        reconstructed_positive_pixels=reconstructed_positive,
        intersection_pixels=intersection,
        iou=intersection / union,
        precision=intersection / reconstructed_positive,
        recall=intersection / source_positive,
    )


# ADD 2026-08-25: Positive YOLO label text를 parse하고 schema bounds를 재검증한다.
def parse_yolo_segmentation_label(
    text: str,
    *,
    valid_class_ids: set[int],
) -> tuple[SegmentationPolygon, ...]:
    """Parse non-empty label lines independently of a training framework."""
    polygons: list[SegmentationPolygon] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        tokens = line.split()
        if len(tokens) < 7 or len(tokens) % 2 == 0:
            raise ValueError(f"Invalid YOLO segmentation label line: {line_number}")
        try:
            class_id = int(tokens[0])
            coordinates = [float(value) for value in tokens[1:]]
        except ValueError as exc:
            raise ValueError(f"Non-numeric YOLO label value on line: {line_number}") from exc
        polygon = SegmentationPolygon(
            class_id=class_id,
            points=tuple(zip(coordinates[::2], coordinates[1::2], strict=True)),
        )
        validate_polygon(polygon, valid_class_ids=valid_class_ids)
        polygons.append(polygon)
    return tuple(polygons)


# ADD 2026-08-25: Fidelity observation을 validation gate와 문서용 distribution으로 집계한다.
def summarize_fidelity(conversions: list[PolygonConversion]) -> dict[str, Any]:
    """Summarize positive sample conversion metrics and topology counts."""
    if not conversions:
        raise ValueError("Fidelity summary requires positive conversions.")

    def distribution(values: list[float]) -> dict[str, float | int]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "min": float(array.min()),
            "p05": float(np.quantile(array, 0.05)),
            "median": float(np.median(array)),
            "mean": float(array.mean()),
            "max": float(array.max()),
        }

    return {
        "sample_count": len(conversions),
        "polygon_count": sum(len(conversion.polygons) for conversion in conversions),
        "vertex_count": sum(conversion.vertex_count for conversion in conversions),
        "hole_count": sum(conversion.hole_count for conversion in conversions),
        "iou": distribution([conversion.iou for conversion in conversions]),
        "precision": distribution([conversion.precision for conversion in conversions]),
        "recall": distribution([conversion.recall for conversion in conversions]),
    }
