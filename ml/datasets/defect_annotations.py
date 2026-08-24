"""Binary-mask annotation analysis for known-defect dataset feasibility."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ml.datasets.manifest import ManifestRecord


@dataclass(frozen=True)
class ComponentAnnotation:
    """One 8-connected mask component and its image-relative geometry."""

    component_index: int
    positive_pixel_count: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int
    height: int
    bbox_area: int
    bbox_area_ratio: float
    mask_bbox_fill_ratio: float
    aspect_ratio: float
    centroid_x: float
    centroid_y: float
    centroid_x_ratio: float
    centroid_y_ratio: float
    touches_edge: bool

    # ADD 2026-08-25: Component box를 normalized YOLO center-x/center-y/width/height로 변환한다.
    def to_yolo_xywh(self, *, image_width: int, image_height: int) -> tuple[float, ...]:
        """Return normalized YOLO box coordinates without assigning a class."""
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive for YOLO normalization.")
        center_x = (self.x_min + self.width / 2.0) / image_width
        center_y = (self.y_min + self.height / 2.0) / image_height
        normalized = (
            center_x,
            center_y,
            self.width / image_width,
            self.height / image_height,
        )
        if any(value <= 0.0 or value > 1.0 for value in normalized):
            raise ValueError("Normalized YOLO box coordinates are outside (0, 1].")
        return normalized


@dataclass(frozen=True)
class SampleAnnotationMetrics:
    """One anomaly mask summarized without loading other dataset samples."""

    dataset_name: str
    category: str
    sample_id: str
    image_path: str
    mask_path: str
    defect_type: str
    image_width: int
    image_height: int
    positive_pixel_count: int
    positive_area_ratio: float
    component_count: int
    largest_component_area: int
    largest_component_area_ratio: float
    largest_component_to_mask_ratio: float
    bbox_x_min: int
    bbox_y_min: int
    bbox_x_max: int
    bbox_y_max: int
    bbox_width: int
    bbox_height: int
    bbox_area: int
    bbox_area_ratio: float
    mask_bbox_fill_ratio: float
    bbox_aspect_ratio: float
    centroid_x_ratio: float
    centroid_y_ratio: float
    touches_edge: bool
    components: tuple[ComponentAnnotation, ...]


# ADD 2026-08-25: Dataset root 밖으로 벗어나지 않는 manifest-relative path를 결정한다.
def resolve_dataset_path(dataset_root: Path, relative_path: str) -> Path:
    """Resolve one required relative dataset path inside the configured root."""
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError("Dataset path must be a non-empty relative path.")
    resolved_root = dataset_root.resolve()
    resolved_path = (resolved_root / relative_path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Dataset path escapes configured root: {relative_path}") from exc
    return resolved_path


# ADD 2026-08-25: Lossless binary PNG mask를 dimension/value/positive-pixel 계약으로 로드한다.
def load_binary_mask(path: Path, *, expected_size: tuple[int, int]) -> NDArray[np.bool_]:
    """Load a strict 0/255 mask as a two-dimensional boolean array."""
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth mask not found: {path}")
    if expected_size[0] <= 0 or expected_size[1] <= 0:
        raise ValueError("Expected mask dimensions must be positive.")

    with Image.open(path) as image:
        if image.size != expected_size:
            raise ValueError(
                f"Mask size mismatch: expected={expected_size}, actual={image.size}, path={path}"
            )
        mask_values = np.asarray(image.convert("L"), dtype=np.uint8)

    unique_values = set(np.unique(mask_values).tolist())
    if not unique_values <= {0, 255}:
        raise ValueError(f"Mask must contain only 0/255 values: {path}")
    binary_mask = cast(NDArray[np.bool_], mask_values == 255)
    if not binary_mask.any():
        raise ValueError(f"Anomaly mask contains no positive pixels: {path}")
    return binary_mask


# ADD 2026-08-25: 8-connected binary component를 deterministic scan order와 geometry로 추출한다.
def extract_mask_components(mask: NDArray[np.bool_]) -> tuple[ComponentAnnotation, ...]:
    """Extract deterministic 8-connected components from a non-empty binary mask."""
    if mask.ndim != 2:
        raise ValueError("Binary mask must have exactly two dimensions.")
    if mask.dtype != np.bool_:
        raise TypeError("Binary mask must use NumPy boolean dtype.")
    if mask.shape[0] <= 0 or mask.shape[1] <= 0 or not mask.any():
        raise ValueError("Binary mask must contain at least one positive pixel.")

    image_height, image_width = mask.shape
    image_area = image_width * image_height
    visited = np.zeros(mask.shape, dtype=np.bool_)
    components: list[ComponentAnnotation] = []

    # Positive coordinate scan으로 background pixel를 Python level에서 순회하지 않는다.
    for start_y, start_x in np.argwhere(mask):
        y = int(start_y)
        x = int(start_x)
        if visited[y, x]:
            continue

        queue: deque[tuple[int, int]] = deque([(y, x)])
        visited[y, x] = True
        area = 0
        x_min = x_max = x
        y_min = y_max = y
        x_sum = 0
        y_sum = 0

        while queue:
            current_y, current_x = queue.pop()
            area += 1
            x_min = min(x_min, current_x)
            x_max = max(x_max, current_x)
            y_min = min(y_min, current_y)
            y_max = max(y_max, current_y)
            x_sum += current_x
            y_sum += current_y

            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    if delta_x == 0 and delta_y == 0:
                        continue
                    neighbor_y = current_y + delta_y
                    neighbor_x = current_x + delta_x
                    if not (0 <= neighbor_y < image_height and 0 <= neighbor_x < image_width):
                        continue
                    if mask[neighbor_y, neighbor_x] and not visited[neighbor_y, neighbor_x]:
                        visited[neighbor_y, neighbor_x] = True
                        queue.append((neighbor_y, neighbor_x))

        width = x_max - x_min + 1
        height = y_max - y_min + 1
        bbox_area = width * height
        centroid_x = x_sum / area
        centroid_y = y_sum / area
        components.append(
            ComponentAnnotation(
                component_index=len(components) + 1,
                positive_pixel_count=area,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                width=width,
                height=height,
                bbox_area=bbox_area,
                bbox_area_ratio=bbox_area / image_area,
                mask_bbox_fill_ratio=area / bbox_area,
                aspect_ratio=width / height,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                centroid_x_ratio=centroid_x / image_width,
                centroid_y_ratio=centroid_y / image_height,
                touches_edge=(
                    x_min == 0
                    or y_min == 0
                    or x_max == image_width - 1
                    or y_max == image_height - 1
                ),
            )
        )

    return tuple(components)


# ADD 2026-08-25: MVTec manifest anomaly record를 normalized mask/component metric으로 변환한다.
def analyze_manifest_annotation(
    *,
    dataset_name: str,
    dataset_root: Path,
    record: ManifestRecord,
) -> SampleAnnotationMetrics:
    """Analyze one manifest anomaly while preserving source path lineage."""
    if not dataset_name.strip():
        raise ValueError("Dataset name must not be blank.")
    if record.label != 1 or record.split != "test" or record.source_split != "test":
        raise ValueError(f"C1 analysis requires an official test anomaly: {record.sample_id}")
    if not record.mask_path:
        raise ValueError(f"Anomaly manifest record has no mask: {record.sample_id}")

    image_path = resolve_dataset_path(dataset_root, record.image_path)
    mask_path = resolve_dataset_path(dataset_root, record.mask_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Anomaly image not found: {image_path}")

    # 단일 sample mask만 로드해 component와 union geometry를 계산한다.
    mask = load_binary_mask(mask_path, expected_size=(record.width, record.height))
    components = extract_mask_components(mask)
    positive_pixel_count = int(mask.sum())
    image_area = record.width * record.height
    positive_y, positive_x = np.nonzero(mask)
    bbox_x_min = int(positive_x.min())
    bbox_x_max = int(positive_x.max())
    bbox_y_min = int(positive_y.min())
    bbox_y_max = int(positive_y.max())
    bbox_width = bbox_x_max - bbox_x_min + 1
    bbox_height = bbox_y_max - bbox_y_min + 1
    bbox_area = bbox_width * bbox_height
    largest_component = max(components, key=lambda component: component.positive_pixel_count)

    return SampleAnnotationMetrics(
        dataset_name=dataset_name,
        category=record.category,
        sample_id=record.sample_id,
        image_path=record.image_path,
        mask_path=record.mask_path,
        defect_type=record.defect_type,
        image_width=record.width,
        image_height=record.height,
        positive_pixel_count=positive_pixel_count,
        positive_area_ratio=positive_pixel_count / image_area,
        component_count=len(components),
        largest_component_area=largest_component.positive_pixel_count,
        largest_component_area_ratio=largest_component.positive_pixel_count / image_area,
        largest_component_to_mask_ratio=(
            largest_component.positive_pixel_count / positive_pixel_count
        ),
        bbox_x_min=bbox_x_min,
        bbox_y_min=bbox_y_min,
        bbox_x_max=bbox_x_max,
        bbox_y_max=bbox_y_max,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        bbox_area=bbox_area,
        bbox_area_ratio=bbox_area / image_area,
        mask_bbox_fill_ratio=positive_pixel_count / bbox_area,
        bbox_aspect_ratio=bbox_width / bbox_height,
        centroid_x_ratio=float(positive_x.mean() / record.width),
        centroid_y_ratio=float(positive_y.mean() / record.height),
        touches_edge=any(component.touches_edge for component in components),
        components=components,
    )


# ADD 2026-08-25: Numeric observation을 stable seven-number distribution summary로 요약한다.
def summarize_distribution(values: list[float | int]) -> dict[str, float | int]:
    """Summarize a non-empty finite numeric sequence."""
    if not values:
        raise ValueError("Cannot summarize an empty value sequence.")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Distribution values must all be finite.")
    return {
        "count": len(values),
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(median(values)),
        "mean": float(fmean(values)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


# ADD 2026-08-25: Defect sample/component metric을 scarcity와 geometry 판단용 summary로 집계한다.
def summarize_defect_metrics(metrics: list[SampleAnnotationMetrics]) -> dict[str, Any]:
    """Aggregate one defect's deterministic sample and component statistics."""
    if not metrics:
        raise ValueError("Defect metrics must not be empty.")
    defect_types = {metric.defect_type for metric in metrics}
    if len(defect_types) != 1:
        raise ValueError("Defect summary cannot mix defect types.")

    components = [component for metric in metrics for component in metric.components]
    resolutions: dict[str, int] = {}
    for metric in metrics:
        resolution = f"{metric.image_width}x{metric.image_height}"
        resolutions[resolution] = resolutions.get(resolution, 0) + 1

    edge_touch_sample_count = sum(metric.touches_edge for metric in metrics)
    edge_touch_component_count = sum(component.touches_edge for component in components)
    multi_component_sample_count = sum(metric.component_count > 1 for metric in metrics)
    return {
        "sample_count": len(metrics),
        "mask_count": len(metrics),
        "image_resolutions": dict(sorted(resolutions.items())),
        "positive_pixel_count": summarize_distribution(
            [metric.positive_pixel_count for metric in metrics]
        ),
        "positive_area_ratio": summarize_distribution(
            [metric.positive_area_ratio for metric in metrics]
        ),
        "component_count_per_sample": summarize_distribution(
            [metric.component_count for metric in metrics]
        ),
        "component_count_total": len(components),
        "multi_component_sample_count": multi_component_sample_count,
        "multi_component_sample_ratio": multi_component_sample_count / len(metrics),
        "largest_component_area": summarize_distribution(
            [metric.largest_component_area for metric in metrics]
        ),
        "largest_component_area_ratio": summarize_distribution(
            [metric.largest_component_area_ratio for metric in metrics]
        ),
        "largest_component_to_mask_ratio": summarize_distribution(
            [metric.largest_component_to_mask_ratio for metric in metrics]
        ),
        "union_bbox_area_ratio": summarize_distribution(
            [metric.bbox_area_ratio for metric in metrics]
        ),
        "union_mask_bbox_fill_ratio": summarize_distribution(
            [metric.mask_bbox_fill_ratio for metric in metrics]
        ),
        "union_bbox_aspect_ratio": summarize_distribution(
            [metric.bbox_aspect_ratio for metric in metrics]
        ),
        "centroid_x_ratio": summarize_distribution([metric.centroid_x_ratio for metric in metrics]),
        "centroid_y_ratio": summarize_distribution([metric.centroid_y_ratio for metric in metrics]),
        "edge_touch_sample_count": edge_touch_sample_count,
        "edge_touch_sample_ratio": edge_touch_sample_count / len(metrics),
        "component_bbox_area_ratio": summarize_distribution(
            [component.bbox_area_ratio for component in components]
        ),
        "component_mask_bbox_fill_ratio": summarize_distribution(
            [component.mask_bbox_fill_ratio for component in components]
        ),
        "component_aspect_ratio": summarize_distribution(
            [component.aspect_ratio for component in components]
        ),
        "edge_touch_component_count": edge_touch_component_count,
        "edge_touch_component_ratio": edge_touch_component_count / len(components),
    }
