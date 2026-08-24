"""Analyze official anomaly masks for known-defect task feasibility."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from ml.datasets.defect_annotations import (
    SampleAnnotationMetrics,
    analyze_manifest_annotation,
    load_binary_mask,
    resolve_dataset_path,
    summarize_defect_metrics,
)
from ml.datasets.manifest import ManifestRecord, read_manifest_csv
from ml.datasets.manifest_validation import validate_manifest_records
from shared.hashing import sha256_file

DEFAULT_DATASET_ROOT = Path("data/raw/mvtec_ad")
DEFAULT_MANIFEST_PATH = Path("data/interim/manifests/mvtec_ad_metal_nut.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/mvtec_ad/metal_nut/known_defect_feasibility")
METAL_NUT_DEFECTS = ("bent", "color", "flip", "scratch")

SAMPLE_FIELDS = (
    "dataset_name",
    "category",
    "sample_id",
    "image_path",
    "mask_path",
    "defect_type",
    "image_width",
    "image_height",
    "positive_pixel_count",
    "positive_area_ratio",
    "component_count",
    "largest_component_area",
    "largest_component_area_ratio",
    "largest_component_to_mask_ratio",
    "bbox_x_min",
    "bbox_y_min",
    "bbox_x_max",
    "bbox_y_max",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_area_ratio",
    "mask_bbox_fill_ratio",
    "bbox_aspect_ratio",
    "centroid_x_ratio",
    "centroid_y_ratio",
    "touches_edge",
    "component_bboxes_json",
)

COMPONENT_FIELDS = (
    "dataset_name",
    "category",
    "sample_id",
    "defect_type",
    "component_index",
    "positive_pixel_count",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "width",
    "height",
    "bbox_area",
    "bbox_area_ratio",
    "mask_bbox_fill_ratio",
    "aspect_ratio",
    "centroid_x",
    "centroid_y",
    "centroid_x_ratio",
    "centroid_y_ratio",
    "touches_edge",
    "yolo_center_x",
    "yolo_center_y",
    "yolo_width",
    "yolo_height",
)


@dataclass(frozen=True)
class AnalysisArtifacts:
    """Paths and counts produced by one completed C1 analysis run."""

    sample_count: int
    component_count: int
    defect_counts: dict[str, int]
    output_dir: Path
    sample_metrics_path: Path
    component_metrics_path: Path
    summary_path: Path
    visualization_paths: tuple[Path, ...]


# ADD 2026-08-25: C1에 해당하는 official test anomaly record만 stable order로 선택한다.
def select_analysis_records(
    records: list[ManifestRecord],
    *,
    category: str,
    expected_defects: tuple[str, ...],
) -> list[ManifestRecord]:
    """Select official test anomalies and reject an unexpected defect taxonomy."""
    selected = sorted(
        (
            record
            for record in records
            if record.category == category
            and record.source_split == "test"
            and record.split == "test"
            and record.label == 1
        ),
        key=lambda record: (record.defect_type, record.sample_id),
    )
    if not selected:
        raise ValueError(f"No official test anomaly records found for category: {category}")
    actual_defects = tuple(sorted({record.defect_type for record in selected}))
    if actual_defects != tuple(sorted(expected_defects)):
        raise ValueError(
            f"Unexpected defect taxonomy: expected={sorted(expected_defects)}, "
            f"actual={list(actual_defects)}"
        )
    return selected


# ADD 2026-08-25: Metric range와 fragmentation을 반영한 representative sample을 선택한다.
def select_representative_metrics(
    metrics: list[SampleAnnotationMetrics],
    *,
    count: int,
) -> list[SampleAnnotationMetrics]:
    """Select area-spread, fragmented, and sparse-box examples without randomness."""
    if count <= 0:
        raise ValueError("Representative example count must be positive.")
    if not metrics:
        raise ValueError("Representative selection requires sample metrics.")

    ordered = sorted(metrics, key=lambda metric: (metric.positive_area_ratio, metric.sample_id))
    target_count = min(count, len(ordered))
    spread_indices = np.linspace(0, len(ordered) - 1, num=target_count, dtype=int)
    candidates = [ordered[int(index)] for index in spread_indices]
    candidates.extend(
        (
            max(metrics, key=lambda metric: (metric.component_count, metric.sample_id)),
            min(metrics, key=lambda metric: (metric.mask_bbox_fill_ratio, metric.sample_id)),
        )
    )

    selected_by_id: dict[str, SampleAnnotationMetrics] = {}
    for candidate in candidates:
        selected_by_id.setdefault(candidate.sample_id, candidate)
    for candidate in ordered:
        if len(selected_by_id) >= target_count:
            break
        selected_by_id.setdefault(candidate.sample_id, candidate)

    selected_ids = set(selected_by_id)
    return [metric for metric in ordered if metric.sample_id in selected_ids][:target_count]


# ADD 2026-08-25: Original, mask overlay와 component box를 defect별 comparison montage로 저장한다.
def render_defect_visualization(
    *,
    dataset_root: Path,
    metrics: list[SampleAnnotationMetrics],
    output_path: Path,
    example_count: int,
) -> list[str]:
    """Render representative source images without copying raw data."""
    selected = select_representative_metrics(metrics, count=example_count)
    figure, axes = plt.subplots(len(selected), 3, figsize=(12, 3.25 * len(selected)))
    axes_array = np.atleast_2d(axes)

    for row_index, metric in enumerate(selected):
        image_path = resolve_dataset_path(dataset_root, metric.image_path)
        mask_path = resolve_dataset_path(dataset_root, metric.mask_path)
        with Image.open(image_path) as source_image:
            image = np.asarray(source_image.convert("RGB"))
        mask = load_binary_mask(
            mask_path,
            expected_size=(metric.image_width, metric.image_height),
        )

        original_axis, overlay_axis, bbox_axis = axes_array[row_index]
        original_axis.imshow(image)
        overlay_axis.imshow(image)
        overlay_axis.imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.55)
        bbox_axis.imshow(image)
        for component in metric.components:
            bbox_axis.add_patch(
                Rectangle(
                    (component.x_min, component.y_min),
                    component.width,
                    component.height,
                    fill=False,
                    edgecolor="#00e5ff",
                    linewidth=1.8,
                )
            )

        original_axis.set_ylabel(
            f"{Path(metric.image_path).stem}\n"
            f"mask={metric.positive_area_ratio:.3f}\n"
            f"components={metric.component_count}",
            fontsize=9,
        )
        for axis in (original_axis, overlay_axis, bbox_axis):
            axis.set_xticks([])
            axis.set_yticks([])

    axes_array[0, 0].set_title("Original")
    axes_array[0, 1].set_title("Ground-truth mask overlay")
    axes_array[0, 2].set_title("8-connected component boxes")
    figure.suptitle(f"metal_nut / {metrics[0].defect_type}", fontsize=14, y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return [metric.sample_id for metric in selected]


# ADD 2026-08-25: Sample-level analysis metric을 stable CSV schema로 저장한다.
def write_sample_metrics_csv(
    metrics: list[SampleAnnotationMetrics],
    output_path: Path,
) -> None:
    """Write one row per anomaly sample with component boxes embedded as JSON."""
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for metric in metrics:
            values = asdict(metric)
            values.pop("components")
            values["component_bboxes_json"] = json.dumps(
                [
                    {
                        "component_index": component.component_index,
                        "x_min": component.x_min,
                        "y_min": component.y_min,
                        "x_max": component.x_max,
                        "y_max": component.y_max,
                        "width": component.width,
                        "height": component.height,
                    }
                    for component in metric.components
                ],
                separators=(",", ":"),
            )
            writer.writerow(values)


# ADD 2026-08-25: Component-level box geometry와 future YOLO coordinate를 separate CSV로 저장한다.
def write_component_metrics_csv(
    metrics: list[SampleAnnotationMetrics],
    output_path: Path,
) -> None:
    """Write one row per component without assigning supervised class semantics."""
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COMPONENT_FIELDS)
        writer.writeheader()
        for metric in metrics:
            for component in metric.components:
                center_x, center_y, width, height = component.to_yolo_xywh(
                    image_width=metric.image_width,
                    image_height=metric.image_height,
                )
                writer.writerow(
                    {
                        "dataset_name": metric.dataset_name,
                        "category": metric.category,
                        "sample_id": metric.sample_id,
                        "defect_type": metric.defect_type,
                        **asdict(component),
                        "yolo_center_x": center_x,
                        "yolo_center_y": center_y,
                        "yolo_width": width,
                        "yolo_height": height,
                    }
                )


# ADD 2026-08-25: C1 analysis를 validation, lazy processing, atomic artifact 순서로 조율한다.
def analyze_known_defect_feasibility(
    *,
    dataset_name: str,
    dataset_root: Path,
    manifest_path: Path,
    category: str,
    output_dir: Path,
    expected_defects: tuple[str, ...],
    examples_per_defect: int,
) -> AnalysisArtifacts:
    """Create analysis-only metrics and visual evidence from manifest anomalies."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if output_dir.exists():
        raise FileExistsError(f"Analysis output already exists: {output_dir}")
    if examples_per_defect <= 0:
        raise ValueError("Examples per defect must be positive.")

    # Existing official manifest schema, path, mask와 split semantics을 analysis 전에 검증한다.
    all_records = read_manifest_csv(manifest_path)
    records = select_analysis_records(
        all_records,
        category=category,
        expected_defects=expected_defects,
    )
    manifest_report = validate_manifest_records(records, dataset_root)
    if not manifest_report.is_valid:
        raise ValueError(
            "C1 source manifest validation failed:\n" + "\n".join(manifest_report.errors)
        )

    # Mask를 sample 단위로 로드해 component metric만 memory에 유지한다.
    metrics = [
        analyze_manifest_annotation(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            record=record,
        )
        for record in records
    ]
    metrics_by_defect: dict[str, list[SampleAnnotationMetrics]] = defaultdict(list)
    for metric in metrics:
        metrics_by_defect[metric.defect_type].append(metric)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        sample_metrics_path = temporary_dir / "sample_metrics.csv"
        component_metrics_path = temporary_dir / "component_metrics.csv"
        summary_path = temporary_dir / "defect_summary.json"

        # Tabular metric, defect summary와 representative image를 artifact 하나로 구성한다.
        write_sample_metrics_csv(metrics, sample_metrics_path)
        write_component_metrics_csv(metrics, component_metrics_path)
        representative_samples: dict[str, list[str]] = {}
        visualization_names: list[str] = []
        for defect_type in sorted(metrics_by_defect):
            visualization_name = f"{defect_type}_examples.png"
            representative_samples[defect_type] = render_defect_visualization(
                dataset_root=dataset_root,
                metrics=metrics_by_defect[defect_type],
                output_path=temporary_dir / "visualizations" / visualization_name,
                example_count=examples_per_defect,
            )
            visualization_names.append(visualization_name)

        summary: dict[str, Any] = {
            "schema_version": "1",
            "analysis_type": "known_defect_feasibility",
            "dataset_name": dataset_name,
            "category": category,
            "source_protocol": "official_mvtec_ad_test_anomalies_analysis_only",
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "connectivity": 8,
            "bbox_coordinates": "inclusive_pixel_indices",
            "sample_count": len(metrics),
            "mask_count": len(metrics),
            "component_count": sum(metric.component_count for metric in metrics),
            "defect_counts": {
                defect_type: len(metrics_by_defect[defect_type])
                for defect_type in sorted(metrics_by_defect)
            },
            "representative_samples": representative_samples,
            "defects": {
                defect_type: summarize_defect_metrics(metrics_by_defect[defect_type])
                for defect_type in sorted(metrics_by_defect)
            },
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    return AnalysisArtifacts(
        sample_count=len(metrics),
        component_count=sum(metric.component_count for metric in metrics),
        defect_counts={
            defect_type: len(metrics_by_defect[defect_type])
            for defect_type in sorted(metrics_by_defect)
        },
        output_dir=output_dir,
        sample_metrics_path=output_dir / "sample_metrics.csv",
        component_metrics_path=output_dir / "component_metrics.csv",
        summary_path=output_dir / "defect_summary.json",
        visualization_paths=tuple(
            output_dir / "visualizations" / name for name in visualization_names
        ),
    )


# ADD 2026-08-25: C1 actual dataset analysis용 CLI argument를 parsing한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MVTec anomaly masks for known-defect task feasibility."
    )
    parser.add_argument("--dataset-name", default="mvtec_ad")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--category", default="metal_nut")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--examples-per-defect", type=int, default=6)
    return parser.parse_args()


# ADD 2026-08-25: C1 analysis artifact를 생성하고 compact run summary를 출력한다.
def main() -> int:
    args = _parse_args()

    # Existing manifest의 official anomaly subset을 분석 artifact로 변환한다.
    artifacts = analyze_known_defect_feasibility(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        category=args.category,
        output_dir=args.output_dir,
        expected_defects=METAL_NUT_DEFECTS,
        examples_per_defect=args.examples_per_defect,
    )

    print("Known-defect feasibility analysis: PASS")
    print(f"Samples: {artifacts.sample_count}")
    print(f"Components: {artifacts.component_count}")
    print(f"Defect counts: {artifacts.defect_counts}")
    print(f"Output: {artifacts.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
