"""Portable behavior tests for YOLO workbench visual evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ml.evaluation.yolo_segmentation_error_analysis import InstanceMatch, SampleAnalysis
from ml.evaluation.yolo_segmentation_visualization import rank_failure_categories
from ml.experiments.yolo_workbench_visualization import (
    overlay_masks,
    render_eda_distribution,
    render_epoch_curves,
    render_gpu_telemetry,
)


# ADD 2026-08-27: Failure ranking용 minimal serializable analysis fixture를 만든다.
def _analysis(
    sample_id: str,
    *,
    false_negative_count: int = 0,
    false_positive_count: int = 0,
    tags: tuple[str, ...] = (),
    matches: tuple[InstanceMatch, ...] = (),
    negative: bool = False,
    predicted_count: int = 1,
) -> SampleAnalysis:
    return SampleAnalysis(
        sample_id=sample_id,
        ground_truth_class="good" if negative else "bent",
        is_negative=negative,
        ground_truth_instance_count=0 if negative else 1,
        predicted_instance_count=predicted_count,
        expected_class_hit=False,
        true_positive_count=0,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        best_mask_iou=min((item.mask_iou for item in matches), default=None),
        best_box_iou=None,
        predicted_confidence=0.8 if predicted_count else None,
        ground_truth_mask_area_ratio=0.01,
        predicted_mask_area_ratio=0.01,
        ground_truth_component_count=0 if negative else 1,
        size_bucket=None if negative else "small",
        main_error="MISSED_DEFECT" if false_negative_count else "FALSE_POSITIVE",
        secondary_tags=tags,
        predicted_classes=("bent",) if predicted_count else (),
        predicted_instance_confidences=(0.8,) if predicted_count else (),
        matches=matches,
        ground_truth_outcomes=(),
        confusion_pairs=(),
    )


# ADD 2026-08-27: FN/IoU/wrong-class/good-negative routing과 deterministic tie-break를 검증한다.
def test_failure_category_ranking_uses_explicit_semantics() -> None:
    low_match = InstanceMatch(0, 0, 0.51, 0.6, 0.7, 0.8)
    high_match = InstanceMatch(0, 0, 0.8, 0.9, 0.9, 0.9)
    analyses = [
        _analysis("b", false_negative_count=2, predicted_count=0),
        _analysis("a", false_negative_count=2, predicted_count=0),
        _analysis("wrong", tags=("WRONG_CLASS",), matches=(high_match,)),
        _analysis("low", matches=(low_match,)),
        _analysis("good", false_positive_count=2, negative=True),
    ]
    ranked = rank_failure_categories(analyses, top_k=4)
    assert [item.sample_id for item in ranked["worst_fn"]] == ["a", "b"]
    assert ranked["lowest_iou"][0].sample_id == "low"
    assert ranked["wrong_class"][0].sample_id == "wrong"
    assert ranked["good_negative_fp"][0].sample_id == "good"


# ADD 2026-08-27: Headless plot/card helpers가 aligned mask와 non-empty PNG를 만드는지 검증한다.
def test_portable_visualizations_create_non_empty_png(tmp_path: Path) -> None:
    image = Image.new("RGB", (16, 16), "black")
    mask = np.zeros((16, 16), dtype=np.bool_)
    mask[2:6, 3:8] = True
    assert overlay_masks(image, [mask]).size == image.size

    summary = {
        "included_splits": ["train", "val"],
        "image_distribution": {
            "train": {"positive": 2, "good_negative": 1},
            "val": {"positive": 1, "good_negative": 1},
        },
        "class_component_count": {"bent": 2},
        "size_component_count": {"small": 1, "medium": 1, "large": 0},
    }
    eda_path = render_eda_distribution(summary, tmp_path / "eda.png")

    epoch_path = tmp_path / "epoch.jsonl"
    epoch_path.write_text(
        json.dumps(
            {
                "epoch": 1,
                "train_box_loss": 0.4,
                "train_seg_loss": 0.5,
                "val_mask_map50_95": 0.2,
                "epoch_time_seconds": 2.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    curves_path = render_epoch_curves(epoch_path, tmp_path / "curves.png")
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(
        json.dumps(
            {
                "training_wall_clock_seconds": 4.0,
                "pytorch_cuda": {"peak_allocated_bytes": 100, "peak_reserved_bytes": 200},
                "nvidia_smi": {
                    "memory_used_mib": {"mean": 10},
                    "utilization_percent": {"mean": 50},
                    "power_draw_watts": {"mean": 30},
                    "samples": [{"utilization_percent": 50}],
                },
            }
        ),
        encoding="utf-8",
    )
    gpu_path = render_gpu_telemetry(telemetry_path, tmp_path / "gpu.png")
    assert all(path.stat().st_size > 0 for path in (eda_path, curves_path, gpu_path))
