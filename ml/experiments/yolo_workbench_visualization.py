"""Headless visual evidence rendering for the YOLO experiment workbench."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from ml.datasets.segmentation_annotations import rasterize_segmentation_label_instances
from ml.experiments.yolo_augmentation import RepresentationPreview, TransformPreview
from ml.experiments.yolo_workbench import WorkbenchSample

CANVAS = "#101821"
GRID = "#385064"
TEXT = "#f4f7fa"
GT_COLOR = np.array([76, 222, 128], dtype=np.uint8)
PREDICTION_COLOR = np.array([255, 91, 91], dtype=np.uint8)


# ADD 2026-08-27: Immutable image copy에 masks를 합성해 model input과 rendering을 분리한다.
def overlay_masks(
    image: Image.Image,
    masks: Iterable[NDArray[np.bool_]],
    *,
    color: NDArray[np.uint8] = GT_COLOR,
) -> Image.Image:
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    for raw_mask in masks:
        mask = np.asarray(raw_mask, dtype=np.bool_)
        if mask.shape != pixels.shape[:2]:
            raise ValueError("Visualization mask and image dimensions must match.")
        pixels[mask] = (pixels[mask] * 0.45 + color * 0.55).astype(np.uint8)
    return Image.fromarray(pixels)


# ADD 2026-08-27: Notebook evidence를 headless PNG로 저장하고 parent path를 보장한다.
def _save(image: Image.Image, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    if output_path.stat().st_size <= 0:
        raise RuntimeError("Workbench visualization output is empty.")
    return output_path


# ADD 2026-08-27: Compact bar chart로 Manifest-derived EDA counts를 표시한다.
def render_eda_distribution(summary: dict[str, Any], output_path: Path) -> Path:
    if summary.get("included_splits") != ["train", "val"]:
        raise ValueError("EDA visualization rejects non train/validation evidence.")
    rows: list[tuple[str, float]] = []
    for split, counts in summary["image_distribution"].items():
        rows.extend(
            (f"{split} {name}", float(counts[name])) for name in ("positive", "good_negative")
        )
    rows.extend(
        (f"class {name}", float(value)) for name, value in summary["class_component_count"].items()
    )
    rows.extend(
        (f"size {name}", float(value)) for name, value in summary["size_component_count"].items()
    )
    width, row_height = 980, 34
    canvas = Image.new("RGB", (width, 78 + row_height * len(rows)), CANVAS)
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 16), "YOLO TRAIN / VALIDATION DATASET DISTRIBUTION", fill=TEXT)
    draw.text((20, 42), "TEST = SEALED / NOT USED", fill="#ffd166")
    maximum = max((value for _, value in rows), default=1.0)
    for index, (label, value) in enumerate(rows):
        y = 78 + index * row_height
        draw.text((20, y + 8), label, fill=TEXT)
        bar_width = int(650 * value / maximum) if maximum else 0
        draw.rectangle((230, y + 5, 230 + bar_width, y + 25), fill="#40c4ff")
        draw.text((890, y + 8), f"{value:g}", fill=TEXT)
    return _save(canvas, output_path)


# ADD 2026-08-27: Deterministically selected samples의 original/GT overlay와 identity를 렌더한다.
def render_ground_truth_gallery(
    *,
    samples: list[WorkbenchSample],
    dataset_root: Path,
    classes: dict[int, str],
    output_path: Path,
) -> Path:
    if not samples or any(sample.split not in {"train", "val"} for sample in samples):
        raise ValueError("GT gallery accepts only train/validation samples.")
    panel_width, panel_height = 280, 354
    canvas = Image.new("RGB", (panel_width * 2, panel_height * len(samples)), CANVAS)
    draw = ImageDraw.Draw(canvas)
    for row, sample in enumerate(samples):
        with Image.open(dataset_root / sample.image_path) as source:
            original = source.convert("RGB")
        label_text = (dataset_root / sample.label_path).read_text(encoding="utf-8")
        instances = rasterize_segmentation_label_instances(
            label_text,
            image_width=sample.image_width,
            image_height=sample.image_height,
            valid_class_ids=set(classes),
        )
        overlay = overlay_masks(original, (item.mask for item in instances))
        original.thumbnail((panel_width, panel_width))
        overlay.thumbnail((panel_width, panel_width))
        y = row * panel_height
        canvas.paste(original, (0, y))
        canvas.paste(overlay, (panel_width, y))
        draw.text((8, y + 286), f"ORIGINAL | {sample.sample_id} | {sample.split}", fill=TEXT)
        draw.text(
            (8, y + 310),
            f"class={sample.class_name} components={sample.component_count}",
            fill=TEXT,
        )
        draw.text(
            (8, y + 332),
            f"area={list(sample.component_area_ratios)} size={list(sample.size_buckets)}",
            fill=TEXT,
        )
        draw.text((panel_width + 8, y + 286), "GT MASK OVERLAY | green", fill="#4cde80")
    return _save(canvas, output_path)


# ADD 2026-08-27: Original과 pinned Ultralytics transform variants를 같은 sample row에 표시한다.
def render_augmentation_gallery(
    *,
    originals: dict[str, Image.Image],
    previews: list[TransformPreview],
    output_path: Path,
) -> Path:
    sample_ids = list(dict.fromkeys(item.sample_id for item in previews))
    if not sample_ids or any(item.split != "train" for item in previews):
        raise ValueError("Augmentation gallery accepts actual train-transform previews only.")
    variants = max(item.variant for item in previews)
    panel = 250
    canvas = Image.new("RGB", ((variants + 1) * panel, len(sample_ids) * (panel + 62)), CANVAS)
    draw = ImageDraw.Draw(canvas)
    for row, sample_id in enumerate(sample_ids):
        original = originals[sample_id].copy().convert("RGB")
        original.thumbnail((panel, panel))
        y = row * (panel + 62)
        canvas.paste(original, (0, y))
        draw.text((8, y + panel + 8), f"{sample_id} | ORIGINAL", fill=TEXT)
        for preview in sorted(
            (item for item in previews if item.sample_id == sample_id),
            key=lambda item: item.variant,
        ):
            image = overlay_masks(
                Image.fromarray(preview.image_rgb),
                preview.component_masks,
            )
            image.thumbnail((panel, panel))
            x = preview.variant * panel
            canvas.paste(image, (x, y))
            draw.text((x + 8, y + panel + 8), f"ACTUAL AUG #{preview.variant}", fill=TEXT)
            draw.text(
                (x + 8, y + panel + 30),
                f"classes={list(preview.class_ids)} masks={len(preview.component_masks)}",
                fill=TEXT,
            )
    return _save(canvas, output_path)


# ADD 2026-08-27: Original과 actual letterbox 640/1024 representation pixel evidence를 비교한다.
def render_representation_comparison(
    *,
    original: Image.Image,
    original_mask_pixels: tuple[int, ...],
    previews: list[RepresentationPreview],
    output_path: Path,
) -> Path:
    if {item.imgsz for item in previews} != {640, 1024} or len(previews) != 2:
        raise ValueError("Representation comparison requires exactly imgsz 640 and 1024.")
    panel = 380
    canvas = Image.new("RGB", (panel * 3, panel + 116), CANVAS)
    draw = ImageDraw.Draw(canvas)
    source = original.copy().convert("RGB")
    source.thumbnail((panel, panel))
    canvas.paste(source, (0, 0))
    draw.text((10, panel + 8), "ORIGINAL", fill=TEXT)
    draw.text((10, panel + 32), f"mask pixels={list(original_mask_pixels)}", fill=TEXT)
    for index, preview in enumerate(sorted(previews, key=lambda item: item.imgsz), start=1):
        represented = Image.fromarray(preview.image_rgb)
        represented.thumbnail((panel, panel))
        canvas.paste(represented, (index * panel, 0))
        draw.text((index * panel + 10, panel + 8), f"ACTUAL LETTERBOX {preview.imgsz}", fill=TEXT)
        draw.text(
            (index * panel + 10, panel + 32),
            f"mask-grid pixels={list(preview.component_mask_pixels)}",
            fill=TEXT,
        )
        draw.text(
            (index * panel + 10, panel + 54),
            f"input pixels={list(preview.represented_input_pixels)}",
            fill=TEXT,
        )
    draw.text(
        (10, panel + 88),
        "더 많은 pixel 표현이 Small Recall 개선을 보장하지 않으며 validation으로 검증한다.",
        fill="#ffd166",
    )
    return _save(canvas, output_path)


# ADD 2026-08-27: Compact JSONL scalar를 loss/metric/duration curve로 렌더한다.
def render_epoch_curves(epoch_metrics_path: Path, output_path: Path) -> Path:
    records = [
        json.loads(line) for line in epoch_metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    if not records:
        raise ValueError("Epoch curve requires at least one completed epoch.")
    metrics = (
        ("train_box_loss", "box loss", "#4cc9f0"),
        ("train_seg_loss", "seg loss", "#80ed99"),
        ("val_mask_map50_95", "val mask mAP50-95", "#ffd166"),
        ("epoch_time_seconds", "fit-epoch elapsed seconds", "#ff6b6b"),
    )
    width, plot_height = 980, 210
    canvas = Image.new("RGB", (width, 54 + plot_height * len(metrics)), CANVAS)
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 16), "PER-EPOCH TRAINING EVIDENCE", fill=TEXT)
    for row, (field, label, color) in enumerate(metrics):
        values = [(int(item["epoch"]), item.get(field)) for item in records]
        values = [(epoch, float(value)) for epoch, value in values if value is not None]
        top = 54 + row * plot_height
        draw.rectangle((70, top + 20, width - 24, top + plot_height - 30), outline=GRID)
        draw.text((12, top + 4), label, fill=color)
        if not values:
            draw.text((80, top + 82), "not captured", fill=TEXT)
            continue
        minimum, maximum = min(value for _, value in values), max(value for _, value in values)
        scale = maximum - minimum or 1.0
        denominator = max(1, len(records) - 1)
        points = [
            (
                70 + int((epoch - 1) / denominator * (width - 94)),
                top + plot_height - 30 - int((value - minimum) / scale * (plot_height - 50)),
            )
            for epoch, value in values
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=color)
        draw.text((width - 220, top + 4), f"min={minimum:.4g} max={maximum:.4g}", fill=TEXT)
    return _save(canvas, output_path)


# ADD 2026-08-27: Existing resource_telemetry evidence의 sampler/framework boundaries를 시각화한다.
def render_gpu_telemetry(telemetry_path: Path, output_path: Path) -> Path:
    payload = json.loads(telemetry_path.read_text(encoding="utf-8"))
    device = payload.get("nvidia_smi", {})
    torch_cuda = payload.get("pytorch_cuda", {})
    samples = device.get("samples", [])
    canvas = Image.new("RGB", (980, 520), CANVAS)
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 16), "GPU RESOURCE EVIDENCE", fill=TEXT)
    lines = [
        f"training wall clock: {payload.get('training_wall_clock_seconds')} sec",
        f"PyTorch peak allocated bytes: {torch_cuda.get('peak_allocated_bytes')}",
        f"PyTorch peak reserved bytes: {torch_cuda.get('peak_reserved_bytes')}",
        f"device-wide memory MiB: {device.get('memory_used_mib')}",
        f"device-wide utilization %: {device.get('utilization_percent')}",
        f"device-wide power W: {device.get('power_draw_watts')}",
    ]
    for index, line in enumerate(lines):
        draw.text((18, 52 + index * 27), line, fill=TEXT)
    plot_top, plot_bottom = 240, 490
    draw.rectangle((70, plot_top, 950, plot_bottom), outline=GRID)
    if samples:
        points = [
            (
                70 + int(index / max(1, len(samples) - 1) * 880),
                plot_bottom - int(float(sample["utilization_percent"]) / 100.0 * 250),
            )
            for index, sample in enumerate(samples)
        ]
        if len(points) > 1:
            draw.line(points, fill="#4cc9f0", width=3)
        draw.text((70, 214), "sample index vs device-wide GPU utilization", fill="#4cc9f0")
    else:
        draw.text((330, 350), "nvidia-smi samples not captured", fill=TEXT)
    return _save(canvas, output_path)


# ADD 2026-08-27: Generated workbench figures를 compact provenance sidecar로 묶는다.
def write_visualization_manifest(
    *,
    output_path: Path,
    experiment_id: str,
    manifest_sha256: str,
    repository: dict[str, Any],
    entries: list[dict[str, Any]],
) -> Path:
    allowed_sources = {"train", "val", "train_val", "none"}
    if any(entry.get("source_split") not in allowed_sources for entry in entries):
        raise ValueError("Visualization manifest rejects sealed test evidence.")
    payload = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "dataset_manifest_sha256": manifest_sha256,
        "split": "train_val_only",
        "test_split_used": False,
        "repository": repository,
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path
