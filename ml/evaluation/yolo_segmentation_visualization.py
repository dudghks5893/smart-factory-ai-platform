"""Deterministic validation-failure evidence for YOLO segmentation experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.evaluation.yolo_segmentation_error_analysis import (
    GroundTruthInstance,
    PredictedInstance,
    SampleAnalysis,
)

OVERLAY_COLORS = {
    0: np.array([255, 91, 91]),
    1: np.array([255, 190, 72]),
    2: np.array([64, 196, 255]),
}
FAILURE_CATEGORIES = (
    "worst_fn",
    "wrong_class",
    "lowest_iou",
    "under_segmentation",
    "over_segmentation",
    "unmatched_prediction",
    "good_negative_fp",
    "complete_miss",
)


@dataclass(frozen=True)
class FailureGalleryArtifacts:
    """All-sample cards, deterministic category galleries and their sidecar."""

    all_sample_paths: tuple[Path, ...]
    gallery_paths: tuple[Path, ...]
    manifest_path: Path


# ADD 2026-08-27: GT/prediction mask와 bbox/class를 visualization copy에 overlay한다.
def overlay_instances(
    image: NDArray[np.uint8],
    instances: tuple[GroundTruthInstance, ...] | tuple[PredictedInstance, ...],
    *,
    classes: dict[int, str],
    predicted: bool,
) -> Image.Image:
    rendered = image.copy()
    for instance in instances:
        color = OVERLAY_COLORS[instance.class_id]
        rendered[instance.mask] = (0.45 * rendered[instance.mask] + 0.55 * color).astype(np.uint8)
    canvas = Image.fromarray(rendered)
    draw = ImageDraw.Draw(canvas)
    for instance in instances:
        instance_color = tuple(int(value) for value in OVERLAY_COLORS[instance.class_id])
        draw.rectangle(instance.box_xyxy, outline=instance_color, width=3)
        label = classes[instance.class_id]
        if predicted:
            label += f" {instance.confidence:.3f}"  # type: ignore[union-attr]
        draw.text((instance.box_xyxy[0] + 4, instance.box_xyxy[1] + 4), label, fill="white")
    return canvas


# ADD 2026-08-27: Original/GT/prediction과 taxonomy metadata를 readable card로 저장한다.
def save_failure_card(
    *,
    record: DerivedManifestRecord,
    dataset_root: Path,
    ground_truth: tuple[GroundTruthInstance, ...],
    predictions: tuple[PredictedInstance, ...],
    analysis: SampleAnalysis,
    classes: dict[int, str],
    output_path: Path,
    source_split: Literal["val", "test"] = "val",
) -> Path:
    if record.derived_split != source_split:
        raise ValueError("Failure visualization record does not match its explicit namespace.")
    with Image.open(dataset_root / record.image_path) as source:
        image = np.asarray(source.convert("RGB"), dtype=np.uint8)
    original = Image.fromarray(image)
    gt_panel = overlay_instances(image, ground_truth, classes=classes, predicted=False)
    prediction_panel = overlay_instances(image, predictions, classes=classes, predicted=True)
    header_height = 92
    combined = Image.new(
        "RGB",
        (record.image_width * 3, record.image_height + header_height),
        "#101821",
    )
    combined.paste(original, (0, header_height))
    combined.paste(gt_panel, (record.image_width, header_height))
    combined.paste(prediction_panel, (record.image_width * 2, header_height))
    draw = ImageDraw.Draw(combined)
    draw.text((8, 8), f"ORIGINAL | {record.sample_id} | split={source_split}", fill="white")
    draw.text((record.image_width + 8, 8), "GROUND TRUTH | overlay", fill="white")
    draw.text(
        (record.image_width * 2 + 8, 8),
        f"PREDICTION | {analysis.main_error}",
        fill="white",
    )
    draw.text(
        (8, 34),
        f"GT={analysis.ground_truth_class} components={analysis.ground_truth_component_count} "
        f"size={analysis.size_bucket}",
        fill="white",
    )
    draw.text(
        (8, 58),
        f"pred={list(analysis.predicted_classes)} conf={analysis.predicted_confidence} "
        f"best_iou={analysis.best_mask_iou} tags={list(analysis.secondary_tags)}",
        fill="white",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)
    return output_path


# ADD 2026-08-27: Existing taxonomy semantics로 category별 Top-K를 stable하게 rank한다.
def rank_failure_categories(
    analyses: list[SampleAnalysis],
    *,
    top_k: int = 4,
) -> dict[str, list[SampleAnalysis]]:
    if top_k <= 0 or any(getattr(item, "sample_id", "") == "" for item in analyses):
        raise ValueError("Failure category ranking input is invalid.")
    selectors = {
        "worst_fn": lambda item: item.false_negative_count > 0,
        "wrong_class": lambda item: "WRONG_CLASS" in item.secondary_tags,
        "lowest_iou": lambda item: bool(item.matches),
        "under_segmentation": lambda item: "MASK_UNDER_SEGMENTATION" in item.secondary_tags,
        "over_segmentation": lambda item: "MASK_OVER_SEGMENTATION" in item.secondary_tags,
        "unmatched_prediction": lambda item: item.false_positive_count > 0,
        "good_negative_fp": lambda item: item.is_negative and item.false_positive_count > 0,
        "complete_miss": lambda item: (
            item.ground_truth_instance_count > 0 and item.predicted_instance_count == 0
        ),
    }
    ranking = {
        "worst_fn": lambda item: (-item.false_negative_count, item.sample_id),
        "wrong_class": lambda item: (item.sample_id,),
        "lowest_iou": lambda item: (
            min(match.mask_iou for match in item.matches),
            item.sample_id,
        ),
        "under_segmentation": lambda item: (item.sample_id,),
        "over_segmentation": lambda item: (item.sample_id,),
        "unmatched_prediction": lambda item: (-item.false_positive_count, item.sample_id),
        "good_negative_fp": lambda item: (
            -item.false_positive_count,
            -(item.predicted_confidence or 0.0),
            item.sample_id,
        ),
        "complete_miss": lambda item: (-item.false_negative_count, item.sample_id),
    }
    return {
        category: sorted(
            (item for item in analyses if selectors[category](item)),
            key=ranking[category],
        )[:top_k]
        for category in FAILURE_CATEGORIES
    }


# ADD 2026-08-27: Ranked sample cards를 one-file gallery evidence로 결합한다.
def _save_card_gallery(card_paths: list[Path], output_path: Path, title: str) -> Path:
    cards: list[Image.Image] = []
    for path in card_paths:
        with Image.open(path) as image:
            cards.append(image.convert("RGB").copy())
    if not cards:
        raise ValueError("Failure gallery requires at least one card.")
    title_height = 44
    width = max(image.width for image in cards)
    height = title_height + sum(image.height for image in cards)
    gallery = Image.new("RGB", (width, height), "#101821")
    ImageDraw.Draw(gallery).text((12, 14), title, fill="white")
    offset = title_height
    for card in cards:
        gallery.paste(card, (0, offset))
        offset += card.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gallery.save(output_path)
    return output_path


# ADD 2026-08-27: All validation cards와 non-empty deterministic failure galleries를 저장한다.
def render_validation_failure_galleries(
    *,
    records_by_sample: dict[str, DerivedManifestRecord],
    dataset_root: Path,
    ground_truth_by_sample: dict[str, tuple[GroundTruthInstance, ...]],
    predictions_by_sample: dict[str, tuple[PredictedInstance, ...]],
    analyses: list[SampleAnalysis],
    classes: dict[int, str],
    output_dir: Path,
    provenance: dict[str, Any],
    top_k: int = 4,
) -> FailureGalleryArtifacts:
    return _render_failure_galleries(
        records_by_sample=records_by_sample,
        dataset_root=dataset_root,
        ground_truth_by_sample=ground_truth_by_sample,
        predictions_by_sample=predictions_by_sample,
        analyses=analyses,
        classes=classes,
        output_dir=output_dir,
        provenance=provenance,
        source_split="val",
        top_k=top_k,
    )


# ADD 2026-08-27: C4-3가 명시적 test namespace에서 같은 renderer를 재사용할 API를 예약한다.
def render_final_test_failure_galleries(
    *,
    records_by_sample: dict[str, DerivedManifestRecord],
    dataset_root: Path,
    ground_truth_by_sample: dict[str, tuple[GroundTruthInstance, ...]],
    predictions_by_sample: dict[str, tuple[PredictedInstance, ...]],
    analyses: list[SampleAnalysis],
    classes: dict[int, str],
    output_dir: Path,
    provenance: dict[str, Any],
    top_k: int = 4,
) -> FailureGalleryArtifacts:
    return _render_failure_galleries(
        records_by_sample=records_by_sample,
        dataset_root=dataset_root,
        ground_truth_by_sample=ground_truth_by_sample,
        predictions_by_sample=predictions_by_sample,
        analyses=analyses,
        classes=classes,
        output_dir=output_dir,
        provenance=provenance,
        source_split="test",
        top_k=top_k,
    )


# ADD 2026-08-27: Explicit split namespace에서 common card/gallery rendering을 수행한다.
def _render_failure_galleries(
    *,
    records_by_sample: dict[str, DerivedManifestRecord],
    dataset_root: Path,
    ground_truth_by_sample: dict[str, tuple[GroundTruthInstance, ...]],
    predictions_by_sample: dict[str, tuple[PredictedInstance, ...]],
    analyses: list[SampleAnalysis],
    classes: dict[int, str],
    output_dir: Path,
    provenance: dict[str, Any],
    source_split: Literal["val", "test"],
    top_k: int,
) -> FailureGalleryArtifacts:
    if any(record.derived_split != source_split for record in records_by_sample.values()):
        raise ValueError("Failure gallery contains a record outside its explicit split namespace.")
    all_sample_paths: list[Path] = []
    card_by_sample: dict[str, Path] = {}
    for analysis in sorted(analyses, key=lambda item: item.sample_id):
        path = output_dir / "all_samples" / f"{analysis.sample_id}.png"
        card_by_sample[analysis.sample_id] = save_failure_card(
            record=records_by_sample[analysis.sample_id],
            dataset_root=dataset_root,
            ground_truth=ground_truth_by_sample[analysis.sample_id],
            predictions=predictions_by_sample[analysis.sample_id],
            analysis=analysis,
            classes=classes,
            output_path=path,
            source_split=source_split,
        )
        all_sample_paths.append(path)
    ranked = rank_failure_categories(analyses, top_k=top_k)
    gallery_paths: list[Path] = []
    categories: dict[str, Any] = {}
    for category in FAILURE_CATEGORIES:
        selected = ranked[category]
        gallery_path: Path | None = None
        if selected:
            gallery_path = _save_card_gallery(
                [card_by_sample[item.sample_id] for item in selected],
                output_dir / f"{category}_gallery.png",
                f"{source_split.upper()} | {category}",
            )
            gallery_paths.append(gallery_path)
        categories[category] = {
            "ranking_policy": category,
            "selected_sample_ids": [item.sample_id for item in selected],
            "generated_path": gallery_path.as_posix() if gallery_path else None,
        }
    manifest = {
        "schema_version": 1,
        "split": source_split,
        "test_split_used": source_split == "test",
        "legend": "GT/prediction class colors: bent red, color amber, scratch blue",
        "categories": categories,
        "provenance": provenance,
    }
    manifest_path = output_dir / "visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return FailureGalleryArtifacts(
        all_sample_paths=tuple(all_sample_paths),
        gallery_paths=tuple(gallery_paths),
        manifest_path=manifest_path,
    )


# ADD 2026-08-27: Same validation sample의 improvement/regression을 함께 비교한다.
def render_baseline_candidate_comparison(
    *,
    baseline_sample_analysis_path: Path,
    candidate_sample_analysis_path: Path,
    baseline_cards_dir: Path,
    candidate_cards_dir: Path,
    output_dir: Path,
    max_count: int = 8,
) -> Path:
    if max_count <= 0:
        raise ValueError("Comparison gallery max_count must be positive.")
    baseline = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in baseline_sample_analysis_path.read_text(encoding="utf-8").splitlines()
        )
    }
    candidate = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in candidate_sample_analysis_path.read_text(encoding="utf-8").splitlines()
        )
    }
    if set(baseline) != set(candidate):
        raise ValueError("Baseline/Candidate comparison requires identical validation samples.")

    def delta(sample_id: str) -> tuple[int, int, str]:
        before = baseline[sample_id]
        after = candidate[sample_id]
        error_delta = (after["false_negative_count"] + after["false_positive_count"]) - (
            before["false_negative_count"] + before["false_positive_count"]
        )
        small_or_multi = int(
            before.get("size_bucket") == "small"
            or before.get("ground_truth_component_count", 0) > 1
            or before.get("predicted_instance_count", 0) == 0
        )
        return error_delta, -small_or_multi, sample_id

    regressions = sorted(
        (sample_id for sample_id in baseline if delta(sample_id)[0] > 0), key=delta
    )
    regressions.reverse()
    improvements = sorted(
        (sample_id for sample_id in baseline if delta(sample_id)[0] < 0), key=delta
    )
    hypothesis = sorted(
        (
            sample_id
            for sample_id, row in baseline.items()
            if row.get("size_bucket") == "small"
            or row.get("ground_truth_component_count", 0) > 1
            or (
                row.get("ground_truth_instance_count", 0) > 0
                and row.get("predicted_instance_count", 0) == 0
            )
        )
    )
    selected: list[str] = []
    for group in (regressions, improvements, hypothesis, sorted(baseline)):
        for sample_id in group:
            if sample_id not in selected:
                selected.append(sample_id)
            if len(selected) >= max_count:
                break
        if len(selected) >= max_count:
            break
    pair_paths: list[Path] = []
    for sample_id in selected:
        pair_paths.extend(
            (
                baseline_cards_dir / f"{sample_id}.png",
                candidate_cards_dir / f"{sample_id}.png",
            )
        )
    output_path = output_dir / "baseline_vs_candidate_gallery.png"
    _save_card_gallery(pair_paths, output_path, "VALIDATION | BASELINE then CANDIDATE")
    manifest = {
        "schema_version": 1,
        "split": "val",
        "test_split_used": False,
        "selected_sample_ids": selected,
        "regression_sample_ids": [item for item in selected if item in regressions],
        "improvement_sample_ids": [item for item in selected if item in improvements],
        "gallery_path": output_path.as_posix(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path
