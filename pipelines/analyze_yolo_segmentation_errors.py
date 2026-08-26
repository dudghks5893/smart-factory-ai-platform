"""Analyze fixed YOLO baseline errors on the supervised-derived validation split."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image

from ml.datasets.segmentation_annotations import rasterize_segmentation_label_instances
from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord, read_derived_manifest
from ml.evaluation.yolo_segmentation_error_analysis import (
    CONFIDENCE_LEVELS,
    MATCH_IOU_THRESHOLD,
    GroundTruthInstance,
    PredictedInstance,
    SampleAnalysis,
    SizeBucketPolicy,
    aggregate_analysis,
    analyze_sample,
    build_confidence_sweep,
    derive_improvement_hypotheses,
    derive_size_bucket_policy,
    filter_predictions,
    mask_box,
    rank_worst_samples,
    require_validation_records,
)
from ml.evaluation.yolo_segmentation_visualization import (
    render_validation_failure_galleries,
    save_failure_card,
)
from ml.training.device import SUPPORTED_DEVICES
from ml.training.yolo_segmentation import (
    YoloSegmentationBaselineConfig,
    load_yolo_segmentation_config,
    validate_training_dataset,
)
from pipelines.train_yolo_segmentation import DEFAULT_CONFIG_PATH, DEFAULT_DATASET_ROOT
from services.inference.yolo_segmentation_runtime import (
    YoloSegmentationAdapter,
    YoloSegmentationResult,
    YoloSegmentationRuntimeConfig,
    load_yolo_segmentation_runtime,
)
from shared.hashing import sha256_file

DEFAULT_ARTIFACT_DIR = Path(
    "artifacts/runtime/yolo_segmentation/smartfactory_yolo11n_seg_metal_nut_seed42_t4"
)
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/yolo_segmentation/error_analysis")
BASELINE_CONFIDENCE = 0.25
MAX_VISUALIZATIONS = 10
ERROR_TAXONOMY_DEFINITIONS = {
    "TRUE_POSITIVE": "Positive sample without diagnostic error tags.",
    "TRUE_NEGATIVE": "Good-negative sample without predictions.",
    "MISSED_DEFECT": "At least one ground-truth component is unmatched.",
    "FALSE_POSITIVE": "At least one predicted instance is unmatched.",
    "WRONG_CLASS": "Spatial IoU >= 0.5 but GT and prediction classes differ.",
    "LOW_IOU_LOCALIZATION": "Matched IoU < 0.65 or same-class overlap is below match IoU 0.5.",
    "MASK_UNDER_SEGMENTATION": "Matched prediction covers less than 0.75 of GT pixels.",
    "MASK_OVER_SEGMENTATION": "Less than 0.75 of matched prediction pixels overlap GT.",
    "MULTI_COMPONENT_MISS": "A multi-component sample has at least one unmatched GT component.",
}


type RuntimeLoader = Callable[[YoloSegmentationRuntimeConfig], YoloSegmentationAdapter]


@dataclass(frozen=True)
class ErrorAnalysisArtifacts:
    """Machine-readable and visual outputs from one validation-only run."""

    output_dir: Path
    sample_analysis_path: Path
    summary_path: Path
    per_class_path: Path
    confidence_sweep_path: Path
    error_taxonomy_path: Path
    hypotheses_path: Path
    visualization_paths: tuple[Path, ...]


# ADD 2026-08-26: One GT polygon/component를 source-resolution mask instance로 복원한다.
# MODIFY 2026-08-27: Shared polygon rasterization contract를 EDA/diagnostics가 함께 사용한다.
def load_ground_truth_instances(
    record: DerivedManifestRecord,
    *,
    dataset_root: Path,
    valid_class_ids: set[int],
) -> tuple[GroundTruthInstance, ...]:
    if record.derived_split != "val":
        raise ValueError("Ground-truth diagnostics accept only validation records.")
    label_text = (dataset_root / record.label_path).read_text(encoding="utf-8")
    rasterized = rasterize_segmentation_label_instances(
        label_text,
        image_width=record.image_width,
        image_height=record.image_height,
        valid_class_ids=valid_class_ids,
    )
    if record.is_negative:
        if rasterized or record.component_count:
            raise ValueError("Validation negative must not contain GT segmentation instances.")
        return ()
    if len(rasterized) != record.component_count:
        raise ValueError(
            f"GT polygon/component count mismatch for validation sample: {record.sample_id}"
        )
    return tuple(
        GroundTruthInstance(
            class_id=instance.class_id,
            mask=instance.mask,
            box_xyxy=mask_box(instance.mask),
            area_ratio=instance.area_ratio,
        )
        for instance in rasterized
    )


# ADD 2026-08-26: Runtime-neutral result를 confidence sweep용 immutable prediction으로 변환한다.
def normalize_predictions(result: YoloSegmentationResult) -> tuple[PredictedInstance, ...]:
    result.validate()
    return tuple(
        PredictedInstance(
            class_id=instance.class_id,
            confidence=instance.confidence,
            mask=instance.mask,
            box_xyxy=instance.box_xyxy,
        )
        for instance in result.instances
    )


# ADD 2026-08-26: Diagnostic payload를 stable pretty JSON으로 저장한다.
def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ADD 2026-08-26: Sample diagnostics를 identity 순서의 reusable JSONL로 저장한다.
def _write_jsonl(path: Path, analyses: list[SampleAnalysis]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for analysis in sorted(analyses, key=lambda item: item.sample_id):
            file.write(json.dumps(analysis.to_dict(), sort_keys=True) + "\n")


# ADD 2026-08-26: GT/prediction mask와 bbox/class evidence를 source image 위에 그린다.
# ADD 2026-08-26: Required diagnostic scenarios를 우선한 뒤 worst ranking으로 Top-N을 채운다.
def select_visualization_samples(
    analyses: list[SampleAnalysis], *, max_count: int = MAX_VISUALIZATIONS
) -> list[SampleAnalysis]:
    ranked = rank_worst_samples(analyses)
    selectors = (
        lambda item: item.false_negative_count > 0,
        lambda item: item.true_positive_count > 0 and item.best_mask_iou is not None,
        lambda item: "WRONG_CLASS" in item.secondary_tags,
        lambda item: item.is_negative and item.false_positive_count > 0,
        lambda item: item.size_bucket == "small" and item.main_error != "TRUE_POSITIVE",
        lambda item: "MULTI_COMPONENT_MISS" in item.secondary_tags,
    )
    selected: list[SampleAnalysis] = []
    for selector in selectors:
        candidates = [item for item in ranked if selector(item)]
        if candidates and candidates[0] not in selected:
            selected.append(candidates[0])
    for item in ranked:
        if len(selected) >= max_count:
            break
        if item.main_error not in {"TRUE_POSITIVE", "TRUE_NEGATIVE"} and item not in selected:
            selected.append(item)
    return selected[:max_count]


# ADD 2026-08-26: Artifact 검증부터 validation diagnostics 저장까지 조율한다.
# MODIFY 2026-08-27: Fixed size policy와 deterministic validation gallery evidence를 확장한다.
def analyze_yolo_segmentation_errors(
    *,
    config: YoloSegmentationBaselineConfig,
    dataset_root: Path,
    artifact_dir: Path,
    output_dir: Path,
    requested_device: str,
    size_policy_override: SizeBucketPolicy | None = None,
    runtime_loader: RuntimeLoader = load_yolo_segmentation_runtime,
    created_at: str | None = None,
) -> ErrorAnalysisArtifacts:
    if output_dir.exists():
        raise FileExistsError(f"Validation error-analysis output already exists: {output_dir}")

    # 비용이 큰 model load 전에 dataset lineage와 validation-only analysis boundary를 검증한다.
    validate_training_dataset(dataset_root, config.dataset_contract)
    manifest_path = dataset_root / "manifest.csv"
    records = read_derived_manifest(manifest_path)
    validation_records = require_validation_records(
        [record for record in records if record.derived_split == "val"]
    )
    if len(validation_records) != config.dataset_contract.sample_counts["val"]:
        raise ValueError("Derived validation record count does not match the dataset contract.")
    if any(record.derived_split == "test" for record in validation_records):
        raise ValueError("Test leakage detected in validation error-analysis input.")

    model_path = artifact_dir / "model" / "model.pt"
    metadata_path = artifact_dir / "model" / "metadata.json"
    model_sha_before = sha256_file(model_path)
    metadata_sha_before = sha256_file(metadata_path)

    # Runtime loader가 metadata/checkpoint/framework/class SHA 계약을 검증한 뒤 model을 복원한다.
    runtime = runtime_loader(
        YoloSegmentationRuntimeConfig(artifact_dir=artifact_dir, device=requested_device)
    )
    provenance = runtime.provenance
    dataset_manifest_sha = sha256_file(manifest_path)
    if getattr(provenance, "dataset_manifest_sha256", None) != dataset_manifest_sha:
        raise ValueError("Runtime artifact and validation dataset manifest SHA do not match.")

    classes = config.dataset_contract.classes
    ground_truth_by_sample: dict[str, tuple[GroundTruthInstance, ...]] = {}
    predictions_by_sample: dict[str, tuple[PredictedInstance, ...]] = {}
    records_by_sample = {record.sample_id: record for record in validation_records}
    all_ground_truth: list[GroundTruthInstance] = []
    for record in validation_records:
        ground_truth = load_ground_truth_instances(
            record,
            dataset_root=dataset_root,
            valid_class_ids=set(classes),
        )
        ground_truth_by_sample[record.sample_id] = ground_truth
        all_ground_truth.extend(ground_truth)

        # 최저 confidence에서 한 번 추론해 모든 sweep point를 같은 prediction pool로 비교한다.
        with Image.open(dataset_root / record.image_path) as source:
            image_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        predictions_by_sample[record.sample_id] = normalize_predictions(
            runtime.predict(image_rgb, diagnostic_confidence=CONFIDENCE_LEVELS[0])
        )

    size_policy = size_policy_override or derive_size_bucket_policy(all_ground_truth)
    baseline_predictions = filter_predictions(predictions_by_sample, BASELINE_CONFIDENCE)
    analyses = [
        analyze_sample(
            record=record,
            ground_truth=ground_truth_by_sample[record.sample_id],
            predictions=baseline_predictions[record.sample_id],
            classes=classes,
            size_policy=size_policy,
        )
        for record in validation_records
    ]
    aggregate = aggregate_analysis(analyses, classes=classes)
    confidence_sweep = build_confidence_sweep(
        records=validation_records,
        ground_truth_by_sample=ground_truth_by_sample,
        predictions_by_sample=predictions_by_sample,
        classes=classes,
        size_policy=size_policy,
    )
    hypotheses = derive_improvement_hypotheses(aggregate, confidence_sweep)

    if (
        sha256_file(model_path) != model_sha_before
        or sha256_file(metadata_path) != metadata_sha_before
    ):
        raise RuntimeError("Baseline artifact changed during validation analysis.")

    # Validation findings와 provenance를 ignored output tree에 machine-readable하게 저장한다.
    output_dir.mkdir(parents=True, exist_ok=False)
    sample_analysis_path = output_dir / "sample_analysis.jsonl"
    summary_path = output_dir / "summary.json"
    per_class_path = output_dir / "per_class.json"
    confidence_sweep_path = output_dir / "confidence_sweep.json"
    error_taxonomy_path = output_dir / "error_taxonomy.json"
    hypotheses_path = output_dir / "improvement_hypotheses.json"
    summary = {
        "schema_version": 1,
        "analysis_name": "YOLO Segmentation Validation Error Analysis",
        "split": "val",
        "test_split_used": False,
        "baseline_confidence": BASELINE_CONFIDENCE,
        "matching": {
            "method": "class_aware_greedy_max_mask_iou",
            "mask_iou_threshold": MATCH_IOU_THRESHOLD,
            "tie_break": "ground_truth_index_then_prediction_index",
        },
        "size_bucket_policy": asdict(size_policy),
        "aggregate": aggregate,
        "environment": {
            "device": runtime.device,
            "torch_version": str(torch.__version__),
        },
        "provenance": {
            "dataset_manifest_sha256": dataset_manifest_sha,
            "model_sha256": model_sha_before,
            "artifact_metadata_sha256": metadata_sha_before,
        },
        "created_at": created_at or datetime.now(UTC).isoformat(),
    }
    _write_jsonl(sample_analysis_path, analyses)
    _write_json(summary_path, summary)
    _write_json(per_class_path, aggregate["per_class"])
    _write_json(confidence_sweep_path, confidence_sweep)
    _write_json(
        error_taxonomy_path,
        {
            "schema_version": 1,
            "split": "val",
            "definitions": ERROR_TAXONOMY_DEFINITIONS,
            "main_counts": aggregate["error_taxonomy"]["main"],
            "secondary_counts": aggregate["error_taxonomy"]["secondary"],
        },
    )
    _write_json(hypotheses_path, hypotheses)

    visualization_paths: list[Path] = []
    analyses_by_sample = {analysis.sample_id: analysis for analysis in analyses}
    for analysis in select_visualization_samples(analyses):
        record = records_by_sample[analysis.sample_id]
        path = (
            output_dir
            / "visualizations"
            / f"{analysis.sample_id}_{analysis.main_error.lower()}.png"
        )
        save_failure_card(
            record=record,
            dataset_root=dataset_root,
            ground_truth=ground_truth_by_sample[analysis.sample_id],
            predictions=baseline_predictions[analysis.sample_id],
            analysis=analyses_by_sample[analysis.sample_id],
            classes=classes,
            output_path=path,
        )
        visualization_paths.append(path)
    render_validation_failure_galleries(
        records_by_sample=records_by_sample,
        dataset_root=dataset_root,
        ground_truth_by_sample=ground_truth_by_sample,
        predictions_by_sample=baseline_predictions,
        analyses=analyses,
        classes=classes,
        output_dir=output_dir / "visualizations" / "validation_failures",
        provenance=cast(dict[str, Any], summary["provenance"]),
    )
    return ErrorAnalysisArtifacts(
        output_dir=output_dir,
        sample_analysis_path=sample_analysis_path,
        summary_path=summary_path,
        per_class_path=per_class_path,
        confidence_sweep_path=confidence_sweep_path,
        error_taxonomy_path=error_taxonomy_path,
        hypotheses_path=hypotheses_path,
        visualization_paths=tuple(visualization_paths),
    )


# ADD 2026-08-26: Validation-only analysis source, runtime artifact와 output arguments를 정의한다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default="auto")
    return parser.parse_args()


# ADD 2026-08-26: Fixed baseline의 validation diagnostics를 실행하고 artifact 위치만 출력한다.
def main() -> int:
    args = parse_args()
    config = load_yolo_segmentation_config(args.config)
    artifacts = analyze_yolo_segmentation_errors(
        config=config,
        dataset_root=args.dataset,
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        requested_device=args.device,
    )
    print("YOLO segmentation validation error analysis: PASS")
    print(f"Summary: {artifacts.summary_path}")
    print(f"Samples: {artifacts.sample_analysis_path}")
    print(f"Visualizations: {len(artifacts.visualization_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
