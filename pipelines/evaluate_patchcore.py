"""Evaluate PatchCore test predictions with a fixed validation threshold artifact."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from ml.datasets.dataset import MVTecManifestDataset
from ml.datasets.manifest import ManifestRecord, read_manifest_csv
from ml.datasets.manifest_validation import validate_manifest_records
from ml.evaluation.metrics import (
    COMPARISON_OPERATOR,
    EVALUATION_SCHEMA_VERSION,
    calculate_auroc,
    calculate_binary_metrics,
    calculate_per_defect_diagnostics,
)
from ml.evaluation.pixel import load_aligned_ground_truth_masks
from ml.evaluation.predictions import load_prediction_bundle
from ml.evaluation.thresholds import (
    ThresholdArtifact,
    read_threshold_artifact,
    validate_threshold_provenance,
)
from ml.training.patchcore import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    read_artifact_metadata,
)
from ml.training.preprocessing import PatchCorePreprocessor
from shared.hashing import sha256_file

METRICS_FILENAME = "metrics.json"
PER_DEFECT_METRICS_FILENAME = "per_defect_metrics.json"
DEFAULT_EVALUATION_ROOT = Path("outputs/evaluation/patchcore/metrics")


@dataclass(frozen=True)
class EvaluationSummary:
    """Summary of persisted fixed-threshold test evaluation outputs."""

    output_dir: Path
    metrics_path: Path
    per_defect_metrics_path: Path
    image_auroc: float
    pixel_auroc: float


# ADD 2026-08-19: Stored validation threshold로 PatchCore test prediction을 평가한다.
# MODIFY 2026-08-19: Pipeline-local provenance 검사 → threshold domain validator를 재사용한다.
def evaluate_patchcore(
    *,
    test_predictions_path: Path,
    test_anomaly_maps_path: Path,
    thresholds_path: Path,
    dataset_root: Path,
    manifest_path: Path,
    artifact_dir: Path,
    output_dir: Path,
    batch_size: int = 4,
    num_workers: int = 0,
) -> EvaluationSummary:
    """Evaluate test predictions without recalculating or moving thresholds."""
    if output_dir.exists():
        raise FileExistsError(f"Evaluation output directory already exists: {output_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest_path}")

    # Threshold, artifact와 manifest hash를 비교해 calibration provenance를 고정한다.
    thresholds = read_threshold_artifact(thresholds_path)
    artifact_metadata = read_artifact_metadata(artifact_dir)
    manifest_sha256 = sha256_file(manifest_path)
    artifact_metadata_sha256 = sha256_file(artifact_dir / METADATA_FILENAME)
    model_sha256 = sha256_file(artifact_dir / MODEL_FILENAME)
    validate_threshold_provenance(
        thresholds,
        artifact_metadata=artifact_metadata,
        manifest_sha256=manifest_sha256,
        artifact_metadata_sha256=artifact_metadata_sha256,
        model_sha256=model_sha256,
    )

    # Test manifest의 category, label/mask path와 image integrity를 검증한다.
    test_records = [record for record in read_manifest_csv(manifest_path) if record.split == "test"]
    _validate_test_records(
        test_records,
        dataset_root=dataset_root,
        category=artifact_metadata.category,
    )
    expected_map_shape = (1, *artifact_metadata.preprocessing.center_crop_size)

    # Test raw prediction을 manifest 순서로 검증해 contiguous score/map tensor로 로드한다.
    predictions = load_prediction_bundle(
        predictions_path=test_predictions_path,
        anomaly_maps_path=test_anomaly_maps_path,
        expected_records=test_records,
        expected_split="test",
        expected_map_shape=expected_map_shape,
    )
    image_labels = predictions.labels()
    image_auroc = calculate_auroc(image_labels, predictions.scores)
    image_metrics = calculate_binary_metrics(
        image_labels,
        predictions.scores,
        thresholds.image_threshold,
    )

    # 기존 PatchCorePreprocessor로 ground-truth mask를 anomaly map geometry에 정렬한다.
    dataset = MVTecManifestDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="test",
    )
    ground_truth_masks = load_aligned_ground_truth_masks(
        dataset=dataset,
        preprocessor=PatchCorePreprocessor(artifact_metadata.preprocessing),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if ground_truth_masks.shape != predictions.anomaly_maps.shape:
        raise ValueError(
            "Ground-truth mask and anomaly map shape mismatch: "
            f"masks={tuple(ground_truth_masks.shape)}, "
            f"maps={tuple(predictions.anomaly_maps.shape)}"
        )

    # 전체 test pixel을 contiguous vector view로 계산해 Python object copy를 피한다.
    pixel_labels = ground_truth_masks.to(dtype=torch.uint8).reshape(-1)
    pixel_scores = predictions.anomaly_maps.reshape(-1)
    pixel_auroc = calculate_auroc(pixel_labels, pixel_scores)
    pixel_metrics = calculate_binary_metrics(
        pixel_labels,
        pixel_scores,
        thresholds.pixel_threshold,
    )
    per_defect = calculate_per_defect_diagnostics(
        predictions.records,
        predictions.scores,
        thresholds.image_threshold,
    )

    # Fixed-threshold metric과 모든 input hash를 deterministic JSON payload로 구성한다.
    metrics = _build_metrics_payload(
        thresholds=thresholds,
        thresholds_sha256=sha256_file(thresholds_path),
        manifest_sha256=manifest_sha256,
        artifact_metadata_sha256=artifact_metadata_sha256,
        model_sha256=model_sha256,
        test_predictions_sha256=sha256_file(test_predictions_path),
        test_anomaly_maps_sha256=sha256_file(test_anomaly_maps_path),
        image_auroc=image_auroc,
        image_metrics=image_metrics.to_json_dict(),
        pixel_auroc=pixel_auroc,
        pixel_metrics=pixel_metrics.to_json_dict(),
        per_defect=per_defect,
        sample_count=len(predictions.records),
        pixel_count=pixel_labels.numel(),
        created_at=datetime.now(UTC).isoformat(),
    )

    # 전체 metrics와 per-defect view를 overwrite 없이 함께 저장한다.
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / METRICS_FILENAME
    per_defect_metrics_path = output_dir / PER_DEFECT_METRICS_FILENAME
    _write_json(metrics_path, metrics)
    _write_json(per_defect_metrics_path, per_defect)
    return EvaluationSummary(
        output_dir=output_dir,
        metrics_path=metrics_path,
        per_defect_metrics_path=per_defect_metrics_path,
        image_auroc=image_auroc,
        pixel_auroc=pixel_auroc,
    )


# ADD 2026-08-19: Test manifest가 non-empty artifact category record set인지 검증한다.
def _validate_test_records(
    records: list[ManifestRecord],
    *,
    dataset_root: Path,
    category: str,
) -> None:
    if not records:
        raise ValueError("Evaluation test manifest must not be empty.")
    if any(record.category != category for record in records):
        raise ValueError("Test records do not match the PatchCore artifact category.")
    report = validate_manifest_records(records, dataset_root)
    if not report.is_valid:
        raise ValueError("Evaluation manifest validation failed:\n" + "\n".join(report.errors))


# ADD 2026-08-19: Evaluation metric과 provenance를 schema-versioned payload로 구성한다.
def _build_metrics_payload(
    *,
    thresholds: ThresholdArtifact,
    thresholds_sha256: str,
    manifest_sha256: str,
    artifact_metadata_sha256: str,
    model_sha256: str,
    test_predictions_sha256: str,
    test_anomaly_maps_sha256: str,
    image_auroc: float,
    image_metrics: dict[str, Any],
    pixel_auroc: float,
    pixel_metrics: dict[str, Any],
    per_defect: dict[str, dict[str, int | float]],
    sample_count: int,
    pixel_count: int,
    created_at: str,
) -> dict[str, Any]:
    image_level = {
        "auroc": image_auroc,
        "threshold": thresholds.image_threshold,
        "comparison_operator": COMPARISON_OPERATOR,
        **image_metrics,
    }
    pixel_level = {
        "auroc": pixel_auroc,
        "threshold": thresholds.pixel_threshold,
        "comparison_operator": COMPARISON_OPERATOR,
        **pixel_metrics,
    }
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "category": thresholds.category,
        "threshold_artifact": {
            "schema_version": thresholds.schema_version,
            "sha256": thresholds_sha256,
            "strategy": thresholds.strategy,
            "comparison_operator": thresholds.comparison_operator,
            "image_threshold": thresholds.image_threshold,
            "pixel_threshold": thresholds.pixel_threshold,
        },
        "provenance": {
            "manifest_sha256": manifest_sha256,
            "artifact_metadata_sha256": artifact_metadata_sha256,
            "model_sha256": model_sha256,
            "validation_predictions_sha256": thresholds.validation_predictions_sha256,
            "validation_anomaly_maps_sha256": thresholds.validation_anomaly_maps_sha256,
            "test_predictions_sha256": test_predictions_sha256,
            "test_anomaly_maps_sha256": test_anomaly_maps_sha256,
        },
        "sample_counts": {
            "total": sample_count,
            "normal": image_metrics["normal_support"],
            "anomaly": image_metrics["anomaly_support"],
        },
        "pixel_counts": {
            "total": pixel_count,
            "normal": pixel_metrics["normal_support"],
            "anomaly": pixel_metrics["anomaly_support"],
        },
        "image_level": image_level,
        "pixel_level": pixel_level,
        "per_defect": per_defect,
        "created_at": created_at,
    }


# ADD 2026-08-19: Evaluation payload를 finite deterministic JSON으로 저장한다.
def _write_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"Evaluation output already exists: {path}")
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


# ADD 2026-08-19: PatchCore fixed-threshold evaluation CLI 입력을 정의한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PatchCore test predictions with stored validation thresholds."
    )
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--test-anomaly-maps", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/mvtec_ad"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/interim/manifests/mvtec_ad_metal_nut.csv"),
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


# ADD 2026-08-19: Stored threshold evaluation CLI 흐름을 조정하고 metric 경로를 출력한다.
def main() -> int:
    args = _parse_args()
    output_dir = args.output_root / args.output_id

    # Test prediction을 threshold 재계산 없이 평가하고 metric artifact를 저장한다.
    summary = evaluate_patchcore(
        test_predictions_path=args.test_predictions,
        test_anomaly_maps_path=args.test_anomaly_maps,
        thresholds_path=args.thresholds,
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        artifact_dir=args.artifact_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print("PatchCore evaluation: PASS")
    print(f"Image AUROC: {summary.image_auroc:.6f}")
    print(f"Pixel AUROC: {summary.pixel_auroc:.6f}")
    print(f"Metrics: {summary.metrics_path}")
    print(f"Per-defect metrics: {summary.per_defect_metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
