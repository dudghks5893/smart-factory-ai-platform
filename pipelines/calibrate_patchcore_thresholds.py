"""Calibrate conservative PatchCore thresholds from normal validation predictions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ml.datasets.manifest import ManifestRecord, read_manifest_csv
from ml.datasets.manifest_validation import validate_manifest_records
from ml.evaluation.predictions import load_prediction_bundle
from ml.evaluation.thresholds import (
    THRESHOLDS_FILENAME,
    ThresholdArtifact,
    calibrate_max_normal_validation,
    write_threshold_artifact,
)
from ml.training.patchcore import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    read_artifact_metadata,
)
from shared.hashing import sha256_file

DEFAULT_THRESHOLD_ROOT = Path("outputs/evaluation/patchcore/thresholds")


@dataclass(frozen=True)
class ThresholdCalibrationSummary:
    """Summary of one completed validation-only threshold calibration."""

    output_dir: Path
    thresholds_path: Path
    thresholds: ThresholdArtifact


# ADD 2026-08-19: Normal-only validation prediction에서 fixed PatchCore threshold를 저장한다.
def calibrate_patchcore_thresholds(
    *,
    validation_predictions_path: Path,
    validation_anomaly_maps_path: Path,
    dataset_root: Path,
    manifest_path: Path,
    artifact_dir: Path,
    output_dir: Path,
) -> ThresholdCalibrationSummary:
    """Validate provenance and calibrate max-normal validation thresholds."""
    if output_dir.exists():
        raise FileExistsError(f"Threshold output directory already exists: {output_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Calibration manifest not found: {manifest_path}")

    # Artifact와 manifest provenance를 model loading 없이 먼저 검증한다.
    artifact_metadata = read_artifact_metadata(artifact_dir)
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != artifact_metadata.manifest_sha256:
        raise ValueError("Calibration manifest SHA-256 does not match the PatchCore artifact.")

    # Validation manifest record의 category, normal-only label과 file integrity를 검증한다.
    validation_records = [
        record for record in read_manifest_csv(manifest_path) if record.split == "validation"
    ]
    _validate_calibration_records(
        validation_records,
        dataset_root=dataset_root,
        category=artifact_metadata.category,
    )
    expected_map_shape = (1, *artifact_metadata.preprocessing.center_crop_size)

    # Prediction metadata와 tensor map을 validation manifest에 정확히 대조한다.
    predictions = load_prediction_bundle(
        predictions_path=validation_predictions_path,
        anomaly_maps_path=validation_anomaly_maps_path,
        expected_records=validation_records,
        expected_split="validation",
        expected_map_shape=expected_map_shape,
    )

    # Validation maxima와 모든 source artifact hash를 threshold provenance로 고정한다.
    thresholds = calibrate_max_normal_validation(
        predictions=predictions,
        artifact_metadata=artifact_metadata,
        artifact_metadata_sha256=sha256_file(artifact_dir / METADATA_FILENAME),
        model_sha256=sha256_file(artifact_dir / MODEL_FILENAME),
        validation_predictions_sha256=sha256_file(validation_predictions_path),
        validation_anomaly_maps_sha256=sha256_file(validation_anomaly_maps_path),
        created_at=datetime.now(UTC).isoformat(),
    )

    # Threshold output directory와 단일 JSON contract를 overwrite 없이 저장한다.
    output_dir.mkdir(parents=True, exist_ok=False)
    thresholds_path = output_dir / THRESHOLDS_FILENAME
    write_threshold_artifact(thresholds, thresholds_path)
    return ThresholdCalibrationSummary(
        output_dir=output_dir,
        thresholds_path=thresholds_path,
        thresholds=thresholds,
    )


# ADD 2026-08-19: Calibration manifest가 non-empty normal validation split인지 검증한다.
def _validate_calibration_records(
    records: list[ManifestRecord],
    *,
    dataset_root: Path,
    category: str,
) -> None:
    if not records:
        raise ValueError("Calibration validation manifest must not be empty.")
    if any(record.category != category for record in records):
        raise ValueError("Calibration records do not match the PatchCore artifact category.")
    if any(record.label != 0 for record in records):
        raise ValueError("max_normal_validation requires validation labels to all be 0.")

    report = validate_manifest_records(records, dataset_root)
    if not report.is_valid:
        raise ValueError("Calibration manifest validation failed:\n" + "\n".join(report.errors))


# ADD 2026-08-19: Threshold calibration CLI 입력을 정의하고 파싱한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate PatchCore thresholds from normal validation predictions."
    )
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--validation-anomaly-maps", type=Path, required=True)
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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_THRESHOLD_ROOT)
    parser.add_argument("--output-id", required=True)
    return parser.parse_args()


# ADD 2026-08-19: Validation calibration CLI 흐름을 조정하고 결과 경로를 출력한다.
def main() -> int:
    args = _parse_args()
    output_dir = args.output_root / args.output_id

    # Validation prediction만 사용해 threshold artifact를 생성한다.
    summary = calibrate_patchcore_thresholds(
        validation_predictions_path=args.validation_predictions,
        validation_anomaly_maps_path=args.validation_anomaly_maps,
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        artifact_dir=args.artifact_dir,
        output_dir=output_dir,
    )
    print("PatchCore threshold calibration: PASS")
    print(f"Strategy: {summary.thresholds.strategy}")
    print(f"Comparison: score {summary.thresholds.comparison_operator} threshold")
    print(f"Validation samples: {summary.thresholds.validation_sample_count}")
    print(f"Thresholds: {summary.thresholds_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
