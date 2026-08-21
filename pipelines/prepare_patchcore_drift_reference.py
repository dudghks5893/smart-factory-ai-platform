"""Prepare a validation-normal PatchCore drift reference artifact."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ml.datasets.manifest import read_manifest_csv
from ml.drift.patchcore import (
    DEFAULT_PSI_BIN_COUNT,
    DEFAULT_PSI_EPSILON,
    REFERENCE_FILENAME,
    DriftReference,
    build_drift_reference,
    load_validation_normal_scores,
    validate_artifact_id,
    write_drift_reference,
)
from ml.evaluation.thresholds import read_threshold_artifact, validate_threshold_provenance
from ml.training.patchcore import METADATA_FILENAME, MODEL_FILENAME, read_artifact_metadata
from shared.hashing import sha256_file

DEFAULT_DRIFT_REFERENCE_ROOT = Path("outputs/drift/reference/patchcore")


@dataclass(frozen=True)
class DriftReferencePreparationSummary:
    """Paths and contract returned after one reference artifact is committed."""

    output_dir: Path
    reference_path: Path
    reference: DriftReference


# ADD 2026-08-21: Validation-normal score와 full artifact provenance로 drift reference를 저장한다.
def prepare_patchcore_drift_reference(
    *,
    validation_predictions_path: Path,
    artifact_dir: Path,
    thresholds_path: Path,
    manifest_path: Path,
    output_dir: Path,
    reference_id: str,
    psi_bin_count: int = DEFAULT_PSI_BIN_COUNT,
    psi_epsilon: float = DEFAULT_PSI_EPSILON,
    created_at: str | None = None,
) -> DriftReferencePreparationSummary:
    """Validate validation-only lineage and persist one immutable score baseline."""
    validate_artifact_id(reference_id)
    if output_dir.exists():
        raise FileExistsError(f"Drift reference output directory already exists: {output_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Drift reference manifest not found: {manifest_path}")

    # Threshold, model artifact와 exact manifest provenance를 model loading 없이 검증한다.
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
    validation_predictions_sha256 = sha256_file(validation_predictions_path)
    if validation_predictions_sha256 != thresholds.validation_predictions_sha256:
        raise ValueError("Reference predictions do not match threshold calibration provenance.")

    # Manifest validation split과 prediction metadata를 대조해 normal score만 추출한다.
    validation_records = [
        record for record in read_manifest_csv(manifest_path) if record.split == "validation"
    ]
    score_values = load_validation_normal_scores(
        validation_predictions_path,
        expected_records=validation_records,
        category=thresholds.category,
    )

    # Fixed PSI bins와 descriptive statistics를 full lineage reference로 고정한다.
    reference = build_drift_reference(
        reference_id=reference_id,
        thresholds=thresholds,
        threshold_artifact_sha256=sha256_file(thresholds_path),
        score_values=score_values,
        psi_bin_count=psi_bin_count,
        psi_epsilon=psi_epsilon,
        created_at=created_at or datetime.now(UTC).isoformat(),
    )

    # Reference directory와 JSON을 overwrite 없이 마지막 단계에서 저장한다.
    output_dir.mkdir(parents=True, exist_ok=False)
    reference_path = output_dir / REFERENCE_FILENAME
    write_drift_reference(reference, reference_path)
    return DriftReferencePreparationSummary(
        output_dir=output_dir,
        reference_path=reference_path,
        reference=reference,
    )


# ADD 2026-08-21: PatchCore drift reference CLI 입력과 PSI bin 설정을 정의한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a validation-normal PatchCore production drift reference."
    )
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DRIFT_REFERENCE_ROOT)
    parser.add_argument("--reference-id", required=True)
    parser.add_argument("--psi-bin-count", type=int, default=DEFAULT_PSI_BIN_COUNT)
    parser.add_argument("--psi-epsilon", type=float, default=DEFAULT_PSI_EPSILON)
    return parser.parse_args()


# ADD 2026-08-21: Validation reference 준비 흐름을 조정하고 artifact 경로를 출력한다.
def main() -> int:
    args = _parse_args()

    # Validation-only source와 calibration provenance에서 reference artifact를 생성한다.
    summary = prepare_patchcore_drift_reference(
        validation_predictions_path=args.validation_predictions,
        artifact_dir=args.artifact_dir,
        thresholds_path=args.thresholds,
        manifest_path=args.manifest,
        output_dir=args.output_root / args.reference_id,
        reference_id=args.reference_id,
        psi_bin_count=args.psi_bin_count,
        psi_epsilon=args.psi_epsilon,
    )
    print("PatchCore drift reference: PASS")
    print(f"Source: {summary.reference.source_split}/{summary.reference.source_label}")
    print(f"Samples: {summary.reference.sample_count}")
    print(f"Reference: {summary.reference_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
