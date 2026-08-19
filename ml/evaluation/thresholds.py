"""PatchCore max-normal validation threshold artifact contract."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.evaluation.metrics import COMPARISON_OPERATOR
from ml.evaluation.predictions import PredictionBundle
from ml.training.patchcore import PatchCoreArtifactMetadata

THRESHOLD_SCHEMA_VERSION = 1
THRESHOLD_STRATEGY = "max_normal_validation"
THRESHOLDS_FILENAME = "thresholds.json"


@dataclass(frozen=True)
class ThresholdArtifact:
    """Auditable fixed thresholds calibrated from normal validation records."""

    schema_version: int
    model_name: str
    category: str
    strategy: str
    comparison_operator: str
    image_threshold: float
    pixel_threshold: float
    validation_sample_count: int
    validation_pixel_count: int
    manifest_sha256: str
    artifact_metadata: PatchCoreArtifactMetadata
    artifact_metadata_sha256: str
    model_sha256: str
    validation_predictions_sha256: str
    validation_anomaly_maps_sha256: str
    created_at: str

    # ADD 2026-08-19: Threshold artifact를 inspectable JSON mapping으로 변환한다.
    def to_json_dict(self) -> dict[str, Any]:
        """Convert the threshold artifact to a stable JSON-compatible mapping."""
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "category": self.category,
            "strategy": self.strategy,
            "comparison_operator": self.comparison_operator,
            "image_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
            "validation_sample_count": self.validation_sample_count,
            "validation_pixel_count": self.validation_pixel_count,
            "manifest_sha256": self.manifest_sha256,
            "artifact_metadata": self.artifact_metadata.to_json_dict(),
            "artifact_metadata_sha256": self.artifact_metadata_sha256,
            "model_sha256": self.model_sha256,
            "validation_predictions_sha256": self.validation_predictions_sha256,
            "validation_anomaly_maps_sha256": self.validation_anomaly_maps_sha256,
            "created_at": self.created_at,
        }

    # ADD 2026-08-19: JSON threshold artifact를 typed contract로 검증해 복원한다.
    @classmethod
    def from_json_dict(cls, raw: object) -> ThresholdArtifact:
        """Validate and construct a threshold artifact loaded from JSON."""
        if not isinstance(raw, dict):
            raise ValueError("Threshold artifact root must be a mapping.")
        try:
            artifact = cls(
                schema_version=_required_integer(raw["schema_version"], "schema_version"),
                model_name=_required_string(raw["model_name"], "model_name"),
                category=_required_string(raw["category"], "category"),
                strategy=_required_string(raw["strategy"], "strategy"),
                comparison_operator=_required_string(
                    raw["comparison_operator"],
                    "comparison_operator",
                ),
                image_threshold=_required_float(raw["image_threshold"], "image_threshold"),
                pixel_threshold=_required_float(raw["pixel_threshold"], "pixel_threshold"),
                validation_sample_count=_required_integer(
                    raw["validation_sample_count"],
                    "validation_sample_count",
                ),
                validation_pixel_count=_required_integer(
                    raw["validation_pixel_count"],
                    "validation_pixel_count",
                ),
                manifest_sha256=_required_string(raw["manifest_sha256"], "manifest_sha256"),
                artifact_metadata=PatchCoreArtifactMetadata.from_json_dict(
                    raw["artifact_metadata"]
                ),
                artifact_metadata_sha256=_required_string(
                    raw["artifact_metadata_sha256"],
                    "artifact_metadata_sha256",
                ),
                model_sha256=_required_string(raw["model_sha256"], "model_sha256"),
                validation_predictions_sha256=_required_string(
                    raw["validation_predictions_sha256"],
                    "validation_predictions_sha256",
                ),
                validation_anomaly_maps_sha256=_required_string(
                    raw["validation_anomaly_maps_sha256"],
                    "validation_anomaly_maps_sha256",
                ),
                created_at=_required_string(raw["created_at"], "created_at"),
            )
        except KeyError as exc:
            raise ValueError("Threshold artifact is missing required fields.") from exc
        artifact.validate()
        return artifact

    # ADD 2026-08-19: Threshold policy, provenance, count와 finite value invariant를 검증한다.
    def validate(self) -> None:
        """Validate fixed-threshold policy and provenance invariants."""
        if self.schema_version != THRESHOLD_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported threshold schema_version: {self.schema_version}; "
                f"expected {THRESHOLD_SCHEMA_VERSION}."
            )
        if self.strategy != THRESHOLD_STRATEGY:
            raise ValueError(f"Unsupported threshold strategy: {self.strategy}")
        if self.comparison_operator != COMPARISON_OPERATOR:
            raise ValueError(f"Threshold comparison_operator must be '{COMPARISON_OPERATOR}'.")
        if not math.isfinite(self.image_threshold) or not math.isfinite(self.pixel_threshold):
            raise ValueError("Threshold values must be finite.")
        if self.validation_sample_count <= 0 or self.validation_pixel_count <= 0:
            raise ValueError("Threshold validation counts must be positive.")
        self.artifact_metadata.validate()
        if self.model_name != self.artifact_metadata.model_name:
            raise ValueError("Threshold model_name does not match artifact metadata.")
        if self.category != self.artifact_metadata.category:
            raise ValueError("Threshold category does not match artifact metadata.")
        if self.manifest_sha256 != self.artifact_metadata.manifest_sha256:
            raise ValueError("Threshold manifest SHA-256 does not match artifact metadata.")
        for field, value in (
            ("manifest_sha256", self.manifest_sha256),
            ("artifact_metadata_sha256", self.artifact_metadata_sha256),
            ("model_sha256", self.model_sha256),
            ("validation_predictions_sha256", self.validation_predictions_sha256),
            ("validation_anomaly_maps_sha256", self.validation_anomaly_maps_sha256),
        ):
            _validate_sha256(value, field)


# ADD 2026-08-19: Normal-only validation score의 maxima로 conservative threshold를 생성한다.
def calibrate_max_normal_validation(
    *,
    predictions: PredictionBundle,
    artifact_metadata: PatchCoreArtifactMetadata,
    artifact_metadata_sha256: str,
    model_sha256: str,
    validation_predictions_sha256: str,
    validation_anomaly_maps_sha256: str,
    created_at: str,
) -> ThresholdArtifact:
    """Calibrate image and pixel thresholds from normal validation maxima only."""
    if not predictions.records:
        raise ValueError("Validation predictions must not be empty.")
    if any(record.split != "validation" for record in predictions.records):
        raise ValueError("Threshold calibration accepts only validation predictions.")
    if any(record.label != 0 for record in predictions.records):
        raise ValueError(
            "max_normal_validation requires every validation prediction label to be 0."
        )

    artifact = ThresholdArtifact(
        schema_version=THRESHOLD_SCHEMA_VERSION,
        model_name=artifact_metadata.model_name,
        category=artifact_metadata.category,
        strategy=THRESHOLD_STRATEGY,
        comparison_operator=COMPARISON_OPERATOR,
        image_threshold=float(predictions.scores.max().item()),
        pixel_threshold=float(predictions.anomaly_maps.max().item()),
        validation_sample_count=len(predictions.records),
        validation_pixel_count=predictions.anomaly_maps.numel(),
        manifest_sha256=artifact_metadata.manifest_sha256,
        artifact_metadata=artifact_metadata,
        artifact_metadata_sha256=artifact_metadata_sha256,
        model_sha256=model_sha256,
        validation_predictions_sha256=validation_predictions_sha256,
        validation_anomaly_maps_sha256=validation_anomaly_maps_sha256,
        created_at=created_at,
    )
    artifact.validate()
    return artifact


# ADD 2026-08-19: Threshold JSON을 검증해 evaluation contract로 복원한다.
def read_threshold_artifact(path: Path) -> ThresholdArtifact:
    """Read and validate a threshold artifact without recalibrating it."""
    if not path.is_file():
        raise FileNotFoundError(f"Threshold artifact not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read threshold artifact: {path}") from exc
    return ThresholdArtifact.from_json_dict(raw)


# ADD 2026-08-19: Threshold artifact를 overwrite 없이 deterministic JSON으로 저장한다.
def write_threshold_artifact(artifact: ThresholdArtifact, path: Path) -> None:
    """Persist one threshold artifact without overwriting an existing file."""
    artifact.validate()
    if path.exists():
        raise FileExistsError(f"Threshold artifact already exists: {path}")
    path.write_text(
        json.dumps(
            artifact.to_json_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


# ADD 2026-08-19: Threshold JSON field가 non-empty string인지 검증한다.
def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Threshold field '{field}' must be a non-empty string.")
    return value


# ADD 2026-08-19: Threshold JSON field가 bool이 아닌 integer인지 검증한다.
def _required_integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"Threshold field '{field}' must be an integer.")
    return value


# ADD 2026-08-19: Threshold JSON field가 finite number인지 검증한다.
def _required_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Threshold field '{field}' must be a finite number.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"Threshold field '{field}' must be a finite number.")
    return converted


# ADD 2026-08-19: Provenance field가 SHA-256 hex digest인지 검증한다.
def _validate_sha256(value: str, field: str) -> None:
    if len(value) != 64:
        raise ValueError(f"Threshold {field} must be a SHA-256 hex digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"Threshold {field} must be a SHA-256 hex digest.") from exc
