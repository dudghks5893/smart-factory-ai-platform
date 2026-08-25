"""Environment-backed configuration for the PatchCore serving process."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from ml.training.device import SUPPORTED_DEVICES
from services.inference.runtime import PatchCoreRuntimeConfig
from services.inference.yolo_segmentation_runtime import (
    YoloSegmentationRuntimeConfig,
    validate_diagnostic_confidence,
)

DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_YOLO_SEGMENTATION_DIAGNOSTIC_CONFIDENCE = 0.25


# ADD 2026-08-20: CLI와 application이 같은 required DATABASE_URL environment 계약을 사용한다.
def required_database_url(environ: Mapping[str, str] | None = None) -> str:
    """Return a non-empty database URL without logging credentials."""
    values = os.environ if environ is None else environ
    database_url = values.get("DATABASE_URL", "").strip()
    if not database_url:
        raise ValueError("DATABASE_URL is required.")
    return database_url


# ADD 2026-08-26: Environment boolean을 ambiguous truthiness 없이 strict parsing한다.
def _environment_boolean(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{name} must be 'true' or 'false'.")


@dataclass(frozen=True)
class ServingSettings:
    """Validated application settings kept separate from experiment configuration."""

    artifact_dir: Path
    thresholds_path: Path
    database_url: str
    model_device: str = "auto"
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    yolo_segmentation_enabled: bool = False
    yolo_segmentation_artifact_dir: Path | None = None
    yolo_segmentation_device: str = "auto"
    yolo_segmentation_diagnostic_confidence: float = DEFAULT_YOLO_SEGMENTATION_DIAGNOSTIC_CONFIDENCE

    # ADD 2026-08-19: Environment mapping에서 serving path/device/upload 설정을 로드한다.
    # MODIFY 2026-08-20: Required DATABASE_URL과 PostgreSQL/SQLite driver validation을 추가한다.
    # MODIFY 2026-08-26: Optional YOLO enablement/artifact/device/confidence 설정을 추가한다.
    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> ServingSettings:
        """Load serving settings and reject missing required artifact paths."""
        values = os.environ if environ is None else environ
        missing = [
            name
            for name in (
                "PATCHCORE_ARTIFACT_DIR",
                "PATCHCORE_THRESHOLDS_PATH",
                "DATABASE_URL",
            )
            if not values.get(name, "").strip()
        ]
        if missing:
            raise ValueError(
                "Missing required serving environment variables: " + ", ".join(missing)
            )
        try:
            max_upload_bytes = int(values.get("MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES)))
        except ValueError as exc:
            raise ValueError("MAX_UPLOAD_BYTES must be an integer.") from exc
        try:
            yolo_confidence = float(
                values.get(
                    "YOLO_SEGMENTATION_CONFIDENCE",
                    str(DEFAULT_YOLO_SEGMENTATION_DIAGNOSTIC_CONFIDENCE),
                )
            )
        except ValueError as exc:
            raise ValueError("YOLO_SEGMENTATION_CONFIDENCE must be numeric.") from exc
        yolo_enabled = _environment_boolean(
            values.get("YOLO_SEGMENTATION_ENABLED", "false"),
            name="YOLO_SEGMENTATION_ENABLED",
        )
        yolo_artifact_value = values.get("YOLO_SEGMENTATION_ARTIFACT_DIR", "").strip()

        settings = cls(
            artifact_dir=Path(values["PATCHCORE_ARTIFACT_DIR"]),
            thresholds_path=Path(values["PATCHCORE_THRESHOLDS_PATH"]),
            database_url=required_database_url(values),
            model_device=values.get("MODEL_DEVICE", "auto"),
            max_upload_bytes=max_upload_bytes,
            yolo_segmentation_enabled=yolo_enabled,
            yolo_segmentation_artifact_dir=(
                Path(yolo_artifact_value) if yolo_artifact_value else None
            ),
            yolo_segmentation_device=values.get("YOLO_SEGMENTATION_DEVICE", "auto"),
            yolo_segmentation_diagnostic_confidence=yolo_confidence,
        )
        settings.validate()
        return settings

    # ADD 2026-08-19: Serving device와 upload size invariant를 검증한다.
    # MODIFY 2026-08-20: Required database URL과 psycopg 3 production driver 계약을 검증한다.
    # MODIFY 2026-08-26: Enabled YOLO runtime의 required path와 operating point를 검증한다.
    def validate(self) -> None:
        """Validate settings before startup accesses model files."""
        if self.model_device not in SUPPORTED_DEVICES:
            raise ValueError(f"MODEL_DEVICE must be one of {SUPPORTED_DEVICES}.")
        if self.max_upload_bytes <= 0:
            raise ValueError("MAX_UPLOAD_BYTES must be positive.")
        if self.yolo_segmentation_device not in SUPPORTED_DEVICES:
            raise ValueError(f"YOLO_SEGMENTATION_DEVICE must be one of {SUPPORTED_DEVICES}.")
        if self.yolo_segmentation_enabled and self.yolo_segmentation_artifact_dir is None:
            raise ValueError(
                "YOLO_SEGMENTATION_ARTIFACT_DIR is required when YOLO segmentation is enabled."
            )
        validate_diagnostic_confidence(self.yolo_segmentation_diagnostic_confidence)
        try:
            driver_name = make_url(self.database_url).drivername
        except (ArgumentError, TypeError, ValueError) as exc:
            raise ValueError("DATABASE_URL must be a valid SQLAlchemy URL.") from exc
        if driver_name != "postgresql+psycopg" and not driver_name.startswith("sqlite"):
            raise ValueError("DATABASE_URL must use postgresql+psycopg or SQLite for tests.")

    # ADD 2026-08-19: API settings에서 transport-independent runtime config를 생성한다.
    def runtime_config(self) -> PatchCoreRuntimeConfig:
        """Return only the settings required by the inference layer."""
        self.validate()
        return PatchCoreRuntimeConfig(
            artifact_dir=self.artifact_dir,
            thresholds_path=self.thresholds_path,
            device=self.model_device,
        )

    # ADD 2026-08-26: Enabled YOLO settings에서 transport-independent runtime config를 생성한다.
    def yolo_segmentation_runtime_config(self) -> YoloSegmentationRuntimeConfig:
        """Return the required YOLO artifact/device settings or reject disabled use."""
        self.validate()
        if not self.yolo_segmentation_enabled or self.yolo_segmentation_artifact_dir is None:
            raise ValueError("YOLO segmentation runtime is not enabled.")
        return YoloSegmentationRuntimeConfig(
            artifact_dir=self.yolo_segmentation_artifact_dir,
            device=self.yolo_segmentation_device,
        )
