"""Environment-backed configuration for the PatchCore serving process."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ml.training.device import SUPPORTED_DEVICES
from services.inference.runtime import PatchCoreRuntimeConfig

DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class ServingSettings:
    """Validated application settings kept separate from experiment configuration."""

    artifact_dir: Path
    thresholds_path: Path
    model_device: str = "auto"
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES

    # ADD 2026-08-19: Environment mapping에서 serving path/device/upload 설정을 로드한다.
    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> ServingSettings:
        """Load serving settings and reject missing required artifact paths."""
        values = os.environ if environ is None else environ
        missing = [
            name
            for name in ("PATCHCORE_ARTIFACT_DIR", "PATCHCORE_THRESHOLDS_PATH")
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

        settings = cls(
            artifact_dir=Path(values["PATCHCORE_ARTIFACT_DIR"]),
            thresholds_path=Path(values["PATCHCORE_THRESHOLDS_PATH"]),
            model_device=values.get("MODEL_DEVICE", "auto"),
            max_upload_bytes=max_upload_bytes,
        )
        settings.validate()
        return settings

    # ADD 2026-08-19: Serving device와 upload size invariant를 검증한다.
    def validate(self) -> None:
        """Validate settings before startup accesses model files."""
        if self.model_device not in SUPPORTED_DEVICES:
            raise ValueError(f"MODEL_DEVICE must be one of {SUPPORTED_DEVICES}.")
        if self.max_upload_bytes <= 0:
            raise ValueError("MAX_UPLOAD_BYTES must be positive.")

    # ADD 2026-08-19: API settings에서 transport-independent runtime config를 생성한다.
    def runtime_config(self) -> PatchCoreRuntimeConfig:
        """Return only the settings required by the inference layer."""
        self.validate()
        return PatchCoreRuntimeConfig(
            artifact_dir=self.artifact_dir,
            thresholds_path=self.thresholds_path,
            device=self.model_device,
        )
