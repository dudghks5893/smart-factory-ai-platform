"""Unit tests for environment-backed serving configuration."""

from pathlib import Path

import pytest

from services.api.config import ServingSettings


# ADD 2026-08-19: Required path와 optional device/upload environment 값을 parsing한다.
def test_serving_settings_load_from_environment() -> None:
    settings = ServingSettings.from_environment(
        {
            "PATCHCORE_ARTIFACT_DIR": "artifacts/model-a",
            "PATCHCORE_THRESHOLDS_PATH": "outputs/thresholds-a.json",
            "MODEL_DEVICE": "mps",
            "MAX_UPLOAD_BYTES": "2048",
        }
    )

    assert settings.artifact_dir == Path("artifacts/model-a")
    assert settings.thresholds_path == Path("outputs/thresholds-a.json")
    assert settings.model_device == "mps"
    assert settings.max_upload_bytes == 2048


# ADD 2026-08-19: Missing path와 unavailable-style invalid config를 startup 전에 거부한다.
def test_serving_settings_reject_missing_or_invalid_values() -> None:
    with pytest.raises(ValueError, match="PATCHCORE_ARTIFACT_DIR"):
        ServingSettings.from_environment({})
    with pytest.raises(ValueError, match="MODEL_DEVICE"):
        ServingSettings.from_environment(
            {
                "PATCHCORE_ARTIFACT_DIR": "artifact",
                "PATCHCORE_THRESHOLDS_PATH": "thresholds.json",
                "MODEL_DEVICE": "gpu",
            }
        )
    with pytest.raises(ValueError, match="MAX_UPLOAD_BYTES"):
        ServingSettings.from_environment(
            {
                "PATCHCORE_ARTIFACT_DIR": "artifact",
                "PATCHCORE_THRESHOLDS_PATH": "thresholds.json",
                "MAX_UPLOAD_BYTES": "0",
            }
        )
