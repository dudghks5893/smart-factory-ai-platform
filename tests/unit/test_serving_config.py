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
            "DATABASE_URL": "postgresql+psycopg://user:password@localhost/database",
            "MODEL_DEVICE": "mps",
            "MAX_UPLOAD_BYTES": "2048",
        }
    )

    assert settings.artifact_dir == Path("artifacts/model-a")
    assert settings.thresholds_path == Path("outputs/thresholds-a.json")
    assert settings.database_url.startswith("postgresql+psycopg://")
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
                "DATABASE_URL": "sqlite+pysqlite:///:memory:",
                "MODEL_DEVICE": "gpu",
            }
        )
    with pytest.raises(ValueError, match="MAX_UPLOAD_BYTES"):
        ServingSettings.from_environment(
            {
                "PATCHCORE_ARTIFACT_DIR": "artifact",
                "PATCHCORE_THRESHOLDS_PATH": "thresholds.json",
                "DATABASE_URL": "sqlite+pysqlite:///:memory:",
                "MAX_UPLOAD_BYTES": "0",
            }
        )
    with pytest.raises(ValueError, match="DATABASE_URL"):
        ServingSettings.from_environment(
            {
                "PATCHCORE_ARTIFACT_DIR": "artifact",
                "PATCHCORE_THRESHOLDS_PATH": "thresholds.json",
                "DATABASE_URL": "postgresql://user:password@localhost/database",
            }
        )


# ADD 2026-08-26: Optional YOLO environment를 enabled runtime config로 strict parsing한다.
def test_serving_settings_parse_enabled_yolo_segmentation() -> None:
    settings = ServingSettings.from_environment(
        {
            "PATCHCORE_ARTIFACT_DIR": "artifacts/patchcore",
            "PATCHCORE_THRESHOLDS_PATH": "outputs/thresholds.json",
            "DATABASE_URL": "sqlite+pysqlite:///:memory:",
            "YOLO_SEGMENTATION_ENABLED": "true",
            "YOLO_SEGMENTATION_ARTIFACT_DIR": "artifacts/yolo-runtime",
            "YOLO_SEGMENTATION_DEVICE": "mps",
            "YOLO_SEGMENTATION_CONFIDENCE": "0.25",
        }
    )
    runtime_config = settings.yolo_segmentation_runtime_config()
    assert settings.yolo_segmentation_enabled is True
    assert runtime_config.artifact_dir == Path("artifacts/yolo-runtime")
    assert runtime_config.device == "mps"
    assert settings.yolo_segmentation_diagnostic_confidence == 0.25


# ADD 2026-08-26: Disabled default와 enabled missing/invalid YOLO settings를 검증한다.
def test_serving_settings_validate_yolo_enablement_policy() -> None:
    disabled = ServingSettings.from_environment(
        {
            "PATCHCORE_ARTIFACT_DIR": "artifacts/patchcore",
            "PATCHCORE_THRESHOLDS_PATH": "outputs/thresholds.json",
            "DATABASE_URL": "sqlite+pysqlite:///:memory:",
        }
    )
    assert disabled.yolo_segmentation_enabled is False
    with pytest.raises(ValueError, match="not enabled"):
        disabled.yolo_segmentation_runtime_config()

    base = {
        "PATCHCORE_ARTIFACT_DIR": "artifacts/patchcore",
        "PATCHCORE_THRESHOLDS_PATH": "outputs/thresholds.json",
        "DATABASE_URL": "sqlite+pysqlite:///:memory:",
        "YOLO_SEGMENTATION_ENABLED": "true",
    }
    with pytest.raises(ValueError, match="YOLO_SEGMENTATION_ARTIFACT_DIR"):
        ServingSettings.from_environment(base)
    with pytest.raises(ValueError, match="true.*false"):
        ServingSettings.from_environment({**base, "YOLO_SEGMENTATION_ENABLED": "yes"})
    with pytest.raises(ValueError, match="YOLO_SEGMENTATION_DEVICE"):
        ServingSettings.from_environment(
            {
                **base,
                "YOLO_SEGMENTATION_ARTIFACT_DIR": "artifact",
                "YOLO_SEGMENTATION_DEVICE": "metal",
            }
        )
    with pytest.raises(ValueError, match="Diagnostic confidence"):
        ServingSettings.from_environment(
            {
                **base,
                "YOLO_SEGMENTATION_ARTIFACT_DIR": "artifact",
                "YOLO_SEGMENTATION_CONFIDENCE": "1.0",
            }
        )
