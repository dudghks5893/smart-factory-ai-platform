"""Integration tests for optional YOLO segmentation FastAPI serving."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from torch import Tensor

from ml.training.yolo_segmentation import ARTIFACT_SCHEMA_VERSION, YoloArtifactMetadata
from services.api.app import create_app
from services.api.config import ServingSettings
from services.api.schemas import KnownDefectResponse
from services.inference.runtime import (
    InferenceResult,
    ModelRuntime,
    PatchCoreRuntimeConfig,
    ServingProvenance,
)
from services.inference.yolo_segmentation_runtime import (
    YoloSegmentationAdapter,
    YoloSegmentationInstance,
    YoloSegmentationProvenance,
    YoloSegmentationResult,
    YoloSegmentationRuntimeConfig,
)
from tests.persistence_helpers import prepare_sqlite_database

CLASSES = {0: "bent", 1: "color", 2: "scratch"}


class FakePatchCoreRuntime:
    """Existing endpoint fake used to detect PatchCore contract regressions."""

    model_name = "patchcore"
    category = "metal_nut"
    device = "cpu"
    provenance = ServingProvenance(
        manifest_sha256="a" * 64,
        artifact_metadata_sha256="b" * 64,
        model_sha256="c" * 64,
        threshold_artifact_sha256="d" * 64,
    )

    # ADD 2026-08-26: Existing PatchCore endpoint용 deterministic normal prediction을 반환한다.
    def predict(self, image: Tensor) -> InferenceResult:
        assert image.shape == (1, 3, 8, 8)
        return InferenceResult(
            model_name=self.model_name,
            category=self.category,
            is_anomaly=False,
            anomaly_score=30.0,
            threshold=40.0,
            comparison_operator=">",
        )


class FakeYoloRuntime:
    """Project-result fake without Ultralytics or real model files."""

    device = "cpu"

    # ADD 2026-08-26: Empty/multi-instance/failure behavior와 runtime identity를 구성한다.
    def __init__(self, *, empty: bool = False, fail: bool = False) -> None:
        self.metadata = _metadata()
        self.provenance = YoloSegmentationProvenance(
            dataset_manifest_sha256="a" * 64,
            dataset_semantic_fingerprint_sha256="b" * 64,
            artifact_metadata_sha256="c" * 64,
            model_sha256="d" * 64,
            framework_version="8.4.128",
        )
        self.empty = empty
        self.fail = fail
        self.predict_calls = 0

    # ADD 2026-08-26: RGB input/confidence를 확인하고 normalized result 또는 failure를 반환한다.
    def predict(
        self,
        image_rgb: np.ndarray,
        *,
        diagnostic_confidence: float,
    ) -> YoloSegmentationResult:
        self.predict_calls += 1
        if self.fail:
            raise RuntimeError("private accelerator stack detail")
        assert image_rgb.shape == (8, 8, 3)
        assert diagnostic_confidence == 0.25
        instances: tuple[YoloSegmentationInstance, ...] = ()
        if not self.empty:
            bent_mask = np.zeros((8, 8), dtype=np.bool_)
            bent_mask[1:5, 1:5] = True
            scratch_mask = np.zeros((8, 8), dtype=np.bool_)
            scratch_mask[4:7, 4:7] = True
            instances = (
                YoloSegmentationInstance(
                    class_id=0,
                    class_name="bent",
                    confidence=0.95,
                    box_xyxy=(1.0, 1.0, 5.0, 5.0),
                    mask=bent_mask,
                ),
                YoloSegmentationInstance(
                    class_id=2,
                    class_name="scratch",
                    confidence=0.75,
                    box_xyxy=(4.0, 4.0, 7.0, 7.0),
                    mask=scratch_mask,
                ),
            )
        return YoloSegmentationResult(
            image_width=8,
            image_height=8,
            device=self.device,
            inference_ms=12.5,
            instances=instances,
        )


# ADD 2026-08-26: Fake YOLO runtime metadata를 actual artifact schema로 생성한다.
def _metadata() -> YoloArtifactMetadata:
    return YoloArtifactMetadata(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        model_name="yolo11n-seg.pt",
        task="segment",
        architecture="yolo11n-seg",
        category="metal_nut",
        classes=CLASSES,
        seed=42,
        dataset_manifest_sha256="a" * 64,
        dataset_semantic_fingerprint_sha256="b" * 64,
        training_config={"training": {"imgsz": 640}},
        created_at="2026-08-25T00:00:00+00:00",
        framework="ultralytics",
        framework_version="8.4.128",
        torch_version="2.13.0",
        device="cuda:0",
        best_epoch=60,
        source_checkpoint="weights/best.pt",
        checkpoint_sha256="d" * 64,
    )


# ADD 2026-08-26: Optional YOLO enablement를 포함한 isolated SQLite app settings를 생성한다.
def _settings(tmp_path: Path, *, enabled: bool = True) -> ServingSettings:
    return ServingSettings(
        artifact_dir=tmp_path / "patchcore",
        thresholds_path=tmp_path / "thresholds.json",
        database_url=prepare_sqlite_database(tmp_path),
        model_device="cpu",
        yolo_segmentation_enabled=enabled,
        yolo_segmentation_artifact_dir=(tmp_path / "yolo" if enabled else None),
        yolo_segmentation_device="cpu",
        yolo_segmentation_diagnostic_confidence=0.25,
    )


# ADD 2026-08-26: In-memory multipart PNG payload를 생성한다.
def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(120, 80, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


# ADD 2026-08-26: Existing PatchCore fake loader를 application lifespan에 주입한다.
def _patchcore_loader(_config: PatchCoreRuntimeConfig) -> ModelRuntime:
    return FakePatchCoreRuntime()


# ADD 2026-08-26: Multi-instance summary와 singleton reuse를 HTTP boundary에서 검증한다.
def test_known_defect_endpoint_response_and_singleton_reuse(tmp_path: Path) -> None:
    runtime = FakeYoloRuntime()
    load_calls: list[YoloSegmentationRuntimeConfig] = []

    # Lifespan이 enabled model을 한 번만 load하는지 기록한다.
    def yolo_loader(config: YoloSegmentationRuntimeConfig) -> YoloSegmentationAdapter:
        load_calls.append(config)
        return cast(YoloSegmentationAdapter, runtime)

    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_patchcore_loader,
        yolo_runtime_loader=yolo_loader,
    )
    with TestClient(app) as client:
        ready = client.get("/ready")
        first = client.post(
            "/v1/known-defects",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )
        second = client.post(
            "/v1/known-defects",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )
        patchcore = client.post(
            "/v1/predictions",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )

    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "model_name": "patchcore",
        "category": "metal_nut",
        "device": "cpu",
    }
    assert first.status_code == second.status_code == 200
    payload = KnownDefectResponse.model_validate(first.json())
    assert payload.model.name == "yolo11n-seg.pt"
    assert payload.model.task == "segment"
    assert payload.image.model_dump() == {"width": 8, "height": 8}
    assert payload.diagnostic_confidence == 0.25
    assert payload.inference_ms == 12.5
    assert [instance.class_name for instance in payload.instances] == ["bent", "scratch"]
    assert payload.instances[0].confidence == 0.95
    assert payload.instances[0].box.model_dump() == {
        "x_min": 1.0,
        "y_min": 1.0,
        "x_max": 5.0,
        "y_max": 5.0,
    }
    assert payload.instances[0].mask.pixel_count == 16
    assert payload.instances[0].mask.area_ratio == 0.25
    assert set(first.json()["instances"][0]["mask"]) == {"pixel_count", "area_ratio"}
    assert "raw_mask" not in first.text
    assert "mask_pixels" not in first.text
    assert "decision" not in first.text
    assert len(load_calls) == 1
    assert runtime.predict_calls == 2

    # Existing PatchCore endpoint의 response shape와 persistence behavior를 유지한다.
    assert patchcore.status_code == 200
    assert set(patchcore.json()) == {
        "inspection_id",
        "model_name",
        "category",
        "is_anomaly",
        "anomaly_score",
        "threshold",
        "comparison_operator",
    }


# ADD 2026-08-26: Empty model prediction이 valid empty instances response인지 검증한다.
def test_known_defect_endpoint_empty_prediction(tmp_path: Path) -> None:
    runtime = FakeYoloRuntime(empty=True)
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_patchcore_loader,
        yolo_runtime_loader=lambda _config: cast(YoloSegmentationAdapter, runtime),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/known-defects",
            files={"image": ("good.png", _png_bytes(), "image/png")},
        )
    assert response.status_code == 200
    assert response.json()["instances"] == []


# ADD 2026-08-26: Malformed/empty/unsupported upload가 shared safe errors를 반환하는지 검증한다.
@pytest.mark.parametrize(
    ("content", "content_type", "status_code", "error_code"),
    [
        (b"not-an-image", "image/png", 400, "invalid_image"),
        (b"", "image/png", 400, "empty_image"),
        (_png_bytes(), "text/plain", 415, "unsupported_media_type"),
    ],
)
def test_known_defect_endpoint_reuses_upload_errors(
    tmp_path: Path,
    content: bytes,
    content_type: str,
    status_code: int,
    error_code: str,
) -> None:
    runtime = FakeYoloRuntime()
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_patchcore_loader,
        yolo_runtime_loader=lambda _config: cast(YoloSegmentationAdapter, runtime),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/known-defects",
            files={"image": ("sample.png", content, content_type)},
        )
    assert response.status_code == status_code
    assert response.json()["error"]["code"] == error_code
    assert runtime.predict_calls == 0


# ADD 2026-08-26: Runtime failure가 internal detail 없는 stable inference error인지 검증한다.
def test_known_defect_endpoint_hides_inference_failure(tmp_path: Path) -> None:
    runtime = FakeYoloRuntime(fail=True)
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_patchcore_loader,
        yolo_runtime_loader=lambda _config: cast(YoloSegmentationAdapter, runtime),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/known-defects",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )
    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "inference_failed",
        "message": "Model inference failed.",
    }
    assert "accelerator" not in response.text


# ADD 2026-08-26: Disabled YOLO가 load 없이 기존 readiness를 유지하고 endpoint는 503인지 검증한다.
def test_disabled_yolo_preserves_readiness_and_endpoint_is_unavailable(tmp_path: Path) -> None:
    yolo_load_count = 0

    # Disabled lifecycle에서 accidental model load를 탐지한다.
    def yolo_loader(_config: YoloSegmentationRuntimeConfig) -> YoloSegmentationAdapter:
        nonlocal yolo_load_count
        yolo_load_count += 1
        return cast(YoloSegmentationAdapter, FakeYoloRuntime())

    app = create_app(
        settings=_settings(tmp_path, enabled=False),
        runtime_loader=_patchcore_loader,
        yolo_runtime_loader=yolo_loader,
    )
    with TestClient(app) as client:
        ready = client.get("/ready")
        response = client.post(
            "/v1/known-defects",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )
    assert ready.status_code == 200
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "known_defect_model_disabled"
    assert yolo_load_count == 0


# ADD 2026-08-26: Enabled missing artifact가 silent disable 없이 startup을 중단하는지 검증한다.
def test_enabled_missing_yolo_artifact_fails_startup(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_patchcore_loader,
    )
    with pytest.raises(FileNotFoundError, match="model.pt and metadata.json"):
        with TestClient(app):
            pass


# ADD 2026-08-26: Enabled runtime loss가 health는 유지하되 readiness/endpoint를 503으로 전환한다.
def test_enabled_yolo_runtime_loss_changes_readiness(tmp_path: Path) -> None:
    runtime = FakeYoloRuntime()
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_patchcore_loader,
        yolo_runtime_loader=lambda _config: cast(YoloSegmentationAdapter, runtime),
    )
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        app.state.yolo_segmentation_runtime = None
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        endpoint = client.post(
            "/v1/known-defects",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )
    assert ready.status_code == endpoint.status_code == 503
    assert ready.json()["error"]["code"] == "known_defect_model_not_ready"
    assert endpoint.json()["error"]["code"] == "known_defect_model_not_ready"
