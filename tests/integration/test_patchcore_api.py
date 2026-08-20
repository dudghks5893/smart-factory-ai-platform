"""Integration tests for the FastAPI PatchCore serving contract."""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from torch import Tensor

from services.api.app import create_app
from services.api.config import ServingSettings
from services.inference.runtime import (
    InferenceResult,
    ModelRuntime,
    PatchCoreRuntimeConfig,
    ServingProvenance,
)
from tests.persistence_helpers import prepare_sqlite_database


class _FakeRuntime:
    """Small injectable image-level runtime with deterministic scores."""

    model_name = "patchcore"
    category = "metal_nut"
    device = "cpu"
    provenance = ServingProvenance(
        manifest_sha256="a" * 64,
        artifact_metadata_sha256="b" * 64,
        model_sha256="c" * 64,
        threshold_artifact_sha256="d" * 64,
    )

    # ADD 2026-08-19: API integration fake의 score, threshold와 failure 상태를 초기화한다.
    def __init__(self, *, score: float = 50.0, threshold: float = 40.0, fail: bool = False):
        self.score = score
        self.threshold = threshold
        self.fail = fail
        self.predict_calls = 0

    # ADD 2026-08-19: Strict threshold 결과를 반환하거나 test inference failure를 발생시킨다.
    def predict(self, image: Tensor) -> InferenceResult:
        self.predict_calls += 1
        if self.fail:
            raise RuntimeError("internal model details")
        assert image.shape == (1, 3, 8, 8)
        return InferenceResult(
            model_name=self.model_name,
            category=self.category,
            is_anomaly=self.score > self.threshold,
            anomaly_score=self.score,
            threshold=self.threshold,
            comparison_operator=">",
        )


# ADD 2026-08-19: Test app용 artifact/device/upload settings를 생성한다.
def _settings(tmp_path: Path, *, max_upload_bytes: int = 1024 * 1024) -> ServingSettings:
    return ServingSettings(
        artifact_dir=tmp_path / "artifact",
        thresholds_path=tmp_path / "thresholds.json",
        database_url=prepare_sqlite_database(tmp_path),
        model_device="cpu",
        max_upload_bytes=max_upload_bytes,
    )


# ADD 2026-08-19: In-memory PNG upload payload를 생성한다.
def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(120, 80, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


# ADD 2026-08-19: Fake runtime을 반환하며 startup load 횟수를 기록하는 loader를 생성한다.
def _runtime_loader(
    runtime: _FakeRuntime,
    calls: list[PatchCoreRuntimeConfig] | None = None,
) -> Callable[[PatchCoreRuntimeConfig], ModelRuntime]:
    # ADD 2026-08-19: Lifespan이 전달한 runtime config와 load 호출을 기록한다.
    def load(config: PatchCoreRuntimeConfig) -> _FakeRuntime:
        if calls is not None:
            calls.append(config)
        return runtime

    return load


# ADD 2026-08-19: Health와 loaded model readiness의 서로 다른 의미를 검증한다.
def test_health_and_ready_when_runtime_is_loaded(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(runtime),
    )

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready")

    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "model_name": "patchcore",
        "category": "metal_nut",
        "device": "cpu",
    }


# ADD 2026-08-19: Lifespan 전 process는 live이지만 model readiness는 503인지 검증한다.
def test_ready_when_runtime_is_unavailable(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(_FakeRuntime()),
    )
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "model_not_ready",
            "message": "Model runtime is not ready.",
        }
    }


# ADD 2026-08-19: Image score가 threshold보다 크거나 같거나 작은 판정 경계를 검증한다.
@pytest.mark.parametrize(
    ("score", "expected_is_anomaly"),
    [(40.1, True), (40.0, False), (39.9, False)],
)
def test_valid_image_uses_strict_threshold(
    tmp_path: Path,
    score: float,
    expected_is_anomaly: bool,
) -> None:
    runtime = _FakeRuntime(score=score, threshold=40.0)
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(runtime),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert UUID(payload.pop("inspection_id"))
    assert payload == {
        "model_name": "patchcore",
        "category": "metal_nut",
        "is_anomaly": expected_is_anomaly,
        "anomaly_score": score,
        "threshold": 40.0,
        "comparison_operator": ">",
    }
    assert "defect_type" not in response.json()
    assert "anomaly_map" not in response.json()


# ADD 2026-08-19: Empty, malformed와 unsupported upload를 구분된 API error로 변환한다.
@pytest.mark.parametrize(
    ("content", "media_type", "status_code", "error_code"),
    [
        (b"", "image/png", 400, "empty_image"),
        (b"not-an-image", "image/png", 400, "invalid_image"),
        (b"GIF89a", "image/gif", 415, "unsupported_media_type"),
    ],
)
def test_invalid_image_uploads_return_public_errors(
    tmp_path: Path,
    content: bytes,
    media_type: str,
    status_code: int,
    error_code: str,
) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(_FakeRuntime()),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions",
            files={"image": ("sample", content, media_type)},
        )

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == error_code


# ADD 2026-08-19: Configured upload limit보다 큰 image payload를 inference 전에 거부한다.
def test_oversized_upload_is_rejected(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path, max_upload_bytes=8),
        runtime_loader=_runtime_loader(_FakeRuntime()),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "image_too_large"


# ADD 2026-08-19: Missing multipart image field를 centralized validation error로 변환한다.
def test_missing_upload_uses_validation_error_contract(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(_FakeRuntime()),
    )

    with TestClient(app) as client:
        response = client.post("/v1/predictions")

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "invalid_request",
        "message": "Request validation failed.",
    }


# ADD 2026-08-19: Runtime exception을 내부 정보 없는 failure response로 변환한다.
def test_runtime_failure_is_mapped_without_internal_details(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(_FakeRuntime(fail=True)),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "inference_failed",
            "message": "Model inference failed.",
        }
    }
    assert "internal model details" not in response.text


# ADD 2026-08-19: Lifespan이 runtime을 한 번만 load하고 여러 request에서 재사용하는지 검증한다.
def test_model_is_loaded_once_and_reused_for_requests(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    load_calls: list[PatchCoreRuntimeConfig] = []
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(runtime, load_calls),
    )

    with TestClient(app) as client:
        for _ in range(2):
            response = client.post(
                "/v1/predictions",
                files={"image": ("sample.png", _png_bytes(), "image/png")},
            )
            assert response.status_code == 200

    assert len(load_calls) == 1
    assert runtime.predict_calls == 2


# ADD 2026-08-19: Startup artifact validation failure가 ready app으로 전환되지 않는지 검증한다.
def test_startup_failure_propagates_before_application_is_ready(tmp_path: Path) -> None:
    # ADD 2026-08-19: Missing artifact를 나타내는 startup failure를 발생시킨다.
    def failing_loader(config: PatchCoreRuntimeConfig) -> _FakeRuntime:
        raise FileNotFoundError(f"missing artifact: {config.artifact_dir}")

    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=failing_loader,
    )

    with pytest.raises(FileNotFoundError, match="missing artifact"):
        with TestClient(app):
            pass
