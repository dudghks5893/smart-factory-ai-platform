"""Prometheus instrumentation integration tests for the FastAPI application."""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from PIL import Image
from prometheus_client.parser import text_string_to_metric_families
from torch import Tensor

from services.api.app import create_app
from services.api.config import ServingSettings
from services.inference.runtime import (
    InferenceResult,
    ModelRuntime,
    PatchCoreRuntimeConfig,
    ServingProvenance,
)
from services.persistence.database import PersistenceError
from services.persistence.inspections import (
    Inspection,
    InspectionCreate,
    InspectionPage,
)
from shared.hashing import sha256_bytes
from tests.persistence_helpers import prepare_sqlite_database


class _Runtime:
    """Mutable deterministic runtime for normal and anomaly metric observations."""

    model_name = "patchcore"
    category = "metal_nut"
    device = "cpu"
    provenance = ServingProvenance(
        manifest_sha256="a" * 64,
        artifact_metadata_sha256="b" * 64,
        model_sha256="c" * 64,
        threshold_artifact_sha256="d" * 64,
    )

    # ADD 2026-08-21: Monitoring test가 변경할 deterministic score를 초기화한다.
    def __init__(self, score: float = 30.0) -> None:
        self.score = score

    # ADD 2026-08-21: Strict threshold 결과를 monitoring 가능한 runtime boundary에서 반환한다.
    def predict(self, image: Tensor) -> InferenceResult:
        return InferenceResult(
            model_name=self.model_name,
            category=self.category,
            is_anomaly=self.score > 40.0,
            anomaly_score=self.score,
            threshold=40.0,
            comparison_operator=">",
        )


class _FailingRepository:
    """Repository double with one stable persistence failure and private detail."""

    # ADD 2026-08-21: Startup schema readiness는 통과해 insert metric failure만 격리한다.
    def check_ready(self) -> None:
        return None

    # ADD 2026-08-21: Insert 시 private detail을 포함한 persistence failure를 발생시킨다.
    def create(self, values: InspectionCreate) -> Inspection:
        raise PersistenceError("private database host and credential")

    # ADD 2026-08-21: Protocol completeness를 위해 사용되지 않는 lookup을 정의한다.
    def get(self, inspection_id: UUID) -> Inspection | None:
        return None

    # ADD 2026-08-21: Protocol completeness를 위해 empty history page를 정의한다.
    def list(
        self,
        *,
        category: str | None,
        is_anomaly: bool | None,
        limit: int,
        offset: int,
    ) -> InspectionPage:
        return InspectionPage(items=(), limit=limit, offset=offset, has_more=False)


# ADD 2026-08-21: Isolated SQLite와 non-secret fake artifact path를 test app에 구성한다.
def _settings(tmp_path: Path, database_name: str = "monitoring.db") -> ServingSettings:
    return ServingSettings(
        artifact_dir=tmp_path / "private-artifact-path",
        thresholds_path=tmp_path / "private-threshold-path.json",
        database_url=prepare_sqlite_database(tmp_path, database_name),
        model_device="cpu",
    )


# ADD 2026-08-21: Test lifespan에서 동일 runtime을 반환하는 loader를 생성한다.
def _runtime_loader(runtime: _Runtime) -> Callable[[PatchCoreRuntimeConfig], ModelRuntime]:
    # ADD 2026-08-21: Runtime config와 무관하게 monitoring test runtime을 재사용한다.
    def load(config: PatchCoreRuntimeConfig) -> ModelRuntime:
        return runtime

    return load


# ADD 2026-08-21: Deterministic in-memory PNG upload를 생성한다.
def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(120, 80, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


# ADD 2026-08-21: Exposition에서 정확한 sample name의 labels와 value를 추출한다.
def _metric_samples(exposition: str, name: str) -> list[tuple[dict[str, str], float]]:
    return [
        (dict(sample.labels), float(sample.value))
        for family in text_string_to_metric_families(exposition)
        for sample in family.samples
        if sample.name == name
    ]


# ADD 2026-08-21: 여러 app factory의 registry isolation과 hidden metrics endpoint를 검증한다.
def test_metrics_endpoint_uses_isolated_registry_and_static_model_info(tmp_path: Path) -> None:
    first_app = create_app(
        settings=_settings(tmp_path, "first.db"),
        runtime_loader=_runtime_loader(_Runtime()),
    )
    second_app = create_app(
        settings=_settings(tmp_path, "second.db"),
        runtime_loader=_runtime_loader(_Runtime()),
    )

    with TestClient(first_app) as first_client, TestClient(second_app) as second_client:
        first_response = first_client.get("/metrics")
        second_response = second_client.get("/metrics")
        openapi = first_client.get("/openapi.json")

    assert first_response.status_code == 200
    assert first_response.headers["content-type"].startswith("text/plain; version=")
    assert second_response.status_code == 200
    assert "/metrics" not in openapi.json()["paths"]
    assert _metric_samples(first_response.text, "smartfactory_model_info") == [
        (
            {
                "category": "metal_nut",
                "device": "cpu",
                "model_name": "patchcore",
                "model_sha256": "c" * 64,
            },
            1.0,
        )
    ]
    assert str(tmp_path) not in first_response.text
    assert "private-artifact-path" not in first_response.text
    assert "private-threshold-path" not in first_response.text


# ADD 2026-08-21: HTTP/domain metric의 bounded label과 sensitive value 비노출을 검증한다.
def test_success_and_validation_metrics_use_bounded_labels(tmp_path: Path) -> None:
    runtime = _Runtime(score=30.0)
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(runtime),
    )
    image = _png_bytes()
    missing_id = uuid4()
    unmatched_token = uuid4()

    with TestClient(app) as client:
        normal = client.post(
            "/v1/predictions",
            files={"image": ("private-filename.png", image, "image/png")},
        )
        runtime.score = 50.0
        anomaly = client.post(
            "/v1/predictions",
            files={"image": ("private-filename.png", image, "image/png")},
        )
        inspection_id = normal.json()["inspection_id"]
        detail = client.get(f"/v1/inspections/{inspection_id}")
        missing = client.get(f"/v1/inspections/{missing_id}")
        invalid = client.post(
            "/v1/predictions",
            files={"image": ("private-invalid.png", b"not-an-image", "image/png")},
        )
        unmatched = client.get(f"/private/{unmatched_token}")
        exposition = client.get("/metrics").text

    assert normal.status_code == 200
    assert normal.json()["is_anomaly"] is False
    assert anomaly.status_code == 200
    assert anomaly.json()["is_anomaly"] is True
    assert detail.status_code == 200
    assert missing.status_code == 404
    assert invalid.status_code == 400
    assert unmatched.status_code == 404
    assert _metric_samples(exposition, "smartfactory_predictions_total") == [
        ({"category": "metal_nut", "result": "normal"}, 1.0),
        ({"category": "metal_nut", "result": "anomaly"}, 1.0),
    ]
    assert _metric_samples(exposition, "smartfactory_inference_duration_seconds_count") == [
        ({"category": "metal_nut", "device": "cpu", "model_name": "patchcore"}, 2.0)
    ]
    assert _metric_samples(exposition, "smartfactory_persistence_duration_seconds_count") == [
        ({"operation": "insert"}, 2.0)
    ]
    request_samples = _metric_samples(exposition, "smartfactory_http_requests_total")
    assert ({"method": "POST", "route": "/v1/predictions", "status_code": "200"}, 2.0) in (
        request_samples
    )
    assert (
        {
            "method": "GET",
            "route": "/v1/inspections/{inspection_id}",
            "status_code": "404",
        },
        1.0,
    ) in request_samples
    assert ({"method": "GET", "route": "unmatched", "status_code": "404"}, 1.0) in (request_samples)
    assert not any(labels["route"] == "/metrics" for labels, _ in request_samples)
    for sensitive_value in (
        inspection_id,
        str(missing_id),
        str(unmatched_token),
        "private-filename.png",
        sha256_bytes(image),
    ):
        assert sensitive_value not in exposition


# ADD 2026-08-21: Persistence failure metric과 기존 safe HTTP 503 contract를 함께 검증한다.
def test_persistence_failure_metrics_preserve_safe_api_response(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(_Runtime(score=50.0)),
        repository_loader=lambda _: _FailingRepository(),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/predictions",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )
        exposition = client.get("/metrics").text

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "persistence_unavailable"
    assert _metric_samples(exposition, "smartfactory_persistence_errors_total") == [
        ({"operation": "insert"}, 1.0)
    ]
    assert _metric_samples(exposition, "smartfactory_persistence_duration_seconds_count") == [
        ({"operation": "insert"}, 1.0)
    ]
    assert (
        {"method": "POST", "route": "/v1/predictions", "status_code": "503"},
        1.0,
    ) in _metric_samples(exposition, "smartfactory_http_requests_total")
    assert _metric_samples(exposition, "smartfactory_predictions_total") == []
    assert "private database host" not in exposition
    assert "credential" not in exposition
