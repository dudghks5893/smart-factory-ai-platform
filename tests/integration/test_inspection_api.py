"""Integration tests for persisted prediction and inspection history API contracts."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

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
from services.persistence.database import (
    PersistenceError,
    create_database_manager,
)
from services.persistence.inspections import (
    Inspection,
    InspectionCreate,
    InspectionPage,
)
from shared.hashing import sha256_bytes
from tests.persistence_helpers import prepare_sqlite_database


class _Runtime:
    """Mutable fake runtime used to create deterministic inspection records."""

    model_name = "patchcore"
    category = "metal_nut"
    device = "cpu"
    provenance = ServingProvenance(
        manifest_sha256="a" * 64,
        artifact_metadata_sha256="b" * 64,
        model_sha256="c" * 64,
        threshold_artifact_sha256="d" * 64,
    )

    # ADD 2026-08-20: API persistence fake의 score와 failure behavior를 초기화한다.
    def __init__(self, *, score: float = 50.0, fail: bool = False) -> None:
        self.score = score
        self.fail = fail
        self.predict_calls = 0

    # ADD 2026-08-20: Strict threshold prediction을 반환하거나 inference failure를 발생시킨다.
    def predict(self, image: Tensor) -> InferenceResult:
        self.predict_calls += 1
        if self.fail:
            raise RuntimeError("private inference detail")
        return InferenceResult(
            model_name=self.model_name,
            category=self.category,
            is_anomaly=self.score > 40.0,
            anomaly_score=self.score,
            threshold=40.0,
            comparison_operator=">",
        )


class _FailingRepository:
    """Repository double that fails every operation without exposing DB details."""

    # ADD 2026-08-20: Startup schema readiness는 통과해 insert failure만 격리한다.
    def check_ready(self) -> None:
        return None

    # ADD 2026-08-20: Insert failure path에 deterministic persistence exception을 발생시킨다.
    def create(self, values: InspectionCreate) -> Inspection:
        raise PersistenceError("private database and credential detail")

    # ADD 2026-08-20: Lookup failure path에 deterministic persistence exception을 발생시킨다.
    def get(self, inspection_id: UUID) -> Inspection | None:
        raise PersistenceError("private database detail")

    # ADD 2026-08-20: History failure path에 deterministic persistence exception을 발생시킨다.
    def list(
        self,
        *,
        category: str | None,
        is_anomaly: bool | None,
        limit: int,
        offset: int,
    ) -> InspectionPage:
        raise PersistenceError("private database detail")


# ADD 2026-08-20: Test app에 isolated SQLite DATABASE_URL과 serving paths를 구성한다.
def _settings(tmp_path: Path) -> ServingSettings:
    return ServingSettings(
        artifact_dir=tmp_path / "artifact",
        thresholds_path=tmp_path / "thresholds.json",
        database_url=prepare_sqlite_database(tmp_path),
        model_device="cpu",
    )


# ADD 2026-08-20: Fake runtime을 process lifecycle에서 한 번 반환하는 loader를 생성한다.
def _runtime_loader(
    runtime: _Runtime,
) -> Callable[[PatchCoreRuntimeConfig], ModelRuntime]:
    # ADD 2026-08-20: Runtime config와 무관하게 동일 fake runtime을 재사용한다.
    def load(config: PatchCoreRuntimeConfig) -> ModelRuntime:
        return runtime

    return load


# ADD 2026-08-20: Deterministic in-memory PNG upload bytes를 생성한다.
def _png_bytes(value: int = 120) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(value, 80, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


# ADD 2026-08-20: Prediction success가 정확히 한 row와 traceable response를 생성하는지 검증한다.
def test_prediction_persists_exactly_one_traceable_inspection(tmp_path: Path) -> None:
    content = _png_bytes()
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(_Runtime()),
    )

    with TestClient(app) as client:
        prediction = client.post(
            "/v1/predictions",
            files={"image": ("not-persisted.png", content, "image/png")},
        )
        history = client.get("/v1/inspections")
        detail = client.get(f"/v1/inspections/{prediction.json()['inspection_id']}")

    assert prediction.status_code == 200
    assert UUID(prediction.json()["inspection_id"])
    assert history.json()["returned_count"] == 1
    assert history.json()["has_more"] is False
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["inspection_id"] == prediction.json()["inspection_id"]
    assert datetime.fromisoformat(payload["created_at"]).tzinfo is not None
    assert payload["image_sha256"] == sha256_bytes(content)
    assert payload["image_size_bytes"] == len(content)
    assert payload["content_type"] == "image/png"
    assert payload["model_sha256"] == "c" * 64
    assert payload["artifact_metadata_sha256"] == "b" * 64
    assert payload["threshold_artifact_sha256"] == "d" * 64
    assert payload["manifest_sha256"] == "a" * 64
    assert "image" not in payload
    assert "image_bytes" not in payload
    assert "filename" not in payload
    assert "defect_type" not in payload


# ADD 2026-08-20: Malformed upload와 inference failure가 inspection row를 생성하지 않는지 검증한다.
@pytest.mark.parametrize(
    ("runtime", "content", "expected_status"),
    [
        (_Runtime(), b"not-an-image", 400),
        (_Runtime(fail=True), _png_bytes(), 500),
    ],
)
def test_failed_prediction_creates_no_inspection(
    tmp_path: Path,
    runtime: _Runtime,
    content: bytes,
    expected_status: int,
) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(runtime),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions",
            files={"image": ("sample.png", content, "image/png")},
        )
        history = client.get("/v1/inspections")

    assert response.status_code == expected_status
    assert history.json()["returned_count"] == 0


# ADD 2026-08-20: Invalid runtime provenance가 inspection을 생성하지 않는지 검증한다.
def test_invalid_runtime_provenance_creates_no_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _Runtime()
    monkeypatch.setattr(runtime, "provenance", object())
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(runtime),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )
        history = client.get("/v1/inspections")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "inference_provenance_unavailable"
    assert history.json()["returned_count"] == 0


# ADD 2026-08-20: History detail 404, filters와 bounded pagination contract를 검증한다.
def test_history_api_filters_pagination_and_unknown_id(tmp_path: Path) -> None:
    runtime = _Runtime(score=30.0)
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(runtime),
    )

    with TestClient(app) as client:
        client.post(
            "/v1/predictions",
            files={"image": ("normal.png", _png_bytes(10), "image/png")},
        )
        runtime.score = 50.0
        client.post(
            "/v1/predictions",
            files={"image": ("anomaly.png", _png_bytes(240), "image/png")},
        )
        anomaly_page = client.get("/v1/inspections?category=metal_nut&is_anomaly=true")
        first_page = client.get("/v1/inspections?limit=1&offset=0")
        second_page = client.get("/v1/inspections?limit=1&offset=1")
        invalid_limit = client.get("/v1/inspections?limit=101")
        missing = client.get(f"/v1/inspections/{uuid4()}")

    assert anomaly_page.status_code == 200
    assert anomaly_page.json()["returned_count"] == 1
    assert anomaly_page.json()["items"][0]["is_anomaly"] is True
    assert first_page.json()["returned_count"] == 1
    assert first_page.json()["has_more"] is True
    assert second_page.json()["returned_count"] == 1
    assert second_page.json()["has_more"] is False
    assert invalid_limit.status_code == 422
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "inspection_not_found"


# ADD 2026-08-20: DB insert failure가 success로 숨겨지지 않고 safe 503을 반환하는지 검증한다.
def test_database_failure_returns_safe_error(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(_Runtime()),
        repository_loader=lambda _: _FailingRepository(),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/predictions",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "persistence_unavailable",
            "message": "Inspection persistence is unavailable.",
        }
    }
    assert "private database" not in response.text
    assert "credential" not in response.text


# ADD 2026-08-20: Health는 유지하되 lost DB connectivity가 readiness를 503으로 전환하는지 검증한다.
def test_readiness_requires_live_database_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = prepare_sqlite_database(tmp_path)
    database = create_database_manager(database_url)
    settings = ServingSettings(
        artifact_dir=tmp_path / "artifact",
        thresholds_path=tmp_path / "thresholds.json",
        database_url=database_url,
        model_device="cpu",
    )
    app = create_app(
        settings=settings,
        runtime_loader=_runtime_loader(_Runtime()),
        database_loader=lambda _: database,
    )

    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200

        # Startup 이후 required dependency loss를 simulated connectivity failure로 전환한다.
        monkeypatch.setattr(
            database,
            "check_connection",
            lambda: (_ for _ in ()).throw(PersistenceError("private database detail")),
        )
        assert client.get("/health").status_code == 200
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "database_not_ready"


# ADD 2026-08-20: Startup DB failure가 model load와 readiness 전에 fail-fast하는지 검증한다.
def test_startup_database_failure_prevents_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = prepare_sqlite_database(tmp_path)
    database = create_database_manager(database_url)
    runtime_load_count = 0

    # ADD 2026-08-20: Startup ordering 검증을 위해 runtime load 호출을 기록한다.
    def load(config: PatchCoreRuntimeConfig) -> ModelRuntime:
        nonlocal runtime_load_count
        runtime_load_count += 1
        return _Runtime()

    # DB connectivity가 runtime load보다 먼저 실패하도록 manager behavior를 교체한다.
    monkeypatch.setattr(
        database,
        "check_connection",
        lambda: (_ for _ in ()).throw(PersistenceError("unreachable")),
    )
    app = create_app(
        settings=ServingSettings(
            artifact_dir=tmp_path / "artifact",
            thresholds_path=tmp_path / "thresholds.json",
            database_url=database_url,
            model_device="cpu",
        ),
        runtime_loader=load,
        database_loader=lambda _: database,
    )

    with pytest.raises(PersistenceError, match="unreachable"):
        with TestClient(app):
            pass
    assert runtime_load_count == 0


# ADD 2026-08-20: Missing migrated table이 model load와 ready state 전에 fail-fast하는지 검증한다.
def test_startup_missing_schema_prevents_model_load(tmp_path: Path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'missing-schema.db'}"
    runtime_load_count = 0

    # ADD 2026-08-20: Schema readiness보다 뒤인 runtime load 호출을 기록한다.
    def load(config: PatchCoreRuntimeConfig) -> ModelRuntime:
        nonlocal runtime_load_count
        runtime_load_count += 1
        return _Runtime()

    app = create_app(
        settings=ServingSettings(
            artifact_dir=tmp_path / "artifact",
            thresholds_path=tmp_path / "thresholds.json",
            database_url=database_url,
            model_device="cpu",
        ),
        runtime_loader=load,
    )

    with pytest.raises(PersistenceError, match="schema readiness"):
        with TestClient(app):
            pass
    assert runtime_load_count == 0
