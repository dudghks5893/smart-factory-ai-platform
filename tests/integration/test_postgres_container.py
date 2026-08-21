"""Real PostgreSQL/Alembic/FastAPI persistence integration for Docker test runs."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import delete, func, make_url, select, text, update
from torch import Tensor

from ml.drift.patchcore import (
    DriftLineage,
    DriftPolicy,
    DriftReference,
    anomaly_ratio,
    build_reference_bin_edges,
    histogram_counts,
    summarize_scores,
    write_drift_reference,
)
from pipelines.analyze_patchcore_drift import analyze_patchcore_drift
from services.api.app import create_app
from services.api.config import ServingSettings
from services.inference.runtime import (
    InferenceResult,
    ModelRuntime,
    PatchCoreRuntimeConfig,
    ServingProvenance,
)
from services.persistence.database import PersistenceError, create_database_manager
from services.persistence.drift import SqlAlchemyDriftWindowRepository
from services.persistence.inspections import InspectionCreate, SqlAlchemyInspectionRepository
from services.persistence.models import InspectionRecord

POSTGRES_INTEGRATION_DATABASE_URL = os.getenv("POSTGRES_INTEGRATION_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(
    not POSTGRES_INTEGRATION_DATABASE_URL,
    reason="requires the dedicated Docker PostgreSQL integration database",
)


class _Runtime:
    """Test-only deterministic inference runtime with valid serving provenance."""

    model_name = "patchcore"
    category = "metal_nut"
    device = "cpu"
    provenance = ServingProvenance(
        manifest_sha256="a" * 64,
        artifact_metadata_sha256="b" * 64,
        model_sha256="c" * 64,
        threshold_artifact_sha256="d" * 64,
    )

    # ADD 2026-08-20: PostgreSQL API test용 strict threshold 결과를 반환한다.
    def predict(self, image: Tensor) -> InferenceResult:
        return InferenceResult(
            model_name=self.model_name,
            category=self.category,
            is_anomaly=True,
            anomaly_score=50.0,
            threshold=40.0,
            comparison_operator=">",
        )


# ADD 2026-08-20: App lifecycle에서 test-only runtime을 재사용하는 loader를 생성한다.
def _runtime_loader() -> Callable[[PatchCoreRuntimeConfig], ModelRuntime]:
    runtime = _Runtime()

    # ADD 2026-08-20: Model artifact 접근 없이 test scope runtime만 반환한다.
    def load(config: PatchCoreRuntimeConfig) -> ModelRuntime:
        return runtime

    return load


# ADD 2026-08-20: PostgreSQL integration request용 작은 PNG upload를 생성한다.
def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(120, 80, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


# ADD 2026-08-20: 실제 PostgreSQL migration, schema, transaction과 FastAPI persistence를 검증한다.
def test_postgres_migration_and_fastapi_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = POSTGRES_INTEGRATION_DATABASE_URL
    alembic_config = Config("alembic.ini")

    # Dedicated ephemeral database에서 downgrade/upgrade lifecycle을 실제로 왕복한다.
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")
    database = create_database_manager(database_url)
    database.check_connection()

    try:
        with database.engine.begin() as connection:
            connection.execute(delete(InspectionRecord))
            migration_revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
            postgres_types = {
                str(row["column_name"]): str(row["data_type"])
                for row in connection.execute(
                    text(
                        "SELECT column_name, data_type FROM information_schema.columns "
                        "WHERE table_schema = 'public' AND table_name = 'inspections' "
                        "AND column_name IN ('id', 'created_at')"
                    )
                ).mappings()
            }
            indexes = set(
                connection.scalars(
                    text(
                        "SELECT indexname FROM pg_indexes "
                        "WHERE schemaname = 'public' AND tablename = 'inspections'"
                    )
                )
            )
            constraints = set(
                connection.scalars(
                    text(
                        "SELECT conname FROM pg_constraint "
                        "WHERE conrelid = 'public.inspections'::regclass"
                    )
                )
            )

        assert migration_revision == "20260820_01"
        assert postgres_types == {
            "created_at": "timestamp with time zone",
            "id": "uuid",
        }
        assert {
            "ix_inspections_created_at",
            "ix_inspections_category_created_at",
            "ix_inspections_anomaly_created_at",
        } <= indexes
        assert {
            "ck_inspections_operator_gt",
            "ck_inspections_image_size_positive",
            "ck_inspections_model_sha256_length",
        } <= constraints

        settings = ServingSettings(
            artifact_dir=tmp_path / "external-model",
            thresholds_path=tmp_path / "external-thresholds.json",
            database_url=database_url,
            model_device="cpu",
        )
        app = create_app(settings=settings, runtime_loader=_runtime_loader())
        image = _png_bytes()

        # Real DB-backed app에서 readiness, insert/commit과 detail/history 조회를 검증한다.
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/ready").status_code == 200
            prediction = client.post(
                "/v1/predictions",
                files={"image": ("sample.png", image, "image/png")},
            )
            inspection_id = UUID(prediction.json()["inspection_id"])
            detail = client.get(f"/v1/inspections/{inspection_id}")
            history = client.get("/v1/inspections")
            malformed = client.post(
                "/v1/predictions",
                files={"image": ("bad.png", b"not-an-image", "image/png")},
            )
            history_after_malformed = client.get("/v1/inspections")

        assert prediction.status_code == 200
        assert inspection_id.version == 4
        assert detail.status_code == 200
        assert datetime.fromisoformat(detail.json()["created_at"]).tzinfo is not None
        assert history.json()["returned_count"] == 1
        assert malformed.status_code == 400
        assert history_after_malformed.json()["returned_count"] == 1

        # DB constraint failure가 repository rollback 후 partial row를 남기지 않는지 확인한다.
        repository = SqlAlchemyInspectionRepository(database.session_factory)
        monkeypatch.setattr(InspectionCreate, "validate", lambda self: None)
        invalid = InspectionCreate(
            model_name="patchcore",
            category="metal_nut",
            is_anomaly=True,
            anomaly_score=50.0,
            threshold=40.0,
            comparison_operator=">",
            image_sha256="e" * 64,
            image_size_bytes=0,
            content_type="image/png",
            model_sha256="f" * 64,
            artifact_metadata_sha256="1" * 64,
            threshold_artifact_sha256="2" * 64,
            manifest_sha256="3" * 64,
            device="cpu",
        )
        with pytest.raises(PersistenceError, match="insert failed"):
            repository.create(invalid)
        with database.engine.connect() as connection:
            assert connection.scalar(select(func.count()).select_from(InspectionRecord)) == 1

        # Unreachable PostgreSQL endpoint가 readiness connectivity error로 fail-fast한다.
        unavailable_url = (
            make_url(database_url)
            .set(port=1)
            .update_query_dict({"connect_timeout": "1"})
            .render_as_string(hide_password=False)
        )
        unavailable_database = create_database_manager(unavailable_url)
        try:
            with pytest.raises(PersistenceError, match="connectivity"):
                unavailable_database.check_connection()
        finally:
            unavailable_database.dispose()
    finally:
        database.dispose()


# ADD 2026-08-21: PostgreSQL drift smoke용 validation-normal reference artifact를 생성한다.
def _drift_reference() -> DriftReference:
    scores = tuple(float(index) for index in range(30))
    edges = build_reference_bin_edges(scores, 10)
    return DriftReference(
        schema_version=1,
        reference_id="postgres-reference",
        model_name="patchcore",
        category="metal_nut",
        lineage=DriftLineage(
            model_sha256="a" * 64,
            artifact_metadata_sha256="b" * 64,
            manifest_sha256="c" * 64,
            threshold_artifact_sha256="d" * 64,
        ),
        source_split="validation",
        source_label="normal",
        validation_predictions_sha256="e" * 64,
        sample_count=30,
        score_values=scores,
        summary=summarize_scores(scores),
        image_threshold=29.0,
        comparison_operator=">",
        reference_anomaly_ratio=anomaly_ratio(scores, threshold=29.0),
        psi_bin_count_requested=10,
        psi_bin_edges=edges,
        reference_bin_counts=histogram_counts(scores, edges),
        psi_epsilon=1e-6,
        created_at="2026-08-21T00:00:00+00:00",
    )


# ADD 2026-08-21: 실제 PostgreSQL synthetic inspection query부터 drift report까지 검증한다.
def test_postgres_patchcore_drift_batch_smoke(tmp_path: Path) -> None:
    command.upgrade(Config("alembic.ini"), "head")
    database = create_database_manager(POSTGRES_INTEGRATION_DATABASE_URL)
    repository = SqlAlchemyInspectionRepository(database.session_factory)
    since = datetime(2026, 8, 21, tzinfo=UTC)
    reference = _drift_reference()
    reference_path = tmp_path / "reference.json"
    write_drift_reference(reference, reference_path)

    try:
        with database.engine.begin() as connection:
            connection.execute(delete(InspectionRecord))
        rows = []
        for score in reference.score_values:
            rows.append(
                repository.create(
                    InspectionCreate(
                        model_name="patchcore",
                        category="metal_nut",
                        is_anomaly=False,
                        anomaly_score=score,
                        threshold=29.0,
                        comparison_operator=">",
                        image_sha256=f"{int(score):064x}",
                        image_size_bytes=100,
                        content_type="image/png",
                        model_sha256="a" * 64,
                        artifact_metadata_sha256="b" * 64,
                        threshold_artifact_sha256="d" * 64,
                        manifest_sha256="c" * 64,
                        device="cpu",
                    )
                )
            )
        with database.engine.begin() as connection:
            for index, row in enumerate(rows):
                connection.execute(
                    update(InspectionRecord)
                    .where(InspectionRecord.id == row.id)
                    .values(created_at=since + timedelta(seconds=index))
                )

        # Real psycopg query와 domain analysis를 통해 immutable report를 생성한다.
        summary = analyze_patchcore_drift(
            repository=SqlAlchemyDriftWindowRepository(database.session_factory),
            reference_path=reference_path,
            output_dir=tmp_path / "drift-output",
            drift_id="postgres-run",
            since=since,
            until=since + timedelta(hours=1),
            policy=DriftPolicy(),
            created_at="2026-08-21T01:00:00+00:00",
        )
    finally:
        database.dispose()

    assert summary.report["status"] == "stable"
    assert summary.report["current_window"]["sample_count"] == 30
    assert summary.report["statistics"]["psi"] == 0.0
