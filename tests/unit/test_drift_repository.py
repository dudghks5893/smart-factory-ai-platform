"""Unit tests for the read-only drift inspection window repository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError

from services.persistence.database import DatabaseManager, PersistenceError, create_database_manager
from services.persistence.drift import SqlAlchemyDriftWindowRepository
from services.persistence.inspections import InspectionCreate, SqlAlchemyInspectionRepository
from services.persistence.models import Base, InspectionRecord


# ADD 2026-08-21: Drift repository test용 complete inspection input을 생성한다.
def _values(
    *,
    score: float,
    category: str = "metal_nut",
    model_name: str = "patchcore",
    model_sha256: str = "a" * 64,
    threshold_sha256: str = "d" * 64,
) -> InspectionCreate:
    return InspectionCreate(
        model_name=model_name,
        category=category,
        is_anomaly=score > 40.0,
        anomaly_score=score,
        threshold=40.0,
        comparison_operator=">",
        image_sha256="1" * 64,
        image_size_bytes=100,
        content_type="image/png",
        model_sha256=model_sha256,
        artifact_metadata_sha256="b" * 64,
        threshold_artifact_sha256=threshold_sha256,
        manifest_sha256="c" * 64,
        device="cpu",
    )


# ADD 2026-08-21: SQLite schema와 write/read repository를 같은 test database에 준비한다.
def _repositories(
    tmp_path: Path,
) -> tuple[
    SqlAlchemyInspectionRepository,
    SqlAlchemyDriftWindowRepository,
    DatabaseManager,
]:
    database = create_database_manager(f"sqlite+pysqlite:///{tmp_path / 'drift.db'}")
    Base.metadata.create_all(database.engine)
    return (
        SqlAlchemyInspectionRepository(database.session_factory),
        SqlAlchemyDriftWindowRepository(database.session_factory),
        database,
    )


# ADD 2026-08-21: Half-open time, category, model name/SHA filter와 stable order를 검증한다.
def test_drift_query_filters_window_category_and_model(tmp_path: Path) -> None:
    writer, reader, database = _repositories(tmp_path)
    since = datetime(2026, 8, 21, tzinfo=UTC)
    try:
        rows = [
            writer.create(_values(score=10.0)),
            writer.create(_values(score=20.0, threshold_sha256="e" * 64)),
            writer.create(_values(score=30.0, category="bottle")),
            writer.create(_values(score=31.0, model_name="other")),
            writer.create(_values(score=32.0, model_sha256="f" * 64)),
            writer.create(_values(score=33.0)),
        ]
        row_times = (
            since,
            since + timedelta(minutes=10),
            since + timedelta(minutes=20),
            since + timedelta(minutes=30),
            since + timedelta(minutes=40),
            since + timedelta(hours=1),
        )
        with database.engine.begin() as connection:
            for row, created_at in zip(rows, row_times, strict=True):
                connection.execute(
                    update(InspectionRecord)
                    .where(InspectionRecord.id == row.id)
                    .values(created_at=created_at)
                )

        observations = reader.list_observations(
            category="metal_nut",
            model_name="patchcore",
            model_sha256="a" * 64,
            since=since,
            until=since + timedelta(hours=1),
        )
    finally:
        database.dispose()

    assert [item.anomaly_score for item in observations] == [10.0, 20.0]
    assert observations[0].created_at.tzinfo is not None
    assert observations[1].lineage.threshold_artifact_sha256 == "e" * 64


# ADD 2026-08-21: Query failure가 credential 없는 safe persistence error가 되는지 검증한다.
def test_drift_query_wraps_database_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, repository, database = _repositories(tmp_path)
    session_type = database.session_factory.class_

    # ADD 2026-08-21: Repository error handling 검증을 위해 DB scalar query를 강제로 실패시킨다.
    def fail_scalars(self: object, statement: object) -> None:
        raise SQLAlchemyError("postgresql://secret@host/database")

    monkeypatch.setattr(session_type, "scalars", fail_scalars)
    try:
        with pytest.raises(PersistenceError, match="Drift inspection window lookup failed") as exc:
            repository.list_observations(
                category="metal_nut",
                model_name="patchcore",
                model_sha256="a" * 64,
                since=datetime(2026, 8, 21, tzinfo=UTC),
                until=datetime(2026, 8, 22, tzinfo=UTC),
            )
    finally:
        database.dispose()

    assert "secret" not in str(exc.value)
