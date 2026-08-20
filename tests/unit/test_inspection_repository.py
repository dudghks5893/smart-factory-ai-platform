"""Unit tests for inspection persistence using an isolated SQLite test database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import inspect, update
from sqlalchemy.exc import SQLAlchemyError

from services.persistence.database import (
    DatabaseManager,
    PersistenceError,
    create_database_manager,
)
from services.persistence.inspections import (
    InspectionCreate,
    SqlAlchemyInspectionRepository,
)
from services.persistence.models import Base, InspectionRecord


# ADD 2026-08-20: Repository test에 사용할 complete inspection input을 생성한다.
def _values(
    *,
    category: str = "metal_nut",
    is_anomaly: bool = True,
    score: float | None = None,
) -> InspectionCreate:
    anomaly_score = (50.0 if is_anomaly else 30.0) if score is None else score
    return InspectionCreate(
        model_name="patchcore",
        category=category,
        is_anomaly=is_anomaly,
        anomaly_score=anomaly_score,
        threshold=40.0,
        comparison_operator=">",
        image_sha256="a" * 64,
        image_size_bytes=1234,
        content_type="image/png",
        model_sha256="b" * 64,
        artifact_metadata_sha256="c" * 64,
        threshold_artifact_sha256="d" * 64,
        manifest_sha256="e" * 64,
        device="cpu",
    )


# ADD 2026-08-20: Test별 SQLite schema와 SQLAlchemy repository를 준비한다.
def _repository(
    tmp_path: Path,
) -> tuple[SqlAlchemyInspectionRepository, DatabaseManager]:
    database = create_database_manager(f"sqlite+pysqlite:///{tmp_path / 'repository.db'}")
    Base.metadata.create_all(database.engine)
    return SqlAlchemyInspectionRepository(database.session_factory), database


# ADD 2026-08-20: Insert가 UUID, UTC timestamp와 prediction/provenance 전체를 보존하는지 검증한다.
def test_insert_and_get_preserve_inspection_contract(tmp_path: Path) -> None:
    repository, database = _repository(tmp_path)
    try:
        created = repository.create(_values())
        loaded = repository.get(created.id)
    finally:
        database.dispose()

    assert created.id.version == 4
    assert created.created_at.tzinfo is not None
    assert created.created_at.utcoffset() == timedelta(0)
    assert loaded == created
    assert created.model_name == "patchcore"
    assert created.anomaly_score == 50.0
    assert created.image_sha256 == "a" * 64
    assert created.model_sha256 == "b" * 64
    assert created.artifact_metadata_sha256 == "c" * 64
    assert created.threshold_artifact_sha256 == "d" * 64
    assert created.manifest_sha256 == "e" * 64


# ADD 2026-08-20: Unknown UUID lookup과 raw image column 부재를 검증한다.
def test_unknown_id_and_schema_exclude_raw_image_bytes(tmp_path: Path) -> None:
    repository, database = _repository(tmp_path)
    try:
        assert repository.get(uuid4()) is None
        column_names = {
            column["name"] for column in inspect(database.engine).get_columns("inspections")
        }
    finally:
        database.dispose()

    assert "image" not in column_names
    assert "image_bytes" not in column_names
    assert "filename" not in column_names


# ADD 2026-08-20: Newest ordering, filters와 limit/offset has-more pagination을 검증한다.
def test_history_ordering_filters_and_pagination(tmp_path: Path) -> None:
    repository, database = _repository(tmp_path)
    try:
        first = repository.create(_values(is_anomaly=False))
        second = repository.create(_values(is_anomaly=True))
        third = repository.create(_values(category="bottle", is_anomaly=True))
        base_time = datetime(2026, 8, 20, tzinfo=UTC)
        with database.engine.begin() as connection:
            for index, inspection in enumerate((first, second, third)):
                connection.execute(
                    update(InspectionRecord)
                    .where(InspectionRecord.id == inspection.id)
                    .values(created_at=base_time + timedelta(seconds=index))
                )

        first_page = repository.list(category=None, is_anomaly=None, limit=2, offset=0)
        second_page = repository.list(category=None, is_anomaly=None, limit=2, offset=2)
        category_page = repository.list(
            category="metal_nut",
            is_anomaly=None,
            limit=10,
            offset=0,
        )
        anomaly_page = repository.list(
            category=None,
            is_anomaly=True,
            limit=10,
            offset=0,
        )
    finally:
        database.dispose()

    assert [item.id for item in first_page.items] == [third.id, second.id]
    assert first_page.has_more is True
    assert [item.id for item in second_page.items] == [first.id]
    assert second_page.has_more is False
    assert {item.id for item in category_page.items} == {first.id, second.id}
    assert {item.id for item in anomaly_page.items} == {second.id, third.id}


# ADD 2026-08-20: Non-finite score와 strict threshold inconsistency를 insert 전에 거부한다.
@pytest.mark.parametrize(
    "values",
    [
        _values(score=float("nan")),
        _values(score=float("inf")),
        _values(is_anomaly=True, score=40.0),
    ],
)
def test_create_validation_rejects_invalid_prediction(
    values: InspectionCreate, tmp_path: Path
) -> None:
    repository, database = _repository(tmp_path)
    try:
        with pytest.raises(ValueError):
            repository.create(values)
    finally:
        database.dispose()


# ADD 2026-08-20: Flush failure가 explicit rollback과 safe persistence exception을 발생시킨다.
def test_insert_failure_rolls_back_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, database = _repository(tmp_path)
    rollback_calls = 0
    session_type = database.session_factory.class_
    original_rollback = session_type.rollback

    # ADD 2026-08-20: Database failure scenario를 위해 flush를 강제로 실패시킨다.
    def fail_flush(self: object, objects: object | None = None) -> None:
        raise SQLAlchemyError("forced flush failure")

    # ADD 2026-08-20: Repository가 호출한 rollback 횟수를 기록한다.
    def track_rollback(self: object) -> None:
        nonlocal rollback_calls
        rollback_calls += 1
        original_rollback(self)  # type: ignore[arg-type]

    monkeypatch.setattr(session_type, "flush", fail_flush)
    monkeypatch.setattr(session_type, "rollback", track_rollback)
    try:
        with pytest.raises(PersistenceError, match="Inspection insert failed"):
            repository.create(_values())
    finally:
        database.dispose()

    assert rollback_calls == 1
