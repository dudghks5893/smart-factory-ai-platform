"""Unit contracts for atomic known-defect parent/child persistence."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import event, func, inspect, select, update

from services.persistence.database import (
    DatabaseManager,
    PersistenceError,
    create_database_manager,
)
from services.persistence.known_defects import (
    KnownDefectCreate,
    KnownDefectInstanceCreate,
    SqlAlchemyKnownDefectRepository,
)
from services.persistence.models import (
    Base,
    KnownDefectInspectionRecord,
    KnownDefectInstanceRecord,
)


# ADD 2026-08-26: Deterministic parent provenance와 ordered multi-instance input을 생성한다.
def _values(*, instances: tuple[KnownDefectInstanceCreate, ...] | None = None) -> KnownDefectCreate:
    return KnownDefectCreate(
        model_name="yolo11n-seg.pt",
        task="segment",
        category="metal_nut",
        device="mps",
        diagnostic_confidence=0.25,
        inference_ms=12.5,
        image_width=8,
        image_height=8,
        image_sha256="a" * 64,
        model_sha256="b" * 64,
        artifact_metadata_sha256="c" * 64,
        dataset_manifest_sha256="d" * 64,
        dataset_semantic_fingerprint_sha256="e" * 64,
        instances=(
            instances
            if instances is not None
            else (
                KnownDefectInstanceCreate(
                    class_id=0,
                    class_name="bent",
                    confidence=0.95,
                    bbox_x_min=1.0,
                    bbox_y_min=1.0,
                    bbox_x_max=5.0,
                    bbox_y_max=5.0,
                    mask_pixel_count=16,
                    mask_area_ratio=0.25,
                ),
                KnownDefectInstanceCreate(
                    class_id=2,
                    class_name="scratch",
                    confidence=0.75,
                    bbox_x_min=4.0,
                    bbox_y_min=4.0,
                    bbox_x_max=7.0,
                    bbox_y_max=7.0,
                    mask_pixel_count=9,
                    mask_area_ratio=9 / 64,
                ),
            )
        ),
    )


# ADD 2026-08-26: Test별 SQLite parent/child schema와 repository를 준비한다.
def _repository(
    tmp_path: Path,
) -> tuple[SqlAlchemyKnownDefectRepository, DatabaseManager]:
    database = create_database_manager(f"sqlite+pysqlite:///{tmp_path / 'known-defects.db'}")
    Base.metadata.create_all(database.engine)
    return SqlAlchemyKnownDefectRepository(database.session_factory), database


# ADD 2026-08-26: Multi-instance insert/detail이 order, lineage와 provenance를 보존하는지 검증한다.
def test_create_and_get_preserve_parent_children_and_provenance(tmp_path: Path) -> None:
    repository, database = _repository(tmp_path)
    try:
        created = repository.create(_values())
        loaded = repository.get(created.inspection.id)
    finally:
        database.dispose()

    assert loaded == created
    assert created.inspection.id.version == 4
    assert created.inspection.created_at.utcoffset() == timedelta(0)
    assert created.inspection.instance_count == 2
    assert created.inspection.image_sha256 == "a" * 64
    assert created.inspection.model_sha256 == "b" * 64
    assert created.inspection.artifact_metadata_sha256 == "c" * 64
    assert created.inspection.dataset_manifest_sha256 == "d" * 64
    assert created.inspection.dataset_semantic_fingerprint_sha256 == "e" * 64
    assert [instance.instance_index for instance in created.instances] == [0, 1]
    assert [instance.class_name for instance in created.instances] == ["bent", "scratch"]


# ADD 2026-08-26: Empty prediction parent와 no-raw-payload schema contract를 검증한다.
def test_zero_instance_parent_is_durable_without_raw_payload_columns(tmp_path: Path) -> None:
    repository, database = _repository(tmp_path)
    try:
        created = repository.create(_values(instances=()))
        loaded = repository.get(created.inspection.id)
        parent_columns = {
            column["name"]
            for column in inspect(database.engine).get_columns("known_defect_inspections")
        }
        child_columns = {
            column["name"]
            for column in inspect(database.engine).get_columns("known_defect_instances")
        }
    finally:
        database.dispose()

    assert loaded is not None
    assert loaded.inspection.instance_count == 0
    assert loaded.instances == ()
    for forbidden in ("image", "image_bytes", "filename", "raw_mask", "polygon"):
        assert forbidden not in parent_columns
        assert forbidden not in child_columns


# ADD 2026-08-26: Parent-only history의 one-query ordering과 pagination을 검증한다.
def test_history_is_newest_first_bounded_and_does_not_hydrate_children(tmp_path: Path) -> None:
    repository, database = _repository(tmp_path)
    select_count = 0

    # History query count만 기록해 per-parent child query가 없는지 확인한다.
    def count_selects(
        connection: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        executemany: bool,
    ) -> None:
        nonlocal select_count
        if statement.lstrip().upper().startswith("SELECT"):
            select_count += 1

    try:
        first = repository.create(_values())
        second = repository.create(_values(instances=()))
        third = repository.create(_values())
        base_time = datetime(2026, 8, 26, tzinfo=UTC)
        with database.engine.begin() as connection:
            for index, detail in enumerate((first, second, third)):
                connection.execute(
                    update(KnownDefectInspectionRecord)
                    .where(KnownDefectInspectionRecord.id == detail.inspection.id)
                    .values(created_at=base_time + timedelta(seconds=index))
                )
        event.listen(database.engine, "before_cursor_execute", count_selects)
        page = repository.list(limit=2, offset=0)
        event.remove(database.engine, "before_cursor_execute", count_selects)
        tail = repository.list(limit=2, offset=2)
    finally:
        database.dispose()

    assert [item.id for item in page.items] == [third.inspection.id, second.inspection.id]
    assert page.has_more is True
    assert [item.id for item in tail.items] == [first.inspection.id]
    assert tail.has_more is False
    assert select_count == 1


# ADD 2026-08-26: Child constraint failure가 flushed parent까지 rollback하는지 검증한다.
def test_child_failure_rolls_back_parent_and_all_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, database = _repository(tmp_path)
    invalid_child = replace(
        _values().instances[0],
        mask_pixel_count=0,
        mask_area_ratio=0.25,
    )
    monkeypatch.setattr(KnownDefectCreate, "validate", lambda self: None)
    try:
        with pytest.raises(PersistenceError, match="insert failed"):
            repository.create(_values(instances=(invalid_child,)))
        with database.engine.connect() as connection:
            parent_count = connection.scalar(
                select(func.count()).select_from(KnownDefectInspectionRecord)
            )
            child_count = connection.scalar(
                select(func.count()).select_from(KnownDefectInstanceRecord)
            )
    finally:
        database.dispose()

    assert parent_count == 0
    assert child_count == 0


# ADD 2026-08-26: Unknown UUID와 invalid provenance/spatial values를 사전 거부한다.
def test_missing_detail_and_create_validation(tmp_path: Path) -> None:
    repository, database = _repository(tmp_path)
    try:
        assert repository.get(uuid4()) is None
        with pytest.raises(ValueError, match="image_sha256"):
            repository.create(replace(_values(), image_sha256="invalid"))
        with pytest.raises(ValueError, match="bbox"):
            invalid = replace(_values().instances[0], bbox_x_max=9.0)
            repository.create(_values(instances=(invalid,)))
    finally:
        database.dispose()
