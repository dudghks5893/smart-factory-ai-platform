"""SQLite structural test for the initial Alembic inspection migration."""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


# ADD 2026-08-20: Initial migration upgrade/downgrade가 inspection schema를 왕복하는지 검증한다.
# MODIFY 2026-08-26: Additive known-defect parent/child revision과 기존 row 보존을 검증한다.
def test_initial_inspection_migration_upgrade_and_downgrade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "migration.db"
    database_url = f"sqlite+pysqlite:///{database_path}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = Config("alembic.ini")

    command.upgrade(config, "20260820_01")
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO inspections ("
                "id, model_name, category, is_anomaly, anomaly_score, threshold, "
                "comparison_operator, image_sha256, image_size_bytes, content_type, "
                "model_sha256, artifact_metadata_sha256, threshold_artifact_sha256, "
                "manifest_sha256, device"
                ") VALUES ("
                ":id, 'patchcore', 'metal_nut', 0, 30, 40, '>', :sha, 100, "
                "'image/png', :sha, :sha, :sha, :sha, 'cpu'"
                ")"
            ),
            {"id": "00000000-0000-4000-8000-000000000001", "sha": "a" * 64},
        )
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    inspector = inspect(engine)
    assert {
        "inspections",
        "known_defect_inspections",
        "known_defect_instances",
    } <= set(inspector.get_table_names())
    assert {index["name"] for index in inspector.get_indexes("inspections")} == {
        "ix_inspections_anomaly_created_at",
        "ix_inspections_category_created_at",
        "ix_inspections_created_at",
    }
    assert {index["name"] for index in inspector.get_indexes("known_defect_inspections")} == {
        "ix_known_defect_inspections_created_at"
    }
    assert {index["name"] for index in inspector.get_indexes("known_defect_instances")} == {
        "ix_known_defect_instances_inspection_id"
    }
    foreign_keys = inspector.get_foreign_keys("known_defect_instances")
    assert len(foreign_keys) == 1
    assert foreign_keys[0]["referred_table"] == "known_defect_inspections"
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM inspections")) == 1
    engine.dispose()

    command.downgrade(config, "20260820_01")
    engine = create_engine(database_url)
    table_names = set(inspect(engine).get_table_names())
    assert "inspections" in table_names
    assert "known_defect_inspections" not in table_names
    assert "known_defect_instances" not in table_names
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM inspections")) == 1
    engine.dispose()

    command.downgrade(config, "base")
    engine = create_engine(database_url)
    assert "inspections" not in inspect(engine).get_table_names()
    engine.dispose()
