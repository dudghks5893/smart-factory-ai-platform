"""SQLite structural test for the initial Alembic inspection migration."""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


# ADD 2026-08-20: Initial migration upgrade/downgrade가 inspection schema를 왕복하는지 검증한다.
def test_initial_inspection_migration_upgrade_and_downgrade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "migration.db"
    database_url = f"sqlite+pysqlite:///{database_path}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = Config("alembic.ini")

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    inspector = inspect(engine)
    assert "inspections" in inspector.get_table_names()
    assert {index["name"] for index in inspector.get_indexes("inspections")} == {
        "ix_inspections_anomaly_created_at",
        "ix_inspections_category_created_at",
        "ix_inspections_created_at",
    }
    engine.dispose()

    command.downgrade(config, "base")
    engine = create_engine(database_url)
    assert "inspections" not in inspect(engine).get_table_names()
    engine.dispose()
