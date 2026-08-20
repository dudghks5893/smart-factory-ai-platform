"""SQLite-only schema helpers for isolated persistence contract tests."""

from pathlib import Path

from services.persistence.database import create_database_manager
from services.persistence.models import Base


# ADD 2026-08-20: Temporary SQLite file에 test-only inspection schema를 생성한다.
def prepare_sqlite_database(tmp_path: Path, name: str = "inspections.db") -> str:
    """Create tables only for one isolated test database and return its URL."""
    database_url = f"sqlite+pysqlite:///{tmp_path / name}"
    database = create_database_manager(database_url)
    try:
        Base.metadata.create_all(database.engine)
    finally:
        database.dispose()
    return database_url
