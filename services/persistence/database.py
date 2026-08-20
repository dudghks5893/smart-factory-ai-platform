"""SQLAlchemy engine and request-session lifecycle for application persistence."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker


class PersistenceError(RuntimeError):
    """Database operation failure safe to map to a generic API error."""


@dataclass
class DatabaseManager:
    """Process-local engine and factory for independent request work units."""

    engine: Engine
    session_factory: sessionmaker[Session]

    # ADD 2026-08-20: Required database dependency에 SELECT 1 connectivity를 확인한다.
    def check_connection(self) -> None:
        """Fail when the configured database cannot complete a trivial query."""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            raise PersistenceError("Database connectivity check failed.") from exc

    # ADD 2026-08-20: Application shutdown에서 pooled database connection을 정리한다.
    def dispose(self) -> None:
        """Dispose all engine connections owned by this process."""
        self.engine.dispose()


# ADD 2026-08-20: DATABASE_URL로 pooled engine과 request별 Session factory를 생성한다.
def create_database_manager(database_url: str) -> DatabaseManager:
    """Build persistence resources without creating or migrating database tables."""
    try:
        make_url(database_url)
        engine = create_engine(database_url, pool_pre_ping=True)
    except (TypeError, ValueError, SQLAlchemyError) as exc:
        raise ValueError("DATABASE_URL is not a valid SQLAlchemy database URL.") from exc
    return DatabaseManager(
        engine=engine,
        session_factory=sessionmaker(bind=engine, expire_on_commit=False),
    )
