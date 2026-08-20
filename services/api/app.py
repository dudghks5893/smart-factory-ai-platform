"""FastAPI application factory for process-local PatchCore serving."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from services.api.config import ServingSettings
from services.api.errors import install_exception_handlers
from services.api.routes import router
from services.inference.runtime import ModelRuntime, PatchCoreRuntimeConfig, load_patchcore_runtime
from services.persistence.database import DatabaseManager, create_database_manager
from services.persistence.inspections import (
    InspectionRepository,
    SqlAlchemyInspectionRepository,
)

type RuntimeLoader = Callable[[PatchCoreRuntimeConfig], ModelRuntime]
type DatabaseLoader = Callable[[str], DatabaseManager]
type RepositoryLoader = Callable[[DatabaseManager], InspectionRepository]


# ADD 2026-08-20: Database manager의 request Session factory로 inspection repository를 생성한다.
def load_inspection_repository(database: DatabaseManager) -> InspectionRepository:
    """Build the production repository without creating database schema."""
    return SqlAlchemyInspectionRepository(database.session_factory)


# ADD 2026-08-19: Lifespan startup과 injectable runtime loader를 가진 FastAPI app을 생성한다.
# MODIFY 2026-08-20: Required DB connectivity/repository lifecycle을 model readiness에 통합한다.
def create_app(
    *,
    settings: ServingSettings | None = None,
    runtime_loader: RuntimeLoader = load_patchcore_runtime,
    database_loader: DatabaseLoader = create_database_manager,
    repository_loader: RepositoryLoader = load_inspection_repository,
) -> FastAPI:
    """Create an app that requires database and model readiness during startup."""

    # ADD 2026-08-19: Startup load가 완료된 뒤에만 ready 상태로 전환한다.
    # MODIFY 2026-08-20: DB connection을 먼저 확인하고 shutdown에서 engine을 dispose한다.
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        active_settings = settings or ServingSettings.from_environment()
        active_settings.validate()

        # Required DB connectivity를 확인한 뒤 request-isolated repository를 준비한다.
        database = database_loader(active_settings.database_url)
        try:
            database.check_connection()
            inspection_repository = repository_loader(database)
            inspection_repository.check_ready()

            # Artifact와 threshold를 검증하고 process-local runtime을 정확히 한 번 생성한다.
            runtime = runtime_loader(active_settings.runtime_config())
            application.state.serving_settings = active_settings
            application.state.database = database
            application.state.inspection_repository = inspection_repository
            application.state.serving_runtime = runtime
            yield
        finally:
            application.state.serving_runtime = None
            application.state.inspection_repository = None
            application.state.database = None
            database.dispose()

    app = FastAPI(
        title="SmartFactory PatchCore Inference API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.serving_settings = None
    app.state.serving_runtime = None
    app.state.database = None
    app.state.inspection_repository = None
    install_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()
