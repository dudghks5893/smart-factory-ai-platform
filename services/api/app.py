"""FastAPI application factory for process-local PatchCore serving."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from services.api.config import ServingSettings
from services.api.errors import install_exception_handlers
from services.api.routes import router
from services.api.websockets import (
    CombinedInspectionEventBroadcaster,
    InspectionEventBroadcaster,
    KnownDefectEventBroadcaster,
)
from services.inference.runtime import ModelRuntime, PatchCoreRuntimeConfig, load_patchcore_runtime
from services.inference.yolo_segmentation_runtime import (
    YoloSegmentationAdapter,
    YoloSegmentationRuntimeConfig,
    load_yolo_segmentation_runtime,
)
from services.monitoring.metrics import MonitoringMetrics, metrics_endpoint
from services.monitoring.middleware import HttpMetricsMiddleware
from services.persistence.combined_inspections import (
    CombinedInspectionRepository,
    SqlAlchemyCombinedInspectionRepository,
)
from services.persistence.database import DatabaseManager, create_database_manager
from services.persistence.inspections import (
    InspectionRepository,
    SqlAlchemyInspectionRepository,
)
from services.persistence.known_defects import (
    KnownDefectRepository,
    SqlAlchemyKnownDefectRepository,
)

type RuntimeLoader = Callable[[PatchCoreRuntimeConfig], ModelRuntime]
type YoloRuntimeLoader = Callable[[YoloSegmentationRuntimeConfig], YoloSegmentationAdapter]
type DatabaseLoader = Callable[[str], DatabaseManager]
type RepositoryLoader = Callable[[DatabaseManager], InspectionRepository]
type KnownDefectRepositoryLoader = Callable[[DatabaseManager], KnownDefectRepository]
type CombinedInspectionRepositoryLoader = Callable[[DatabaseManager], CombinedInspectionRepository]

DEFAULT_LIVE_MONITOR_DIR = Path(__file__).resolve().parents[2] / "apps" / "live_monitor"


# ADD 2026-08-20: Database manager의 request Session factory로 inspection repository를 생성한다.
def load_inspection_repository(database: DatabaseManager) -> InspectionRepository:
    """Build the production repository without creating database schema."""
    return SqlAlchemyInspectionRepository(database.session_factory)


# ADD 2026-08-26: Database Session factory로 known-defect parent/child repository를 생성한다.
def load_known_defect_repository(database: DatabaseManager) -> KnownDefectRepository:
    """Build the production known-defect repository without creating schema."""
    return SqlAlchemyKnownDefectRepository(database.session_factory)


# ADD 2026-08-26: Database Session factory로 atomic combined-inspection repository를 생성한다.
def load_combined_inspection_repository(
    database: DatabaseManager,
) -> CombinedInspectionRepository:
    """Build the production combined repository without creating schema."""
    return SqlAlchemyCombinedInspectionRepository(database.session_factory)


# ADD 2026-08-19: Lifespan startup과 injectable runtime loader를 가진 FastAPI app을 생성한다.
# MODIFY 2026-08-25: WebSocket lifecycle과 optional browser monitor static mount를 추가한다.
# MODIFY 2026-08-26: Optional enabled YOLO singleton을 startup/readiness lifecycle에 추가한다.
# MODIFY 2026-08-26: Known-defect repository와 별도 event channel을 lifecycle에 추가한다.
# MODIFY 2026-08-26: Combined correlation repository를 required persistence lifecycle에 추가한다.
# MODIFY 2026-08-26: Combined decision WebSocket channel을 process lifecycle에 추가한다.
def create_app(
    *,
    settings: ServingSettings | None = None,
    runtime_loader: RuntimeLoader = load_patchcore_runtime,
    yolo_runtime_loader: YoloRuntimeLoader = load_yolo_segmentation_runtime,
    database_loader: DatabaseLoader = create_database_manager,
    repository_loader: RepositoryLoader = load_inspection_repository,
    known_defect_repository_loader: KnownDefectRepositoryLoader = (load_known_defect_repository),
    combined_inspection_repository_loader: CombinedInspectionRepositoryLoader = (
        load_combined_inspection_repository
    ),
    inspection_event_broadcaster: InspectionEventBroadcaster | None = None,
    known_defect_event_broadcaster: KnownDefectEventBroadcaster | None = None,
    combined_inspection_event_broadcaster: CombinedInspectionEventBroadcaster | None = None,
    live_monitor_dir: Path = DEFAULT_LIVE_MONITOR_DIR,
) -> FastAPI:
    """Create an app that requires database and model readiness during startup."""

    monitoring_metrics = MonitoringMetrics()
    event_broadcaster = inspection_event_broadcaster or InspectionEventBroadcaster()
    known_defect_broadcaster = known_defect_event_broadcaster or KnownDefectEventBroadcaster()
    combined_broadcaster = (
        combined_inspection_event_broadcaster or CombinedInspectionEventBroadcaster()
    )

    # ADD 2026-08-19: Startup load가 완료된 뒤에만 ready 상태로 전환한다.
    # MODIFY 2026-08-21: Loaded model identity를 app-local Info metric에 한 번 게시한다.
    # MODIFY 2026-08-26: Enabled YOLO load 성공도 ready 전환의 필수 조건으로 포함한다.
    # MODIFY 2026-08-26: Known-defect schema와 WebSocket channel을 required lifecycle에 포함한다.
    # MODIFY 2026-08-26: Combined correlation schema readiness와 state cleanup을 포함한다.
    # MODIFY 2026-08-26: Combined decision event connection cleanup을 포함한다.
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
            known_defect_repository = known_defect_repository_loader(database)
            known_defect_repository.check_ready()
            combined_inspection_repository = combined_inspection_repository_loader(database)
            combined_inspection_repository.check_ready()

            # Artifact와 threshold를 검증하고 process-local runtime을 정확히 한 번 생성한다.
            runtime = runtime_loader(active_settings.runtime_config())
            yolo_runtime = (
                yolo_runtime_loader(active_settings.yolo_segmentation_runtime_config())
                if active_settings.yolo_segmentation_enabled
                else None
            )
            monitoring_metrics.set_model_info(runtime)
            application.state.serving_settings = active_settings
            application.state.database = database
            application.state.inspection_repository = inspection_repository
            application.state.known_defect_repository = known_defect_repository
            application.state.combined_inspection_repository = combined_inspection_repository
            application.state.serving_runtime = runtime
            application.state.yolo_segmentation_runtime = yolo_runtime
            yield
        finally:
            await event_broadcaster.close_all()
            await known_defect_broadcaster.close_all()
            await combined_broadcaster.close_all()
            application.state.yolo_segmentation_runtime = None
            application.state.serving_runtime = None
            application.state.inspection_repository = None
            application.state.known_defect_repository = None
            application.state.combined_inspection_repository = None
            application.state.database = None
            database.dispose()

    app = FastAPI(
        title="SmartFactory PatchCore Inference API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.serving_settings = None
    app.state.serving_runtime = None
    app.state.yolo_segmentation_runtime = None
    app.state.database = None
    app.state.inspection_repository = None
    app.state.known_defect_repository = None
    app.state.combined_inspection_repository = None
    app.state.inspection_event_broadcaster = event_broadcaster
    app.state.known_defect_event_broadcaster = known_defect_broadcaster
    app.state.combined_inspection_event_broadcaster = combined_broadcaster
    app.state.monitoring_metrics = monitoring_metrics
    app.add_middleware(HttpMetricsMiddleware, metrics=monitoring_metrics)
    install_exception_handlers(app)
    app.add_api_route(
        "/metrics",
        metrics_endpoint,
        methods=["GET"],
        include_in_schema=False,
    )
    app.include_router(router)

    # API와 같은 origin에서 REST/WebSocket을 사용하도록 available asset만 mount한다.
    if live_monitor_dir.is_dir():
        app.mount(
            "/live",
            StaticFiles(directory=live_monitor_dir, html=True),
            name="live-monitor",
        )
    return app


app = create_app()
