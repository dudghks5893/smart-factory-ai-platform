"""FastAPI application factory for process-local PatchCore serving."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from services.api.config import ServingSettings
from services.api.errors import install_exception_handlers
from services.api.routes import router
from services.inference.runtime import ModelRuntime, PatchCoreRuntimeConfig, load_patchcore_runtime

type RuntimeLoader = Callable[[PatchCoreRuntimeConfig], ModelRuntime]


# ADD 2026-08-19: Lifespan startup과 injectable runtime loader를 가진 FastAPI app을 생성한다.
def create_app(
    *,
    settings: ServingSettings | None = None,
    runtime_loader: RuntimeLoader = load_patchcore_runtime,
) -> FastAPI:
    """Create an app that restores one model runtime during startup."""

    # ADD 2026-08-19: Startup load가 완료된 뒤에만 ready 상태로 전환한다.
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        active_settings = settings or ServingSettings.from_environment()
        active_settings.validate()

        # Artifact와 threshold를 검증하고 process-local runtime을 정확히 한 번 생성한다.
        runtime = runtime_loader(active_settings.runtime_config())
        application.state.serving_settings = active_settings
        application.state.serving_runtime = runtime
        yield

    app = FastAPI(
        title="SmartFactory PatchCore Inference API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.serving_settings = None
    app.state.serving_runtime = None
    install_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()
