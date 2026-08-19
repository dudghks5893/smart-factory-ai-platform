"""HTTP routes for PatchCore liveness, readiness, and image inference."""

from __future__ import annotations

import logging
from typing import Annotated, Literal, cast

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from services.api.config import ServingSettings
from services.api.errors import ApiError
from services.api.images import decode_uploaded_image
from services.api.schemas import HealthResponse, InferenceResponse, ReadinessResponse
from services.inference.runtime import ModelRuntime

LOGGER = logging.getLogger(__name__)
router = APIRouter()


# ADD 2026-08-19: Process liveness를 model lifecycle과 독립적으로 반환한다.
@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report that the API process can handle requests."""
    return HealthResponse(status="ok")


# ADD 2026-08-19: Startup에서 model이 복원된 경우에만 readiness를 반환한다.
@router.get("/ready", response_model=ReadinessResponse)
async def ready(request: Request) -> ReadinessResponse:
    """Report loaded model identity or return a model-not-ready error."""
    runtime = _runtime_from_request(request)
    return ReadinessResponse(
        status="ready",
        model_name=runtime.model_name,
        category=runtime.category,
        device=runtime.device,
    )


# ADD 2026-08-19: Multipart image를 검증하고 shared runtime의 image-level prediction을 반환한다.
@router.post("/v1/predictions", response_model=InferenceResponse)
async def create_prediction(
    request: Request,
    image: Annotated[UploadFile, File(description="JPEG or PNG inspection image")],
) -> InferenceResponse:
    """Run one PatchCore image prediction without returning the anomaly map."""
    runtime = _runtime_from_request(request)
    settings = _settings_from_request(request)

    # Upload를 bounded read하고 application-side temporary file 없이 tensor로 decode한다.
    try:
        content = await image.read(settings.max_upload_bytes + 1)
    finally:
        await image.close()
    image_tensor = decode_uploaded_image(
        content,
        content_type=image.content_type,
        max_upload_bytes=settings.max_upload_bytes,
    )

    # Blocking model inference는 threadpool에서 실행하고 내부 failure를 client contract로 변환한다.
    try:
        prediction = await run_in_threadpool(runtime.predict, image_tensor)
    except Exception as exc:
        LOGGER.exception("PatchCore request inference failed", exc_info=exc)
        raise ApiError(500, "inference_failed", "Model inference failed.") from exc
    return InferenceResponse(
        model_name=prediction.model_name,
        category=prediction.category,
        is_anomaly=prediction.is_anomaly,
        anomaly_score=prediction.anomaly_score,
        threshold=prediction.threshold,
        comparison_operator=cast(Literal[">"], prediction.comparison_operator),
    )


# ADD 2026-08-19: Request state에서 ready runtime을 가져오고 unavailable 상태를 503으로 변환한다.
def _runtime_from_request(request: Request) -> ModelRuntime:
    runtime = getattr(request.app.state, "serving_runtime", None)
    if runtime is None:
        raise ApiError(503, "model_not_ready", "Model runtime is not ready.")
    return cast(ModelRuntime, runtime)


# ADD 2026-08-19: Startup에서 검증한 serving settings를 request state에서 반환한다.
def _settings_from_request(request: Request) -> ServingSettings:
    settings = getattr(request.app.state, "serving_settings", None)
    if not isinstance(settings, ServingSettings):
        raise ApiError(503, "model_not_ready", "Model runtime is not ready.")
    return settings
