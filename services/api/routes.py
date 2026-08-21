"""HTTP routes for PatchCore liveness, readiness, and image inference."""

from __future__ import annotations

import logging
from typing import Annotated, Literal, cast
from uuid import UUID

from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from services.api.config import ServingSettings
from services.api.errors import ApiError
from services.api.images import decode_uploaded_image
from services.api.schemas import (
    HealthResponse,
    InferenceResponse,
    InspectionHistoryResponse,
    InspectionResponse,
    ReadinessResponse,
)
from services.inference.runtime import (
    ModelRuntime,
    require_serving_provenance,
)
from services.monitoring.metrics import MonitoringMetrics
from services.persistence.database import DatabaseManager, PersistenceError
from services.persistence.inspections import (
    Inspection,
    InspectionCreate,
    InspectionRepository,
)
from shared.hashing import sha256_bytes

LOGGER = logging.getLogger(__name__)
router = APIRouter()


# ADD 2026-08-19: Process liveness를 model lifecycle과 독립적으로 반환한다.
@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report that the API process can handle requests."""
    return HealthResponse(status="ok")


# ADD 2026-08-19: Startup에서 model이 복원된 경우에만 readiness를 반환한다.
# MODIFY 2026-08-20: Required database connectivity를 readiness 응답 전에 재확인한다.
@router.get("/ready", response_model=ReadinessResponse)
async def ready(request: Request) -> ReadinessResponse:
    """Report model identity only while the required database is reachable."""
    runtime = _runtime_from_request(request)
    database = _database_from_request(request)
    repository = _repository_from_request(request)
    try:
        await run_in_threadpool(database.check_connection)
        await run_in_threadpool(repository.check_ready)
    except PersistenceError as exc:
        raise ApiError(503, "database_not_ready", "Required database is not ready.") from exc
    return ReadinessResponse(
        status="ready",
        model_name=runtime.model_name,
        category=runtime.category,
        device=runtime.device,
    )


# ADD 2026-08-19: Multipart image를 검증하고 shared runtime의 image-level prediction을 반환한다.
# MODIFY 2026-08-21: Inference와 insert 경계를 측정하고 persisted result metric을 기록한다.
@router.post("/v1/predictions", response_model=InferenceResponse)
async def create_prediction(
    request: Request,
    image: Annotated[UploadFile, File(description="JPEG or PNG inspection image")],
) -> InferenceResponse:
    """Run one PatchCore image prediction without returning the anomaly map."""
    runtime = _runtime_from_request(request)
    settings = _settings_from_request(request)
    repository = _repository_from_request(request)
    monitoring = _monitoring_from_request(request)

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
        with monitoring.track_inference(
            model_name=runtime.model_name,
            category=runtime.category,
            device=runtime.device,
        ):
            prediction = await run_in_threadpool(runtime.predict, image_tensor)
    except Exception as exc:
        LOGGER.exception("PatchCore request inference failed", exc_info=exc)
        raise ApiError(500, "inference_failed", "Model inference failed.") from exc

    # 검증된 runtime provenance와 in-memory upload hash로 inspection을 한 번 저장한다.
    try:
        provenance = require_serving_provenance(runtime)
    except TypeError as exc:
        raise ApiError(
            500,
            "inference_provenance_unavailable",
            "Model provenance is unavailable.",
        ) from exc
    try:
        with monitoring.track_persistence(operation="insert"):
            inspection = await run_in_threadpool(
                repository.create,
                InspectionCreate(
                    model_name=prediction.model_name,
                    category=prediction.category,
                    is_anomaly=prediction.is_anomaly,
                    anomaly_score=prediction.anomaly_score,
                    threshold=prediction.threshold,
                    comparison_operator=prediction.comparison_operator,
                    image_sha256=sha256_bytes(content),
                    image_size_bytes=len(content),
                    content_type=(image.content_type or "").lower(),
                    model_sha256=provenance.model_sha256,
                    artifact_metadata_sha256=provenance.artifact_metadata_sha256,
                    threshold_artifact_sha256=provenance.threshold_artifact_sha256,
                    manifest_sha256=provenance.manifest_sha256,
                    device=runtime.device,
                ),
            )
    except PersistenceError:
        monitoring.record_persistence_error(operation="insert")
        raise
    monitoring.record_prediction(
        category=prediction.category,
        is_anomaly=prediction.is_anomaly,
    )
    return InferenceResponse(
        inspection_id=inspection.id,
        model_name=prediction.model_name,
        category=prediction.category,
        is_anomaly=prediction.is_anomaly,
        anomaly_score=prediction.anomaly_score,
        threshold=prediction.threshold,
        comparison_operator=cast(Literal[">"], prediction.comparison_operator),
    )


# ADD 2026-08-20: UUID로 persisted inspection detail을 조회하고 missing row를 404로 변환한다.
@router.get("/v1/inspections/{inspection_id}", response_model=InspectionResponse)
async def get_inspection(request: Request, inspection_id: UUID) -> InspectionResponse:
    """Return one persisted inspection by its stable UUID."""
    repository = _repository_from_request(request)
    inspection = await run_in_threadpool(repository.get, inspection_id)
    if inspection is None:
        raise ApiError(404, "inspection_not_found", "Inspection was not found.")
    return _inspection_response(inspection)


# ADD 2026-08-20: Inspection history를 optional filter와 bounded offset pagination으로 조회한다.
@router.get("/v1/inspections", response_model=InspectionHistoryResponse)
async def list_inspections(
    request: Request,
    category: Annotated[str | None, Query(min_length=1, max_length=100)] = None,
    is_anomaly: bool | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> InspectionHistoryResponse:
    """Return newest inspections with deterministic ordering and has-more metadata."""
    repository = _repository_from_request(request)
    page = await run_in_threadpool(
        repository.list,
        category=category,
        is_anomaly=is_anomaly,
        limit=limit,
        offset=offset,
    )
    return InspectionHistoryResponse(
        items=[_inspection_response(item) for item in page.items],
        limit=page.limit,
        offset=page.offset,
        returned_count=len(page.items),
        has_more=page.has_more,
    )


# ADD 2026-08-20: Persistence domain item을 public inspection response로 변환한다.
def _inspection_response(inspection: Inspection) -> InspectionResponse:
    return InspectionResponse(
        inspection_id=inspection.id,
        created_at=inspection.created_at,
        model_name=inspection.model_name,
        category=inspection.category,
        is_anomaly=inspection.is_anomaly,
        anomaly_score=inspection.anomaly_score,
        threshold=inspection.threshold,
        comparison_operator=cast(Literal[">"], inspection.comparison_operator),
        image_sha256=inspection.image_sha256,
        image_size_bytes=inspection.image_size_bytes,
        content_type=inspection.content_type,
        model_sha256=inspection.model_sha256,
        artifact_metadata_sha256=inspection.artifact_metadata_sha256,
        threshold_artifact_sha256=inspection.threshold_artifact_sha256,
        manifest_sha256=inspection.manifest_sha256,
        device=inspection.device,
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


# ADD 2026-08-20: Request state에서 required database manager를 반환한다.
def _database_from_request(request: Request) -> DatabaseManager:
    database = getattr(request.app.state, "database", None)
    if not isinstance(database, DatabaseManager):
        raise ApiError(503, "database_not_ready", "Required database is not ready.")
    return database


# ADD 2026-08-20: Request state에서 inspection repository를 가져온다.
def _repository_from_request(request: Request) -> InspectionRepository:
    repository = getattr(request.app.state, "inspection_repository", None)
    if repository is None:
        raise ApiError(503, "database_not_ready", "Required database is not ready.")
    return cast(InspectionRepository, repository)


# ADD 2026-08-21: Request app에서 instance-isolated monitoring registry를 반환한다.
def _monitoring_from_request(request: Request) -> MonitoringMetrics:
    monitoring = getattr(request.app.state, "monitoring_metrics", None)
    if not isinstance(monitoring, MonitoringMetrics):
        raise RuntimeError("Application monitoring registry is unavailable.")
    return monitoring
