"""HTTP and WebSocket routes for PatchCore inspection serving."""

from __future__ import annotations

import logging
from typing import Annotated, Literal, cast
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool

from services.api.config import ServingSettings
from services.api.errors import ApiError
from services.api.images import decode_uploaded_image, decode_uploaded_rgb_array
from services.api.schemas import (
    HealthResponse,
    InferenceResponse,
    InspectionCreatedEvent,
    InspectionCreatedPayload,
    InspectionHistoryResponse,
    InspectionResponse,
    KnownDefectBoundingBox,
    KnownDefectImageSummary,
    KnownDefectInstanceResponse,
    KnownDefectMaskSummary,
    KnownDefectModelIdentity,
    KnownDefectResponse,
    ReadinessResponse,
)
from services.api.websockets import InspectionEventBroadcaster
from services.inference.runtime import (
    ModelRuntime,
    require_serving_provenance,
)
from services.inference.yolo_segmentation_runtime import (
    YoloSegmentationAdapter,
    YoloSegmentationResult,
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
# MODIFY 2026-08-26: Enabled YOLO singleton도 aggregate readiness에 포함한다.
@router.get("/ready", response_model=ReadinessResponse)
async def ready(request: Request) -> ReadinessResponse:
    """Report model identity only while the required database is reachable."""
    runtime = _runtime_from_request(request)
    _require_yolo_readiness(request)
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
# MODIFY 2026-08-25: Commit된 inspection event를 response 이후 best-effort broadcast한다.
# MODIFY 2026-08-26: Shared bounded upload read를 YOLO endpoint와 재사용한다.
@router.post("/v1/predictions", response_model=InferenceResponse)
async def create_prediction(
    request: Request,
    background_tasks: BackgroundTasks,
    image: Annotated[UploadFile, File(description="JPEG or PNG inspection image")],
) -> InferenceResponse:
    """Run one PatchCore image prediction without returning the anomaly map."""
    runtime = _runtime_from_request(request)
    settings = _settings_from_request(request)
    repository = _repository_from_request(request)
    monitoring = _monitoring_from_request(request)

    # Upload를 bounded read하고 application-side temporary file 없이 tensor로 decode한다.
    content = await _read_bounded_upload(image, max_upload_bytes=settings.max_upload_bytes)
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

    # Commit된 domain value로만 event를 만들고 HTTP response 이후 background broadcast를 예약한다.
    event = _inspection_created_event(inspection)
    broadcaster = _broadcaster_from_request(request)
    background_tasks.add_task(broadcaster.broadcast, event)
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


# ADD 2026-08-26: Multipart image를 enabled YOLO singleton으로 known-defect segmentation한다.
@router.post("/v1/known-defects", response_model=KnownDefectResponse)
async def create_known_defect_prediction(
    request: Request,
    image: Annotated[UploadFile, File(description="JPEG or PNG inspection image")],
) -> KnownDefectResponse:
    """Return normalized known-defect instances without persistence or final disposition."""
    runtime = _yolo_runtime_from_request(request)
    settings = _settings_from_request(request)
    monitoring = _monitoring_from_request(request)

    # PatchCore와 같은 bounded media/format 정책으로 upload를 RGB array까지 decode한다.
    content = await _read_bounded_upload(image, max_upload_bytes=settings.max_upload_bytes)
    image_rgb = decode_uploaded_rgb_array(
        content,
        content_type=image.content_type,
        max_upload_bytes=settings.max_upload_bytes,
    )

    # Blocking singleton inference를 threadpool에서 실행하고 internal detail은 숨긴다.
    try:
        with monitoring.track_inference(
            model_name=runtime.metadata.model_name,
            category=runtime.metadata.category,
            device=runtime.device,
        ):
            prediction = await run_in_threadpool(
                runtime.predict,
                image_rgb,
                diagnostic_confidence=settings.yolo_segmentation_diagnostic_confidence,
            )
    except Exception as exc:
        LOGGER.exception("YOLO segmentation request inference failed", exc_info=exc)
        raise ApiError(500, "inference_failed", "Model inference failed.") from exc
    return _known_defect_response(
        runtime=runtime,
        prediction=prediction,
        diagnostic_confidence=settings.yolo_segmentation_diagnostic_confidence,
    )


# ADD 2026-08-25: Server-push connection을 등록하고 peer disconnect까지 command 없이 유지한다.
@router.websocket("/v1/ws/inspections")
async def stream_inspections(websocket: WebSocket) -> None:
    """Push best-effort inspection.created events from this API process."""
    broadcaster = _broadcaster_from_websocket(websocket)
    await broadcaster.connect(websocket)
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        await broadcaster.disconnect(websocket)


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


# ADD 2026-08-25: Committed inspection을 compact device-independent live event로 변환한다.
def _inspection_created_event(inspection: Inspection) -> InspectionCreatedEvent:
    return InspectionCreatedEvent(
        inspection=InspectionCreatedPayload(
            inspection_id=inspection.id,
            model_name=inspection.model_name,
            category=inspection.category,
            is_anomaly=inspection.is_anomaly,
            anomaly_score=inspection.anomaly_score,
            threshold=inspection.threshold,
            comparison_operator=cast(Literal[">"], inspection.comparison_operator),
            device=inspection.device,
            created_at=inspection.created_at,
        )
    )


# ADD 2026-08-26: Runtime result를 raw mask 없는 known-defect API schema로 변환한다.
def _known_defect_response(
    *,
    runtime: YoloSegmentationAdapter,
    prediction: YoloSegmentationResult,
    diagnostic_confidence: float,
) -> KnownDefectResponse:
    return KnownDefectResponse(
        model=KnownDefectModelIdentity(
            name=runtime.metadata.model_name,
            task="segment",
            category=runtime.metadata.category,
            device=runtime.device,
        ),
        image=KnownDefectImageSummary(
            width=prediction.image_width,
            height=prediction.image_height,
        ),
        diagnostic_confidence=diagnostic_confidence,
        inference_ms=prediction.inference_ms,
        instances=[
            KnownDefectInstanceResponse(
                class_id=instance.class_id,
                class_name=instance.class_name,
                confidence=instance.confidence,
                box=KnownDefectBoundingBox(
                    x_min=instance.box_xyxy[0],
                    y_min=instance.box_xyxy[1],
                    x_max=instance.box_xyxy[2],
                    y_max=instance.box_xyxy[3],
                ),
                mask=KnownDefectMaskSummary(
                    pixel_count=instance.mask_pixel_count,
                    area_ratio=instance.mask_area_ratio,
                ),
            )
            for instance in prediction.instances
        ],
    )


# ADD 2026-08-26: UploadFile을 bounded read하고 inference 전에 handle을 닫는다.
async def _read_bounded_upload(image: UploadFile, *, max_upload_bytes: int) -> bytes:
    try:
        return await image.read(max_upload_bytes + 1)
    finally:
        await image.close()


# ADD 2026-08-19: Request state에서 ready runtime을 가져오고 unavailable 상태를 503으로 변환한다.
def _runtime_from_request(request: Request) -> ModelRuntime:
    runtime = getattr(request.app.state, "serving_runtime", None)
    if runtime is None:
        raise ApiError(503, "model_not_ready", "Model runtime is not ready.")
    return cast(ModelRuntime, runtime)


# ADD 2026-08-26: Request state에서 enabled YOLO runtime을 가져오거나 stable 503을 반환한다.
def _yolo_runtime_from_request(request: Request) -> YoloSegmentationAdapter:
    settings = _settings_from_request(request)
    if not settings.yolo_segmentation_enabled:
        raise ApiError(
            503,
            "known_defect_model_disabled",
            "Known-defect model is disabled.",
        )
    runtime = getattr(request.app.state, "yolo_segmentation_runtime", None)
    if runtime is None:
        raise ApiError(
            503,
            "known_defect_model_not_ready",
            "Known-defect model runtime is not ready.",
        )
    return cast(YoloSegmentationAdapter, runtime)


# ADD 2026-08-26: Enabled YOLO component가 없으면 aggregate readiness를 503으로 전환한다.
def _require_yolo_readiness(request: Request) -> None:
    settings = _settings_from_request(request)
    if (
        settings.yolo_segmentation_enabled
        and getattr(
            request.app.state,
            "yolo_segmentation_runtime",
            None,
        )
        is None
    ):
        raise ApiError(
            503,
            "known_defect_model_not_ready",
            "Known-defect model runtime is not ready.",
        )


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


# ADD 2026-08-25: Request app의 process-local inspection event broadcaster를 반환한다.
def _broadcaster_from_request(request: Request) -> InspectionEventBroadcaster:
    broadcaster = getattr(request.app.state, "inspection_event_broadcaster", None)
    if not isinstance(broadcaster, InspectionEventBroadcaster):
        raise RuntimeError("Inspection event broadcaster is unavailable.")
    return broadcaster


# ADD 2026-08-25: WebSocket app의 process-local inspection event broadcaster를 반환한다.
def _broadcaster_from_websocket(websocket: WebSocket) -> InspectionEventBroadcaster:
    broadcaster = getattr(websocket.app.state, "inspection_event_broadcaster", None)
    if not isinstance(broadcaster, InspectionEventBroadcaster):
        raise RuntimeError("Inspection event broadcaster is unavailable.")
    return broadcaster
