"""HTTP and WebSocket routes for PatchCore inspection serving."""

from __future__ import annotations

import logging
from typing import Annotated, Literal, cast
from uuid import UUID, uuid4

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
from services.api.images import (
    decode_uploaded_image,
    decode_uploaded_inference_inputs,
    decode_uploaded_rgb_array,
)
from services.api.schemas import (
    CombinedInspectionImageResponse,
    CombinedInspectionResponse,
    CombinedInspectionTimings,
    CombinedPatchCoreResponse,
    HealthResponse,
    InferenceResponse,
    InspectionCreatedEvent,
    InspectionCreatedPayload,
    InspectionHistoryResponse,
    InspectionResponse,
    KnownDefectBoundingBox,
    KnownDefectCreatedEvent,
    KnownDefectCreatedPayload,
    KnownDefectDetailResponse,
    KnownDefectHistoryItemResponse,
    KnownDefectHistoryResponse,
    KnownDefectImageSummary,
    KnownDefectInstanceResponse,
    KnownDefectMaskSummary,
    KnownDefectModelIdentity,
    KnownDefectPersistedInstanceResponse,
    KnownDefectResponse,
    ReadinessResponse,
)
from services.api.websockets import (
    InspectionEventBroadcaster,
    KnownDefectEventBroadcaster,
)
from services.inference.combined import run_combined_inference
from services.inference.runtime import (
    InferenceResult,
    ModelRuntime,
    ServingProvenance,
    require_serving_provenance,
)
from services.inference.yolo_segmentation_runtime import (
    YoloSegmentationAdapter,
    YoloSegmentationProvenance,
    YoloSegmentationResult,
)
from services.monitoring.metrics import MonitoringMetrics
from services.persistence.combined_inspections import (
    CombinedInspection,
    CombinedInspectionCreate,
    CombinedInspectionRepository,
)
from services.persistence.database import DatabaseManager, PersistenceError
from services.persistence.inspections import (
    Inspection,
    InspectionCreate,
    InspectionRepository,
)
from services.persistence.known_defects import (
    KnownDefectCreate,
    KnownDefectInspection,
    KnownDefectInspectionDetail,
    KnownDefectInstance,
    KnownDefectInstanceCreate,
    KnownDefectRepository,
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
# MODIFY 2026-08-26: Known-defect parent/child schema readiness도 재확인한다.
# MODIFY 2026-08-26: Combined correlation schema readiness도 재확인한다.
@router.get("/ready", response_model=ReadinessResponse)
async def ready(request: Request) -> ReadinessResponse:
    """Report model identity only while the required database is reachable."""
    runtime = _runtime_from_request(request)
    _require_yolo_readiness(request)
    database = _database_from_request(request)
    repository = _repository_from_request(request)
    known_defect_repository = _known_defect_repository_from_request(request)
    combined_repository = _combined_repository_from_request(request)
    try:
        await run_in_threadpool(database.check_connection)
        await run_in_threadpool(repository.check_ready)
        await run_in_threadpool(known_defect_repository.check_ready)
        await run_in_threadpool(combined_repository.check_ready)
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
# MODIFY 2026-08-26: Combined endpoint와 persistence-value builder를 공유한다.
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
                _patchcore_create_values(
                    prediction=prediction,
                    runtime=runtime,
                    provenance=provenance,
                    image_sha256=sha256_bytes(content),
                    image_size_bytes=len(content),
                    content_type=(image.content_type or "").lower(),
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
# MODIFY 2026-08-26: Result를 atomically persist하고 commit 뒤 event를 예약한다.
# MODIFY 2026-08-26: Combined endpoint와 normalized persistence-value builder를 공유한다.
@router.post("/v1/known-defects", response_model=KnownDefectResponse)
async def create_known_defect_prediction(
    request: Request,
    background_tasks: BackgroundTasks,
    image: Annotated[UploadFile, File(description="JPEG or PNG inspection image")],
) -> KnownDefectResponse:
    """Persist normalized known-defect instances without a final disposition."""
    runtime = _yolo_runtime_from_request(request)
    settings = _settings_from_request(request)
    repository = _known_defect_repository_from_request(request)
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

    # Runtime-validated provenance와 compact instance summary를 한 DB transaction으로 저장한다.
    provenance = _require_yolo_provenance(runtime)
    try:
        with monitoring.track_persistence(operation="insert"):
            persisted = await run_in_threadpool(
                repository.create,
                _known_defect_create_values(
                    runtime=runtime,
                    prediction=prediction,
                    provenance=provenance,
                    diagnostic_confidence=settings.yolo_segmentation_diagnostic_confidence,
                    image_sha256=sha256_bytes(content),
                ),
            )
    except PersistenceError:
        monitoring.record_persistence_error(operation="insert")
        raise

    # Commit된 durable object로 event를 만들고 response 이후 best-effort broadcast한다.
    event = _known_defect_created_event(persisted)
    broadcaster = _known_defect_broadcaster_from_request(request)
    background_tasks.add_task(broadcaster.broadcast, event)
    return _known_defect_response(
        inspection_id=persisted.inspection.id,
        runtime=runtime,
        prediction=prediction,
        diagnostic_confidence=settings.yolo_segmentation_diagnostic_confidence,
    )


# ADD 2026-08-26: One decoded upload에서 PatchCore와 YOLO를 병렬 실행하고 원자적으로 저장한다.
@router.post("/v1/combined-inspections", response_model=CombinedInspectionResponse)
async def create_combined_inspection(
    request: Request,
    background_tasks: BackgroundTasks,
    image: Annotated[UploadFile, File(description="JPEG or PNG inspection image")],
) -> CombinedInspectionResponse:
    """Persist two independent model observations without deriving a disposition."""
    combined_inspection_id = uuid4()
    patchcore_runtime = _runtime_from_request(request)
    yolo_runtime = _yolo_runtime_from_request(request)
    settings = _settings_from_request(request)
    repository = _combined_repository_from_request(request)
    monitoring = _monitoring_from_request(request)

    # Upload를 한 번만 읽고 검증한 RGB decode에서 양쪽 runtime input을 파생한다.
    content = await _read_bounded_upload(image, max_upload_bytes=settings.max_upload_bytes)
    decoded = decode_uploaded_inference_inputs(
        content,
        content_type=image.content_type,
        max_upload_bytes=settings.max_upload_bytes,
    )
    image_sha256 = sha256_bytes(content)
    content_type = (image.content_type or "").lower()

    # 각 runtime의 기존 instance lock을 유지하면서 서로 다른 threadpool worker에서 실행한다.
    def predict_patchcore() -> InferenceResult:
        with monitoring.track_inference(
            model_name=patchcore_runtime.model_name,
            category=patchcore_runtime.category,
            device=patchcore_runtime.device,
        ):
            return patchcore_runtime.predict(decoded.patchcore_tensor)

    def predict_known_defects() -> YoloSegmentationResult:
        with monitoring.track_inference(
            model_name=yolo_runtime.metadata.model_name,
            category=yolo_runtime.metadata.category,
            device=yolo_runtime.device,
        ):
            return yolo_runtime.predict(
                decoded.yolo_rgb,
                diagnostic_confidence=settings.yolo_segmentation_diagnostic_confidence,
            )

    try:
        inference = await run_combined_inference(
            predict_patchcore,
            predict_known_defects,
        )
    except Exception as exc:
        LOGGER.exception("Combined inspection inference failed", exc_info=exc)
        raise ApiError(500, "inference_failed", "Model inference failed.") from exc
    patchcore_prediction = inference.patchcore

    # 두 provenance와 model output을 검증한 뒤 양쪽 child와 correlation을 한 번에 commit한다.
    try:
        patchcore_provenance = require_serving_provenance(patchcore_runtime)
    except TypeError as exc:
        raise ApiError(
            500,
            "inference_provenance_unavailable",
            "Model provenance is unavailable.",
        ) from exc
    yolo_provenance = _require_yolo_provenance(yolo_runtime)
    patchcore_values = _patchcore_create_values(
        prediction=patchcore_prediction,
        runtime=patchcore_runtime,
        provenance=patchcore_provenance,
        image_sha256=image_sha256,
        image_size_bytes=len(content),
        content_type=content_type,
    )
    known_defect_values = _known_defect_create_values(
        runtime=yolo_runtime,
        prediction=inference.known_defect,
        provenance=yolo_provenance,
        diagnostic_confidence=settings.yolo_segmentation_diagnostic_confidence,
        image_sha256=image_sha256,
    )
    try:
        with monitoring.track_persistence(operation="insert"):
            persisted = await run_in_threadpool(
                repository.create,
                CombinedInspectionCreate(
                    id=combined_inspection_id,
                    image_sha256=image_sha256,
                    image_width=decoded.width,
                    image_height=decoded.height,
                    image_size_bytes=len(content),
                    content_type=content_type,
                    patchcore_inference_ms=inference.patchcore_inference_ms,
                    orchestration_ms=inference.orchestration_ms,
                    patchcore=patchcore_values,
                    known_defect=known_defect_values,
                ),
            )
    except PersistenceError:
        monitoring.record_persistence_error(operation="insert")
        raise

    # Commit 후 기존 독립 WebSocket channel에 각 child event를 그대로 게시한다.
    background_tasks.add_task(
        _broadcaster_from_request(request).broadcast,
        _inspection_created_event(persisted.patchcore),
    )
    background_tasks.add_task(
        _known_defect_broadcaster_from_request(request).broadcast,
        _known_defect_created_event(persisted.known_defect),
    )
    monitoring.record_prediction(
        category=persisted.patchcore.category,
        is_anomaly=persisted.patchcore.is_anomaly,
    )
    return _combined_inspection_response(persisted)


# ADD 2026-08-26: Combined UUID로 correlation과 두 durable child result를 복구한다.
@router.get(
    "/v1/combined-inspections/{combined_inspection_id}",
    response_model=CombinedInspectionResponse,
)
async def get_combined_inspection(
    request: Request,
    combined_inspection_id: UUID,
) -> CombinedInspectionResponse:
    repository = _combined_repository_from_request(request)
    combined = await run_in_threadpool(repository.get, combined_inspection_id)
    if combined is None:
        raise ApiError(
            404,
            "combined_inspection_not_found",
            "Combined inspection was not found.",
        )
    return _combined_inspection_response(combined)


# ADD 2026-08-26: Known-defect parent를 child hydration 없이 newest-first 조회한다.
@router.get("/v1/known-defects", response_model=KnownDefectHistoryResponse)
async def list_known_defect_inspections(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> KnownDefectHistoryResponse:
    repository = _known_defect_repository_from_request(request)
    page = await run_in_threadpool(repository.list, limit=limit, offset=offset)
    return KnownDefectHistoryResponse(
        items=[_known_defect_history_item(item) for item in page.items],
        limit=page.limit,
        offset=page.offset,
        returned_count=len(page.items),
        has_more=page.has_more,
    )


# ADD 2026-08-26: Persisted known-defect parent provenance와 ordered children을 복구한다.
@router.get(
    "/v1/known-defects/{inspection_id}",
    response_model=KnownDefectDetailResponse,
)
async def get_known_defect_inspection(
    request: Request,
    inspection_id: UUID,
) -> KnownDefectDetailResponse:
    repository = _known_defect_repository_from_request(request)
    detail = await run_in_threadpool(repository.get, inspection_id)
    if detail is None:
        raise ApiError(
            404,
            "known_defect_inspection_not_found",
            "Known-defect inspection was not found.",
        )
    return _known_defect_detail_response(detail)


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


# ADD 2026-08-26: Existing inspection channel과 분리된 known-defect notification stream을 제공한다.
@router.websocket("/v1/ws/known-defects")
async def stream_known_defects(websocket: WebSocket) -> None:
    """Push committed known_defect.created notifications on a separate channel."""
    broadcaster = _known_defect_broadcaster_from_websocket(websocket)
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
    inspection_id: UUID,
    runtime: YoloSegmentationAdapter,
    prediction: YoloSegmentationResult,
    diagnostic_confidence: float,
) -> KnownDefectResponse:
    return KnownDefectResponse(
        inspection_id=inspection_id,
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


# ADD 2026-08-26: Runtime prediction과 shared upload provenance를 PatchCore insert 값으로 변환한다.
def _patchcore_create_values(
    *,
    prediction: InferenceResult,
    runtime: ModelRuntime,
    provenance: ServingProvenance,
    image_sha256: str,
    image_size_bytes: int,
    content_type: str,
) -> InspectionCreate:
    return InspectionCreate(
        model_name=prediction.model_name,
        category=prediction.category,
        is_anomaly=prediction.is_anomaly,
        anomaly_score=prediction.anomaly_score,
        threshold=prediction.threshold,
        comparison_operator=prediction.comparison_operator,
        image_sha256=image_sha256,
        image_size_bytes=image_size_bytes,
        content_type=content_type,
        model_sha256=provenance.model_sha256,
        artifact_metadata_sha256=provenance.artifact_metadata_sha256,
        threshold_artifact_sha256=provenance.threshold_artifact_sha256,
        manifest_sha256=provenance.manifest_sha256,
        device=runtime.device,
    )


# ADD 2026-08-26: Normalized YOLO result를 raw mask 없는 durable parent/child 값으로 변환한다.
def _known_defect_create_values(
    *,
    runtime: YoloSegmentationAdapter,
    prediction: YoloSegmentationResult,
    provenance: YoloSegmentationProvenance,
    diagnostic_confidence: float,
    image_sha256: str,
) -> KnownDefectCreate:
    return KnownDefectCreate(
        model_name=runtime.metadata.model_name,
        task=runtime.metadata.task,
        category=runtime.metadata.category,
        device=runtime.device,
        diagnostic_confidence=diagnostic_confidence,
        inference_ms=prediction.inference_ms,
        image_width=prediction.image_width,
        image_height=prediction.image_height,
        image_sha256=image_sha256,
        model_sha256=provenance.model_sha256,
        artifact_metadata_sha256=provenance.artifact_metadata_sha256,
        dataset_manifest_sha256=provenance.dataset_manifest_sha256,
        dataset_semantic_fingerprint_sha256=(provenance.dataset_semantic_fingerprint_sha256),
        instances=tuple(
            KnownDefectInstanceCreate(
                class_id=instance.class_id,
                class_name=instance.class_name,
                confidence=instance.confidence,
                bbox_x_min=instance.box_xyxy[0],
                bbox_y_min=instance.box_xyxy[1],
                bbox_x_max=instance.box_xyxy[2],
                bbox_y_max=instance.box_xyxy[3],
                mask_pixel_count=instance.mask_pixel_count,
                mask_area_ratio=instance.mask_area_ratio,
            )
            for instance in prediction.instances
        ),
    )


# ADD 2026-08-26: Durable correlation과 child rows를 recoverable combined response로 변환한다.
def _combined_inspection_response(
    combined: CombinedInspection,
) -> CombinedInspectionResponse:
    patchcore = combined.patchcore
    return CombinedInspectionResponse(
        combined_inspection_id=combined.id,
        created_at=combined.created_at,
        image=CombinedInspectionImageResponse(
            width=combined.image_width,
            height=combined.image_height,
            sha256=combined.image_sha256,
        ),
        patchcore=CombinedPatchCoreResponse(
            inspection_id=patchcore.id,
            model_name=patchcore.model_name,
            category=patchcore.category,
            device=patchcore.device,
            is_anomaly=patchcore.is_anomaly,
            anomaly_score=patchcore.anomaly_score,
            threshold=patchcore.threshold,
            comparison_operator=cast(Literal[">"], patchcore.comparison_operator),
        ),
        known_defects=_known_defect_detail_response(combined.known_defect),
        timings=CombinedInspectionTimings(
            patchcore_inference_ms=combined.patchcore_inference_ms,
            yolo_inference_ms=combined.known_defect.inspection.inference_ms,
            orchestration_ms=combined.orchestration_ms,
        ),
    )


# ADD 2026-08-26: Persisted parent를 child-free history response로 변환한다.
def _known_defect_history_item(
    inspection: KnownDefectInspection,
) -> KnownDefectHistoryItemResponse:
    return KnownDefectHistoryItemResponse(
        inspection_id=inspection.id,
        created_at=inspection.created_at,
        model=KnownDefectModelIdentity(
            name=inspection.model_name,
            task="segment",
            category=inspection.category,
            device=inspection.device,
        ),
        image=KnownDefectImageSummary(
            width=inspection.image_width,
            height=inspection.image_height,
        ),
        diagnostic_confidence=inspection.diagnostic_confidence,
        inference_ms=inspection.inference_ms,
        instance_count=inspection.instance_count,
    )


# ADD 2026-08-26: Persisted child를 raw mask 없는 ordered detail schema로 변환한다.
def _known_defect_persisted_instance(
    instance: KnownDefectInstance,
) -> KnownDefectPersistedInstanceResponse:
    return KnownDefectPersistedInstanceResponse(
        instance_id=instance.id,
        instance_index=instance.instance_index,
        class_id=instance.class_id,
        class_name=instance.class_name,
        confidence=instance.confidence,
        box=KnownDefectBoundingBox(
            x_min=instance.bbox_x_min,
            y_min=instance.bbox_y_min,
            x_max=instance.bbox_x_max,
            y_max=instance.bbox_y_max,
        ),
        mask=KnownDefectMaskSummary(
            pixel_count=instance.mask_pixel_count,
            area_ratio=instance.mask_area_ratio,
        ),
    )


# ADD 2026-08-26: Durable parent provenance와 ordered children을 REST detail로 변환한다.
def _known_defect_detail_response(
    detail: KnownDefectInspectionDetail,
) -> KnownDefectDetailResponse:
    inspection = detail.inspection
    return KnownDefectDetailResponse(
        inspection_id=inspection.id,
        created_at=inspection.created_at,
        model=KnownDefectModelIdentity(
            name=inspection.model_name,
            task="segment",
            category=inspection.category,
            device=inspection.device,
        ),
        image=KnownDefectImageSummary(
            width=inspection.image_width,
            height=inspection.image_height,
        ),
        diagnostic_confidence=inspection.diagnostic_confidence,
        inference_ms=inspection.inference_ms,
        image_sha256=inspection.image_sha256,
        model_sha256=inspection.model_sha256,
        artifact_metadata_sha256=inspection.artifact_metadata_sha256,
        dataset_manifest_sha256=inspection.dataset_manifest_sha256,
        dataset_semantic_fingerprint_sha256=(inspection.dataset_semantic_fingerprint_sha256),
        instance_count=inspection.instance_count,
        instances=[_known_defect_persisted_instance(instance) for instance in detail.instances],
    )


# ADD 2026-08-26: Committed detail에서 compact unique-class live event를 생성한다.
def _known_defect_created_event(
    detail: KnownDefectInspectionDetail,
) -> KnownDefectCreatedEvent:
    inspection = detail.inspection
    classes = list(dict.fromkeys(instance.class_name for instance in detail.instances))
    return KnownDefectCreatedEvent(
        inspection=KnownDefectCreatedPayload(
            inspection_id=inspection.id,
            model_name=inspection.model_name,
            category=inspection.category,
            device=inspection.device,
            diagnostic_confidence=inspection.diagnostic_confidence,
            instance_count=inspection.instance_count,
            classes=classes,
            created_at=inspection.created_at,
        )
    )


# ADD 2026-08-26: Runtime provenance를 persistence 전에 concrete validated type으로 요구한다.
def _require_yolo_provenance(
    runtime: YoloSegmentationAdapter,
) -> YoloSegmentationProvenance:
    provenance = runtime.provenance
    if not isinstance(provenance, YoloSegmentationProvenance):
        raise ApiError(
            500,
            "inference_provenance_unavailable",
            "Model provenance is unavailable.",
        )
    return provenance


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


# ADD 2026-08-26: Request app에서 migrated known-defect repository를 반환한다.
def _known_defect_repository_from_request(request: Request) -> KnownDefectRepository:
    repository = getattr(request.app.state, "known_defect_repository", None)
    if repository is None:
        raise ApiError(503, "database_not_ready", "Required database is not ready.")
    return cast(KnownDefectRepository, repository)


# ADD 2026-08-26: Request app에서 migrated combined-inspection repository를 반환한다.
def _combined_repository_from_request(request: Request) -> CombinedInspectionRepository:
    repository = getattr(request.app.state, "combined_inspection_repository", None)
    if repository is None:
        raise ApiError(503, "database_not_ready", "Required database is not ready.")
    return cast(CombinedInspectionRepository, repository)


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


# ADD 2026-08-26: Request app에서 dedicated known-defect event channel을 반환한다.
def _known_defect_broadcaster_from_request(
    request: Request,
) -> KnownDefectEventBroadcaster:
    broadcaster = getattr(request.app.state, "known_defect_event_broadcaster", None)
    if not isinstance(broadcaster, KnownDefectEventBroadcaster):
        raise RuntimeError("Known-defect event broadcaster is unavailable.")
    return broadcaster


# ADD 2026-08-26: WebSocket app에서 dedicated known-defect event channel을 반환한다.
def _known_defect_broadcaster_from_websocket(
    websocket: WebSocket,
) -> KnownDefectEventBroadcaster:
    broadcaster = getattr(websocket.app.state, "known_defect_event_broadcaster", None)
    if not isinstance(broadcaster, KnownDefectEventBroadcaster):
        raise RuntimeError("Known-defect event broadcaster is unavailable.")
    return broadcaster
