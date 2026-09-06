"""C6-6B internal ingest and live WebSocket routes for DeepStream observations."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Request, WebSocket, WebSocketDisconnect, status

from services.api.streaming_schemas import (
    StreamingKnownDefectObservedEvent,
    StreamingObservationAcceptedResponse,
)
from services.api.websockets import StreamingKnownDefectEventBroadcaster
from services.streaming.yolo_deepstream_service_integration import (
    EXPECTED_INTERNAL_ENDPOINT,
    EXPECTED_WEBSOCKET_ENDPOINT,
)

router = APIRouter()


# ADD 2026-09-06: DeepStream observation을 persistence/re-inference 없이 API process로 수용한다.
@router.post(
    EXPECTED_INTERNAL_ENDPOINT,
    response_model=StreamingObservationAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    include_in_schema=False,
)
async def ingest_streaming_known_defect(
    request: Request,
    background_tasks: BackgroundTasks,
    event: StreamingKnownDefectObservedEvent,
) -> StreamingObservationAcceptedResponse:
    """Acknowledge one validated observation and schedule process-local live broadcast."""
    broadcaster = _streaming_broadcaster_from_request(request)
    background_tasks.add_task(broadcaster.broadcast, event)
    return StreamingObservationAcceptedResponse(
        observation_id=event.observation.observation_id,
    )


# ADD 2026-09-06: persisted known-defect와 분리된 streaming WebSocket을 제공한다.
@router.websocket(EXPECTED_WEBSOCKET_ENDPOINT)
async def stream_streaming_known_defects(websocket: WebSocket) -> None:
    """Push best-effort streaming_known_defect.observed events."""
    broadcaster = _streaming_broadcaster_from_websocket(websocket)
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


# ADD 2026-09-06: Request app state에서 dedicated streaming broadcaster를 반환한다.
def _streaming_broadcaster_from_request(
    request: Request,
) -> StreamingKnownDefectEventBroadcaster:
    broadcaster = getattr(
        request.app.state,
        "streaming_known_defect_event_broadcaster",
        None,
    )
    if not isinstance(broadcaster, StreamingKnownDefectEventBroadcaster):
        raise RuntimeError("Streaming known-defect event broadcaster is unavailable.")
    return broadcaster


# ADD 2026-09-06: WebSocket app state에서 dedicated streaming broadcaster를 반환한다.
def _streaming_broadcaster_from_websocket(
    websocket: WebSocket,
) -> StreamingKnownDefectEventBroadcaster:
    broadcaster = getattr(
        websocket.app.state,
        "streaming_known_defect_event_broadcaster",
        None,
    )
    if not isinstance(broadcaster, StreamingKnownDefectEventBroadcaster):
        raise RuntimeError("Streaming known-defect event broadcaster is unavailable.")
    return broadcaster
