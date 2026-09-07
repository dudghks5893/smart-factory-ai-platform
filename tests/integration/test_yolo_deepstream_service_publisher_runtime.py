"""Actual network runtime smoke for the C6-6C worker publisher."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

import uvicorn
from fastapi import FastAPI
from websockets.sync.client import connect

from services.api.streaming_routes import router as streaming_router
from services.api.websockets import StreamingKnownDefectEventBroadcaster
from services.streaming.yolo_deepstream_service_integration import (
    EXPECTED_WEBSOCKET_ENDPOINT,
    StreamingKnownDefectObservation,
    load_service_integration_config,
)
from services.streaming.yolo_deepstream_service_publisher import (
    DeepStreamFrameSnapshot,
    DeepStreamInstanceSnapshot,
    LatestEventWinsPublisher,
    build_streaming_observation,
)

CONFIG = Path("configs/streaming/yolo_deepstream_service_integration.json")
HOST = "127.0.0.1"
NETWORK_TIMEOUT_SECONDS = 5.0
SOURCE_ID = "line-01-camera-01"
SESSION_ID = UUID("11111111-1111-4111-8111-111111111111")
OBSERVATION_ID = UUID("22222222-2222-4222-8222-222222222222")
OBSERVED_AT = datetime(2026, 9, 7, 1, 30, tzinfo=UTC)


def _runtime_app() -> FastAPI:
    app = FastAPI()
    app.state.streaming_known_defect_event_broadcaster = StreamingKnownDefectEventBroadcaster()
    app.include_router(streaming_router)
    return app


@contextmanager
def _running_service_bridge() -> Iterator[int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, 0))
    sock.listen(128)
    port = int(sock.getsockname()[1])
    server = uvicorn.Server(
        uvicorn.Config(
            _runtime_app(),
            host=HOST,
            port=port,
            log_level="warning",
            access_log=False,
            lifespan="off",
        )
    )
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [sock]},
        name="c6-6c2-uvicorn",
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + NETWORK_TIMEOUT_SECONDS
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=NETWORK_TIMEOUT_SECONDS)
        sock.close()
        raise AssertionError("C6-6C2 service bridge did not start.")
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=NETWORK_TIMEOUT_SECONDS)
        sock.close()
        assert not thread.is_alive(), "C6-6C2 service bridge did not stop cleanly."


@contextmanager
def _bound_non_listening_port() -> Iterator[int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, 0))
    try:
        yield int(sock.getsockname()[1])
    finally:
        sock.close()


def _observation(*, frame_number: int = 42) -> StreamingKnownDefectObservation:
    config = load_service_integration_config(CONFIG)
    snapshot = DeepStreamFrameSnapshot(
        frame_number=frame_number,
        pts_ns=1_400_000_000 + frame_number,
        image_width=640,
        image_height=640,
        instances=(
            DeepStreamInstanceSnapshot(
                class_id=2,
                confidence=0.91,
                left=10.0,
                top=20.0,
                width=100.0,
                height=80.0,
                mask_pixel_count=1000,
            ),
        ),
    )
    return build_streaming_observation(
        snapshot,
        source_id=SOURCE_ID,
        stream_session_id=SESSION_ID,
        config=config,
        observed_at=OBSERVED_AT,
        observation_id=OBSERVATION_ID,
    )


# ADD 2026-09-07: 실제 publisher POST가 FastAPI와 WebSocket에 도달하는지 검증한다.
def test_actual_publisher_reaches_service_bridge_and_websocket() -> None:
    config = load_service_integration_config(CONFIG)
    observation = _observation()
    expected_event = observation.to_payload(config)

    with _running_service_bridge() as port:
        publisher = LatestEventWinsPublisher(
            service_base_url=f"http://{HOST}:{port}",
            config=config,
        )
        websocket_url = f"ws://{HOST}:{port}{EXPECTED_WEBSOCKET_ENDPOINT}"
        with connect(
            websocket_url,
            open_timeout=NETWORK_TIMEOUT_SECONDS,
            close_timeout=NETWORK_TIMEOUT_SECONDS,
        ) as websocket:
            publisher.start()
            try:
                publisher.submit(observation)
                assert publisher.wait_until_idle(timeout_seconds=NETWORK_TIMEOUT_SECONDS)
                event = cast(
                    dict[str, object],
                    json.loads(websocket.recv(timeout=NETWORK_TIMEOUT_SECONDS)),
                )
            finally:
                publisher.close(timeout_seconds=NETWORK_TIMEOUT_SECONDS)

    metrics = publisher.metrics_snapshot()
    assert event == expected_event
    assert metrics.generated_total == 1
    assert metrics.delivered_total == 1
    assert metrics.dropped_total == 0
    assert metrics.delivery_errors_total == 0
    assert metrics.delivery_retries_total == 0
    assert metrics.bridge_up == 1
    assert metrics.pending_events == 0
    assert metrics.seconds_since_delivery is not None

    serialized = json.dumps(event, sort_keys=True)
    for forbidden in ("raw_frame", "raw_mask", "rtsp_uri", "image_sha256"):
        assert forbidden not in serialized


# ADD 2026-09-07: Connection refusal의 delivery exhaustion 격리를 검증한다.
def test_actual_network_delivery_exhaustion_is_failure_isolated() -> None:
    config = load_service_integration_config(CONFIG)
    observation = _observation(frame_number=99)

    with _bound_non_listening_port() as port:
        publisher = LatestEventWinsPublisher(
            service_base_url=f"http://{HOST}:{port}",
            config=config,
        )
        publisher.start()
        submit_started = time.monotonic()
        publisher.submit(observation)
        submit_elapsed = time.monotonic() - submit_started
        delivery_started = time.monotonic()
        publisher.close(timeout_seconds=NETWORK_TIMEOUT_SECONDS)
        delivery_elapsed = time.monotonic() - delivery_started

    metrics = publisher.metrics_snapshot()
    assert submit_elapsed < 0.25
    assert 0.65 <= delivery_elapsed < NETWORK_TIMEOUT_SECONDS
    assert metrics.generated_total == 1
    assert metrics.delivered_total == 0
    assert metrics.dropped_total == 1
    assert metrics.delivery_errors_total == 4
    assert metrics.delivery_retries_total == 3
    assert metrics.bridge_up == 0
    assert metrics.pending_events == 0
    assert metrics.seconds_since_delivery is None
