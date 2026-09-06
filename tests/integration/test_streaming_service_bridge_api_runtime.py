"""Loopback runtime smoke for the C6-6B DeepStream service bridge."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import UUID, uuid4

import uvicorn
from fastapi import FastAPI
from websockets.sync.client import connect

from services.api.streaming_routes import router as streaming_router
from services.api.websockets import StreamingKnownDefectEventBroadcaster
from services.streaming.yolo_deepstream_service_integration import (
    EXPECTED_DECODER_ID,
    EXPECTED_ENGINE_SHA256,
    EXPECTED_EVENT_TYPE,
    EXPECTED_INTERNAL_ENDPOINT,
    EXPECTED_LABELS_SHA256,
    EXPECTED_PARSER_SHA256,
    EXPECTED_WEBSOCKET_ENDPOINT,
)

HOST = "127.0.0.1"
SERVER_START_TIMEOUT_SECONDS = 5.0
SERVER_STOP_TIMEOUT_SECONDS = 5.0
NETWORK_TIMEOUT_SECONDS = 5.0


def _runtime_app() -> FastAPI:
    app = FastAPI()
    app.state.streaming_known_defect_event_broadcaster = StreamingKnownDefectEventBroadcaster()
    app.include_router(streaming_router)
    return app


@contextmanager
def _running_server() -> Iterator[int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, 0))
    sock.listen(128)
    port = int(sock.getsockname()[1])

    config = uvicorn.Config(
        _runtime_app(),
        host=HOST,
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="off",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [sock]},
        name="c6-6b2-uvicorn",
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=SERVER_STOP_TIMEOUT_SECONDS)
        sock.close()
        raise AssertionError("C6-6B2 uvicorn server did not start within timeout.")

    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=SERVER_STOP_TIMEOUT_SECONDS)
        sock.close()
        assert not thread.is_alive(), "C6-6B2 uvicorn server did not stop cleanly."


def _event_payload(*, source_id: str) -> dict[str, Any]:
    observation_id = uuid4()
    session_id = uuid4()
    image_area = 640 * 640
    mask_pixels = 1000
    return {
        "schema_version": "1",
        "type": EXPECTED_EVENT_TYPE,
        "observation": {
            "observation_id": str(observation_id),
            "source_id": source_id,
            "stream_session_id": str(session_id),
            "frame_number": 42,
            "pts_ns": 1_400_000_000,
            "observed_at": "2026-09-06T02:00:00+00:00",
            "image": {"width": 640, "height": 640},
            "runtime": {
                "decoder_id": EXPECTED_DECODER_ID,
                "engine_sha256": EXPECTED_ENGINE_SHA256,
                "parser_sha256": EXPECTED_PARSER_SHA256,
                "labels_sha256": EXPECTED_LABELS_SHA256,
            },
            "diagnostic_confidence": 0.25,
            "instances": [
                {
                    "class_id": 0,
                    "class_name": "bent",
                    "confidence": 0.95,
                    "box": {
                        "x_min": 10.0,
                        "y_min": 20.0,
                        "x_max": 110.0,
                        "y_max": 120.0,
                    },
                    "mask": {
                        "pixel_count": mask_pixels,
                        "area_ratio": mask_pixels / image_area,
                    },
                }
            ],
        },
    }


def _post_json(port: int, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = Request(
        f"http://{HOST}:{port}{EXPECTED_INTERNAL_ENDPOINT}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
        body = json.loads(response.read().decode("utf-8"))
        return response.status, body


# ADD 2026-09-06: 실제 loopback HTTP→WebSocket 경로에서 validated live event 전달을 검증한다.
def test_loopback_http_ingest_broadcasts_exact_event_to_websocket() -> None:
    payload = _event_payload(source_id="line-01-camera-01")
    observation_id = payload["observation"]["observation_id"]

    with _running_server() as port:
        websocket_url = f"ws://{HOST}:{port}{EXPECTED_WEBSOCKET_ENDPOINT}"
        with connect(
            websocket_url,
            open_timeout=NETWORK_TIMEOUT_SECONDS,
            close_timeout=NETWORK_TIMEOUT_SECONDS,
        ) as websocket:
            status_code, acknowledgement = _post_json(port, payload)
            event = json.loads(websocket.recv(timeout=NETWORK_TIMEOUT_SECONDS))

    assert status_code == 202
    assert acknowledgement == {
        "status": "accepted",
        "observation_id": observation_id,
    }
    assert UUID(acknowledgement["observation_id"])
    expected_event = json.loads(json.dumps(payload))
    expected_event["observation"]["observed_at"] = "2026-09-06T02:00:00Z"
    assert event == expected_event
    assert event["type"] == EXPECTED_EVENT_TYPE
    assert event["observation"]["runtime"]["engine_sha256"] == EXPECTED_ENGINE_SHA256
    assert event["observation"]["instances"][0]["mask"]["pixel_count"] == 1000
    serialized = json.dumps(event, sort_keys=True)
    for forbidden in ("raw_frame", "raw_mask", "rtsp_uri", "image_sha256"):
        assert forbidden not in serialized


# ADD 2026-09-06: RTSP credential 형태 source ID가 HTTP 422로 거부되는지 검증한다.
def test_loopback_http_rejects_rtsp_credential_like_source_id() -> None:
    payload = _event_payload(source_id="rtsp://user:pass@camera/stream")

    with _running_server() as port:
        try:
            _post_json(port, payload)
        except HTTPError as exc:
            status_code = exc.code
            error_body = json.loads(exc.read().decode("utf-8"))
        else:
            raise AssertionError("Credential-like RTSP source ID was unexpectedly accepted.")

    assert status_code == 422
    assert error_body["detail"]
