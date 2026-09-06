"""Static contracts for the dependency-light C6-6B1 service bridge API source."""

from __future__ import annotations

import ast
from pathlib import Path

from services.streaming.yolo_deepstream_service_integration import (
    EXPECTED_INTERNAL_ENDPOINT,
    EXPECTED_WEBSOCKET_ENDPOINT,
    load_service_integration_config,
)

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "services/api/app.py"
WEBSOCKETS = ROOT / "services/api/websockets.py"
SCHEMAS = ROOT / "services/api/streaming_schemas.py"
ROUTES = ROOT / "services/api/streaming_routes.py"
DOC = ROOT / "docs/vision/YOLO_REALTIME_STREAMING.md"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_source(path: Path, name: str) -> str:
    text = _source(path)
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(text, node) or ""
    raise AssertionError(f"Function not found: {name}")


# ADD 2026-09-06: C6-6B1이 C6-6A의 reserved internal/WS endpoint를 그대로 상속하는지 검증한다.
def test_service_bridge_uses_sealed_c6_6a_endpoints() -> None:
    config = load_service_integration_config()

    assert config.process_boundary.internal_endpoint == EXPECTED_INTERNAL_ENDPOINT
    assert config.process_boundary.websocket_endpoint == EXPECTED_WEBSOCKET_ENDPOINT
    routes = _source(ROUTES)
    assert "EXPECTED_INTERNAL_ENDPOINT" in routes
    assert "EXPECTED_WEBSOCKET_ENDPOINT" in routes
    assert "/v1/known-defects" not in routes
    assert "/v1/ws/known-defects" not in routes


# ADD 2026-09-06: Internal ingest의 202 ack 뒤 background broadcast 예약만 검증한다.
def test_internal_ingest_has_no_inference_or_persistence_path() -> None:
    body = _function_source(ROUTES, "ingest_streaming_known_defect")

    assert "background_tasks.add_task" in body
    assert "broadcaster.broadcast" in body
    assert "StreamingObservationAcceptedResponse" in body
    for forbidden in (
        "repository",
        "persist",
        "predict(",
        "run_in_threadpool",
        "UploadFile",
        "sha256_bytes",
        "database",
    ):
        assert forbidden not in body


# ADD 2026-09-06: Streaming schema가 raw media/URI 없이 exact provenance와 geometry를 검증하게 한다.
def test_streaming_schema_is_compact_and_fail_closed() -> None:
    schemas = _source(SCHEMAS)
    tree = ast.parse(schemas)
    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}

    assert {
        "StreamingDefectBoundingBox",
        "StreamingDefectMaskSummary",
        "StreamingDefectInstanceEvent",
        "StreamingObservationImage",
        "StreamingRuntimeIdentity",
        "StreamingKnownDefectObservationPayload",
        "StreamingKnownDefectObservedEvent",
        "StreamingObservationAcceptedResponse",
    } <= class_names
    assert 'extra="forbid"' in schemas
    assert "EXPECTED_ENGINE_SHA256" in schemas
    assert "EXPECTED_PARSER_SHA256" in schemas
    assert "EXPECTED_LABELS_SHA256" in schemas
    assert "diagnostic_confidence != 0.25" in schemas
    assert "raw_frame" not in schemas
    assert "raw_mask" not in schemas
    assert "rtsp_uri" not in schemas


# ADD 2026-09-06: Dedicated broadcaster가 existing durable event channels와 별도 type인지 검증한다.
def test_streaming_broadcaster_is_dedicated() -> None:
    websockets = _source(WEBSOCKETS)

    assert "class StreamingKnownDefectEventBroadcaster(" in websockets
    assert "EventBroadcaster[StreamingKnownDefectObservedEvent]" in websockets
    assert "class KnownDefectEventBroadcaster" in websockets
    assert "class CombinedInspectionEventBroadcaster" in websockets


# ADD 2026-09-06: App factory의 streaming router/broadcaster lifecycle 등록을 검증한다.
def test_app_registers_streaming_router_and_broadcaster_lifecycle() -> None:
    app = _source(APP)

    assert "from services.api.streaming_routes import router as streaming_router" in app
    assert "StreamingKnownDefectEventBroadcaster" in app
    assert "streaming_known_defect_event_broadcaster" in app
    assert "await streaming_broadcaster.close_all()" in app
    assert "app.state.streaming_known_defect_event_broadcaster" in app
    assert "app.include_router(streaming_router)" in app


# ADD 2026-09-06: C6-6B1 문서가 API source 완료와 runtime 미실행 경계를 명시하는지 검증한다.
def test_documentation_records_b1_source_only_scope() -> None:
    doc = _source(DOC)

    assert "C6-6B1 API_SOURCE_COMMITTED / API_RUNTIME_PENDING" in doc
    assert "## 30. C6-6B1 Service bridge API source" in doc
    assert "API runtime smoke executed: `false`" in doc
    assert "persistence used: `false`" in doc
    assert "`C6-6B2 SERVICE_BRIDGE_API_RUNTIME`" in doc
