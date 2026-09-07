"""Static contract tests for the Live Monitor DeepStream streaming domain."""

from pathlib import Path

from services.streaming.yolo_deepstream_service_integration import (
    EXPECTED_WEBSOCKET_ENDPOINT,
)

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "apps/live_monitor/index.html"
APP = ROOT / "apps/live_monitor/app.js"
STATE = ROOT / "apps/live_monitor/state.js"


# ADD 2026-09-07: UI copy와 DOM IDs가 streaming frame을 non-persisted domain으로 명시하게 한다.
def test_live_monitor_declares_non_persisted_streaming_domain() -> None:
    html = INDEX.read_text(encoding="utf-8")

    assert 'id="streaming-connection-state"' in html
    assert 'id="streaming-latest-observation"' in html
    assert 'id="streaming-observation-feed"' in html
    assert "no per-frame persistence" in html
    assert "Frames are not recovered from PostgreSQL." in html


# ADD 2026-09-07: Browser streaming endpoint가 backend contract와 동일한 path를 사용하게 한다.
def test_live_monitor_streaming_endpoint_matches_backend_contract() -> None:
    state = STATE.read_text(encoding="utf-8")

    assert EXPECTED_WEBSOCKET_ENDPOINT in state
    assert "streaming_known_defect.observed" in state
    assert "parseStreamingObservationEvent" in state
    assert "mergeStreamingObservations" in state


# ADD 2026-09-07: Fourth live-only channel이 persisted REST sync를 재사용하지 않게 한다.
def test_live_monitor_streaming_app_is_live_only() -> None:
    app = APP.read_text(encoding="utf-8")

    assert "connectStreaming()" in app
    assert "acceptStreamingLiveMessage" in app
    assert 'elements.streamingLastSync.textContent = "LIVE · non-persisted"' in app
    assert 'fetch("/v1/streaming' not in app
