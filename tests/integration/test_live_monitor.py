"""Integration contracts for same-origin browser-native inspection monitoring."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from services.api.app import create_app


# ADD 2026-08-25: Browser monitor HTML/CSS/modules이 API same-origin path에서 제공되는지 검증한다.
def test_live_monitor_assets_are_served_from_api_origin() -> None:
    client = TestClient(create_app())

    page = client.get("/live/")
    stylesheet = client.get("/live/styles.css")
    application = client.get("/live/app.js")
    state = client.get("/live/state.js")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "Live Inspection Monitor" in page.text
    assert '<script type="module" src="./app.js"></script>' in page.text
    assert stylesheet.status_code == 200
    assert application.status_code == 200
    assert state.status_code == 200
    assert 'fetch("/v1/inspections?limit=100&offset=0"' in application.text
    assert "/v1/ws/inspections" in state.text
    assert application.text.index("new WebSocket") < application.text.index(
        'fetch("/v1/inspections?limit=100&offset=0"'
    )
    assert "bufferedInspections" in application.text
    assert "scheduleReconnect" in application.text
    assert "connectAndSynchronize();" in application.text
    assert "localhost" not in application.text
    assert "localhost" not in state.text


# ADD 2026-08-25: Static asset directory가 없어도 API liveness가 독립적으로 유지되는지 검증한다.
def test_missing_live_monitor_assets_do_not_break_api_liveness(tmp_path: Path) -> None:
    client = TestClient(create_app(live_monitor_dir=tmp_path / "missing"))

    health = client.get("/health")
    monitor = client.get("/live/")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert monitor.status_code == 404
