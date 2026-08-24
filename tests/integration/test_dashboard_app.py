"""Lightweight Streamlit AppTest smoke for dashboard graceful empty states."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from apps.dashboard import clients as dashboard_clients
from examples.dashboard_demo.api import app as demo_api


# ADD 2026-08-21: Unavailable API와 missing drift에서도 Streamlit app이 렌더링되는지 검증한다.
def test_dashboard_app_starts_with_unavailable_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # AppTest process가 즉시 connection-refused를 받아 network dependency 없이 error UI를 검증한다.
    monkeypatch.setenv("DASHBOARD_API_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("DASHBOARD_REQUEST_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setenv("DRIFT_REPORT_DIR", str(tmp_path / "missing"))
    app_path = Path(__file__).resolve().parents[2] / "apps" / "dashboard" / "app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "Smart Factory AI Operations Dashboard"
    assert any("Inspection API is unavailable" in element.value for element in app.error)
    assert any("No drift report available" in element.value for element in app.info)


# ADD 2026-08-24: Synthetic API와 drift fixture의 populated UI/banner/detail을 검증한다.
def test_dashboard_app_renders_populated_synthetic_demo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    api_client = TestClient(demo_api)

    # Streamlit AppTest에서도 실제 demo FastAPI routing/response schema를 사용한다.
    def demo_transport(url: str, _timeout_seconds: float) -> object:
        parsed = urlsplit(url)
        response = api_client.get(f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path)
        response.raise_for_status()
        return response.json()

    monkeypatch.setattr(dashboard_clients, "_read_json_url", demo_transport)
    monkeypatch.setenv("DASHBOARD_API_BASE_URL", "http://dashboard-demo-api.local")
    monkeypatch.setenv(
        "DRIFT_REPORT_DIR",
        str(root / "examples" / "dashboard_demo" / "fixtures" / "drift"),
    )
    monkeypatch.setenv("DASHBOARD_ENV_LABEL", "DEMO — SYNTHETIC DATA")
    app_path = root / "apps" / "dashboard" / "app.py"

    dashboard = AppTest.from_file(str(app_path)).run(timeout=10)

    assert not dashboard.exception
    assert any("DEMO — SYNTHETIC DATA" in element.value for element in dashboard.warning)
    metric_values = {metric.label: metric.value for metric in dashboard.metric}
    assert metric_values["Recent Inspections"] == "100"
    assert metric_values["Normal Predictions"] == "88"
    assert metric_values["Anomaly Predictions"] == "12"
    assert metric_values["Anomaly Ratio"] == "12.0%"
    assert metric_values["Latest Drift Status"] == "WARNING"
    assert any("Latest status: WARNING" in element.value for element in dashboard.warning)

    inspection_select = next(
        element for element in dashboard.selectbox if element.label == "Inspection"
    )
    inspection_select.select_index(0)
    dashboard.run(timeout=10)

    assert not dashboard.exception
    assert dashboard.json
    assert "patchcore-demo-synthetic" in dashboard.json[0].value
