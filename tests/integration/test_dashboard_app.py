"""Lightweight Streamlit AppTest smoke for dashboard graceful empty states."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


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
