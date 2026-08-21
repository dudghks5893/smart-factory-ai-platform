"""Unit tests for dashboard API, KPI, presentation, and drift artifact logic."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from apps.dashboard.clients import DashboardApiError, InspectionApiClient, InspectionItem
from apps.dashboard.config import DashboardSettings
from apps.dashboard.drift import (
    MalformedDriftReportError,
    load_latest_drift_report,
)
from apps.dashboard.presentation import (
    calculate_inspection_kpis,
    filter_inspections,
    inspection_detail_fields,
    inspection_table_rows,
    score_trend_rows,
)

BASE_TIME = datetime(2026, 8, 21, tzinfo=UTC)


# ADD 2026-08-21: Dashboard test용 internally consistent inspection API payload를 생성한다.
def _inspection_payload(
    *,
    inspection_id: str = "00000000-0000-0000-0000-000000000001",
    seconds: int = 0,
    score: float = 1.0,
    threshold: float = 2.0,
    category: str = "metal_nut",
    model_sha: str = "a" * 64,
) -> dict[str, object]:
    return {
        "inspection_id": inspection_id,
        "created_at": (BASE_TIME + timedelta(seconds=seconds)).isoformat(),
        "model_name": "patchcore",
        "category": category,
        "is_anomaly": score > threshold,
        "anomaly_score": score,
        "threshold": threshold,
        "comparison_operator": ">",
        "image_sha256": "9" * 64,
        "image_size_bytes": 123,
        "content_type": "image/png",
        "model_sha256": model_sha,
        "artifact_metadata_sha256": "b" * 64,
        "threshold_artifact_sha256": "c" * 64,
        "manifest_sha256": "d" * 64,
        "device": "cpu",
    }


# ADD 2026-08-21: Dashboard test용 complete drift v1 JSON payload를 생성한다.
def _drift_payload(
    *,
    drift_id: str = "drift-1",
    status: str = "stable",
    created_at: str = "2026-08-21T01:00:00+00:00",
) -> dict[str, Any]:
    summary = {
        "mean": 1.0,
        "std": 0.5,
        "min": 0.0,
        "max": 2.0,
        "p50": 1.0,
        "p90": 1.8,
        "p95": 1.9,
        "p99": 1.98,
    }
    return {
        "schema_version": 1,
        "drift_id": drift_id,
        "model_name": "patchcore",
        "category": "metal_nut",
        "lineage": {},
        "reference": {
            "reference_id": "reference-1",
            "sample_count": 30,
            "source_split": "validation",
            "source_label": "normal",
            "summary": summary,
        },
        "current_window": {
            "since": "2026-08-21T00:00:00+00:00",
            "until": "2026-08-21T01:00:00+00:00",
            "boundary": "since_inclusive_until_exclusive",
            "sample_count": 30,
            "summary": {**summary, "mean": 1.2, "p95": 2.1},
        },
        "statistics": {
            "psi": 0.05,
            "current_bin_counts": [3] * 10,
            "anomaly_ratio": 0.1,
            "reference_anomaly_ratio": 0.0,
            "anomaly_ratio_absolute_delta": 0.1,
            "mean_delta": 0.2,
            "p50_delta": 0.0,
            "p95_delta": 0.2,
        },
        "policy": {},
        "status": status,
        "created_at": created_at,
    }


# ADD 2026-08-21: Dashboard config defaults와 explicit environment override를 검증한다.
def test_dashboard_settings_load_and_validate_environment(tmp_path: Path) -> None:
    settings = DashboardSettings.from_environment(
        {
            "DASHBOARD_API_BASE_URL": "http://api:8000/",
            "DRIFT_REPORT_DIR": str(tmp_path),
            "GRAFANA_URL": "https://grafana.example.test/",
            "DASHBOARD_REQUEST_TIMEOUT_SECONDS": "2.5",
        }
    )

    assert settings.api_base_url == "http://api:8000"
    assert settings.drift_report_dir == tmp_path
    assert settings.grafana_url == "https://grafana.example.test"
    assert settings.request_timeout_seconds == 2.5


# ADD 2026-08-21: Empty artifact path, embedded credential와 non-finite timeout을 거부한다.
@pytest.mark.parametrize(
    "environ",
    [
        {"DRIFT_REPORT_DIR": ""},
        {"DASHBOARD_API_BASE_URL": "http://user:secret@api:8000"},
        {"DASHBOARD_REQUEST_TIMEOUT_SECONDS": "nan"},
    ],
)
def test_dashboard_settings_reject_unsafe_values(environ: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        DashboardSettings.from_environment(environ)


# ADD 2026-08-21: Inspection client가 filter query와 dashboard-safe response parsing을 적용한다.
def test_inspection_client_parses_history_and_filters_query() -> None:
    requested: list[tuple[str, float]] = []

    def transport(url: str, timeout: float) -> object:
        requested.append((url, timeout))
        return {
            "items": [_inspection_payload()],
            "limit": 50,
            "offset": 0,
            "returned_count": 1,
            "has_more": False,
        }

    client = InspectionApiClient(
        "http://api:8000",
        timeout_seconds=3.0,
        transport=transport,
    )
    page = client.list_inspections(
        category="metal_nut",
        is_anomaly=False,
        limit=50,
    )

    assert page.returned_count == 1
    assert page.items[0].created_at.tzinfo == UTC
    assert requested == [
        (
            "http://api:8000/v1/inspections?limit=50&offset=0&category=metal_nut&is_anomaly=false",
            3.0,
        )
    ]


# ADD 2026-08-21: API timeout, HTTP failure와 malformed response의 safe error 변환을 검증한다.
@pytest.mark.parametrize(
    ("transport", "message"),
    [
        (lambda _url, _timeout: (_ for _ in ()).throw(TimeoutError()), "timed out"),
        (
            lambda _url, _timeout: (_ for _ in ()).throw(
                DashboardApiError("Inspection API returned HTTP 503.")
            ),
            "HTTP 503",
        ),
        (lambda _url, _timeout: {"unexpected": True}, "malformed"),
    ],
)
def test_inspection_client_normalizes_failure_states(
    transport: Any,
    message: str,
) -> None:
    client = InspectionApiClient(
        "http://api:8000",
        timeout_seconds=1.0,
        transport=transport,
    )

    with pytest.raises(DashboardApiError, match=message):
        client.list_inspections()


# ADD 2026-08-21: Empty inspection sample이 zero KPI와 safe empty presentation을 만드는지 검증한다.
def test_empty_inspection_sample_is_safe() -> None:
    kpis = calculate_inspection_kpis(())

    assert kpis.recent_inspections == 0
    assert kpis.anomaly_ratio == 0.0
    assert score_trend_rows(()) == []
    assert inspection_table_rows(()) == []


# ADD 2026-08-21: Mixed prediction과 lineage sample의 KPI/filter/trend 계산을 검증한다.
def test_mixed_inspection_sample_preserves_record_thresholds_and_model_count() -> None:
    newest = InspectionItem.from_json_dict(
        _inspection_payload(
            inspection_id="00000000-0000-0000-0000-000000000002",
            seconds=2,
            score=4.0,
            threshold=3.0,
            category="bottle",
            model_sha="e" * 64,
        )
    )
    oldest = InspectionItem.from_json_dict(_inspection_payload(score=1.0, threshold=2.0))
    items = (newest, oldest)

    kpis = calculate_inspection_kpis(items)
    anomaly_only = filter_inspections(items, category=None, result="anomaly", limit=100)
    trend = score_trend_rows(items)

    assert (kpis.normal_count, kpis.anomaly_count, kpis.anomaly_ratio) == (1, 1, 0.5)
    assert kpis.model_versions == 2
    assert anomaly_only == (newest,)
    assert [row["threshold"] for row in trend] == [2.0, 3.0]


# ADD 2026-08-21: Projection이 raw image metadata나 unsupported defect type을 숨기는지 검증한다.
def test_dashboard_projection_excludes_raw_image_and_defect_type() -> None:
    item = InspectionItem.from_json_dict(_inspection_payload())
    table_text = json.dumps(inspection_table_rows((item,)))
    detail_text = json.dumps(inspection_detail_fields(item))

    assert "image_sha256" not in vars(item)
    assert "999999" not in table_text + detail_text
    assert "Defect Type" not in table_text + detail_text
    assert "Model SHA" in detail_text


# ADD 2026-08-21: created_at이 filesystem mtime보다 latest drift 선택을 결정하는지 검증한다.
def test_latest_drift_report_uses_created_at_not_mtime(tmp_path: Path) -> None:
    older_path = tmp_path / "z-older" / "drift.json"
    newer_path = tmp_path / "a-newer" / "drift.json"
    older_path.parent.mkdir()
    newer_path.parent.mkdir()
    older_path.write_text(
        json.dumps(_drift_payload(drift_id="older", created_at="2026-08-21T01:00:00+00:00")),
        encoding="utf-8",
    )
    newer_path.write_text(
        json.dumps(_drift_payload(drift_id="newer", created_at="2026-08-21T02:00:00+00:00")),
        encoding="utf-8",
    )
    older_path.touch()

    latest = load_latest_drift_report(tmp_path)

    assert latest is not None
    assert latest.drift_id == "newer"


# ADD 2026-08-21: 모든 supported drift status가 dashboard projection에서 보존되는지 검증한다.
@pytest.mark.parametrize("status", ["stable", "warning", "drift", "insufficient_data"])
def test_supported_drift_statuses_are_parsed(tmp_path: Path, status: str) -> None:
    report_path = tmp_path / "run" / "drift.json"
    report_path.parent.mkdir()
    report_path.write_text(json.dumps(_drift_payload(status=status)), encoding="utf-8")

    report = load_latest_drift_report(tmp_path)

    assert report is not None
    assert report.status == status


# ADD 2026-08-21: Missing drift root와 empty root가 non-fatal no-report 상태인지 검증한다.
def test_missing_and_empty_drift_report_states_return_none(tmp_path: Path) -> None:
    assert load_latest_drift_report(tmp_path / "missing") is None
    assert load_latest_drift_report(tmp_path) is None


# ADD 2026-08-21: Malformed candidate를 older valid report로 숨기지 않는지 검증한다.
def test_malformed_drift_report_is_explicit_error(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid" / "drift.json"
    malformed_path = tmp_path / "malformed" / "drift.json"
    valid_path.parent.mkdir()
    malformed_path.parent.mkdir()
    valid_path.write_text(json.dumps(_drift_payload()), encoding="utf-8")
    malformed_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(MalformedDriftReportError, match="malformed/drift.json"):
        load_latest_drift_report(tmp_path)


# ADD 2026-08-21: Inspection detail endpoint가 UUID를 사용하고 safe fields만 복원하는지 검증한다.
def test_inspection_client_gets_detail_by_uuid() -> None:
    requested: list[str] = []

    def transport(url: str, _timeout: float) -> object:
        requested.append(url)
        return _inspection_payload()

    client = InspectionApiClient(
        "http://api:8000",
        timeout_seconds=1.0,
        transport=transport,
    )
    inspection_id = UUID("00000000-0000-0000-0000-000000000001")

    detail = client.get_inspection(inspection_id)

    assert detail.inspection_id == inspection_id
    assert requested == [f"http://api:8000/v1/inspections/{inspection_id}"]
