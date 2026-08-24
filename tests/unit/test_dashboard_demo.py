"""Contracts for the isolated synthetic dashboard portfolio demo."""

from __future__ import annotations

import json
import re
from datetime import UTC
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.dashboard.clients import InspectionItem, InspectionPage
from apps.dashboard.config import DashboardSettings
from apps.dashboard.drift import load_latest_drift_report
from examples.dashboard_demo.api import app
from examples.dashboard_demo.fixtures import (
    DEMO_ANOMALY_COUNT,
    DEMO_INSPECTION_COUNT,
    DEMO_NORMAL_COUNT,
    DEMO_THRESHOLD,
    SYNTHETIC_INSPECTIONS,
)
from ml.drift.patchcore import DriftLineage, DriftPolicy, ScoreSummary


# ADD 2026-08-24: Demo inspection의 count, identity, UTC ordering과 score 판정 계약을 검증한다.
def test_synthetic_inspections_are_deterministic_and_internally_consistent() -> None:
    normal = tuple(item for item in SYNTHETIC_INSPECTIONS if not item.is_anomaly)
    anomaly = tuple(item for item in SYNTHETIC_INSPECTIONS if item.is_anomaly)

    assert len(SYNTHETIC_INSPECTIONS) == DEMO_INSPECTION_COUNT == 100
    assert (len(normal), len(anomaly)) == (DEMO_NORMAL_COUNT, DEMO_ANOMALY_COUNT) == (88, 12)
    assert len({item.inspection_id for item in SYNTHETIC_INSPECTIONS}) == 100
    assert all(item.created_at.tzinfo == UTC for item in SYNTHETIC_INSPECTIONS)
    assert list(SYNTHETIC_INSPECTIONS) == sorted(
        SYNTHETIC_INSPECTIONS,
        key=lambda item: item.created_at,
        reverse=True,
    )
    assert all(25.0 <= item.anomaly_score < DEMO_THRESHOLD for item in normal)
    assert all(DEMO_THRESHOLD < item.anomaly_score < 60.0 for item in anomaly)
    assert all(
        item.is_anomaly is (item.anomaly_score > item.threshold) for item in normal + anomaly
    )
    assert {item.model_name for item in SYNTHETIC_INSPECTIONS} == {"patchcore-demo-synthetic"}
    assert {item.device for item in SYNTHETIC_INSPECTIONS} == {"demo-cpu"}


# ADD 2026-08-24: Demo API의 existing list/detail/filter/pagination schema를 검증한다.
def test_demo_api_serves_dashboard_contract_without_runtime_dependencies() -> None:
    client = TestClient(app)

    health = client.get("/health")
    history = client.get("/v1/inspections?limit=100")
    anomaly_history = client.get("/v1/inspections?is_anomaly=true&limit=100")
    missing_category = client.get("/v1/inspections?category=not-present&limit=100")
    second_page = client.get("/v1/inspections?limit=7&offset=7")

    assert health.json() == {"status": "ok"}
    page = InspectionPage.from_json_dict(history.json())
    assert (page.returned_count, page.has_more) == (100, False)
    assert InspectionPage.from_json_dict(anomaly_history.json()).returned_count == 12
    assert InspectionPage.from_json_dict(missing_category.json()).returned_count == 0
    assert InspectionPage.from_json_dict(second_page.json()).returned_count == 7

    inspection_id = page.items[0].inspection_id
    detail = client.get(f"/v1/inspections/{inspection_id}")
    missing = client.get("/v1/inspections/00000000-0000-0000-0000-000000000000")

    assert detail.status_code == 200
    assert InspectionItem.from_json_dict(detail.json()).inspection_id == inspection_id
    assert missing.status_code == 404


# ADD 2026-08-24: Demo drift JSON이 STEP 10 schema와 screenshot용 warning 값을 보존하는지 검증한다.
def test_demo_drift_fixture_matches_current_schema_and_demo_lineage() -> None:
    root = Path(__file__).resolve().parents[2]
    report_root = root / "examples" / "dashboard_demo" / "fixtures" / "drift"
    report_path = report_root / "synthetic-warning-v1" / "drift.json"
    raw = json.loads(report_path.read_text(encoding="utf-8"))

    lineage = DriftLineage.from_json_dict(raw["lineage"])
    ScoreSummary.from_json_dict(raw["reference"]["summary"])
    ScoreSummary.from_json_dict(raw["current_window"]["summary"])
    policy = DriftPolicy(**raw["policy"])
    policy.validate()
    report = load_latest_drift_report(report_root)

    assert report is not None
    assert (report.status, report.psi) == ("warning", 0.17)
    assert (report.reference_mean, report.current_mean) == (30.0, 35.0)
    assert (report.reference_p95, report.current_p95) == (36.0, 44.0)
    assert (report.reference_anomaly_ratio, report.current_anomaly_ratio) == (0.02, 0.12)
    assert report.current_sample_count == DEMO_INSPECTION_COUNT
    assert sum(raw["statistics"]["current_bin_counts"]) == DEMO_INSPECTION_COUNT
    first = SYNTHETIC_INSPECTIONS[0]
    assert lineage.model_sha256 == first.model_sha256
    assert lineage.artifact_metadata_sha256 == first.artifact_metadata_sha256
    assert lineage.threshold_artifact_sha256 == first.threshold_artifact_sha256
    assert lineage.manifest_sha256 == first.manifest_sha256


# ADD 2026-08-24: Generic environment label의 opt-in banner contract와 safe default를 검증한다.
def test_dashboard_environment_label_is_optional_and_single_line() -> None:
    default_settings = DashboardSettings.from_environment({})
    demo_settings = DashboardSettings.from_environment(
        {"DASHBOARD_ENV_LABEL": "  DEMO — SYNTHETIC DATA  "}
    )

    assert default_settings.environment_label is None
    assert demo_settings.environment_label == "DEMO — SYNTHETIC DATA"
    with pytest.raises(ValueError, match="single-line"):
        DashboardSettings.from_environment({"DASHBOARD_ENV_LABEL": "DEMO\nUNSAFE"})


# ADD 2026-08-24: Makefile과 documentation이 root import와 두-process demo UX에 합의하는지 검증한다.
def test_dashboard_entrypoint_and_documentation_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    dashboard_document = (root / "docs" / "dashboard" / "DASHBOARD.md").read_text(encoding="utf-8")

    assert re.search(r'^dashboard:\n\tPYTHONPATH="\$\(CURDIR\)" ', makefile, re.MULTILINE)
    assert "dashboard-demo-api:" in makefile
    assert "dashboard-demo:" in makefile
    assert 'DASHBOARD_ENV_LABEL="DEMO — SYNTHETIC DATA"' in makefile
    assert "\nmake dashboard\n" in readme
    assert "make dashboard-demo-api" in readme
    assert "make dashboard-demo" in readme
    assert "make dashboard-demo-api" in dashboard_document
    assert "make dashboard-demo" in dashboard_document


# ADD 2026-08-24: Demo payload에 credential, real path 또는 raw image body가 없는지 검증한다.
def test_demo_payload_contains_no_secret_or_raw_data_fields() -> None:
    payload = json.dumps(
        [item.model_dump(mode="json") for item in SYNTHETIC_INSPECTIONS],
        sort_keys=True,
    ).lower()

    assert not any(term in payload for term in ("password", "api_key", "access_token", "raw_image"))
    assert "/users/" not in payload
    assert "smartfactory-dashboard-demo" not in payload
