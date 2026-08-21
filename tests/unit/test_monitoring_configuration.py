from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


# ADD 2026-08-21: Repository root를 monitoring configuration test에 반환한다.
def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ADD 2026-08-21: Prometheus scrape interval과 bounded API target contract를 검증한다.
def test_prometheus_scrape_configuration() -> None:
    config = yaml.safe_load(
        (_project_root() / "monitoring/prometheus/prometheus.yml").read_text(encoding="utf-8")
    )

    assert config["global"] == {
        "scrape_interval": "15s",
        "evaluation_interval": "15s",
    }
    assert config["scrape_configs"] == [
        {
            "job_name": "smartfactory-api",
            "metrics_path": "/metrics",
            "static_configs": [{"targets": ["api:8000"]}],
        }
    ]


# ADD 2026-08-21: Grafana datasource와 file dashboard provider가 internal DNS를 사용하는지 검증한다.
def test_grafana_provisioning_configuration() -> None:
    root = _project_root() / "monitoring/grafana/provisioning"
    datasource = yaml.safe_load((root / "datasources/prometheus.yaml").read_text(encoding="utf-8"))[
        "datasources"
    ][0]
    provider = yaml.safe_load((root / "dashboards/dashboard.yaml").read_text(encoding="utf-8"))[
        "providers"
    ][0]

    assert datasource["uid"] == "prometheus"
    assert datasource["url"] == "http://prometheus:9090"
    assert datasource["isDefault"] is True
    assert datasource["editable"] is False
    assert provider["options"]["path"] == "/etc/grafana/dashboards"
    assert provider["disableDeletion"] is True
    assert provider["editable"] is False


# ADD 2026-08-21: Provisioned dashboard panel과 PromQL이 실제 metric catalog만 참조하는지 검증한다.
def test_grafana_dashboard_metric_contract() -> None:
    dashboard: dict[str, Any] = json.loads(
        (
            _project_root() / "monitoring/grafana/dashboards/smartfactory-api-overview.json"
        ).read_text(encoding="utf-8")
    )
    panels = {panel["title"]: panel for panel in dashboard["panels"]}

    assert dashboard["title"] == "Smart Factory AI — API Overview"
    assert set(panels) == {
        "API Request Rate",
        "HTTP Error Rate",
        "HTTP p50",
        "HTTP p95",
        "HTTP p99",
        "Prediction Rate",
        "Anomaly Ratio",
        "Inference p95",
        "Persistence p95",
        "Prometheus Target Up",
    }
    expressions = "\n".join(
        target["expr"] for panel in panels.values() for target in panel["targets"]
    )
    expected_metrics = {
        "smartfactory_http_requests_total",
        "smartfactory_http_request_duration_seconds_bucket",
        "smartfactory_predictions_total",
        "smartfactory_inference_duration_seconds_bucket",
        "smartfactory_persistence_duration_seconds_bucket",
    }
    assert all(metric in expressions for metric in expected_metrics)
    assert expressions.count("histogram_quantile") == 5
    assert "clamp_min" in panels["HTTP Error Rate"]["targets"][0]["expr"]
    assert "clamp_min" in panels["Anomaly Ratio"]["targets"][0]["expr"]
    assert panels["Prometheus Target Up"]["targets"][0]["expr"] == (
        'max(up{job="smartfactory-api"})'
    )


# ADD 2026-08-21: Compose monitoring image, ports, readonly config와 named volume을 검증한다.
def test_monitoring_compose_contract() -> None:
    compose = yaml.safe_load((_project_root() / "compose.yaml").read_text(encoding="utf-8"))
    prometheus = compose["services"]["prometheus"]
    grafana = compose["services"]["grafana"]

    assert prometheus["image"] == "prom/prometheus:v3.12.0"
    assert grafana["image"] == "grafana/grafana:13.1.0"
    assert prometheus["ports"] == ["${PROMETHEUS_PORT:-9090}:9090"]
    assert grafana["ports"] == ["${GRAFANA_PORT:-3000}:3000"]
    assert "depends_on" not in prometheus
    assert "depends_on" not in grafana
    assert all(
        mount["read_only"]
        for mount in prometheus["volumes"] + grafana["volumes"]
        if isinstance(mount, dict)
    )
    assert "prometheus_data:/prometheus" in prometheus["volumes"]
    assert "grafana_data:/var/lib/grafana" in grafana["volumes"]
    assert grafana["environment"]["GF_AUTH_ANONYMOUS_ENABLED"] == "false"
    assert grafana["environment"]["GF_PLUGINS_PREINSTALL_DISABLED"] == "true"
