"""Streamlit entrypoint for the Smart Factory AI operations dashboard."""

from __future__ import annotations

from typing import cast
from uuid import UUID

import streamlit as st

from apps.dashboard.clients import DashboardApiError, InspectionApiClient, InspectionItem
from apps.dashboard.config import DashboardSettings
from apps.dashboard.drift import DriftReport, MalformedDriftReportError, load_latest_drift_report
from apps.dashboard.presentation import (
    ResultFilter,
    calculate_inspection_kpis,
    filter_inspections,
    inspection_detail_fields,
    inspection_table_rows,
    score_trend_rows,
)


# ADD 2026-08-21: Dashboard KPI를 현재 조회·필터된 prediction sample 기준으로 표시한다.
def _render_kpis(items: tuple[InspectionItem, ...], drift_report: DriftReport | None) -> None:
    kpis = calculate_inspection_kpis(items)
    columns = st.columns(5)
    columns[0].metric("Recent Inspections", kpis.recent_inspections)
    columns[1].metric("Normal Predictions", kpis.normal_count)
    columns[2].metric("Anomaly Predictions", kpis.anomaly_count)
    columns[3].metric("Anomaly Ratio", f"{kpis.anomaly_ratio:.1%}")
    columns[4].metric(
        "Latest Drift Status",
        "NO REPORT" if drift_report is None else drift_report.status.upper(),
    )
    st.caption(
        "Counts and anomaly ratio describe only the inspections currently shown. "
        "They are AI prediction statistics, not confirmed factory defect rates. "
        f"Model versions in view: {kpis.model_versions}."
    )


# ADD 2026-08-21: Drift status, distribution statistics와 interpretation boundary를 표시한다.
def _render_drift(report: DriftReport | None, error: str | None) -> None:
    st.subheader("Drift Status")
    if error is not None:
        st.error(error)
        return
    if report is None:
        st.info("No drift report available.")
        return

    status_message = f"Latest status: {report.status.upper()}"
    if report.status == "stable":
        st.success(status_message)
    elif report.status == "warning":
        st.warning(status_message)
    elif report.status == "drift":
        st.error(status_message)
    else:
        st.info(status_message)

    columns = st.columns(4)
    columns[0].metric("PSI", f"{report.psi:.4f}")
    columns[1].metric("Reference Samples", report.reference_sample_count)
    columns[2].metric("Current Samples", report.current_sample_count)
    columns[3].metric("Category", report.category)
    st.table(
        [
            {
                "Population": "Reference",
                "Mean": report.reference_mean,
                "P95": report.reference_p95,
                "Anomaly Ratio": report.reference_anomaly_ratio,
            },
            {
                "Population": "Current",
                "Mean": report.current_mean,
                "P95": report.current_p95,
                "Anomaly Ratio": report.current_anomaly_ratio,
            },
        ]
    )
    st.caption(
        "Window (UTC): "
        f"{report.window_start.isoformat()} to {report.window_end.isoformat()} · "
        f"Report created: {report.created_at.isoformat()}"
    )
    st.caption(
        "Drift detection does not prove model accuracy degradation. Anomaly ratio is an AI "
        "prediction ratio, not a confirmed defect rate. Ground truth is not available here."
    )


# ADD 2026-08-21: Score trend와 recent inspection table의 safe empty state를 렌더링한다.
def _render_inspections(items: tuple[InspectionItem, ...]) -> None:
    st.subheader("Anomaly Score Trend")
    if not items:
        st.info("No inspection data available.")
        return
    st.line_chart(
        score_trend_rows(items),
        x="created_at",
        y=["anomaly_score", "threshold"],
        x_label="Created At (UTC)",
        y_label="Score",
    )
    st.caption(
        "The threshold series uses each inspection's stored threshold, so mixed model or "
        "threshold lineages are not represented by one false global threshold."
    )
    st.subheader("Recent Inspections")
    st.dataframe(inspection_table_rows(items), hide_index=True, width="stretch")


# ADD 2026-08-21: Existing detail endpoint로 선택 inspection의 lineage를 필요할 때만 조회한다.
def _render_detail(client: InspectionApiClient, items: tuple[InspectionItem, ...]) -> None:
    st.subheader("Inspection Detail")
    if not items:
        st.caption("Select an inspection after history data becomes available.")
        return
    labels = {
        item.inspection_id: (
            f"{item.created_at.isoformat()} · {item.category} · "
            f"{'Anomaly' if item.is_anomaly else 'Normal'} · {str(item.inspection_id)[:8]}"
        )
        for item in items
    }
    selected = st.selectbox(
        "Inspection",
        options=list(labels),
        index=None,
        format_func=lambda inspection_id: labels[inspection_id],
        placeholder="Choose an inspection",
    )
    if selected is None:
        return
    try:
        detail = client.get_inspection(cast(UUID, selected))
    except DashboardApiError as exc:
        st.error(str(exc))
        return
    st.json(inspection_detail_fields(detail))
    st.caption("Raw product images and anomaly overlays are not stored by the current API.")


# ADD 2026-08-21: Dashboard config, API client, artifact reader와 overview UI를 조율한다.
def main() -> None:
    """Render one manually refreshed operations overview without direct database access."""
    st.set_page_config(page_title="Smart Factory AI Operations", layout="wide")
    st.title("Smart Factory AI Operations Dashboard")
    st.caption("Internal overview of persisted PatchCore predictions and batch drift reports.")

    try:
        settings = DashboardSettings.from_environment()
    except ValueError as exc:
        st.error(f"Dashboard configuration is invalid: {exc}")
        return

    # Manual rerun is the refresh boundary; no background or second-level polling is used.
    st.sidebar.header("Inspection Filters")
    st.sidebar.button("Refresh Data", use_container_width=True)
    recent_count = st.sidebar.select_slider(
        "Recent record count",
        options=[20, 50, 100],
        value=100,
    )
    client = InspectionApiClient(
        settings.api_base_url,
        timeout_seconds=settings.request_timeout_seconds,
    )
    try:
        page = client.list_inspections(limit=recent_count)
        recent_items = page.items
        api_error = None
    except DashboardApiError as exc:
        recent_items = ()
        api_error = str(exc)

    categories = sorted({item.category for item in recent_items})
    category_label = st.sidebar.selectbox("Category", ["All", *categories])
    result_label = st.sidebar.selectbox("Prediction result", ["All", "Normal", "Anomaly"])
    result_filter = cast(ResultFilter, result_label.lower())
    displayed_items = filter_inspections(
        recent_items,
        category=None if category_label == "All" else category_label,
        result=result_filter,
        limit=recent_count,
    )

    # Drift access를 API와 분리해 한 source의 실패가 다른 source를 숨기지 않게 한다.
    try:
        drift_report = load_latest_drift_report(settings.drift_report_dir)
        drift_error = None
    except MalformedDriftReportError as exc:
        drift_report = None
        drift_error = str(exc)

    if api_error is not None:
        st.error(api_error)
    _render_kpis(displayed_items, drift_report)
    _render_inspections(displayed_items)
    _render_detail(client, displayed_items)
    _render_drift(drift_report, drift_error)

    st.subheader("Service Monitoring")
    st.link_button("Open Grafana Monitoring", settings.grafana_url)
    st.caption(
        "Grafana owns request rate, HTTP/inference/persistence latency, error rate, and service "
        "availability. This dashboard does not duplicate those operational metric charts."
    )


main()
