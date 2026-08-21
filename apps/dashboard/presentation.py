"""Pure presentation models for inspection KPIs, charts, tables, and detail."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from typing import Literal

from apps.dashboard.clients import InspectionItem

ResultFilter = Literal["all", "normal", "anomaly"]


@dataclass(frozen=True)
class InspectionKpis:
    """Counts calculated only from the currently displayed inspection sample."""

    recent_inspections: int
    normal_count: int
    anomaly_count: int
    anomaly_ratio: float
    model_versions: int


# ADD 2026-08-21: Displayed inspection sample에서 prediction KPI를 계산한다.
def calculate_inspection_kpis(items: tuple[InspectionItem, ...]) -> InspectionKpis:
    """Calculate safe empty-state counts and anomaly prediction ratio."""
    anomaly_count = sum(item.is_anomaly for item in items)
    total = len(items)
    return InspectionKpis(
        recent_inspections=total,
        normal_count=total - anomaly_count,
        anomaly_count=anomaly_count,
        anomaly_ratio=0.0 if total == 0 else anomaly_count / total,
        model_versions=len({item.model_sha256 for item in items}),
    )


# ADD 2026-08-21: Recent API sample에 category/result/count UI filter를 적용한다.
def filter_inspections(
    items: tuple[InspectionItem, ...],
    *,
    category: str | None,
    result: ResultFilter,
    limit: int,
) -> tuple[InspectionItem, ...]:
    """Return newest matching items without changing the API ordering."""
    if limit <= 0:
        raise ValueError("Inspection display limit must be positive.")
    if result not in {"all", "normal", "anomaly"}:
        raise ValueError("Unsupported inspection result filter.")
    selected = (
        item
        for item in items
        if (category is None or item.category == category)
        and (result == "all" or item.is_anomaly is (result == "anomaly"))
    )
    return tuple(list(selected)[:limit])


# ADD 2026-08-21: Record별 threshold를 보존한 UTC score trend rows를 생성한다.
def score_trend_rows(items: tuple[InspectionItem, ...]) -> list[dict[str, object]]:
    """Return oldest-first chart rows safe for mixed model/threshold lineages."""
    return [
        {
            "created_at": item.created_at.astimezone(UTC),
            "anomaly_score": item.anomaly_score,
            "threshold": item.threshold,
        }
        for item in sorted(items, key=lambda item: (item.created_at, item.inspection_id))
    ]


# ADD 2026-08-21: Inspection overview table의 non-sensitive columns를 명시적으로 구성한다.
def inspection_table_rows(items: tuple[InspectionItem, ...]) -> list[dict[str, object]]:
    """Return newest-first rows without UUID, image hash, or unsupported defect type."""
    return [
        {
            "Created At (UTC)": _utc_text(item),
            "Category": item.category,
            "Anomaly Score": item.anomaly_score,
            "Threshold": item.threshold,
            "Result": "Anomaly" if item.is_anomaly else "Normal",
            "Model Name": item.model_name,
            "Device": item.device,
        }
        for item in items
    ]


# ADD 2026-08-21: Selected inspection의 prediction과 lineage detail만 노출한다.
def inspection_detail_fields(item: InspectionItem) -> dict[str, object]:
    """Return explicit detail fields while excluding unavailable raw image data."""
    return {
        "Inspection ID": str(item.inspection_id),
        "Created At (UTC)": _utc_text(item),
        "Category": item.category,
        "Result": "Anomaly" if item.is_anomaly else "Normal",
        "Anomaly Score": item.anomaly_score,
        "Threshold": item.threshold,
        "Decision Rule": "score > threshold",
        "Model Name": item.model_name,
        "Device": item.device,
        "Model SHA": item.model_sha256,
        "Artifact Metadata SHA": item.artifact_metadata_sha256,
        "Threshold Artifact SHA": item.threshold_artifact_sha256,
        "Manifest SHA": item.manifest_sha256,
    }


# ADD 2026-08-21: Inspection timestamp를 explicit UTC text로 변환한다.
def _utc_text(item: InspectionItem) -> str:
    return item.created_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
