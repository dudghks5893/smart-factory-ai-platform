"""Validated latest-report discovery for immutable PatchCore drift artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

DriftStatus = Literal["stable", "warning", "drift", "insufficient_data"]
SUPPORTED_DRIFT_STATUSES = {"stable", "warning", "drift", "insufficient_data"}


class MalformedDriftReportError(ValueError):
    """A discovered drift artifact cannot be safely interpreted."""


@dataclass(frozen=True)
class DriftReport:
    """Dashboard projection of one validated drift report artifact."""

    path: Path
    drift_id: str
    model_name: str
    category: str
    status: DriftStatus
    psi: float
    reference_sample_count: int
    current_sample_count: int
    reference_mean: float
    current_mean: float | None
    reference_p95: float
    current_p95: float | None
    reference_anomaly_ratio: float
    current_anomaly_ratio: float
    window_start: datetime
    window_end: datetime
    created_at: datetime

    # ADD 2026-08-21: Drift JSON schema에서 dashboard 표시용 validated projection을 생성한다.
    @classmethod
    def from_json_dict(cls, raw: object, *, path: Path) -> DriftReport:
        """Reject malformed, non-finite, naive-time, or unsupported drift reports."""
        root = _mapping(raw, "drift report")
        if _integer(root.get("schema_version"), "schema_version") != 1:
            raise ValueError("Unsupported drift report schema_version.")
        status_value = _string(root.get("status"), "status")
        if status_value not in SUPPORTED_DRIFT_STATUSES:
            raise ValueError("Unsupported drift status.")
        reference = _mapping(root.get("reference"), "reference")
        reference_summary = _mapping(reference.get("summary"), "reference.summary")
        current = _mapping(root.get("current_window"), "current_window")
        if _string(current.get("boundary"), "current_window.boundary") != (
            "since_inclusive_until_exclusive"
        ):
            raise ValueError("Unsupported drift window boundary.")
        current_count = _nonnegative_integer(current.get("sample_count"), "current sample_count")
        current_summary_raw = current.get("summary")
        if current_summary_raw is None:
            if current_count != 0:
                raise ValueError("Current summary is required for a non-empty window.")
            current_mean = None
            current_p95 = None
        else:
            current_summary = _mapping(current_summary_raw, "current_window.summary")
            current_mean = _finite_float(current_summary.get("mean"), "current mean")
            current_p95 = _finite_float(current_summary.get("p95"), "current p95")
        statistics = _mapping(root.get("statistics"), "statistics")
        window_start = _aware_datetime(current.get("since"), "current_window.since")
        window_end = _aware_datetime(current.get("until"), "current_window.until")
        if window_start >= window_end:
            raise ValueError("Drift report window must satisfy since < until.")
        reference_ratio = _ratio(
            statistics.get("reference_anomaly_ratio"),
            "reference anomaly ratio",
        )
        current_ratio = _ratio(statistics.get("anomaly_ratio"), "current anomaly ratio")
        psi = _finite_float(statistics.get("psi"), "PSI")
        if psi < 0:
            raise ValueError("PSI must be non-negative.")
        return cls(
            path=path,
            drift_id=_string(root.get("drift_id"), "drift_id"),
            model_name=_string(root.get("model_name"), "model_name"),
            category=_string(root.get("category"), "category"),
            status=cast(DriftStatus, status_value),
            psi=psi,
            reference_sample_count=_positive_integer(
                reference.get("sample_count"),
                "reference sample count",
            ),
            current_sample_count=current_count,
            reference_mean=_finite_float(reference_summary.get("mean"), "reference mean"),
            current_mean=current_mean,
            reference_p95=_finite_float(reference_summary.get("p95"), "reference p95"),
            current_p95=current_p95,
            reference_anomaly_ratio=reference_ratio,
            current_anomaly_ratio=current_ratio,
            window_start=window_start.astimezone(UTC),
            window_end=window_end.astimezone(UTC),
            created_at=_aware_datetime(root.get("created_at"), "created_at").astimezone(UTC),
        )


# ADD 2026-08-21: Configured artifact root에서 report metadata를 검증해 최신 report를 선택한다.
def load_latest_drift_report(report_root: Path) -> DriftReport | None:
    """Return the latest report by created_at and deterministic metadata, never mtime."""
    if not report_root.exists():
        return None
    if not report_root.is_dir():
        raise MalformedDriftReportError("Configured drift report path is not a directory.")
    paths = tuple(sorted(report_root.glob("*/drift.json")))
    if not paths:
        return None

    # 모든 candidate schema를 검증해야 malformed 최신 파일을 이전 값으로 숨기지 않는다.
    reports: list[DriftReport] = []
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            reports.append(DriftReport.from_json_dict(raw, path=path))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            relative_path = path.relative_to(report_root)
            raise MalformedDriftReportError(f"Malformed drift report: {relative_path}") from exc
    return select_latest_drift_report(reports)


# ADD 2026-08-21: created_at 동률을 drift id와 path로 결정해 latest 선택을 재현한다.
def select_latest_drift_report(reports: Sequence[DriftReport]) -> DriftReport | None:
    """Select deterministically without consulting mutable filesystem timestamps."""
    if not reports:
        return None
    return max(reports, key=lambda report: (report.created_at, report.drift_id, str(report.path)))


# ADD 2026-08-21: Drift JSON object 계약을 runtime mapping으로 좁힌다.
def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object.")
    return value


# ADD 2026-08-21: Required drift string을 검증한다.
def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string.")
    return value


# ADD 2026-08-21: Drift integer를 boolean coercion 없이 검증한다.
def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer.")
    return value


# ADD 2026-08-21: Nonnegative count contract를 검증한다.
def _nonnegative_integer(value: object, field: str) -> int:
    result = _integer(value, field)
    if result < 0:
        raise ValueError(f"{field} must be non-negative.")
    return result


# ADD 2026-08-21: Positive reference count contract를 검증한다.
def _positive_integer(value: object, field: str) -> int:
    result = _integer(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive.")
    return result


# ADD 2026-08-21: Drift numeric field를 finite float로 검증한다.
def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number.")
    return result


# ADD 2026-08-21: Ratio field가 finite [0, 1] 범위인지 검증한다.
def _ratio(value: object, field: str) -> float:
    result = _finite_float(value, field)
    if not 0 <= result <= 1:
        raise ValueError(f"{field} must be in [0, 1].")
    return result


# ADD 2026-08-21: Drift timestamp가 timezone-aware ISO-8601인지 검증한다.
def _aware_datetime(value: object, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(_string(value, field))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 datetime.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone offset.")
    return parsed
