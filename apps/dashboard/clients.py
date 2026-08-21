"""Purpose-specific HTTP client and response contracts for inspection history."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from uuid import UUID

from shared.hashing import is_sha256_digest

MAX_API_RESPONSE_BYTES = 2 * 1024 * 1024
type JsonTransport = Callable[[str, float], object]


class DashboardApiError(RuntimeError):
    """Non-sensitive API failure safe to show in the internal dashboard."""


@dataclass(frozen=True)
class InspectionItem:
    """Dashboard-safe inspection fields excluding raw-image metadata."""

    inspection_id: UUID
    created_at: datetime
    model_name: str
    category: str
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    comparison_operator: Literal[">"]
    model_sha256: str
    artifact_metadata_sha256: str
    threshold_artifact_sha256: str
    manifest_sha256: str
    device: str

    # ADD 2026-08-21: FastAPI inspection payload를 dashboard-safe immutable item으로 복원한다.
    @classmethod
    def from_json_dict(cls, raw: object) -> InspectionItem:
        """Validate fields needed by overview and lineage detail views."""
        payload = _mapping(raw, "inspection")
        created_at = _aware_datetime(payload.get("created_at"), "created_at")
        comparison_operator = _string(payload.get("comparison_operator"), "comparison_operator")
        if comparison_operator != ">":
            raise ValueError("comparison_operator must be '>'.")
        item = cls(
            inspection_id=_uuid(payload.get("inspection_id"), "inspection_id"),
            created_at=created_at.astimezone(UTC),
            model_name=_string(payload.get("model_name"), "model_name"),
            category=_string(payload.get("category"), "category"),
            is_anomaly=_boolean(payload.get("is_anomaly"), "is_anomaly"),
            anomaly_score=_finite_float(payload.get("anomaly_score"), "anomaly_score"),
            threshold=_finite_float(payload.get("threshold"), "threshold"),
            comparison_operator=">",
            model_sha256=_sha256(payload.get("model_sha256"), "model_sha256"),
            artifact_metadata_sha256=_sha256(
                payload.get("artifact_metadata_sha256"),
                "artifact_metadata_sha256",
            ),
            threshold_artifact_sha256=_sha256(
                payload.get("threshold_artifact_sha256"),
                "threshold_artifact_sha256",
            ),
            manifest_sha256=_sha256(payload.get("manifest_sha256"), "manifest_sha256"),
            device=_string(payload.get("device"), "device"),
        )
        if item.is_anomaly is not (item.anomaly_score > item.threshold):
            raise ValueError("inspection result violates score > threshold.")
        return item


@dataclass(frozen=True)
class InspectionPage:
    """Validated bounded inspection history page returned by FastAPI."""

    items: tuple[InspectionItem, ...]
    limit: int
    offset: int
    returned_count: int
    has_more: bool

    # ADD 2026-08-21: Inspection history envelope와 pagination metadata를 검증한다.
    @classmethod
    def from_json_dict(cls, raw: object) -> InspectionPage:
        """Restore one API page and reject malformed pagination metadata."""
        payload = _mapping(raw, "inspection history")
        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            raise ValueError("inspection history items must be an array.")
        items = tuple(InspectionItem.from_json_dict(item) for item in raw_items)
        limit = _integer(payload.get("limit"), "limit")
        offset = _integer(payload.get("offset"), "offset")
        returned_count = _integer(payload.get("returned_count"), "returned_count")
        has_more = _boolean(payload.get("has_more"), "has_more")
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("inspection history pagination is outside its bounds.")
        if returned_count != len(items) or returned_count > limit:
            raise ValueError("inspection history returned_count is inconsistent.")
        return cls(items, limit, offset, returned_count, has_more)


class InspectionApiClient:
    """Bounded synchronous client used only for inspection read endpoints."""

    # ADD 2026-08-21: Validated endpoint, timeout과 injectable JSON transport를 보관한다.
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float,
        transport: JsonTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport or _read_json_url

    # ADD 2026-08-21: Existing bounded history endpoint를 optional filter와 함께 조회한다.
    def list_inspections(
        self,
        *,
        category: str | None = None,
        is_anomaly: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> InspectionPage:
        """Return one validated newest-first inspection page."""
        query: dict[str, str | int] = {"limit": limit, "offset": offset}
        if category is not None:
            query["category"] = category
        if is_anomaly is not None:
            query["is_anomaly"] = str(is_anomaly).lower()
        raw = self._request(f"/v1/inspections?{urlencode(query)}")
        try:
            return InspectionPage.from_json_dict(raw)
        except (TypeError, ValueError) as exc:
            raise DashboardApiError("Inspection API returned a malformed response.") from exc

    # ADD 2026-08-21: Existing UUID detail endpoint에서 inspection lineage를 조회한다.
    def get_inspection(self, inspection_id: UUID) -> InspectionItem:
        """Return one validated inspection detail without retaining raw-image fields."""
        raw = self._request(f"/v1/inspections/{quote(str(inspection_id), safe='')}")
        try:
            return InspectionItem.from_json_dict(raw)
        except (TypeError, ValueError) as exc:
            raise DashboardApiError("Inspection API returned a malformed response.") from exc

    # ADD 2026-08-21: Transport failure를 dashboard-safe unavailable error로 변환한다.
    def _request(self, path: str) -> object:
        try:
            return self._transport(f"{self._base_url}{path}", self._timeout_seconds)
        except DashboardApiError:
            raise
        except TimeoutError as exc:
            raise DashboardApiError("Inspection API request timed out.") from exc
        except Exception as exc:
            raise DashboardApiError("Inspection API is unavailable.") from exc


# ADD 2026-08-21: Bounded GET response를 JSON으로 decode하고 HTTP/network error를 정규화한다.
def _read_json_url(url: str, timeout_seconds: float) -> object:
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            content = response.read(MAX_API_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise DashboardApiError(f"Inspection API returned HTTP {exc.code}.") from exc
    except TimeoutError as exc:
        raise DashboardApiError("Inspection API request timed out.") from exc
    except (URLError, OSError) as exc:
        raise DashboardApiError("Inspection API is unavailable.") from exc
    if len(content) > MAX_API_RESPONSE_BYTES:
        raise DashboardApiError("Inspection API response exceeded the size limit.")
    try:
        return json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DashboardApiError("Inspection API returned invalid JSON.") from exc


# ADD 2026-08-21: JSON object 계약을 runtime mapping으로 좁힌다.
def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object.")
    return value


# ADD 2026-08-21: Required non-empty JSON string을 검증한다.
def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string.")
    return value


# ADD 2026-08-21: JSON boolean을 integer coercion 없이 검증한다.
def _boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean.")
    return value


# ADD 2026-08-21: JSON integer를 boolean coercion 없이 검증한다.
def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer.")
    return value


# ADD 2026-08-21: JSON number를 finite float로 검증한다.
def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number.")
    return result


# ADD 2026-08-21: API UUID string을 typed identifier로 복원한다.
def _uuid(value: object, field: str) -> UUID:
    try:
        return UUID(_string(value, field))
    except ValueError as exc:
        raise ValueError(f"{field} must be a UUID.") from exc


# ADD 2026-08-21: API timestamp가 timezone-aware ISO-8601인지 검증한다.
def _aware_datetime(value: object, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(_string(value, field))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 datetime.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone offset.")
    return parsed


# ADD 2026-08-21: Lineage digest를 SHA-256 hex contract로 검증한다.
def _sha256(value: object, field: str) -> str:
    digest = _string(value, field)
    if not is_sha256_digest(digest):
        raise ValueError(f"{field} must be a SHA-256 digest.")
    return digest
