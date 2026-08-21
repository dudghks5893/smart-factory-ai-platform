"""Environment-backed configuration for the operations dashboard."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_DRIFT_REPORT_DIR = Path("outputs/drift/patchcore")
DEFAULT_GRAFANA_URL = "http://localhost:3000"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 5.0


# ADD 2026-08-21: Dashboard HTTP endpoint가 browser/service 용도로 안전한 형태인지 검증한다.
def _validated_http_url(value: str, name: str) -> str:
    """Return one normalized HTTP(S) URL without query or fragment state."""
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be an absolute HTTP(S) URL.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{name} must not embed credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{name} must not include a query or fragment.")
    return normalized


@dataclass(frozen=True)
class DashboardSettings:
    """Validated dashboard endpoints, artifact root, and network timeout."""

    api_base_url: str = DEFAULT_API_BASE_URL
    drift_report_dir: Path = DEFAULT_DRIFT_REPORT_DIR
    grafana_url: str = DEFAULT_GRAFANA_URL
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS

    # ADD 2026-08-21: Process environment에서 dashboard runtime 설정을 로드한다.
    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> DashboardSettings:
        """Load dashboard settings without requiring model or database credentials."""
        values = os.environ if environ is None else environ
        drift_report_dir = values.get("DRIFT_REPORT_DIR", str(DEFAULT_DRIFT_REPORT_DIR))
        if not drift_report_dir.strip():
            raise ValueError("DRIFT_REPORT_DIR must not be empty.")
        try:
            timeout = float(
                values.get(
                    "DASHBOARD_REQUEST_TIMEOUT_SECONDS",
                    str(DEFAULT_REQUEST_TIMEOUT_SECONDS),
                )
            )
        except ValueError as exc:
            raise ValueError("DASHBOARD_REQUEST_TIMEOUT_SECONDS must be numeric.") from exc
        settings = cls(
            api_base_url=values.get("DASHBOARD_API_BASE_URL", DEFAULT_API_BASE_URL),
            drift_report_dir=Path(drift_report_dir),
            grafana_url=values.get("GRAFANA_URL", DEFAULT_GRAFANA_URL),
            request_timeout_seconds=timeout,
        )
        return settings.validated()

    # ADD 2026-08-21: Dashboard path, endpoint와 finite timeout invariant를 검증한다.
    def validated(self) -> DashboardSettings:
        """Return normalized settings or reject invalid runtime configuration."""
        if (
            not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
            or self.request_timeout_seconds > 60
        ):
            raise ValueError("DASHBOARD_REQUEST_TIMEOUT_SECONDS must be in (0, 60].")
        if not str(self.drift_report_dir).strip():
            raise ValueError("DRIFT_REPORT_DIR must not be empty.")
        return DashboardSettings(
            api_base_url=_validated_http_url(
                self.api_base_url,
                "DASHBOARD_API_BASE_URL",
            ),
            drift_report_dir=self.drift_report_dir,
            grafana_url=_validated_http_url(self.grafana_url, "GRAFANA_URL"),
            request_timeout_seconds=self.request_timeout_seconds,
        )
