"""Analyze a PostgreSQL inspection window against a PatchCore drift reference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.drift.patchcore import (
    DEFAULT_ANOMALY_RATIO_DRIFT_THRESHOLD,
    DEFAULT_ANOMALY_RATIO_WARNING_THRESHOLD,
    DEFAULT_MINIMUM_SAMPLE_COUNT,
    DEFAULT_PSI_DRIFT_THRESHOLD,
    DEFAULT_PSI_WARNING_THRESHOLD,
    DRIFT_FILENAME,
    DriftPolicy,
    build_drift_report,
    read_drift_reference,
    validate_artifact_id,
    write_drift_report,
)
from services.api.config import required_database_url
from services.persistence.database import create_database_manager
from services.persistence.drift import DriftWindowRepository, SqlAlchemyDriftWindowRepository

DEFAULT_DRIFT_OUTPUT_ROOT = Path("outputs/drift/patchcore")


@dataclass(frozen=True)
class DriftAnalysisSummary:
    """Paths and status returned after one immutable drift report is committed."""

    output_dir: Path
    report_path: Path
    report: dict[str, Any]


# ADD 2026-08-21: Persisted inspection window를 reference와 비교해 drift report를 저장한다.
def analyze_patchcore_drift(
    *,
    repository: DriftWindowRepository,
    reference_path: Path,
    output_dir: Path,
    drift_id: str,
    since: datetime,
    until: datetime,
    policy: DriftPolicy,
    created_at: str | None = None,
) -> DriftAnalysisSummary:
    """Query an isolated production window and persist one batch drift report."""
    validate_artifact_id(drift_id)
    policy.validate()
    normalized_since, normalized_until = _normalize_window(since, until)
    if output_dir.exists():
        raise FileExistsError(f"Drift report output directory already exists: {output_dir}")

    # Reference artifact를 검증하고 model/category/model SHA로 DB scan 범위를 제한한다.
    reference = read_drift_reference(reference_path)
    observations = repository.list_observations(
        category=reference.category,
        model_name=reference.model_name,
        model_sha256=reference.lineage.model_sha256,
        since=normalized_since,
        until=normalized_until,
    )

    # Query 결과의 remaining lineage를 대조한 뒤 statistics와 status를 계산한다.
    report = build_drift_report(
        drift_id=drift_id,
        reference=reference,
        observations=observations,
        since=normalized_since,
        until=normalized_until,
        policy=policy,
        created_at=created_at or datetime.now(UTC).isoformat(),
    )

    # 분석 성공 후에만 immutable output directory와 report를 생성한다.
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / DRIFT_FILENAME
    write_drift_report(report, report_path)
    return DriftAnalysisSummary(output_dir=output_dir, report_path=report_path, report=report)


# ADD 2026-08-21: ISO-8601 CLI timestamp를 timezone-aware UTC datetime으로 변환한다.
def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone offset")
    return parsed.astimezone(UTC)


# ADD 2026-08-21: Drift window를 UTC half-open interval로 정규화하고 순서를 검증한다.
def _normalize_window(since: datetime, until: datetime) -> tuple[datetime, datetime]:
    if since.tzinfo is None or since.utcoffset() is None:
        raise ValueError("Drift since must be timezone-aware.")
    if until.tzinfo is None or until.utcoffset() is None:
        raise ValueError("Drift until must be timezone-aware.")
    normalized_since = since.astimezone(UTC)
    normalized_until = until.astimezone(UTC)
    if normalized_since >= normalized_until:
        raise ValueError("Drift time window requires since < until.")
    return normalized_since, normalized_until


# ADD 2026-08-21: PostgreSQL drift batch CLI의 window와 operational policy 입력을 정의한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze PatchCore production drift from persisted inspection history."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--since", type=_parse_datetime, required=True)
    parser.add_argument("--until", type=_parse_datetime, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DRIFT_OUTPUT_ROOT)
    parser.add_argument("--drift-id", required=True)
    parser.add_argument("--minimum-sample-count", type=int, default=DEFAULT_MINIMUM_SAMPLE_COUNT)
    parser.add_argument(
        "--psi-warning-threshold", type=float, default=DEFAULT_PSI_WARNING_THRESHOLD
    )
    parser.add_argument("--psi-drift-threshold", type=float, default=DEFAULT_PSI_DRIFT_THRESHOLD)
    parser.add_argument(
        "--anomaly-ratio-warning-threshold",
        type=float,
        default=DEFAULT_ANOMALY_RATIO_WARNING_THRESHOLD,
    )
    parser.add_argument(
        "--anomaly-ratio-drift-threshold",
        type=float,
        default=DEFAULT_ANOMALY_RATIO_DRIFT_THRESHOLD,
    )
    return parser.parse_args()


# ADD 2026-08-21: Database lifecycle과 drift query/report CLI 흐름을 조정한다.
def main() -> int:
    args = _parse_args()
    policy = DriftPolicy(
        minimum_sample_count=args.minimum_sample_count,
        psi_warning_threshold=args.psi_warning_threshold,
        psi_drift_threshold=args.psi_drift_threshold,
        anomaly_ratio_warning_threshold=args.anomaly_ratio_warning_threshold,
        anomaly_ratio_drift_threshold=args.anomaly_ratio_drift_threshold,
    )

    # Required DATABASE_URL로 batch 전용 repository를 열고 완료 후 pool을 정리한다.
    database = create_database_manager(required_database_url())
    try:
        summary = analyze_patchcore_drift(
            repository=SqlAlchemyDriftWindowRepository(database.session_factory),
            reference_path=args.reference,
            output_dir=args.output_root / args.drift_id,
            drift_id=args.drift_id,
            since=args.since,
            until=args.until,
            policy=policy,
        )
    finally:
        database.dispose()

    print("PatchCore production drift analysis: PASS")
    print(f"Status: {summary.report['status']}")
    print(f"Samples: {summary.report['current_window']['sample_count']}")
    print(f"PSI: {summary.report['statistics']['psi']:.6f}")
    print(f"Report: {summary.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
