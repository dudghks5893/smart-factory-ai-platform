"""Read-only inspection window repository for PatchCore batch drift."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import Select, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from ml.drift.patchcore import DriftLineage, DriftObservation
from services.persistence.database import PersistenceError
from services.persistence.models import InspectionRecord


class DriftWindowRepository(Protocol):
    """Narrow persistence contract consumed by the batch drift pipeline."""

    # ADD 2026-08-21: Category/model/time으로 격리된 production score window를 조회한다.
    def list_observations(
        self,
        *,
        category: str,
        model_name: str,
        model_sha256: str,
        since: datetime,
        until: datetime,
    ) -> tuple[DriftObservation, ...]:
        """Return one deterministic half-open inspection window."""
        ...


class SqlAlchemyDriftWindowRepository:
    """SQLAlchemy read adapter that does not persist drift reports."""

    # ADD 2026-08-21: Batch query별 독립 Session을 생성할 factory를 보관한다.
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    # ADD 2026-08-21: Category/model/time filter와 stable order로 drift observation을 조회한다.
    def list_observations(
        self,
        *,
        category: str,
        model_name: str,
        model_sha256: str,
        since: datetime,
        until: datetime,
    ) -> tuple[DriftObservation, ...]:
        """Read score and lineage columns without loading unrelated image provenance."""
        if not category or not model_name:
            raise ValueError("Drift query category and model_name must not be empty.")
        if since.tzinfo is None or until.tzinfo is None or since >= until:
            raise ValueError("Drift query requires a valid timezone-aware time window.")

        statement: Select[tuple[InspectionRecord]] = (
            select(InspectionRecord)
            .where(
                InspectionRecord.category == category,
                InspectionRecord.model_name == model_name,
                InspectionRecord.model_sha256 == model_sha256,
                InspectionRecord.created_at >= since,
                InspectionRecord.created_at < until,
            )
            .order_by(InspectionRecord.created_at.asc(), InspectionRecord.id.asc())
        )
        with self._session_factory() as session:
            try:
                records = tuple(session.scalars(statement))
            except SQLAlchemyError as exc:
                raise PersistenceError("Drift inspection window lookup failed.") from exc
        return tuple(_to_drift_observation(record) for record in records)


# ADD 2026-08-21: ORM inspection을 narrow timezone-aware drift observation으로 변환한다.
def _to_drift_observation(record: InspectionRecord) -> DriftObservation:
    created_at = record.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    else:
        created_at = created_at.astimezone(UTC)
    return DriftObservation(
        created_at=created_at,
        model_name=record.model_name,
        category=record.category,
        anomaly_score=record.anomaly_score,
        is_anomaly=record.is_anomaly,
        lineage=DriftLineage(
            model_sha256=record.model_sha256,
            artifact_metadata_sha256=record.artifact_metadata_sha256,
            manifest_sha256=record.manifest_sha256,
            threshold_artifact_sha256=record.threshold_artifact_sha256,
        ),
    )
