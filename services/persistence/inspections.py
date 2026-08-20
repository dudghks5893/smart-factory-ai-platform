"""Inspection persistence domain contracts and SQLAlchemy repository."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from services.persistence.database import PersistenceError
from services.persistence.models import InspectionRecord
from shared.hashing import is_sha256_digest


@dataclass(frozen=True)
class InspectionCreate:
    """Validated values required to persist one completed inference."""

    model_name: str
    category: str
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    comparison_operator: str
    image_sha256: str
    image_size_bytes: int
    content_type: str
    model_sha256: str
    artifact_metadata_sha256: str
    threshold_artifact_sha256: str
    manifest_sha256: str
    device: str

    # ADD 2026-08-20: Inspection prediction과 input/model provenance invariant를 검증한다.
    def validate(self) -> None:
        """Reject incomplete, non-finite, or inconsistent inspection values."""
        for field, value in (
            ("model_name", self.model_name),
            ("category", self.category),
            ("content_type", self.content_type),
            ("device", self.device),
        ):
            if not value:
                raise ValueError(f"Inspection {field} must not be empty.")
        if not math.isfinite(self.anomaly_score) or not math.isfinite(self.threshold):
            raise ValueError("Inspection score and threshold must be finite.")
        if self.comparison_operator != ">":
            raise ValueError("Inspection comparison_operator must be '>'.")
        if self.is_anomaly is not (self.anomaly_score > self.threshold):
            raise ValueError("Inspection result violates the strict score > threshold contract.")
        if self.image_size_bytes <= 0:
            raise ValueError("Inspection image_size_bytes must be positive.")
        for field, digest in (
            ("image_sha256", self.image_sha256),
            ("model_sha256", self.model_sha256),
            ("artifact_metadata_sha256", self.artifact_metadata_sha256),
            ("threshold_artifact_sha256", self.threshold_artifact_sha256),
            ("manifest_sha256", self.manifest_sha256),
        ):
            if not is_sha256_digest(digest):
                raise ValueError(f"Inspection {field} must be a SHA-256 hex digest.")


@dataclass(frozen=True)
class Inspection:
    """Transport-independent persisted inspection history item."""

    id: UUID
    created_at: datetime
    model_name: str
    category: str
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    comparison_operator: str
    image_sha256: str
    image_size_bytes: int
    content_type: str
    model_sha256: str
    artifact_metadata_sha256: str
    threshold_artifact_sha256: str
    manifest_sha256: str
    device: str


@dataclass(frozen=True)
class InspectionPage:
    """Offset page with one-query has-more metadata."""

    items: tuple[Inspection, ...]
    limit: int
    offset: int
    has_more: bool


class InspectionRepository(Protocol):
    """Persistence contract consumed by FastAPI routes and test doubles."""

    # ADD 2026-08-20: Required inspection table을 lightweight query로 확인한다.
    def check_ready(self) -> None:
        """Fail when inspection persistence cannot serve queries."""
        ...

    # ADD 2026-08-20: Completed prediction을 transaction 하나로 저장한다.
    def create(self, values: InspectionCreate) -> Inspection:
        """Insert and commit exactly one inspection."""
        ...

    # ADD 2026-08-20: UUID로 inspection 하나를 조회한다.
    def get(self, inspection_id: UUID) -> Inspection | None:
        """Return one inspection or None."""
        ...

    # ADD 2026-08-20: Optional filter와 deterministic pagination으로 inspection을 조회한다.
    def list(
        self,
        *,
        category: str | None,
        is_anomaly: bool | None,
        limit: int,
        offset: int,
    ) -> InspectionPage:
        """Return newest inspections and whether another row exists."""
        ...


class SqlAlchemyInspectionRepository:
    """Request-isolated SQLAlchemy implementation of inspection persistence."""

    # ADD 2026-08-20: Request work unit을 생성할 Session factory를 보관한다.
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    # ADD 2026-08-20: Migrated inspection table과 query path를 readiness 전에 확인한다.
    def check_ready(self) -> None:
        """Verify that the required inspection table can execute a query."""
        with self._session_factory() as session:
            try:
                session.execute(select(InspectionRecord.id).limit(1))
            except SQLAlchemyError as exc:
                raise PersistenceError("Inspection schema readiness check failed.") from exc

    # ADD 2026-08-20: Inspection insert를 commit하고 failure 시 rollback한다.
    def create(self, values: InspectionCreate) -> Inspection:
        """Insert one row and expose only a committed domain value."""
        values.validate()
        record = InspectionRecord(**vars(values))
        with self._session_factory() as session:
            try:
                session.add(record)
                session.flush()
                inspection = _to_inspection(record)
                session.commit()
            except SQLAlchemyError as exc:
                session.rollback()
                raise PersistenceError("Inspection insert failed.") from exc
        return inspection

    # ADD 2026-08-20: UUID primary key로 inspection을 독립 Session에서 조회한다.
    def get(self, inspection_id: UUID) -> Inspection | None:
        """Return one inspection without retaining ORM state after the request."""
        with self._session_factory() as session:
            try:
                record = session.get(InspectionRecord, inspection_id)
            except SQLAlchemyError as exc:
                raise PersistenceError("Inspection lookup failed.") from exc
            return None if record is None else _to_inspection(record)

    # ADD 2026-08-20: Filter, newest-first ordering과 limit+1 pagination을 적용한다.
    def list(
        self,
        *,
        category: str | None,
        is_anomaly: bool | None,
        limit: int,
        offset: int,
    ) -> InspectionPage:
        """Query one deterministic page without issuing an aggregate count."""
        statement: Select[tuple[InspectionRecord]] = select(InspectionRecord)
        if category is not None:
            statement = statement.where(InspectionRecord.category == category)
        if is_anomaly is not None:
            statement = statement.where(InspectionRecord.is_anomaly.is_(is_anomaly))
        statement = (
            statement.order_by(
                InspectionRecord.created_at.desc(),
                InspectionRecord.id.desc(),
            )
            .offset(offset)
            .limit(limit + 1)
        )

        with self._session_factory() as session:
            try:
                records = tuple(session.scalars(statement))
            except SQLAlchemyError as exc:
                raise PersistenceError("Inspection history lookup failed.") from exc
        return InspectionPage(
            items=tuple(_to_inspection(record) for record in records[:limit]),
            limit=limit,
            offset=offset,
            has_more=len(records) > limit,
        )


# ADD 2026-08-20: ORM record를 timezone-aware immutable inspection domain으로 변환한다.
def _to_inspection(record: InspectionRecord) -> Inspection:
    created_at = record.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    else:
        created_at = created_at.astimezone(UTC)
    return Inspection(
        id=record.id,
        created_at=created_at,
        model_name=record.model_name,
        category=record.category,
        is_anomaly=record.is_anomaly,
        anomaly_score=record.anomaly_score,
        threshold=record.threshold,
        comparison_operator=record.comparison_operator,
        image_sha256=record.image_sha256,
        image_size_bytes=record.image_size_bytes,
        content_type=record.content_type,
        model_sha256=record.model_sha256,
        artifact_metadata_sha256=record.artifact_metadata_sha256,
        threshold_artifact_sha256=record.threshold_artifact_sha256,
        manifest_sha256=record.manifest_sha256,
        device=record.device,
    )
