"""Atomic persistence contract for correlated PatchCore and YOLO results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from services.decision.models import DecisionResult, Disposition, ReasonCode
from services.persistence.database import PersistenceError
from services.persistence.decisions import (
    InspectionDecision,
    add_inspection_decision,
    decision_from_record,
)
from services.persistence.inspections import (
    Inspection,
    InspectionCreate,
    _to_inspection,
    add_inspection,
)
from services.persistence.known_defects import (
    KnownDefectCreate,
    KnownDefectInspectionDetail,
    _to_instance,
    add_known_defect_inspection,
)
from services.persistence.known_defects import (
    _to_inspection as _to_known_inspection,
)
from services.persistence.models import (
    CombinedInspectionRecord,
    InspectionDecisionRecord,
    InspectionRecord,
    KnownDefectInspectionRecord,
    KnownDefectInstanceRecord,
)
from shared.hashing import is_sha256_digest


@dataclass(frozen=True)
class CombinedInspectionCreate:
    """One correlation and both child aggregates committed as a single unit."""

    id: UUID
    image_sha256: str
    image_width: int
    image_height: int
    image_size_bytes: int
    content_type: str
    patchcore_inference_ms: float
    orchestration_ms: float
    patchcore: InspectionCreate
    known_defect: KnownDefectCreate
    decision: DecisionResult

    # ADD 2026-08-26: Shared-image identity와 timing 및 child invariant를 함께 검증한다.
    def validate(self) -> None:
        if not is_sha256_digest(self.image_sha256):
            raise ValueError("Combined inspection image_sha256 must be a SHA-256 digest.")
        if self.image_width <= 0 or self.image_height <= 0 or self.image_size_bytes <= 0:
            raise ValueError("Combined inspection image dimensions and size must be positive.")
        if not self.content_type:
            raise ValueError("Combined inspection content_type must not be empty.")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (self.patchcore_inference_ms, self.orchestration_ms)
        ):
            raise ValueError("Combined inspection timings must be finite and non-negative.")
        self.patchcore.validate()
        self.known_defect.validate()
        if self.patchcore.image_sha256 != self.image_sha256:
            raise ValueError("PatchCore child must reference the combined image digest.")
        if self.known_defect.image_sha256 != self.image_sha256:
            raise ValueError("Known-defect child must reference the combined image digest.")
        if self.patchcore.image_size_bytes != self.image_size_bytes:
            raise ValueError("PatchCore child must reference the combined image size.")
        if (self.known_defect.image_width, self.known_defect.image_height) != (
            self.image_width,
            self.image_height,
        ):
            raise ValueError("Known-defect child dimensions must match the combined image.")
        self.decision.evidence.validate()
        if self.decision.evidence.patchcore.is_anomaly is not self.patchcore.is_anomaly:
            raise ValueError("Decision PatchCore prediction must match its durable child.")
        if not math.isclose(
            self.decision.evidence.patchcore.score,
            self.patchcore.anomaly_score,
            rel_tol=0.0,
            abs_tol=0.0,
        ) or not math.isclose(
            self.decision.evidence.patchcore.threshold,
            self.patchcore.threshold,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("Decision PatchCore score evidence must match its durable child.")
        child_classes = tuple(instance.class_name for instance in self.known_defect.instances)
        if (
            self.decision.evidence.known_defects.instance_count != len(self.known_defect.instances)
            or self.decision.evidence.known_defects.instance_classes != child_classes
        ):
            raise ValueError("Decision known-defect evidence must match its durable children.")


@dataclass(frozen=True)
class CombinedInspection:
    """Committed correlation plus both durable child aggregates."""

    id: UUID
    created_at: datetime
    image_sha256: str
    image_width: int
    image_height: int
    image_size_bytes: int
    content_type: str
    patchcore_inference_ms: float
    orchestration_ms: float
    patchcore: Inspection
    known_defect: KnownDefectInspectionDetail
    decision: InspectionDecision


@dataclass(frozen=True)
class CombinedInspectionSummary:
    """Child-free decision summary for bounded newest-first recovery."""

    id: UUID
    created_at: datetime
    patchcore_is_anomaly: bool
    known_defect_instance_count: int
    disposition: Disposition
    reason_code: ReasonCode
    policy_name: str
    policy_version: str


@dataclass(frozen=True)
class CombinedInspectionPage:
    """Offset page that uses limit+1 instead of an aggregate count query."""

    items: tuple[CombinedInspectionSummary, ...]
    limit: int
    offset: int
    has_more: bool


class CombinedInspectionRepository(Protocol):
    """Repository boundary used by combined inspection routes and test doubles."""

    def check_ready(self) -> None: ...

    def create(self, values: CombinedInspectionCreate) -> CombinedInspection: ...

    def get(self, combined_inspection_id: UUID) -> CombinedInspection | None: ...

    def list(self, *, limit: int, offset: int) -> CombinedInspectionPage: ...


class SqlAlchemyCombinedInspectionRepository:
    """SQLAlchemy unit of work that prevents partially durable combined results."""

    # ADD 2026-08-26: Request-isolated transaction factory를 보관한다.
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    # ADD 2026-08-26: Correlation schema와 query path를 startup/readiness에서 확인한다.
    def check_ready(self) -> None:
        with self._session_factory() as session:
            try:
                session.execute(select(CombinedInspectionRecord.id).limit(1))
                session.execute(select(InspectionDecisionRecord.id).limit(1))
            except SQLAlchemyError as exc:
                raise PersistenceError(
                    "Combined inspection schema readiness check failed."
                ) from exc

    # ADD 2026-08-26: 두 child aggregate와 correlation을 하나의 transaction으로 commit한다.
    # MODIFY 2026-08-26: Policy decision을 같은 transaction의 required final row로 추가한다.
    def create(self, values: CombinedInspectionCreate) -> CombinedInspection:
        values.validate()
        with self._session_factory() as session:
            try:
                patchcore = add_inspection(session, values.patchcore)
                known_defect = add_known_defect_inspection(session, values.known_defect)
                record = CombinedInspectionRecord(
                    id=values.id,
                    patchcore_inspection_id=patchcore.id,
                    known_defect_inspection_id=known_defect.inspection.id,
                    image_sha256=values.image_sha256,
                    image_width=values.image_width,
                    image_height=values.image_height,
                    image_size_bytes=values.image_size_bytes,
                    content_type=values.content_type,
                    patchcore_inference_ms=values.patchcore_inference_ms,
                    orchestration_ms=values.orchestration_ms,
                )
                session.add(record)
                session.flush()
                decision = add_inspection_decision(
                    session,
                    combined_inspection_id=record.id,
                    decision=values.decision,
                )
                combined = _to_combined(record, patchcore, known_defect, decision)
                session.commit()
            except SQLAlchemyError as exc:
                session.rollback()
                raise PersistenceError("Combined inspection insert failed.") from exc
        return combined

    # ADD 2026-08-26: Correlation과 양쪽 child를 같은 read Session에서 복구한다.
    def get(self, combined_inspection_id: UUID) -> CombinedInspection | None:
        with self._session_factory() as session:
            try:
                record = session.get(CombinedInspectionRecord, combined_inspection_id)
                if record is None:
                    return None
                patchcore_record = session.get(InspectionRecord, record.patchcore_inspection_id)
                known_record = session.get(
                    KnownDefectInspectionRecord, record.known_defect_inspection_id
                )
                if patchcore_record is None or known_record is None:
                    raise PersistenceError("Combined inspection references missing child rows.")
                decision_record = session.scalar(
                    select(InspectionDecisionRecord).where(
                        InspectionDecisionRecord.combined_inspection_id == combined_inspection_id
                    )
                )
                if decision_record is None:
                    raise PersistenceError("Combined inspection is missing its decision row.")
                children = tuple(
                    session.scalars(
                        select(KnownDefectInstanceRecord)
                        .where(
                            KnownDefectInstanceRecord.inspection_id
                            == record.known_defect_inspection_id
                        )
                        .order_by(KnownDefectInstanceRecord.instance_index.asc())
                    )
                )
                return _to_combined(
                    record,
                    _to_inspection(patchcore_record),
                    KnownDefectInspectionDetail(
                        inspection=_to_known_inspection(known_record),
                        instances=tuple(_to_instance(child) for child in children),
                    ),
                    decision_from_record(decision_record),
                )
            except SQLAlchemyError as exc:
                raise PersistenceError("Combined inspection lookup failed.") from exc

    # ADD 2026-08-26: Decision snapshot join으로 child-free newest-first history를 반환한다.
    def list(self, *, limit: int, offset: int) -> CombinedInspectionPage:
        statement = (
            select(CombinedInspectionRecord, InspectionDecisionRecord)
            .join(
                InspectionDecisionRecord,
                InspectionDecisionRecord.combined_inspection_id == CombinedInspectionRecord.id,
            )
            .order_by(
                CombinedInspectionRecord.created_at.desc(),
                CombinedInspectionRecord.id.desc(),
            )
            .offset(offset)
            .limit(limit + 1)
        )
        with self._session_factory() as session:
            try:
                records = tuple(session.execute(statement).all())
            except SQLAlchemyError as exc:
                raise PersistenceError("Combined inspection history lookup failed.") from exc
        return CombinedInspectionPage(
            items=tuple(_to_summary(combined, decision) for combined, decision in records[:limit]),
            limit=limit,
            offset=offset,
            has_more=len(records) > limit,
        )


# ADD 2026-08-26: ORM correlation을 timezone-aware durable aggregate로 변환한다.
def _to_combined(
    record: CombinedInspectionRecord,
    patchcore: Inspection,
    known_defect: KnownDefectInspectionDetail,
    decision: InspectionDecision,
) -> CombinedInspection:
    created_at = record.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    else:
        created_at = created_at.astimezone(UTC)
    return CombinedInspection(
        id=record.id,
        created_at=created_at,
        image_sha256=record.image_sha256,
        image_width=record.image_width,
        image_height=record.image_height,
        image_size_bytes=record.image_size_bytes,
        content_type=record.content_type,
        patchcore_inference_ms=record.patchcore_inference_ms,
        orchestration_ms=record.orchestration_ms,
        patchcore=patchcore,
        known_defect=known_defect,
        decision=decision,
    )


# ADD 2026-08-26: Correlation과 decision snapshot을 child-free history domain으로 변환한다.
def _to_summary(
    combined: CombinedInspectionRecord,
    decision: InspectionDecisionRecord,
) -> CombinedInspectionSummary:
    created_at = combined.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    else:
        created_at = created_at.astimezone(UTC)
    return CombinedInspectionSummary(
        id=combined.id,
        created_at=created_at,
        patchcore_is_anomaly=decision.patchcore_is_anomaly,
        known_defect_instance_count=decision.known_defect_instance_count,
        disposition=Disposition(decision.disposition),
        reason_code=ReasonCode(decision.reason_code),
        policy_name=decision.policy_name,
        policy_version=decision.policy_version,
    )
