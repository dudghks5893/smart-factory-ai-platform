"""Durable manufacturing decision snapshot helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import Session

from services.decision.models import (
    DecisionResult,
    Disposition,
    ReasonCode,
)
from services.decision.policy import reason_explanation
from services.persistence.models import InspectionDecisionRecord


@dataclass(frozen=True)
class InspectionDecision:
    """Committed decision identity, policy and minimal model-evidence snapshot."""

    id: UUID
    combined_inspection_id: UUID
    created_at: datetime
    disposition: Disposition
    policy_name: str
    policy_version: str
    reason_code: ReasonCode
    reason: str
    patchcore_is_anomaly: bool
    patchcore_score: float
    patchcore_threshold: float
    known_defect_instance_count: int


# ADD 2026-08-26: Caller-owned combined transaction에 validated decision snapshot을 flush한다.
def add_inspection_decision(
    session: Session,
    *,
    combined_inspection_id: UUID,
    decision: DecisionResult,
) -> InspectionDecision:
    """Flush one policy result without committing the caller's transaction."""
    decision.evidence.validate()
    if not decision.policy.name or not decision.policy.version:
        raise ValueError("Decision policy identity must not be empty.")
    if decision.reason != reason_explanation(decision.reason_code):
        raise ValueError("Decision explanation does not match its stable reason code.")
    record = InspectionDecisionRecord(
        combined_inspection_id=combined_inspection_id,
        disposition=decision.disposition.value,
        policy_name=decision.policy.name,
        policy_version=decision.policy.version,
        reason_code=decision.reason_code.value,
        patchcore_is_anomaly=decision.evidence.patchcore.is_anomaly,
        patchcore_score=decision.evidence.patchcore.score,
        patchcore_threshold=decision.evidence.patchcore.threshold,
        known_defect_instance_count=decision.evidence.known_defects.instance_count,
    )
    session.add(record)
    session.flush()
    return decision_from_record(record)


# ADD 2026-08-26: ORM decision을 timezone-aware immutable persistence domain으로 변환한다.
def decision_from_record(record: InspectionDecisionRecord) -> InspectionDecision:
    created_at = record.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    else:
        created_at = created_at.astimezone(UTC)
    reason_code = ReasonCode(record.reason_code)
    return InspectionDecision(
        id=record.id,
        combined_inspection_id=record.combined_inspection_id,
        created_at=created_at,
        disposition=Disposition(record.disposition),
        policy_name=record.policy_name,
        policy_version=record.policy_version,
        reason_code=reason_code,
        reason=reason_explanation(reason_code),
        patchcore_is_anomaly=record.patchcore_is_anomaly,
        patchcore_score=record.patchcore_score,
        patchcore_threshold=record.patchcore_threshold,
        known_defect_instance_count=record.known_defect_instance_count,
    )
