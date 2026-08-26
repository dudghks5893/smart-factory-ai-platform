"""Pure deterministic manufacturing decision evaluation."""

from services.decision.models import (
    DecisionInput,
    DecisionResult,
    Disposition,
    ReasonCode,
)
from services.decision.policy import MODEL_AGREEMENT_POLICY_V1, reason_explanation


# ADD 2026-08-26: Policy v1 truth table을 normalized model evidence에 deterministic하게 적용한다.
def decide_manufacturing_inspection(evidence: DecisionInput) -> DecisionResult:
    """Apply the experimental model-agreement baseline without confidence heuristics."""
    evidence.validate()
    patchcore_anomaly = evidence.patchcore.is_anomaly
    has_known_defect = evidence.known_defects.instance_count > 0

    if not patchcore_anomaly and not has_known_defect:
        disposition = Disposition.PASS
        reason_code = ReasonCode.NO_ANOMALY_EVIDENCE
    elif patchcore_anomaly and not has_known_defect:
        disposition = Disposition.REVIEW
        reason_code = ReasonCode.UNKNOWN_ANOMALY
    elif not patchcore_anomaly and has_known_defect:
        disposition = Disposition.REVIEW
        reason_code = ReasonCode.MODEL_DISAGREEMENT
    else:
        disposition = Disposition.REJECT
        reason_code = ReasonCode.CONFIRMED_KNOWN_DEFECT

    return DecisionResult(
        disposition=disposition,
        policy=MODEL_AGREEMENT_POLICY_V1,
        reason_code=reason_code,
        reason=reason_explanation(reason_code),
        evidence=evidence,
    )
