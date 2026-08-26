"""Versioned experimental model-agreement policy definitions."""

from services.decision.models import DecisionPolicyIdentity, ReasonCode

MODEL_AGREEMENT_POLICY_V1 = DecisionPolicyIdentity(
    name="model_agreement",
    version="1",
)

REASON_EXPLANATIONS: dict[ReasonCode, str] = {
    ReasonCode.NO_ANOMALY_EVIDENCE: (
        "Neither PatchCore nor the known-defect model produced anomaly evidence."
    ),
    ReasonCode.UNKNOWN_ANOMALY: (
        "PatchCore detected an anomaly without a matching known-defect instance."
    ),
    ReasonCode.MODEL_DISAGREEMENT: (
        "Known-defect instances were detected without PatchCore anomaly evidence."
    ),
    ReasonCode.CONFIRMED_KNOWN_DEFECT: (
        "PatchCore anomaly evidence and known-defect instances are both present."
    ),
}


# ADD 2026-08-26: Stable reason code의 human-readable v1 explanation을 반환한다.
def reason_explanation(reason_code: ReasonCode) -> str:
    return REASON_EXPLANATIONS[reason_code]
