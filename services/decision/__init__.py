"""Project-owned manufacturing decision domain."""

from services.decision.engine import decide_manufacturing_inspection
from services.decision.models import (
    DecisionInput,
    DecisionResult,
    Disposition,
    KnownDefectDecisionEvidence,
    ModelPrediction,
    PatchCoreDecisionEvidence,
    ReasonCode,
)
from services.decision.policy import MODEL_AGREEMENT_POLICY_V1

__all__ = [
    "MODEL_AGREEMENT_POLICY_V1",
    "DecisionInput",
    "DecisionResult",
    "Disposition",
    "KnownDefectDecisionEvidence",
    "ModelPrediction",
    "PatchCoreDecisionEvidence",
    "ReasonCode",
    "decide_manufacturing_inspection",
]
