"""Transport-independent inputs and outputs for manufacturing decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class Disposition(StrEnum):
    """Manufacturing disposition emitted by a versioned policy."""

    PASS = "PASS"
    REJECT = "REJECT"
    REVIEW = "REVIEW"


class ReasonCode(StrEnum):
    """Stable explanation codes for Decision Policy v1."""

    NO_ANOMALY_EVIDENCE = "NO_ANOMALY_EVIDENCE"
    UNKNOWN_ANOMALY = "UNKNOWN_ANOMALY"
    MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"
    CONFIRMED_KNOWN_DEFECT = "CONFIRMED_KNOWN_DEFECT"


class ModelPrediction(StrEnum):
    """PatchCore model output, distinct from a manufacturing disposition."""

    NORMAL = "NORMAL"
    ANOMALY = "ANOMALY"


@dataclass(frozen=True)
class PatchCoreDecisionEvidence:
    """Minimal validated PatchCore evidence consumed by the policy."""

    is_anomaly: bool
    score: float
    threshold: float

    # ADD 2026-08-26: Strict score > threshold state와 finite score evidence를 검증한다.
    def validate(self) -> None:
        if not math.isfinite(self.score) or not math.isfinite(self.threshold):
            raise ValueError("PatchCore decision evidence must be finite.")
        if self.is_anomaly is not (self.score > self.threshold):
            raise ValueError("PatchCore evidence violates strict score > threshold semantics.")

    @property
    def prediction(self) -> ModelPrediction:
        return ModelPrediction.ANOMALY if self.is_anomaly else ModelPrediction.NORMAL


@dataclass(frozen=True)
class KnownDefectDecisionEvidence:
    """Minimal YOLO instance evidence without confidence-based decision tuning."""

    instance_count: int
    instance_classes: tuple[str, ...]

    # ADD 2026-08-26: Instance count/class alignment을 decision 전에 fail-fast 검증한다.
    def validate(self) -> None:
        if self.instance_count < 0 or self.instance_count != len(self.instance_classes):
            raise ValueError("Known-defect count must match the supplied instance classes.")
        if any(not class_name for class_name in self.instance_classes):
            raise ValueError("Known-defect class names must be non-empty canonical values.")

    @property
    def classes(self) -> tuple[str, ...]:
        """Return a deterministic unique class summary without changing instance count."""
        return tuple(sorted(set(self.instance_classes)))


@dataclass(frozen=True)
class DecisionInput:
    """Normalized model evidence accepted by a manufacturing policy."""

    patchcore: PatchCoreDecisionEvidence
    known_defects: KnownDefectDecisionEvidence

    # ADD 2026-08-26: Both independent model evidence contracts를 함께 검증한다.
    def validate(self) -> None:
        self.patchcore.validate()
        self.known_defects.validate()


@dataclass(frozen=True)
class DecisionPolicyIdentity:
    """Stable name/version pair stored with every durable decision."""

    name: str
    version: str


@dataclass(frozen=True)
class DecisionResult:
    """Immutable explainable output of one deterministic policy evaluation."""

    disposition: Disposition
    policy: DecisionPolicyIdentity
    reason_code: ReasonCode
    reason: str
    evidence: DecisionInput
