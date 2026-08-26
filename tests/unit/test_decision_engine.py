"""Truth-table and validation contracts for Decision Policy v1."""

import pytest

from services.decision import (
    MODEL_AGREEMENT_POLICY_V1,
    DecisionInput,
    Disposition,
    KnownDefectDecisionEvidence,
    PatchCoreDecisionEvidence,
    ReasonCode,
    decide_manufacturing_inspection,
)


def _evidence(
    *,
    is_anomaly: bool,
    score: float,
    classes: tuple[str, ...],
) -> DecisionInput:
    return DecisionInput(
        patchcore=PatchCoreDecisionEvidence(
            is_anomaly=is_anomaly,
            score=score,
            threshold=40.0,
        ),
        known_defects=KnownDefectDecisionEvidence(
            instance_count=len(classes),
            instance_classes=classes,
        ),
    )


# ADD 2026-08-26: Experimental model-agreement baseline의 네 truth-table branch를 검증한다.
@pytest.mark.parametrize(
    ("evidence", "disposition", "reason_code"),
    [
        (
            _evidence(is_anomaly=False, score=30.0, classes=()),
            Disposition.PASS,
            ReasonCode.NO_ANOMALY_EVIDENCE,
        ),
        (
            _evidence(is_anomaly=True, score=50.0, classes=()),
            Disposition.REVIEW,
            ReasonCode.UNKNOWN_ANOMALY,
        ),
        (
            _evidence(is_anomaly=False, score=30.0, classes=("color",)),
            Disposition.REVIEW,
            ReasonCode.MODEL_DISAGREEMENT,
        ),
        (
            _evidence(is_anomaly=True, score=50.0, classes=("bent",)),
            Disposition.REJECT,
            ReasonCode.CONFIRMED_KNOWN_DEFECT,
        ),
    ],
)
def test_decision_policy_v1_truth_table(
    evidence: DecisionInput,
    disposition: Disposition,
    reason_code: ReasonCode,
) -> None:
    result = decide_manufacturing_inspection(evidence)

    assert result.disposition is disposition
    assert result.reason_code is reason_code
    assert result.policy == MODEL_AGREEMENT_POLICY_V1
    assert result.reason


# ADD 2026-08-26: 동일 evidence의 output determinism과 duplicate-class summary를 검증한다.
def test_decision_is_deterministic_and_class_summary_is_unique() -> None:
    evidence = _evidence(
        is_anomaly=True,
        score=50.0,
        classes=("scratch", "bent", "bent"),
    )

    first = decide_manufacturing_inspection(evidence)
    second = decide_manufacturing_inspection(evidence)

    assert first == second
    assert first.evidence.known_defects.instance_count == 3
    assert first.evidence.known_defects.classes == ("bent", "scratch")


# ADD 2026-08-26: YOLO confidence가 decision input/threshold가 아님을 검증한다.
def test_decision_input_has_no_yolo_confidence_threshold() -> None:
    fields = KnownDefectDecisionEvidence.__dataclass_fields__

    assert set(fields) == {"instance_count", "instance_classes"}


# ADD 2026-08-26: Inconsistent PatchCore/YOLO normalized state를 policy 전에 fail-fast한다.
@pytest.mark.parametrize(
    "evidence",
    [
        DecisionInput(
            patchcore=PatchCoreDecisionEvidence(
                is_anomaly=False,
                score=50.0,
                threshold=40.0,
            ),
            known_defects=KnownDefectDecisionEvidence(0, ()),
        ),
        DecisionInput(
            patchcore=PatchCoreDecisionEvidence(
                is_anomaly=True,
                score=50.0,
                threshold=40.0,
            ),
            known_defects=KnownDefectDecisionEvidence(2, ("bent",)),
        ),
        DecisionInput(
            patchcore=PatchCoreDecisionEvidence(
                is_anomaly=True,
                score=50.0,
                threshold=40.0,
            ),
            known_defects=KnownDefectDecisionEvidence(1, ("",)),
        ),
    ],
)
def test_decision_rejects_invalid_model_evidence(evidence: DecisionInput) -> None:
    with pytest.raises(ValueError):
        decide_manufacturing_inspection(evidence)
