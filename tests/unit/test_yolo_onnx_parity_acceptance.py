"""Synthetic contracts for the frozen C5-2 ONNX FP32 acceptance policy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.deployment.yolo_onnx import FrozenYoloSource, YoloOnnxExportMetadata
from ml.deployment.yolo_onnx_parity import (
    BackendPrediction,
    RuntimeTensorObservation,
    YoloOnnxParityEvidence,
    build_parity_evidence,
    build_sample_parity,
)
from ml.deployment.yolo_onnx_parity_acceptance import (
    ACCEPTED_STATE,
    EXPECTED_EXPORT_CONFIG_SHA256,
    EXPECTED_FROZEN_MANIFEST_SHA256,
    EXPECTED_ONNX_SHA256,
    EXPECTED_SOURCE_EXPERIMENT_ID,
    EXPECTED_SOURCE_MODEL_SHA256,
    REJECTED_STATE,
    YoloOnnxParityAcceptanceResult,
    assess_yolo_onnx_parity_acceptance,
    load_yolo_onnx_parity_acceptance_policy,
    load_yolo_onnx_parity_evidence,
)
from ml.evaluation.final_benchmark import RepositoryProvenance
from ml.evaluation.yolo_segmentation_error_analysis import PredictedInstance, mask_box
from shared.hashing import sha256_file

POLICY_PATH = Path("configs/deployment/yolo_onnx_fp32_parity_acceptance.yaml")


# ADD 2026-09-02: Validation-only derived manifest fixture를 만든다.
def _record() -> DerivedManifestRecord:
    return DerivedManifestRecord(
        dataset_name="fixture",
        dataset_version="v1",
        derived_task="yolo_segmentation",
        source_manifest_sha256="a" * 64,
        source_split="train",
        source_manifest_split="train",
        source_image_path="raw/image.png",
        source_mask_path="raw/mask.png",
        category="metal_nut",
        sample_id="sample-001",
        defect_type="bent",
        target_class="bent",
        target_class_id="0",
        derived_split="val",
        is_negative=False,
        image_width=8,
        image_height=8,
        image_path="images/val/sample.png",
        label_path="labels/val/sample.txt",
        image_sha256="b" * 64,
        mask_sha256="c" * 64,
        polygon_count=1,
        component_count=1,
        hole_count=0,
        polygon_vertex_count=4,
        round_trip_iou="1.0",
        pixel_precision="1.0",
        pixel_recall="1.0",
    )


# ADD 2026-09-02: Small synthetic mask prediction fixture를 만든다.
def _instance(
    *,
    class_id: int = 0,
    confidence: float = 0.9,
    row: int = 1,
    column: int = 1,
) -> PredictedInstance:
    mask = np.zeros((8, 8), dtype=np.bool_)
    mask[row : row + 2, column : column + 2] = True
    return PredictedInstance(
        class_id=class_id,
        confidence=confidence,
        mask=mask,
        box_xyxy=mask_box(mask),
    )


# ADD 2026-09-02: Finite backend tensor schema fixture를 만든다.
def _tensors(*, shape: tuple[int, ...] = (1, 8, 8)) -> tuple[RuntimeTensorObservation, ...]:
    return (
        RuntimeTensorObservation("boxes.xyxy", "float32", (1, 4), True),
        RuntimeTensorObservation("boxes.cls", "float32", (1,), True),
        RuntimeTensorObservation("boxes.conf", "float32", (1,), True),
        RuntimeTensorObservation("masks.data", "uint8", shape, True),
    )


# ADD 2026-09-02: Exact C5 identity를 가진 synthetic parity evidence를 만든다.
def _evidence(
    *,
    pytorch_instances: tuple[PredictedInstance, ...] | None = None,
    onnx_instances: tuple[PredictedInstance, ...] | None = None,
    pytorch_tensors: tuple[RuntimeTensorObservation, ...] | None = None,
    onnx_tensors: tuple[RuntimeTensorObservation, ...] | None = None,
) -> YoloOnnxParityEvidence:
    pytorch_instances = pytorch_instances or (_instance(),)
    onnx_instances = onnx_instances or (_instance(confidence=0.900001),)
    sample = build_sample_parity(
        record=_record(),
        pytorch=BackendPrediction(
            instances=pytorch_instances,
            tensors=pytorch_tensors or _tensors(),
        ),
        onnx=BackendPrediction(
            instances=onnx_instances,
            tensors=onnx_tensors or _tensors(),
        ),
    )
    source = cast(
        FrozenYoloSource,
        SimpleNamespace(
            candidate=SimpleNamespace(
                selected_experiment_id=EXPECTED_SOURCE_EXPERIMENT_ID,
                model_sha256=EXPECTED_SOURCE_MODEL_SHA256,
            ),
            manifest_sha256=EXPECTED_FROZEN_MANIFEST_SHA256,
        ),
    )
    metadata = cast(
        YoloOnnxExportMetadata,
        SimpleNamespace(
            onnx_sha256=EXPECTED_ONNX_SHA256,
            export_config_sha256=EXPECTED_EXPORT_CONFIG_SHA256,
        ),
    )
    return build_parity_evidence(
        parity_id="fixture-parity",
        created_at="2026-09-02T12:00:00+09:00",
        source=source,
        export_metadata=metadata,
        samples=(sample,),
        provenance=RepositoryProvenance(git_commit="1" * 40, working_tree_dirty=False),
    )


# ADD 2026-09-02: Committed policy fixture로 synthetic evidence를 평가한다.
def _assess(evidence: YoloOnnxParityEvidence) -> YoloOnnxParityAcceptanceResult:
    policy = load_yolo_onnx_parity_acceptance_policy(POLICY_PATH)
    return assess_yolo_onnx_parity_acceptance(
        evidence=evidence,
        policy=policy,
        policy_sha256=sha256_file(POLICY_PATH),
        parity_evidence_sha256=hashlib.sha256(evidence.to_json_bytes()).hexdigest(),
        policy_provenance=RepositoryProvenance(git_commit="2" * 40, working_tree_dirty=False),
    )


# ADD 2026-09-02: Policy v1 경계 안의 evidence가 accepted되는지 검증한다.
def test_valid_parity_evidence_is_accepted() -> None:
    result = _assess(_evidence())
    assert result.accepted is True
    assert result.state == ACCEPTED_STATE
    assert all(check.passed for check in result.checks)


@pytest.mark.parametrize(
    ("field", "mapping"),
    [
        ("confidence_abs_error", {"count": 1, "min": 0.0, "mean": 0.0002, "max": 0.0002}),
        ("box_iou", {"count": 1, "min": 0.998, "mean": 0.998, "max": 0.998}),
        ("mask_iou", {"count": 1, "min": 0.998, "mean": 0.998, "max": 0.998}),
    ],
)
# ADD 2026-09-02: 승인 numeric tolerance를 넘는 regression을 거부한다.
def test_numeric_policy_rejects_threshold_regression(
    field: str,
    mapping: dict[str, float | int | None],
) -> None:
    evidence = _evidence()
    if field == "confidence_abs_error":
        evidence = replace(evidence, confidence_abs_error=mapping)
    elif field == "box_iou":
        evidence = replace(evidence, box_iou=mapping)
    else:
        evidence = replace(evidence, mask_iou=mapping)
    result = _assess(evidence)
    assert result.accepted is False
    assert result.state == REJECTED_STATE


# ADD 2026-09-02: Class disagreement를 structural rejection으로 검증한다.
def test_class_disagreement_is_rejected() -> None:
    result = _assess(
        _evidence(
            pytorch_instances=(_instance(class_id=0),),
            onnx_instances=(_instance(class_id=1),),
        )
    )
    assert result.accepted is False
    assert any(check.name == "class_agreement_rate" and not check.passed for check in result.checks)


# ADD 2026-09-02: Backend prediction count mismatch를 거부한다.
def test_prediction_count_mismatch_is_rejected() -> None:
    result = _assess(
        _evidence(
            pytorch_instances=(
                _instance(row=1, column=1),
                _instance(row=5, column=5),
            ),
            onnx_instances=(_instance(row=1, column=1),),
        )
    )
    assert result.accepted is False
    assert any(
        check.name == "prediction_count_match" and not check.passed for check in result.checks
    )
    assert any(check.name == "zero_unmatched" and not check.passed for check in result.checks)


# ADD 2026-09-02: Count가 같아도 unmatched instance가 있으면 거부한다.
def test_equal_counts_with_unmatched_predictions_are_rejected() -> None:
    result = _assess(
        _evidence(
            pytorch_instances=(
                _instance(row=1, column=1),
                _instance(row=5, column=5),
            ),
            onnx_instances=(
                _instance(row=1, column=1),
                _instance(row=1, column=5),
            ),
        )
    )
    assert result.accepted is False
    assert any(check.name == "zero_unmatched" and not check.passed for check in result.checks)


# ADD 2026-09-02: Backend tensor schema mismatch를 거부한다.
def test_tensor_schema_mismatch_is_rejected() -> None:
    result = _assess(
        _evidence(
            onnx_tensors=_tensors(shape=(1, 4, 4)),
        )
    )
    assert result.accepted is False
    assert any(check.name == "tensor_schema_match" and not check.passed for check in result.checks)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("test_used", True),
        ("test_split_used", True),
        ("split", "test"),
    ],
)
# ADD 2026-09-02: Test/non-val lifecycle evidence를 fail closed한다.
def test_unsafe_lifecycle_evidence_fails_closed(field: str, value: object) -> None:
    evidence = _evidence()
    if field == "test_used":
        evidence = replace(evidence, test_used=cast(bool, value))
    elif field == "test_split_used":
        evidence = replace(evidence, test_split_used=cast(bool, value))
    else:
        evidence = replace(evidence, split=cast(str, value))
    with pytest.raises(ValueError):
        _assess(evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("onnx_sha256", "9" * 64),
        ("source_model_sha256", "8" * 64),
        ("frozen_manifest_sha256", "7" * 64),
        ("export_config_sha256", "6" * 64),
    ],
)
# ADD 2026-09-02: Frozen/model/export/ONNX identity mismatch를 거부한다.
def test_wrong_artifact_identity_is_rejected(field: str, value: str) -> None:
    evidence = _evidence()
    if field == "onnx_sha256":
        evidence = replace(evidence, onnx_sha256=value)
    elif field == "source_model_sha256":
        evidence = replace(evidence, source_model_sha256=value)
    elif field == "frozen_manifest_sha256":
        evidence = replace(evidence, frozen_manifest_sha256=value)
    else:
        evidence = replace(evidence, export_config_sha256=value)
    result = _assess(evidence)
    assert result.accepted is False
    assert any(check.name == field and not check.passed for check in result.checks)


# ADD 2026-09-02: NaN aggregate metric을 fail closed한다.
def test_non_finite_aggregate_metric_fails_closed() -> None:
    evidence = replace(
        _evidence(),
        confidence_abs_error={"count": 1, "min": 0.0, "mean": 0.0, "max": float("nan")},
    )
    with pytest.raises(ValueError):
        _assess(evidence)


# ADD 2026-09-02: Missing parity metric field를 strict loader에서 거부한다.
def test_missing_metric_field_is_rejected_by_json_loader(tmp_path: Path) -> None:
    raw = json.loads(_evidence().to_json_bytes())
    del raw["mask_iou"]
    path = tmp_path / "parity.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="fields do not match"):
        load_yolo_onnx_parity_evidence(path)


# ADD 2026-09-02: Acceptance result serialization determinism을 검증한다.
def test_acceptance_serialization_is_deterministic() -> None:
    result = _assess(_evidence())
    assert result.to_json_bytes() == result.to_json_bytes()
