"""Frozen C5-3 TensorRT FP16 acceptance policy and pure evaluator contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ml.deployment.yolo_tensorrt_parity_acceptance import (
    ACCEPTED_STATE,
    DEFAULT_ACCEPTANCE_POLICY,
    EXPECTED_ENGINE_SHA256,
    REJECTED_STATE,
    build_yolo_tensorrt_parity_acceptance_result,
    load_yolo_tensorrt_parity_acceptance_policy,
)
from ml.evaluation.final_benchmark import RepositoryProvenance

POLICY_COMMIT = "1" * 40


# ADD 2026-09-02: Prospective acceptance unit test용 valid tensor observation을 만든다.
def _tensor(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": "torch.float32",
        "shape": [1],
        "finite": True,
    }


# ADD 2026-09-02: One matched prediction sample의 minimal valid evidence를 만든다.
def _sample() -> dict[str, Any]:
    prediction = {
        "prediction_index": 0,
        "class_id": 0,
        "confidence": 0.9,
        "box_xyxy": [1.0, 2.0, 3.0, 4.0],
        "mask_shape": [10, 10],
        "mask_foreground_pixels": 10,
        "mask_sha256": "a" * 64,
    }
    match = {
        "reference_index": 0,
        "candidate_index": 0,
        "class_agreement": True,
        "confidence_abs_error": 0.001,
        "box_iou": 0.999,
        "mask_iou": 0.999,
    }
    return {
        "sample_id": "metal_nut_test_bent_000",
        "split": "val",
        "pytorch_prediction_count": 1,
        "tensorrt_prediction_count": 1,
        "matched_instance_count": 1,
        "unmatched_pytorch_count": 0,
        "unmatched_tensorrt_count": 0,
        "pytorch_tensors": [_tensor("boxes")],
        "tensorrt_tensors": [_tensor("boxes")],
        "pytorch_predictions": [prediction],
        "tensorrt_predictions": [prediction],
        "matches": [match],
    }


# ADD 2026-09-02: Characterization보다 여유를 둔 prospective TensorRT parity fixture를 만든다.
def _evidence() -> dict[str, Any]:
    samples = [_sample()]
    for index in range(1, 28):
        sample = deepcopy(samples[0])
        sample["sample_id"] = f"sample_{index:02d}"
        samples.append(sample)
    return {
        "schema_version": 1,
        "parity_id": "c5_3c_prospective_unit",
        "state": "TENSORRT_FP16_METRICS_COLLECTED_ACCEPTANCE_PENDING",
        "created_at": "2026-09-02T00:00:00+00:00",
        "source_experiment_id": "c4_2c_yolo11n_seg_crop350_nomosaic_maskratio2_seed42",
        "frozen_manifest_sha256": (
            "2a26b1bc03a1876f828e12a625c69c76af5e8c5713e3f64be699feffe2e8aa09"
        ),
        "source_model_sha256": ("e3fd10cdd708d31421feacfc5d694cb638e0ea60672e08796391b33aecf67155"),
        "source_onnx_sha256": ("f916325bb126d174de9c1fdfc24802eec11c46014f723fbf3ba3b3c1755c1490"),
        "engine_sha256": EXPECTED_ENGINE_SHA256,
        "tensorrt_config_sha256": (
            "edc135932e9367f67b9179dbbd47b01da6fa07db878a7f8af73b491718b517c9"
        ),
        "split": "val",
        "test_used": False,
        "test_split_used": False,
        "sample_count": 28,
        "pytorch_prediction_count": 28,
        "tensorrt_prediction_count": 28,
        "matched_instance_count": 28,
        "unmatched_pytorch_count": 0,
        "unmatched_tensorrt_count": 0,
        "class_agreement_count": 28,
        "class_agreement_rate": 1.0,
        "confidence_abs_error": {
            "count": 28,
            "min": 0.0001,
            "mean": 0.001,
            "max": 0.005,
        },
        "box_iou": {
            "count": 28,
            "min": 0.985,
            "mean": 0.999,
            "max": 1.0,
        },
        "mask_iou": {
            "count": 28,
            "min": 0.997,
            "mean": 0.999,
            "max": 1.0,
        },
        "latency": {
            "scope": "ultralytics_end_to_end_single_image",
            "sample_id": "metal_nut_test_bent_000",
            "warmup_iterations": 10,
            "measured_iterations": 50,
            "pytorch_latency_ms": {
                "count": 50,
                "min": 30.0,
                "mean": 33.0,
                "p50": 33.0,
                "p95": 34.0,
                "max": 35.0,
            },
            "tensorrt_latency_ms": {
                "count": 50,
                "min": 25.0,
                "mean": 27.0,
                "p50": 27.0,
                "p95": 29.0,
                "max": 30.0,
            },
            "speedup_ratio": 33.0 / 27.0,
        },
        "structural_gates_passed": True,
        "numeric_acceptance": "PENDING_TENSORRT_FP16_TOLERANCE_APPROVAL",
        "samples": samples,
        "environment": {
            "python_version": "3.12.13",
            "platform": "fixture",
            "python_implementation": "cpython",
            "torch_version": "2.13.0+cu130",
            "ultralytics_version": "8.4.128",
            "tensorrt_version": "10.13.3.9.post1",
            "cuda_runtime_version": "13.0",
            "gpu_name": "Tesla T4",
            "gpu_compute_capability": "7.5",
            "pytorch_device": "cuda:0",
            "tensorrt_device": "cuda:0",
        },
        "repository": {
            "git_commit": POLICY_COMMIT,
            "working_tree_dirty": False,
        },
    }


# ADD 2026-09-02: Repository policy가 characterization 관측값에 과적합하지 않은 margin인지 검증한다.
def test_tensorrt_acceptance_policy_freezes_reviewed_margins() -> None:
    policy = load_yolo_tensorrt_parity_acceptance_policy(DEFAULT_ACCEPTANCE_POLICY)
    assert policy.identity.engine_sha256 == EXPECTED_ENGINE_SHA256
    assert policy.numeric.max_confidence_abs_error == 0.01
    assert policy.numeric.min_box_iou == 0.98
    assert policy.numeric.min_mask_iou == 0.995
    assert policy.performance.min_speedup_ratio == 1.05
    assert policy.runtime.gpu_name == "Tesla T4"
    assert policy.structural.require_evidence_commit_match_policy is True


# ADD 2026-09-02: Valid prospective evidence가 frozen acceptance gate를 통과하는지 검증한다.
def test_tensorrt_acceptance_accepts_valid_prospective_evidence() -> None:
    policy = load_yolo_tensorrt_parity_acceptance_policy(DEFAULT_ACCEPTANCE_POLICY)
    result = build_yolo_tensorrt_parity_acceptance_result(
        evidence=_evidence(),
        policy=policy,
        policy_sha256="b" * 64,
        parity_evidence_sha256="c" * 64,
        policy_repository=RepositoryProvenance(
            git_commit=POLICY_COMMIT,
            working_tree_dirty=False,
        ),
    )
    assert result.accepted is True
    assert result.state == ACCEPTED_STATE
    assert all(check.passed for check in result.checks)


# ADD 2026-09-02: Pre-policy characterization evidence를 prospective PASS로 재사용하지 못하게 한다.
def test_tensorrt_acceptance_rejects_pre_policy_evidence_commit() -> None:
    policy = load_yolo_tensorrt_parity_acceptance_policy(DEFAULT_ACCEPTANCE_POLICY)
    evidence = _evidence()
    evidence["repository"]["git_commit"] = "2" * 40
    result = build_yolo_tensorrt_parity_acceptance_result(
        evidence=evidence,
        policy=policy,
        policy_sha256="b" * 64,
        parity_evidence_sha256="c" * 64,
        policy_repository=RepositoryProvenance(
            git_commit=POLICY_COMMIT,
            working_tree_dirty=False,
        ),
    )
    assert result.accepted is False
    assert result.state == REJECTED_STATE
    failed = {check.name for check in result.checks if not check.passed}
    assert failed == {"evidence_commit_matches_policy"}


# ADD 2026-09-02: FP16 box parity가 approved floor 아래로 내려가면 fail closed하는지 검증한다.
def test_tensorrt_acceptance_rejects_box_iou_regression() -> None:
    policy = load_yolo_tensorrt_parity_acceptance_policy(DEFAULT_ACCEPTANCE_POLICY)
    evidence = _evidence()
    evidence["box_iou"]["min"] = 0.979
    result = build_yolo_tensorrt_parity_acceptance_result(
        evidence=evidence,
        policy=policy,
        policy_sha256="b" * 64,
        parity_evidence_sha256="c" * 64,
        policy_repository=RepositoryProvenance(
            git_commit=POLICY_COMMIT,
            working_tree_dirty=False,
        ),
    )
    assert result.accepted is False
    failed = {check.name for check in result.checks if not check.passed}
    assert "box_iou_min" in failed


# ADD 2026-09-02: TensorRT가 meaningful speedup을 잃으면 deployment acceptance를 거부한다.
def test_tensorrt_acceptance_rejects_insufficient_speedup() -> None:
    policy = load_yolo_tensorrt_parity_acceptance_policy(DEFAULT_ACCEPTANCE_POLICY)
    evidence = _evidence()
    evidence["latency"]["tensorrt_latency_ms"]["mean"] = 32.0
    evidence["latency"]["speedup_ratio"] = 1.03125
    result = build_yolo_tensorrt_parity_acceptance_result(
        evidence=evidence,
        policy=policy,
        policy_sha256="b" * 64,
        parity_evidence_sha256="c" * 64,
        policy_repository=RepositoryProvenance(
            git_commit=POLICY_COMMIT,
            working_tree_dirty=False,
        ),
    )
    assert result.accepted is False
    failed = {check.name for check in result.checks if not check.passed}
    assert "speedup_ratio" in failed


# ADD 2026-09-02: Non-finite backend tensor를 aggregate flag만으로 숨기지 못하게 한다.
def test_tensorrt_acceptance_rejects_nonfinite_tensor() -> None:
    policy = load_yolo_tensorrt_parity_acceptance_policy(DEFAULT_ACCEPTANCE_POLICY)
    evidence = _evidence()
    evidence["samples"][0]["tensorrt_tensors"][0]["finite"] = False
    result = build_yolo_tensorrt_parity_acceptance_result(
        evidence=evidence,
        policy=policy,
        policy_sha256="b" * 64,
        parity_evidence_sha256="c" * 64,
        policy_repository=RepositoryProvenance(
            git_commit=POLICY_COMMIT,
            working_tree_dirty=False,
        ),
    )
    assert result.accepted is False
    failed = {check.name for check in result.checks if not check.passed}
    assert "all_tensors_finite" in failed
