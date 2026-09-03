"""Frozen C5-4D TensorRT INT8 acceptance policy and pure evaluator contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ml.deployment.yolo_tensorrt_int8_parity_acceptance import (
    ACCEPTED_STATE,
    DEFAULT_ACCEPTANCE_POLICY,
    EXPECTED_ENGINE_SHA256,
    REJECTED_STATE,
    build_yolo_tensorrt_int8_parity_acceptance_result,
    load_yolo_tensorrt_int8_parity_acceptance_policy,
)
from ml.evaluation.final_benchmark import RepositoryProvenance

POLICY_COMMIT = "1" * 40


def _tensor(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": "torch.float32",
        "shape": [1],
        "finite": True,
    }


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
        "confidence_abs_error": 0.05,
        "box_iou": 0.96,
        "mask_iou": 0.96,
    }
    return {
        "sample_id": "metal_nut_val_bent_000",
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


# ADD 2026-09-04: C5-4D policy보다 여유 있게 통과하는 fresh prospective INT8 fixture를 만든다.
def _evidence() -> dict[str, Any]:
    samples = [_sample()]
    for index in range(1, 28):
        sample = deepcopy(samples[0])
        sample["sample_id"] = f"sample_{index:02d}"
        samples.append(sample)
    return {
        "schema_version": 1,
        "characterization_id": "c5_4c_yolo11n_seg_tensorrt_int8_validation",
        "state": "TENSORRT_INT8_METRICS_COLLECTED_ACCEPTANCE_PENDING",
        "created_at": "2026-09-04T00:00:00+00:00",
        "source_experiment_id": "c4_2c_yolo11n_seg_crop350_nomosaic_maskratio2_seed42",
        "frozen_manifest_sha256": (
            "2a26b1bc03a1876f828e12a625c69c76af5e8c5713e3f64be699feffe2e8aa09"
        ),
        "source_model_sha256": ("e3fd10cdd708d31421feacfc5d694cb638e0ea60672e08796391b33aecf67155"),
        "int8_quantization_config_sha256": (
            "18309302e45855e506628bb5e262886fc2cb366f8758fc100c55aaf6dbf3c37a"
        ),
        "engine_export_id": "c5_4b2_yolo11n_seg_tensorrt_int8_qdq",
        "engine_sha256": EXPECTED_ENGINE_SHA256,
        "engine_metadata_sha256": (
            "d44de78cc89fea67d6b351c2ba92f76dda0242386f4b6f14e216740ca682461e"
        ),
        "engine_config_sha256": (
            "63eebcac04d11c9247bf7543fe18d0798758ab20cc734d2b18bfbece4eaf6b41"
        ),
        "engine_evidence_zip_sha256": (
            "0cba556981b12a95b25feb324d0ff02b9cadeda6bde056b46e27eb7698f66b00"
        ),
        "engine_build_repository_commit": "7835291c8fb123eba6acfa839977f94093c2f3ac",
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
            "min": 0.001,
            "mean": 0.03,
            "max": 0.11,
        },
        "box_iou": {
            "count": 28,
            "min": 0.94,
            "mean": 0.97,
            "max": 0.999,
        },
        "mask_iou": {
            "count": 28,
            "min": 0.95,
            "mean": 0.98,
            "max": 0.999,
        },
        "latency": {
            "scope": "ultralytics_end_to_end_single_image",
            "sample_id": "metal_nut_val_bent_000",
            "warmup_iterations": 10,
            "measured_iterations": 50,
            "pytorch_latency_ms": {
                "count": 50,
                "min": 30.0,
                "mean": 31.0,
                "p50": 31.0,
                "p95": 32.0,
                "max": 33.0,
            },
            "tensorrt_latency_ms": {
                "count": 50,
                "min": 26.0,
                "mean": 27.5,
                "p50": 27.5,
                "p95": 29.0,
                "max": 30.0,
            },
            "speedup_ratio": 31.0 / 27.5,
        },
        "historical_fp16": {
            "comparison_mode": "historical_context_only_runtime_mismatch",
            "acceptance_state": "TENSORRT_FP16_PARITY_ACCEPTED",
            "engine_sha256": ("9bbbe5297e6cc55bcea877a79f45485ee7e1e5e6a831ad5276aedc8e3d904037"),
            "policy_sha256": ("4f8f81a70417e380062358a9f3888d4fe0fa236fdfbc7b04da2616356833bfd9"),
            "tensorrt_version": "10.13.3.9.post1",
            "cuda_runtime_version": "13.0",
            "gpu_name": "Tesla T4",
            "gpu_compute_capability": "7.5",
            "torch_version": "2.13.0+cu130",
            "ultralytics_version": "8.4.128",
            "pytorch_mean_latency_ms": 31.130911420002576,
            "tensorrt_fp16_mean_latency_ms": 25.844023020001714,
            "speedup_ratio": 1.2045690949860681,
        },
        "structural_gates_passed": True,
        "numeric_acceptance": "PENDING_TENSORRT_INT8_TOLERANCE_APPROVAL",
        "samples": samples,
        "environment": {
            "python_version": "3.12.13",
            "platform": "fixture",
            "python_implementation": "cpython",
            "torch_version": "2.10.0+cu128",
            "ultralytics_version": "8.4.128",
            "tensorrt_version": "10.13.3.9.post1",
            "cuda_runtime_version": "12.8",
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


# ADD 2026-09-04: C5-4C exact 관측값을 복사하지 않은 reviewed margin을 검증한다.
def test_int8_acceptance_policy_freezes_reviewed_margins() -> None:
    policy = load_yolo_tensorrt_int8_parity_acceptance_policy(DEFAULT_ACCEPTANCE_POLICY)
    assert policy.identity.engine_sha256 == EXPECTED_ENGINE_SHA256
    assert policy.numeric.max_confidence_abs_error == 0.12
    assert policy.numeric.min_box_iou == 0.93
    assert policy.numeric.min_mask_iou == 0.93
    assert policy.performance.min_speedup_ratio == 1.05
    assert policy.runtime.cuda_runtime_version == "12.8"
    assert policy.runtime.gpu_name == "Tesla T4"
    assert policy.structural.require_evidence_commit_match_policy is True


# ADD 2026-09-04: Valid fresh policy-commit evidence가 frozen INT8 gate를 통과하는지 검증한다.
def test_int8_acceptance_accepts_valid_prospective_evidence() -> None:
    policy = load_yolo_tensorrt_int8_parity_acceptance_policy(DEFAULT_ACCEPTANCE_POLICY)
    result = build_yolo_tensorrt_int8_parity_acceptance_result(
        evidence=_evidence(),
        policy=policy,
        policy_sha256="b" * 64,
        characterization_evidence_sha256="c" * 64,
        policy_repository=RepositoryProvenance(
            git_commit=POLICY_COMMIT,
            working_tree_dirty=False,
        ),
    )
    assert result.accepted is True
    assert result.state == ACCEPTED_STATE
    assert all(check.passed for check in result.checks)


# ADD 2026-09-04: Pre-policy C5-4C characterization을 prospective PASS로 재사용하지 못하게 한다.
def test_int8_acceptance_rejects_pre_policy_evidence_commit() -> None:
    policy = load_yolo_tensorrt_int8_parity_acceptance_policy(DEFAULT_ACCEPTANCE_POLICY)
    evidence = _evidence()
    evidence["repository"]["git_commit"] = "2" * 40
    result = build_yolo_tensorrt_int8_parity_acceptance_result(
        evidence=evidence,
        policy=policy,
        policy_sha256="b" * 64,
        characterization_evidence_sha256="c" * 64,
        policy_repository=RepositoryProvenance(
            git_commit=POLICY_COMMIT,
            working_tree_dirty=False,
        ),
    )
    assert result.accepted is False
    assert result.state == REJECTED_STATE
    failed = {check.name for check in result.checks if not check.passed}
    assert failed == {"evidence_commit_matches_policy"}


# ADD 2026-09-04: INT8 confidence deviation이 approved ceiling을 넘으면 fail closed한다.
def test_int8_acceptance_rejects_confidence_regression() -> None:
    policy = load_yolo_tensorrt_int8_parity_acceptance_policy(DEFAULT_ACCEPTANCE_POLICY)
    evidence = _evidence()
    evidence["confidence_abs_error"]["max"] = 0.121
    result = build_yolo_tensorrt_int8_parity_acceptance_result(
        evidence=evidence,
        policy=policy,
        policy_sha256="b" * 64,
        characterization_evidence_sha256="c" * 64,
        policy_repository=RepositoryProvenance(POLICY_COMMIT, False),
    )
    assert result.accepted is False
    failed = {check.name for check in result.checks if not check.passed}
    assert "confidence_abs_error_max" in failed


# ADD 2026-09-04: INT8 box/mask spatial parity가 approved floor 아래면 fail closed한다.
def test_int8_acceptance_rejects_spatial_regression() -> None:
    policy = load_yolo_tensorrt_int8_parity_acceptance_policy(DEFAULT_ACCEPTANCE_POLICY)
    evidence = _evidence()
    evidence["box_iou"]["min"] = 0.929
    evidence["mask_iou"]["min"] = 0.929
    result = build_yolo_tensorrt_int8_parity_acceptance_result(
        evidence=evidence,
        policy=policy,
        policy_sha256="b" * 64,
        characterization_evidence_sha256="c" * 64,
        policy_repository=RepositoryProvenance(POLICY_COMMIT, False),
    )
    assert result.accepted is False
    failed = {check.name for check in result.checks if not check.passed}
    assert {"box_iou_min", "mask_iou_min"}.issubset(failed)


# ADD 2026-09-04: TensorRT INT8가 meaningful speedup을 잃으면 deployment acceptance를 거부한다.
def test_int8_acceptance_rejects_insufficient_speedup() -> None:
    policy = load_yolo_tensorrt_int8_parity_acceptance_policy(DEFAULT_ACCEPTANCE_POLICY)
    evidence = _evidence()
    evidence["latency"]["tensorrt_latency_ms"]["mean"] = 30.0
    evidence["latency"]["speedup_ratio"] = 31.0 / 30.0
    result = build_yolo_tensorrt_int8_parity_acceptance_result(
        evidence=evidence,
        policy=policy,
        policy_sha256="b" * 64,
        characterization_evidence_sha256="c" * 64,
        policy_repository=RepositoryProvenance(POLICY_COMMIT, False),
    )
    assert result.accepted is False
    failed = {check.name for check in result.checks if not check.passed}
    assert "speedup_ratio" in failed


# ADD 2026-09-04: Non-finite backend tensor를 aggregate metrics로 숨기지 못하게 한다.
def test_int8_acceptance_rejects_nonfinite_tensor() -> None:
    policy = load_yolo_tensorrt_int8_parity_acceptance_policy(DEFAULT_ACCEPTANCE_POLICY)
    evidence = _evidence()
    evidence["samples"][0]["tensorrt_tensors"][0]["finite"] = False
    result = build_yolo_tensorrt_int8_parity_acceptance_result(
        evidence=evidence,
        policy=policy,
        policy_sha256="b" * 64,
        characterization_evidence_sha256="c" * 64,
        policy_repository=RepositoryProvenance(POLICY_COMMIT, False),
    )
    assert result.accepted is False
    failed = {check.name for check in result.checks if not check.passed}
    assert "all_tensors_finite" in failed
