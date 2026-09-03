"""Unit tests for C6-3C TensorRT INT8 streaming acceptance policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from services.streaming.yolo_tensorrt_int8_streaming_acceptance import (
    ACCEPTED_STATE,
    DEFAULT_STREAMING_ACCEPTANCE_POLICY,
    REJECTED_STATE,
    StreamingPerformancePolicy,
    evaluate_streaming_acceptance,
    load_streaming_acceptance_policy,
)


def _evidence(commit: str) -> dict[str, Any]:
    return {
        "characterization_id": "c6_3b_yolo11n_seg_tensorrt_int8_streaming_v1",
        "state": "TENSORRT_INT8_STREAMING_METRICS_COLLECTED_ACCEPTANCE_PENDING",
        "repository": {
            "git_commit": commit,
            "working_tree_dirty_before_run": False,
        },
        "backend": {
            "engine_sha256": ("4f397d59741f4efb7832087030b890a0fe059a657d074a3b07cdeb54493e8971"),
            "engine_rebuilt": False,
        },
        "runtime": {
            "tensorrt_version": "10.13.3.9.post1",
            "cuda_runtime_version": "12.8",
            "gpu_name": "Tesla T4",
            "gpu_compute_capability": "7.5",
            "torch_version": "2.10.0+cu128",
            "ultralytics_version": "8.4.128",
        },
        "stream": {
            "framerate": 30,
            "width": 640,
            "height": 640,
            "pixel_format": "BGR",
            "queue_max_buffers": 1,
            "queue_leaky": "downstream",
            "appsink_max_buffers": 1,
            "appsink_drop": True,
            "appsink_sync": False,
        },
        "metrics": {
            "frame_counts": {
                "source_buffers": 180,
                "processed_frames": 180,
                "dropped_frames": 0,
                "drop_rate": 0.0,
            },
            "frame_adapter_latency_ms": {"p95": 0.7},
            "inference_latency_ms": {"mean": 11.0, "p95": 12.0},
            "processing_latency_ms": {"mean": 11.8, "p95": 13.0},
            "observed_processed_fps": 30.0,
            "processing_capacity_fps_from_mean": 84.0,
            "source_frame_period_ms": 33.333333333333336,
        },
        "engine_rebuilt": False,
        "dataset_used": False,
        "validation_used": False,
        "test_used": False,
        "final_test_used": False,
        "deepstream_used": False,
    }


def _write_repo(tmp_path: Path, commit: str) -> Path:
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "c6-test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "C6 Test"],
        cwd=tmp_path,
        check=True,
    )
    marker = tmp_path / "marker.txt"
    marker.write_text("c6\n", encoding="utf-8")
    subprocess.run(["git", "add", "marker.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    if commit != actual:
        return tmp_path
    return tmp_path


# ADD 2026-09-04: Repository canonical policy가 exact thresholds/provenance로 load되는지 검증한다.
def test_repository_policy_loads() -> None:
    policy = load_streaming_acceptance_policy(DEFAULT_STREAMING_ACCEPTANCE_POLICY)
    policy.validate()
    assert policy.performance.max_drop_rate == 0.01
    assert policy.performance.min_processed_frames == 179
    assert policy.performance.max_processing_p95_ms == 16.0
    assert policy.performance.min_processing_capacity_fps == 70.0


# ADD 2026-09-04: Threshold mutation을 policy validation이 차단하는지 검증한다.
def test_performance_policy_rejects_mutation() -> None:
    with pytest.raises(ValueError, match="performance policy"):
        StreamingPerformancePolicy(
            max_drop_rate=0.02,
            min_processed_frames=179,
            min_observed_processed_fps=29.0,
            max_frame_adapter_p95_ms=1.5,
            max_inference_mean_ms=13.0,
            max_inference_p95_ms=15.0,
            max_processing_mean_ms=14.0,
            max_processing_p95_ms=16.0,
            min_processing_capacity_fps=70.0,
            require_processing_p95_below_source_period=True,
        ).validate()


# ADD 2026-09-04: Fresh passing evidence가 모든 prospective gates를 통과하는지 검증한다.
def test_evaluate_accepts_passing_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = load_streaming_acceptance_policy(DEFAULT_STREAMING_ACCEPTANCE_POLICY)
    evidence_path = tmp_path / "characterization.json"

    monkeypatch.setattr(
        "services.streaming.yolo_tensorrt_int8_streaming_acceptance._git_output",
        lambda _repo, *args: "" if args == ("status", "--porcelain") else "a" * 40,
    )
    evidence_path.write_text(json.dumps(_evidence("a" * 40)), encoding="utf-8")

    result = evaluate_streaming_acceptance(
        policy=policy,
        evidence_path=evidence_path,
        repo=tmp_path,
    )
    assert result["state"] == ACCEPTED_STATE
    assert result["all_gates_passed"] is True


# ADD 2026-09-04: Processing p95 threshold 초과가 prospective rejection으로 이어지는지 검증한다.
def test_evaluate_rejects_slow_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = load_streaming_acceptance_policy(DEFAULT_STREAMING_ACCEPTANCE_POLICY)
    evidence = _evidence("b" * 40)
    evidence["metrics"]["processing_latency_ms"]["p95"] = 17.0
    evidence_path = tmp_path / "characterization.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    monkeypatch.setattr(
        "services.streaming.yolo_tensorrt_int8_streaming_acceptance._git_output",
        lambda _repo, *args: "" if args == ("status", "--porcelain") else "b" * 40,
    )

    result = evaluate_streaming_acceptance(
        policy=policy,
        evidence_path=evidence_path,
        repo=tmp_path,
    )
    assert result["state"] == REJECTED_STATE
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
    assert "processing_p95_ms" in failed


# ADD 2026-09-04: Evidence commit이 current policy commit과 다르면 structural gate가 실패한다.
def test_evaluate_rejects_commit_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = load_streaming_acceptance_policy(DEFAULT_STREAMING_ACCEPTANCE_POLICY)
    evidence_path = tmp_path / "characterization.json"
    evidence_path.write_text(json.dumps(_evidence("c" * 40)), encoding="utf-8")

    monkeypatch.setattr(
        "services.streaming.yolo_tensorrt_int8_streaming_acceptance._git_output",
        lambda _repo, *args: "" if args == ("status", "--porcelain") else "d" * 40,
    )

    result = evaluate_streaming_acceptance(
        policy=policy,
        evidence_path=evidence_path,
        repo=tmp_path,
    )
    assert result["state"] == REJECTED_STATE
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
    assert "evidence_commit_matches_current_repository" in failed
