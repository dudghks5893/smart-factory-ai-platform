"""Synthetic contracts for C5-4C TensorRT INT8 characterization."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.deployment.yolo_onnx import FrozenYoloSource
from ml.deployment.yolo_onnx_parity import BackendPrediction, RuntimeTensorObservation
from ml.deployment.yolo_tensorrt_int8_parity import (
    INT8_CHARACTERIZATION_STATE,
    build_int8_characterization_evidence,
    load_yolo_tensorrt_int8_characterization_config,
)
from ml.deployment.yolo_tensorrt_parity import (
    TensorRtLatencyBenchmark,
    build_tensorrt_sample_parity,
    summarize_latency_ms,
)
from ml.evaluation.final_benchmark import RepositoryProvenance
from ml.evaluation.yolo_segmentation_error_analysis import PredictedInstance, mask_box


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


def _instance(*, confidence: float) -> PredictedInstance:
    mask = np.zeros((8, 8), dtype=np.bool_)
    mask[1:3, 1:3] = True
    return PredictedInstance(
        class_id=0,
        confidence=confidence,
        mask=mask,
        box_xyxy=mask_box(mask),
    )


def _tensors() -> tuple[RuntimeTensorObservation, ...]:
    return (
        RuntimeTensorObservation("boxes.xyxy", "float32", (1, 4), True),
        RuntimeTensorObservation("boxes.cls", "float32", (1,), True),
        RuntimeTensorObservation("boxes.conf", "float32", (1,), True),
        RuntimeTensorObservation("masks.data", "float32", (1, 8, 8), True),
    )


def _latency() -> TensorRtLatencyBenchmark:
    pytorch = summarize_latency_ms([10.0] * 50)
    int8 = summarize_latency_ms([4.0] * 50)
    return TensorRtLatencyBenchmark(
        scope="ultralytics_end_to_end_single_image",
        sample_id="sample-001",
        warmup_iterations=10,
        measured_iterations=50,
        pytorch_latency_ms=pytorch,
        tensorrt_latency_ms=int8,
        speedup_ratio=2.5,
    )


# ADD 2026-09-04: C5-4C config가 exact B2 engine과 val-only metrics-first policy를 고정한다.
def test_int8_characterization_config_freezes_exact_b2_identity() -> None:
    config = load_yolo_tensorrt_int8_characterization_config(
        config_path := Path("configs/deployment/yolo_tensorrt_int8_characterization.yaml")
    )
    assert config.config_path == config_path.resolve()
    assert (
        config.source.engine_sha256
        == "4f397d59741f4efb7832087030b890a0fe059a657d074a3b07cdeb54493e8971"
    )
    assert (
        config.source.engine_metadata_sha256
        == "d44de78cc89fea67d6b351c2ba92f76dda0242386f4b6f14e216740ca682461e"
    )
    assert config.characterization.split == "val"
    assert config.characterization.sample_count == 28
    assert config.characterization.numeric_thresholds is None
    assert config.characterization.test_used is False
    assert config.characterization.test_split_used is False
    assert config.historical_fp16.comparison_mode == "historical_context_only_runtime_mismatch"


# ADD 2026-09-04: Numeric threshold를 characterization 전에 넣는 회귀를 거부한다.
def test_int8_characterization_policy_rejects_numeric_thresholds() -> None:
    config = load_yolo_tensorrt_int8_characterization_config(
        Path("configs/deployment/yolo_tensorrt_int8_characterization.yaml")
    )
    policy = replace(
        config.characterization,
        numeric_thresholds=cast(Any, {"max_confidence_abs_error": 0.1}),
    )
    with pytest.raises(ValueError, match="policy"):
        policy.validate()


# ADD 2026-09-04: Historical FP16 baseline은 same-runtime 성능 gate로 오인되지 않게 한다.
def test_historical_fp16_baseline_is_context_only() -> None:
    config = load_yolo_tensorrt_int8_characterization_config(
        Path("configs/deployment/yolo_tensorrt_int8_characterization.yaml")
    )
    assert config.historical_fp16.cuda_runtime_version == "13.0"
    assert config.source.engine_cuda_runtime_version == "12.8"
    assert config.historical_fp16.comparison_mode == "historical_context_only_runtime_mismatch"


# ADD 2026-09-04: Aggregate evidence가 INT8 acceptance pending 상태를 유지한다.
def test_int8_evidence_is_metrics_first_acceptance_pending() -> None:
    config = load_yolo_tensorrt_int8_characterization_config(
        Path("configs/deployment/yolo_tensorrt_int8_characterization.yaml")
    )
    pytorch = BackendPrediction(instances=(_instance(confidence=0.90),), tensors=_tensors())
    int8 = BackendPrediction(instances=(_instance(confidence=0.88),), tensors=_tensors())
    sample = build_tensorrt_sample_parity(
        record=_record(),
        pytorch=pytorch,
        tensorrt=int8,
    )
    source = cast(
        FrozenYoloSource,
        SimpleNamespace(
            candidate=SimpleNamespace(
                selected_experiment_id="c4_2c_yolo11n_seg_crop350_nomosaic_maskratio2_seed42",
                model_sha256="b" * 64,
            ),
            manifest_sha256="a" * 64,
        ),
    )
    evidence = build_int8_characterization_evidence(
        config=config,
        created_at="2026-09-04T01:00:00+09:00",
        source=source,
        samples=tuple(sample for _ in range(28)),
        latency=_latency(),
        provenance=RepositoryProvenance(git_commit="1" * 40, working_tree_dirty=False),
        environment={
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
    )
    assert evidence.state == INT8_CHARACTERIZATION_STATE
    assert evidence.sample_count == 28
    assert evidence.numeric_acceptance == "PENDING_TENSORRT_INT8_TOLERANCE_APPROVAL"
    assert evidence.test_used is False
    assert evidence.test_split_used is False
    assert evidence.to_json_bytes() == evidence.to_json_bytes()
