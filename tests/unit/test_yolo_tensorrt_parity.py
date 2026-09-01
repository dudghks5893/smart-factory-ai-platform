"""Synthetic validation-only contracts for PyTorch FP32 versus TensorRT FP16."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.deployment.yolo_onnx import FrozenYoloSource
from ml.deployment.yolo_onnx_parity import BackendPrediction, RuntimeTensorObservation
from ml.deployment.yolo_tensorrt import YoloTensorRtExportMetadata
from ml.deployment.yolo_tensorrt_parity import (
    TensorRtLatencyBenchmark,
    build_tensorrt_parity_evidence,
    build_tensorrt_sample_parity,
    summarize_latency_ms,
)
from ml.evaluation.final_benchmark import RepositoryProvenance
from ml.evaluation.yolo_segmentation_error_analysis import PredictedInstance, mask_box


# ADD 2026-09-02: Validation/test split을 바꿔 쓸 수 있는 manifest record fixture를 만든다.
def _record(*, split: str = "val") -> DerivedManifestRecord:
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
        derived_split=split,
        is_negative=False,
        image_width=8,
        image_height=8,
        image_path=f"images/{split}/sample.png",
        label_path=f"labels/{split}/sample.txt",
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


# ADD 2026-09-02: Small boolean mask에서 normalized prediction fixture를 만든다.
def _instance(
    *,
    class_id: int,
    confidence: float,
    row: int,
    column: int,
) -> PredictedInstance:
    mask = np.zeros((8, 8), dtype=np.bool_)
    mask[row : row + 2, column : column + 2] = True
    return PredictedInstance(
        class_id=class_id,
        confidence=confidence,
        mask=mask,
        box_xyxy=mask_box(mask),
    )


# ADD 2026-09-02: Finite synthetic backend tensor observations를 만든다.
def _tensors() -> tuple[RuntimeTensorObservation, ...]:
    return (
        RuntimeTensorObservation("boxes.xyxy", "float32", (1, 4), True),
        RuntimeTensorObservation("boxes.cls", "float32", (1,), True),
        RuntimeTensorObservation("boxes.conf", "float32", (1,), True),
        RuntimeTensorObservation("masks.data", "float32", (1, 8, 8), True),
    )


# ADD 2026-09-02: Stable 50-sample benchmark fixture를 만든다.
def _latency() -> TensorRtLatencyBenchmark:
    pytorch = summarize_latency_ms([10.0] * 50)
    tensorrt = summarize_latency_ms([5.0] * 50)
    return TensorRtLatencyBenchmark(
        scope="ultralytics_end_to_end_single_image",
        sample_id="sample-001",
        warmup_iterations=10,
        measured_iterations=50,
        pytorch_latency_ms=pytorch,
        tensorrt_latency_ms=tensorrt,
        speedup_ratio=2.0,
    )


# ADD 2026-09-02: TensorRT sample evidence가 unmatched count를 보존하는지 검증한다.
def test_tensorrt_sample_parity_preserves_counts() -> None:
    pytorch = BackendPrediction(
        instances=(
            _instance(class_id=0, confidence=0.9, row=0, column=0),
            _instance(class_id=1, confidence=0.8, row=5, column=5),
        ),
        tensors=_tensors(),
    )
    tensorrt = BackendPrediction(
        instances=(_instance(class_id=0, confidence=0.89, row=0, column=0),),
        tensors=_tensors(),
    )
    sample = build_tensorrt_sample_parity(
        record=_record(),
        pytorch=pytorch,
        tensorrt=tensorrt,
    )
    assert sample.matched_instance_count == 1
    assert sample.unmatched_pytorch_count == 1
    assert sample.unmatched_tensorrt_count == 0
    assert sample.tensorrt_predictions[0].class_id == 0


# ADD 2026-09-02: Test split sample은 evidence builder에서 fail-fast한다.
def test_tensorrt_sample_parity_rejects_test_split() -> None:
    prediction = BackendPrediction(instances=(), tensors=_tensors())
    with pytest.raises(ValueError, match="non-validation"):
        build_tensorrt_sample_parity(
            record=replace(_record(), derived_split="test"),
            pytorch=prediction,
            tensorrt=prediction,
        )


# ADD 2026-09-02: Latency summary가 count/p50/p95를 deterministic하게 기록하는지 검증한다.
def test_latency_summary_records_distribution() -> None:
    result = summarize_latency_ms([1.0, 2.0, 3.0, 4.0])
    assert result["count"] == 4
    assert result["mean"] == 2.5
    assert result["p50"] == 2.5
    assert 3.0 < float(result["p95"]) <= 4.0


# ADD 2026-09-02: Non-positive latency observation을 characterization evidence로 허용하지 않는다.
@pytest.mark.parametrize("values", [[], [0.0], [-1.0], [float("inf")]])
def test_latency_summary_rejects_invalid_values(values: list[float]) -> None:
    with pytest.raises(ValueError, match="latency"):
        summarize_latency_ms(values)


# ADD 2026-09-02: Aggregate evidence가 threshold 없이 FP16 acceptance pending 상태인지 검증한다.
def test_tensorrt_parity_evidence_is_metrics_first() -> None:
    prediction = BackendPrediction(
        instances=(_instance(class_id=0, confidence=0.9, row=0, column=0),),
        tensors=_tensors(),
    )
    sample = build_tensorrt_sample_parity(
        record=_record(),
        pytorch=prediction,
        tensorrt=prediction,
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
    engine_metadata = cast(
        YoloTensorRtExportMetadata,
        SimpleNamespace(
            source_onnx_sha256="c" * 64,
            engine_sha256="d" * 64,
            tensorrt_config_sha256="e" * 64,
        ),
    )
    evidence = build_tensorrt_parity_evidence(
        parity_id="fixture-trt-parity",
        created_at="2026-09-02T12:00:00+09:00",
        source=source,
        engine_metadata=engine_metadata,
        samples=(sample,),
        latency=_latency(),
        provenance=RepositoryProvenance(git_commit="1" * 40, working_tree_dirty=False),
        environment={
            "python_version": "3.12.13",
            "platform": "fixture",
            "python_implementation": "cpython",
            "torch_version": "2.13.0+cu130",
            "ultralytics_version": "8.4.128",
            "tensorrt_version": "10.13.0",
            "cuda_runtime_version": "13.0",
            "gpu_name": "fixture-gpu",
            "gpu_compute_capability": "8.9",
            "pytorch_device": "cuda:0",
            "tensorrt_device": "cuda:0",
        },
    )
    assert evidence.state == "TENSORRT_FP16_METRICS_COLLECTED_ACCEPTANCE_PENDING"
    assert evidence.numeric_acceptance == "PENDING_TENSORRT_FP16_TOLERANCE_APPROVAL"
    assert evidence.to_json_bytes() == evidence.to_json_bytes()
