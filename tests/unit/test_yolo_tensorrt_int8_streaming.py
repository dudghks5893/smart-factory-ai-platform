"""Unit tests for C6-3B TensorRT INT8 streaming characterization contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from services.streaming.yolo_tensorrt_int8_streaming import (
    EXPECTED_CHARACTERIZATION_ACCEPTANCE_STATE,
    EXPECTED_CHARACTERIZATION_ID,
    EXPECTED_OUTPUT_ROOT,
    PythonFrameBoundaryIdentity,
    StreamingBackendIdentity,
    StreamingCharacterizationPolicy,
    StreamingInferenceConfig,
    StreamingRuntimeIdentity,
    StreamingSourceConfig,
    YoloTensorRtInt8StreamingCharacterizationConfig,
    build_streaming_pipeline,
    load_streaming_characterization_config,
    summarize_frame_counts,
    summarize_latency_ms,
    verify_runtime_environment,
)


def _config(tmp_path: Path) -> YoloTensorRtInt8StreamingCharacterizationConfig:
    return YoloTensorRtInt8StreamingCharacterizationConfig(
        schema_version=1,
        characterization_id=EXPECTED_CHARACTERIZATION_ID,
        output_root=EXPECTED_OUTPUT_ROOT,
        python_frame_boundary=PythonFrameBoundaryIdentity(
            acceptance_commit="1a7419ef4c074d4ac1e49fd3deba23922dd8504d",
            validation_sha256=("83db8f1d40bd03ab457ded7829e576e189dbbc1d3fb1fb8384be4976e71929fc"),
            config_sha256=("4c20bfc683e0e20a6bdd015ddfd8ef6d24fceab3f102e390ad55facef8320fe8"),
            frame_sha256=("e1851d821c8e04ae3f7e07e546e50a5b055b2a0ea38be00b9b4e2deac2bc852d"),
            evidence_archive_sha256=(
                "c63860141627e2e0aa44a7cc897acb478352728fa22f7fd01f4ecd2ea087232a"
            ),
        ),
        backend=StreamingBackendIdentity(
            acceptance_state="TENSORRT_INT8_PARITY_ACCEPTED",
            c5_closure_commit="88e9b0b2440e99b6dfd2594bdc9a4947eff75187",
            acceptance_policy_sha256=(
                "938c06a099b681de9ac48d95132f423f5255ba4527f05d3f27f75d9eae5ad56c"
            ),
            engine_export_id="c5_4b2_yolo11n_seg_tensorrt_int8_qdq",
            engine_sha256=("4f397d59741f4efb7832087030b890a0fe059a657d074a3b07cdeb54493e8971"),
            engine_metadata_sha256=(
                "d44de78cc89fea67d6b351c2ba92f76dda0242386f4b6f14e216740ca682461e"
            ),
            engine_config_sha256=(
                "63eebcac04d11c9247bf7543fe18d0798758ab20cc734d2b18bfbece4eaf6b41"
            ),
            engine_evidence_zip_sha256=(
                "0cba556981b12a95b25feb324d0ff02b9cadeda6bde056b46e27eb7698f66b00"
            ),
            engine_build_commit="7835291c8fb123eba6acfa839977f94093c2f3ac",
            engine_rebuild_allowed=False,
        ),
        runtime=StreamingRuntimeIdentity(
            tensorrt_version="10.13.3.9.post1",
            cuda_runtime_version="12.8",
            gpu_name="Tesla T4",
            gpu_compute_capability="7.5",
            torch_version="2.10.0+cu128",
            ultralytics_version="8.4.128",
            device=0,
        ),
        stream=StreamingSourceConfig(
            source="videotestsrc",
            pattern="ball",
            is_live=True,
            do_timestamp=True,
            num_buffers=180,
            width=640,
            height=640,
            framerate=30,
            pixel_format="BGR",
            appsink_name="framesink",
            queue_max_buffers=1,
            queue_leaky="downstream",
            appsink_max_buffers=1,
            appsink_drop=True,
            appsink_sync=False,
        ),
        inference=StreamingInferenceConfig(
            imgsz=640,
            conf=0.001,
            iou=0.7,
            max_det=300,
            retina_masks=False,
            warmup_iterations=10,
        ),
        characterization=StreamingCharacterizationPolicy(
            scope="gstreamer_appsink_numpy_to_ultralytics_tensorrt_int8",
            metrics_only=True,
            numeric_thresholds=None,
            acceptance_state=EXPECTED_CHARACTERIZATION_ACCEPTANCE_STATE,
            dataset_used=False,
            validation_used=False,
            test_used=False,
            final_test_used=False,
            deepstream_used=False,
        ),
        config_path=tmp_path / "config.yaml",
    )


# ADD 2026-09-04: Repository canonical YAML이 frozen C6-3B contract로 load되는지 검증한다.
def test_repository_streaming_characterization_config() -> None:
    config = load_streaming_characterization_config(
        Path("configs/streaming/yolo_tensorrt_int8_streaming_characterization.yaml")
    )
    config.validate()
    assert config.characterization_id == EXPECTED_CHARACTERIZATION_ID
    assert config.output_root == EXPECTED_OUTPUT_ROOT


# ADD 2026-09-04: GStreamer pipeline의 live/latest-frame-wins contract를 검증한다.
def test_build_streaming_pipeline_contract(tmp_path: Path) -> None:
    pipeline = build_streaming_pipeline(_config(tmp_path))
    assert "videotestsrc num-buffers=180 pattern=ball is-live=true do-timestamp=true" in pipeline
    assert "framerate=30/1" in pipeline
    assert "video/x-raw,format=BGR,width=640,height=640" in pipeline
    assert "queue max-size-buffers=1" in pipeline
    assert "leaky=downstream" in pipeline
    assert "appsink name=framesink" in pipeline
    assert "max-buffers=1 drop=true sync=false" in pipeline


# ADD 2026-09-04: Latency summary의 count/min/mean/p50/p95/max 계산을 검증한다.
def test_summarize_latency_ms() -> None:
    summary = summarize_latency_ms([10.0, 20.0, 30.0, 40.0])
    assert summary["count"] == 4
    assert summary["min"] == 10.0
    assert summary["mean"] == 25.0
    assert summary["p50"] == 25.0
    assert summary["max"] == 40.0


# ADD 2026-09-04: Generated minus processed 방식의 drop count/rate 계산을 검증한다.
def test_summarize_frame_counts() -> None:
    summary = summarize_frame_counts(source_buffers=180, processed_frames=171)
    assert summary == {
        "source_buffers": 180,
        "processed_frames": 171,
        "dropped_frames": 9,
        "drop_rate": 0.05,
    }


# ADD 2026-09-04: Runtime mismatch가 exact accepted engine 환경을 벗어나지 못하게 한다.
def test_verify_runtime_environment_rejects_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    environment = {
        "torch_version": "2.10.0+cu128",
        "ultralytics_version": "8.4.128",
        "tensorrt_version": "10.13.3.9.post1",
        "cuda_runtime_version": "12.8",
        "gpu_name": "Tesla T4",
        "gpu_compute_capability": "7.5",
        "device": "cuda:0",
    }
    verify_runtime_environment(config=config, environment=environment)

    environment["gpu_name"] = "Different GPU"
    with pytest.raises(RuntimeError, match="gpu_name"):
        verify_runtime_environment(config=config, environment=environment)


# ADD 2026-09-04: Metrics-only policy가 a priori threshold를 도입하지 못하게 한다.
def test_characterization_policy_rejects_thresholds(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.validate()

    with pytest.raises(ValueError, match="characterization policy"):
        StreamingCharacterizationPolicy(
            scope=config.characterization.scope,
            metrics_only=True,
            numeric_thresholds=cast(Any, {"max_drop_rate": 0.1}),
            acceptance_state=EXPECTED_CHARACTERIZATION_ACCEPTANCE_STATE,
            dataset_used=False,
            validation_used=False,
            test_used=False,
            final_test_used=False,
            deepstream_used=False,
        ).validate()
