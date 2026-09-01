"""Synthetic contracts for exact-ONNX TensorRT FP16 engine foundation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ml.deployment.yolo_tensorrt import (
    DEFAULT_TENSORRT_EXPORT_CONFIG,
    EXPECTED_ONNX_EXPORT_CONFIG_SHA256,
    EXPECTED_ONNX_METADATA_SHA256,
    EXPECTED_ONNX_SHA256,
    TensorRtEngineContract,
    TensorRtTensorContract,
    YoloTensorRtExportMetadata,
    load_yolo_tensorrt_export_config,
)


# ADD 2026-09-02: Valid static TensorRT engine I/O fixture를 만든다.
def _engine() -> TensorRtEngineContract:
    return TensorRtEngineContract(
        io_tensors=(
            TensorRtTensorContract("images", "INPUT", "DataType.FLOAT", (1, 3, 640, 640)),
            TensorRtTensorContract("output0", "OUTPUT", "DataType.HALF", (1, 39, 8400)),
            TensorRtTensorContract("output1", "OUTPUT", "DataType.HALF", (1, 32, 160, 160)),
        ),
        device_memory_size_bytes=1024,
    )


# ADD 2026-09-02: TensorRT export metadata fixture를 exact source/config contract로 만든다.
def _metadata() -> YoloTensorRtExportMetadata:
    config = load_yolo_tensorrt_export_config(DEFAULT_TENSORRT_EXPORT_CONFIG)
    config_payload = asdict(config)
    config_payload.pop("config_path")
    config_payload["output_root"] = str(config.output_root)
    return YoloTensorRtExportMetadata(
        schema_version=1,
        artifact_type="yolo_segmentation_tensorrt",
        export_state="TENSORRT_FP16_ENGINE_BUILT",
        export_id=config.export_id,
        created_at="2026-09-02T12:00:00+09:00",
        source_experiment_id="c4_2c_yolo11n_seg_crop350_nomosaic_maskratio2_seed42",
        frozen_manifest_sha256="a" * 64,
        source_model_sha256="b" * 64,
        source_model_family="yolo11n-seg",
        source_task="segment",
        dataset_manifest_sha256="c" * 64,
        source_onnx_sha256=EXPECTED_ONNX_SHA256,
        source_onnx_metadata_sha256=EXPECTED_ONNX_METADATA_SHA256,
        source_onnx_export_config_sha256=EXPECTED_ONNX_EXPORT_CONFIG_SHA256,
        source_onnx_export_commit="643ed9386a61bd2bf0c041f92a10b809b6d52c3e",
        tensorrt_config_sha256="d" * 64,
        tensorrt_config=config_payload,
        engine_sha256="e" * 64,
        engine_size_bytes=1234,
        engine=asdict(_engine()),
        environment={
            "python_version": "3.12.13",
            "platform": "fixture",
            "python_implementation": "cpython",
            "torch_version": "2.13.0+cu130",
            "ultralytics_version": "8.4.128",
            "tensorrt_version": "10.13.0",
            "cuda_runtime_version": "13.0",
            "cuda_available": "true",
            "gpu_name": "fixture-gpu",
            "gpu_compute_capability": "8.9",
            "gpu_total_memory_bytes": "123456789",
        },
        repository={"git_commit": "1" * 40, "working_tree_dirty": False},
        test_used=False,
        test_split_used=False,
    )


# ADD 2026-09-02: Repository config가 static FP16 characterization contract인지 검증한다.
def test_tensorrt_config_is_static_fp16_and_metrics_first() -> None:
    config = load_yolo_tensorrt_export_config(DEFAULT_TENSORRT_EXPORT_CONFIG)
    assert config.precision == "fp16"
    assert (config.batch, config.imgsz, config.dynamic) == (1, 640, False)
    assert (config.workspace_gib, config.device) == (4, 0)
    assert config.source_onnx_sha256 == EXPECTED_ONNX_SHA256
    assert config.parity.numeric_thresholds is None
    assert config.parity.test_used is False
    assert config.parity.test_split_used is False
    assert config.parity.benchmark.measured_iterations == 50


# ADD 2026-09-02: Config가 accepted C5-2 ONNX SHA를 바꾸면 fail closed한다.
def test_tensorrt_config_rejects_different_onnx_sha(tmp_path: Path) -> None:
    raw = DEFAULT_TENSORRT_EXPORT_CONFIG.read_text(encoding="utf-8").replace(
        EXPECTED_ONNX_SHA256,
        "0" * 64,
    )
    path = tmp_path / "trt.yaml"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="source ONNX identity"):
        load_yolo_tensorrt_export_config(path)


# ADD 2026-09-02: Dynamic TensorRT build를 repository contract에서 거부한다.
def test_tensorrt_config_rejects_dynamic_build(tmp_path: Path) -> None:
    raw = DEFAULT_TENSORRT_EXPORT_CONFIG.read_text(encoding="utf-8").replace(
        "dynamic: false",
        "dynamic: true",
    )
    path = tmp_path / "trt.yaml"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="static FP16 build parameters"):
        load_yolo_tensorrt_export_config(path)


# ADD 2026-09-02: Engine I/O contract가 accepted ONNX shapes를 보존하는지 검증한다.
def test_tensorrt_engine_contract_accepts_static_segmentation_shapes() -> None:
    config = load_yolo_tensorrt_export_config(DEFAULT_TENSORRT_EXPORT_CONFIG)
    _engine().validate(config=config)


# ADD 2026-09-02: Engine output shape drift를 metadata publication 전에 거부한다.
def test_tensorrt_engine_contract_rejects_changed_output_shape() -> None:
    config = load_yolo_tensorrt_export_config(DEFAULT_TENSORRT_EXPORT_CONFIG)
    engine = TensorRtEngineContract(
        io_tensors=(
            TensorRtTensorContract("images", "INPUT", "DataType.FLOAT", (1, 3, 640, 640)),
            TensorRtTensorContract("output0", "OUTPUT", "DataType.HALF", (1, 39, 8401)),
            TensorRtTensorContract("output1", "OUTPUT", "DataType.HALF", (1, 32, 160, 160)),
        ),
        device_memory_size_bytes=1024,
    )
    with pytest.raises(ValueError, match="output0 shape"):
        engine.validate(config=config)


# ADD 2026-09-02: Metadata strict JSON round-trip이 deterministic한지 검증한다.
def test_tensorrt_metadata_round_trip_is_deterministic() -> None:
    metadata = _metadata()
    payload = metadata.to_json_bytes()
    restored = YoloTensorRtExportMetadata.from_json_dict(json.loads(payload))
    assert restored.to_json_bytes() == payload
    assert json.loads(payload)["environment"]["cuda_available"] == "true"


# ADD 2026-09-02: Dirty repository provenance는 Official engine metadata로 허용하지 않는다.
def test_tensorrt_metadata_rejects_dirty_repository() -> None:
    raw = asdict(_metadata())
    raw["repository"]["working_tree_dirty"] = True
    with pytest.raises(ValueError, match="clean repository"):
        YoloTensorRtExportMetadata.from_json_dict(raw)
