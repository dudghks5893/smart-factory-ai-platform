"""Synthetic contracts for C5-4B2 TensorRT explicit-Q/DQ engine build."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.deployment.yolo_tensorrt_int8_engine import (
    DEFAULT_TENSORRT_INT8_ENGINE_CONFIG,
    Int8EngineBuildPolicy,
    Int8EngineContract,
    Int8EngineTensor,
    _encode_engine_container,
    _read_engine_container,
    _validate_engine_header,
    load_yolo_tensorrt_int8_engine_config,
)


# ADD 2026-09-03: B2 config가 exact successful B1 Q/DQ artifact와 strong typing을 고정한다.
def test_int8_engine_config_freezes_qdq_source_and_explicit_build() -> None:
    config = load_yolo_tensorrt_int8_engine_config(DEFAULT_TENSORRT_INT8_ENGINE_CONFIG)

    assert (
        config.source.qdq_onnx_sha256
        == "d7c9af3ab3c2f71e88de26be71abe80f113f2e1c359d2a532a24079fa9b4dd00"
    )
    assert (
        config.source.qdq_metadata_sha256
        == "8c3b215082ba111d4f932f4e021a9bc11866c49ecec788a52f20b2f9fe244fa7"
    )
    assert (
        config.source.qdq_evidence_zip_sha256
        == "00f925d0ce5f6106d441822e419a039a736831c0d2c13835cfd01b62fad50990"
    )
    assert (
        config.source.qdq_run_summary_sha256
        == "c6b4dd790ae9a2ff312b9336d46c87f0efc03f3a2364ddda0b014a3f4405a60c"
    )
    assert config.source.qdq_run_commit == "8e489c80ef9527a044b100cc96172d179947e051"
    assert (
        config.source.qdq_int8_contract_sha256
        == "18309302e45855e506628bb5e262886fc2cb366f8758fc100c55aaf6dbf3c37a"
    )
    assert config.source.qdq_opset == 19
    assert config.source.quantize_linear_count == 211
    assert config.source.dequantize_linear_count == 211
    assert config.source.calibration_sample_count == 84

    assert config.build.explicit_quantization is True
    assert config.build.strongly_typed_network is True
    assert config.build.builder_int8_flag is False
    assert config.build.builder_fp16_flag is False
    assert config.build.legacy_calibrator is False


# ADD 2026-09-03: INT8/FP16 builder flag 또는 calibrator 회귀를 거부한다.
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("builder_int8_flag", True),
        ("builder_fp16_flag", True),
        ("legacy_calibrator", True),
        ("explicit_quantization", False),
        ("strongly_typed_network", False),
    ),
)
def test_int8_engine_build_policy_rejects_non_explicit_contract(
    field: str,
    value: bool,
) -> None:
    values = {
        "explicit_quantization": True,
        "strongly_typed_network": True,
        "builder_int8_flag": False,
        "builder_fp16_flag": False,
        "legacy_calibrator": False,
    }
    values[field] = value
    policy = Int8EngineBuildPolicy(**values)

    with pytest.raises(ValueError):
        policy.validate()


# ADD 2026-09-03: Ultralytics-compatible engine container가 header와 payload를 보존한다.
def test_int8_engine_container_round_trip(tmp_path: Path) -> None:
    config = load_yolo_tensorrt_int8_engine_config(DEFAULT_TENSORRT_INT8_ENGINE_CONFIG)
    payload = b"synthetic-trt-engine"
    path = tmp_path / "synthetic.engine"
    path.write_bytes(_encode_engine_container(payload, config))

    header, observed_payload = _read_engine_container(path)
    _validate_engine_header(header, config)
    assert observed_payload == payload


# ADD 2026-09-03: Engine I/O shape가 frozen segmentation graph와 일치해야 한다.
def test_int8_engine_contract_accepts_frozen_static_shapes() -> None:
    config = load_yolo_tensorrt_int8_engine_config(DEFAULT_TENSORRT_INT8_ENGINE_CONFIG)
    contract = Int8EngineContract(
        io_tensors=(
            Int8EngineTensor("images", "INPUT", "DataType.FLOAT", (1, 3, 640, 640)),
            Int8EngineTensor("output0", "OUTPUT", "DataType.FLOAT", (1, 39, 8400)),
            Int8EngineTensor("output1", "OUTPUT", "DataType.FLOAT", (1, 32, 160, 160)),
        ),
        device_memory_size_bytes=1,
    )

    contract.validate(config=config)
