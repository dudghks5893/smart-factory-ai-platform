"""Contracts for the C5-4A TensorRT INT8 explicit-Q/DQ foundation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.deployment.yolo_tensorrt_int8 import (
    DEFAULT_TENSORRT_INT8_CONFIG,
    EXPECTED_ONNX_SHA256,
    load_yolo_tensorrt_int8_config,
)


# ADD 2026-09-02: Repository config가 explicit Q/DQ, train-only calibration,
# val-only characterization인지 검증한다.
def test_int8_config_uses_explicit_qdq_train_calibration_and_val_characterization() -> None:
    config = load_yolo_tensorrt_int8_config(DEFAULT_TENSORRT_INT8_CONFIG)

    assert config.precision == "int8"
    assert config.quantizer.explicit_qdq is True
    assert config.quantizer.version == "0.46.0"
    assert config.quantizer.calibration_method == "entropy"

    assert config.calibration.split == "train"
    assert config.calibration.sample_count == 84
    assert config.calibration.validation_used is False
    assert config.calibration.test_used is False
    assert config.calibration.test_split_used is False

    assert config.characterization.split == "val"
    assert config.characterization.sample_count == 28
    assert config.characterization.numeric_thresholds is None
    assert config.characterization.test_used is False
    assert config.characterization.test_split_used is False


# ADD 2026-09-02: INT8 source ONNX가 accepted C5-1 bytes에서 바뀌면 fail closed한다.
def test_int8_config_rejects_different_source_onnx(tmp_path: Path) -> None:
    raw = DEFAULT_TENSORRT_INT8_CONFIG.read_text(encoding="utf-8").replace(
        EXPECTED_ONNX_SHA256,
        "a" * 64,
    )
    path = tmp_path / "int8.yaml"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="onnx_sha256 changed"):
        load_yolo_tensorrt_int8_config(path)


# ADD 2026-09-02: Calibration에 val을 사용하려는 변경을 거부한다.
def test_int8_config_rejects_validation_calibration(tmp_path: Path) -> None:
    raw = DEFAULT_TENSORRT_INT8_CONFIG.read_text(encoding="utf-8").replace(
        "split: train\n  sample_count: 84",
        "split: val\n  sample_count: 84",
        1,
    )
    path = tmp_path / "int8.yaml"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="all 84 train samples"):
        load_yolo_tensorrt_int8_config(path)


# ADD 2026-09-02: Calibration sample을 임의 subset으로 줄이는 변경을 거부한다.
def test_int8_config_rejects_partial_calibration_subset(tmp_path: Path) -> None:
    raw = DEFAULT_TENSORRT_INT8_CONFIG.read_text(encoding="utf-8").replace(
        "sample_count: 84",
        "sample_count: 64",
        1,
    )
    path = tmp_path / "int8.yaml"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="all 84 train samples"):
        load_yolo_tensorrt_int8_config(path)


# ADD 2026-09-02: Deprecated implicit INT8 calibration path로 되돌리는 변경을 거부한다.
def test_int8_config_rejects_non_qdq_quantization(tmp_path: Path) -> None:
    raw = DEFAULT_TENSORRT_INT8_CONFIG.read_text(encoding="utf-8").replace(
        "explicit_qdq: true",
        "explicit_qdq: false",
    )
    path = tmp_path / "int8.yaml"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="explicit Q/DQ"):
        load_yolo_tensorrt_int8_config(path)


# ADD 2026-09-02: Characterization 전에 numeric acceptance threshold를 넣는 변경을 거부한다.
def test_int8_config_rejects_predeclared_numeric_thresholds(tmp_path: Path) -> None:
    raw = DEFAULT_TENSORRT_INT8_CONFIG.read_text(encoding="utf-8").replace(
        "numeric_thresholds: null",
        "numeric_thresholds:\n    confidence_abs_error_max: 0.02",
    )
    path = tmp_path / "int8.yaml"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="characterize INT8 before defining numeric acceptance"):
        load_yolo_tensorrt_int8_config(path)
