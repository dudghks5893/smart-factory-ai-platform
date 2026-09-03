"""Synthetic contracts for C5-4B1 ModelOpt INT8 Q/DQ ONNX generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import ml.deployment.yolo_tensorrt_int8_quantization as int8_quantization
from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.deployment.yolo_tensorrt_int8 import (
    DEFAULT_TENSORRT_INT8_CONFIG,
    load_yolo_tensorrt_int8_config,
)
from ml.deployment.yolo_tensorrt_int8_quantization import (
    CALIBRATION_EXECUTION_PROVIDERS,
    Int8CalibrationDataReader,
    QdqGraphContract,
    calibration_sample_ids_sha256,
    inspect_qdq_graph,
    run_modelopt_quantization,
)


# ADD 2026-09-02: Synthetic derived-manifest record를 calibration identity test용으로 만든다.
def _record(index: int) -> DerivedManifestRecord:
    sample_id = f"sample-{index:03d}"
    return DerivedManifestRecord(
        dataset_name="mvtec_ad_metal_nut_yolo_segmentation",
        dataset_version="v1",
        derived_task="yolo_segmentation",
        source_manifest_sha256="a" * 64,
        source_split="test",
        source_manifest_split="test",
        source_image_path=f"source/{sample_id}.png",
        source_mask_path="",
        category="metal_nut",
        sample_id=sample_id,
        defect_type="good",
        target_class="",
        target_class_id="",
        derived_split="train",
        is_negative=True,
        image_width=700,
        image_height=700,
        image_path=f"images/train/{sample_id}.png",
        label_path=f"labels/train/{sample_id}.txt",
        image_sha256="b" * 64,
        mask_sha256="",
        polygon_count=0,
        component_count=0,
        hole_count=0,
        polygon_vertex_count=0,
        round_trip_iou="",
        pixel_precision="",
        pixel_recall="",
    )


# ADD 2026-09-02: Calibration sample ID digest가 input 순서에 민감하고 deterministic한지 검증한다.
def test_calibration_sample_ids_digest_is_deterministic() -> None:
    records = tuple(_record(index) for index in range(3))
    first = calibration_sample_ids_sha256(records)
    second = calibration_sample_ids_sha256(records)
    reversed_digest = calibration_sample_ids_sha256(tuple(reversed(records)))

    assert first == second
    assert first != reversed_digest
    assert len(first) == 64


# ADD 2026-09-03: ModelOpt 0.46.0용 get_first가 iterator를 소비하지 않는지 검증한다.
def test_calibration_reader_get_first_is_non_consuming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_yolo_tensorrt_int8_config(DEFAULT_TENSORRT_INT8_CONFIG)
    records = (_record(0), _record(1))
    observed: list[str] = []

    def fake_preprocess(image_path: Path, *, imgsz: int) -> np.ndarray:
        observed.append(image_path.name)
        assert imgsz == 640
        return np.zeros((1, 3, 640, 640), dtype=np.float32)

    monkeypatch.setattr(
        int8_quantization,
        "preprocess_calibration_image",
        fake_preprocess,
    )
    reader = Int8CalibrationDataReader(
        dataset_root=Path("/dataset"),
        records=records,
        config=config,
    )

    first = reader.get_first()
    first_again = reader.get_first()
    next_first = reader.get_next()
    next_second = reader.get_next()

    assert first[config.calibration.input_name].shape == (1, 3, 640, 640)
    assert first_again[config.calibration.input_name].shape == (1, 3, 640, 640)
    assert next_first is not None
    assert next_second is not None
    assert observed == [
        "sample-000.png",
        "sample-000.png",
        "sample-000.png",
        "sample-001.png",
    ]


# ADD 2026-09-02: ModelOpt 호출 kwargs가 frozen INT8 PTQ contract와 일치하는지 검증한다.
def test_modelopt_quantization_uses_frozen_kwargs(tmp_path: Path) -> None:
    config = load_yolo_tensorrt_int8_config(DEFAULT_TENSORRT_INT8_CONFIG)
    source = tmp_path / "source.onnx"
    source.write_bytes(b"source")
    output = tmp_path / "output.onnx"
    calls: dict[str, Any] = {}

    class Reader:
        pass

    reader = Reader()

    def fake_quantizer(onnx_path: str, **kwargs: Any) -> None:
        calls["onnx_path"] = onnx_path
        calls.update(kwargs)
        Path(str(kwargs["output_path"])).write_bytes(b"quantized")

    run_modelopt_quantization(
        source_onnx_path=source,
        output_onnx_path=output,
        calibration_reader=reader,  # type: ignore[arg-type]
        config=config,
        quantizer=fake_quantizer,
    )

    assert calls["onnx_path"] == str(source)
    assert calls["quantize_mode"] == "int8"
    assert calls["calibration_method"] == "entropy"
    assert calls["calibration_eps"] == list(CALIBRATION_EXECUTION_PROVIDERS)
    assert calls["high_precision_dtype"] == "fp16"
    assert calls["simplify"] is False
    assert calls["calibration_data_reader"] is reader
    assert output.read_bytes() == b"quantized"


# ADD 2026-09-02: Synthetic source ONNX graph를 static segmentation I/O contract로 만든다.
def _source_onnx(path: Path) -> None:
    input_info = helper.make_tensor_value_info(
        "images",
        TensorProto.FLOAT,
        [1, 3, 640, 640],
    )
    output_info = helper.make_tensor_value_info(
        "output0",
        TensorProto.FLOAT,
        [1, 39, 8400],
    )
    node = helper.make_node("Identity", ["images"], ["output0"], name="identity")
    graph = helper.make_graph([node], "source", [input_info], [output_info])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.save(model, path)


# ADD 2026-09-02: Synthetic Q/DQ ONNX graph를 explicit quantization structure로 만든다.
def _qdq_onnx(path: Path) -> None:
    input_info = helper.make_tensor_value_info(
        "images",
        TensorProto.FLOAT,
        [1, 3, 640, 640],
    )
    output_info = helper.make_tensor_value_info(
        "output0",
        TensorProto.FLOAT,
        [1, 39, 8400],
    )
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [], [0.1])
    zero = helper.make_tensor("zero", TensorProto.INT8, [], [0])
    nodes = [
        helper.make_node("QuantizeLinear", ["images", "scale", "zero"], ["q"]),
        helper.make_node("DequantizeLinear", ["q", "scale", "zero"], ["dq"]),
        helper.make_node("Identity", ["dq"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes,
        "qdq",
        [input_info],
        [output_info],
        initializer=[scale, zero],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.save(model, path)


# ADD 2026-09-02: Q/DQ inspector가 explicit QuantizeLinear/DequantizeLinear node를 관측한다.
def test_qdq_graph_inspection_and_source_identity(tmp_path: Path) -> None:
    source_path = tmp_path / "source.onnx"
    candidate_path = tmp_path / "candidate.onnx"
    _source_onnx(source_path)
    _qdq_onnx(candidate_path)

    source = inspect_qdq_graph(source_path)
    candidate = inspect_qdq_graph(candidate_path)
    candidate.validate_against(source)

    assert candidate.quantize_linear_count == 1
    assert candidate.dequantize_linear_count == 1
    assert candidate.input_shapes == ((1, 3, 640, 640),)


# ADD 2026-09-02: Q/DQ node가 없는 candidate는 explicit quantization contract를 통과하지 못한다.
def test_qdq_graph_rejects_candidate_without_qdq() -> None:
    source = QdqGraphContract(
        quantize_linear_count=0,
        dequantize_linear_count=0,
        input_names=("images",),
        input_shapes=((1, 3, 640, 640),),
        output_names=("output0",),
        output_shapes=((1, 39, 8400),),
    )
    candidate = source

    with pytest.raises(ValueError, match="explicit Q/DQ"):
        candidate.validate_against(source)
