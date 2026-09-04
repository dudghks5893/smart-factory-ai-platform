"""Unit tests for C6-5C DeepStream/L4 TensorRT INT8 contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from services.streaming.yolo_deepstream_tensorrt_int8 import (
    EXPECTED_OUTPUT_ROOT,
    EXPECTED_QDQ_OPSET,
    EXPECTED_REPO_DIGEST,
    EXPECTED_SOURCE_ARCHIVE_SHA256,
    EXPECTED_TENSORRT_VERSION,
    expected_engine_io,
    load_deepstream_tensorrt_config,
    parse_container_build_payload,
    validate_container_build_payload,
)


# ADD 2026-09-05: Frozen config의 exact C5 Q/DQ lineage와 DeepStream/L4 runtime을 검증한다.
def test_load_deepstream_tensorrt_config() -> None:
    config = load_deepstream_tensorrt_config()

    assert config.build_id == "c6_5c_deepstream_l4_yolo11n_seg_int8_qdq_v1"
    assert config.source.evidence_zip_sha256 == EXPECTED_SOURCE_ARCHIVE_SHA256
    assert config.source.qdq_opset == EXPECTED_QDQ_OPSET
    assert config.runtime.repo_digest == EXPECTED_REPO_DIGEST
    assert config.runtime.tensorrt_python_version == EXPECTED_TENSORRT_VERSION
    assert config.build.output_root == EXPECTED_OUTPUT_ROOT
    assert config.build.plan_filename == "model.plan"
    assert config.policy.final_test_used is False


# ADD 2026-09-05: Characterization plan SHA를 canonical config에 고정하지 않는지 검증한다.
def test_config_does_not_freeze_characterization_plan_sha() -> None:
    text = Path("configs/streaming/yolo_deepstream_tensorrt_int8.json").read_text(encoding="utf-8")

    assert "baa935b62708854f69dbb4e5c0c23af595d2f2704a3c08949c614f7c886c694c" not in text
    assert "ephemeral_plan_sha256" not in text


# ADD 2026-09-05: TensorRT error-level tactic skip를 automatic failure로 취급하지 않는지 검증한다.
def test_diagnostics_policy_uses_structural_failure_gates() -> None:
    config = load_deepstream_tensorrt_config()

    assert config.diagnostics.record_tensorrt_stderr is True
    assert config.diagnostics.error_line_is_automatic_failure is False
    assert config.diagnostics.fatal_conditions == (
        "parser_failure",
        "serialized_plan_build_failure",
        "deserialize_failure",
        "engine_io_contract_mismatch",
        "container_nonzero_exit",
    )


# ADD 2026-09-05: Explicit-Q/DQ build가 precision flag/calibrator 없이 strongly typed인지 검증한다.
def test_build_policy_is_strongly_typed_explicit_qdq() -> None:
    config = load_deepstream_tensorrt_config()

    assert config.build.strongly_typed is True
    assert config.build.explicit_quantization is True
    assert config.build.builder_int8_flag is False
    assert config.build.builder_fp16_flag is False
    assert config.build.legacy_calibrator is False
    assert config.build.workspace_bytes == 4_294_967_296
    assert config.build.persist_raw_plan is True


# ADD 2026-09-05: L4 deployment plan이 기존 C5 engine namespace와 분리되는지 검증한다.
def test_output_namespace_is_separate_from_c5_engine() -> None:
    config = load_deepstream_tensorrt_config()

    assert "deepstream_l4" in str(config.build.output_root)
    assert config.build.output_root == EXPECTED_OUTPUT_ROOT
    assert config.build.output_root != Path(
        "artifacts/deployment/yolo_segmentation/tensorrt_int8/engine"
    )


# ADD 2026-09-05: Frozen external TensorRT engine I/O를 exact name/mode/dtype/shape로 검증한다.
def test_expected_engine_io() -> None:
    assert expected_engine_io() == {
        "images": {
            "mode": "INPUT",
            "dtype": "FLOAT",
            "shape": (1, 3, 640, 640),
        },
        "output0": {
            "mode": "OUTPUT",
            "dtype": "FLOAT",
            "shape": (1, 39, 8400),
        },
        "output1": {
            "mode": "OUTPUT",
            "dtype": "FLOAT",
            "shape": (1, 32, 160, 160),
        },
    }


# ADD 2026-09-05: Successful synthetic TensorRT build payload를 구성한다.
def _valid_payload() -> dict[str, object]:
    config = load_deepstream_tensorrt_config()

    io_tensors = [
        {
            "name": name,
            "mode": contract["mode"],
            "dtype": contract["dtype"],
            "shape": list(cast(tuple[int, ...], contract["shape"])),
        }
        for name, contract in expected_engine_io().items()
    ]

    return {
        "status": "passed",
        "tensorrt_version": config.runtime.tensorrt_python_version,
        "source_qdq_onnx_sha256": config.source.files["model.int8.qdq.onnx"].sha256,
        "source_qdq_onnx_bytes": config.source.files["model.int8.qdq.onnx"].size_bytes,
        "parser_success": True,
        "parser_error_count": 0,
        "strongly_typed": True,
        "workspace_bytes": config.build.workspace_bytes,
        "build_succeeded": True,
        "deserialize_succeeded": True,
        "plan_filename": config.build.plan_filename,
        "plan_sha256": "a" * 64,
        "plan_bytes": 4_900_000,
        "io_tensors": io_tensors,
        "application_inference_executed": False,
        "dataset_used": False,
        "validation_used": False,
        "test_used": False,
        "final_test_used": False,
    }


# ADD 2026-09-05: Single result marker JSON을 parse하고 build-result contract를 검증한다.
def test_parse_and_validate_container_build_payload() -> None:
    config = load_deepstream_tensorrt_config()
    payload = _valid_payload()

    stdout = "diagnostic output\nC6_5C_BUILD_RESULT=" + json.dumps(payload) + "\n"

    parsed = parse_container_build_payload(stdout)
    validate_container_build_payload(parsed, config)


# ADD 2026-09-05: Parser failure payload를 structural failure로 fail-closed 처리한다.
def test_validate_container_build_payload_rejects_parser_failure() -> None:
    config = load_deepstream_tensorrt_config()
    payload = _valid_payload()
    payload["parser_success"] = False

    with pytest.raises(
        ValueError,
        match="structural contract failed",
    ):
        validate_container_build_payload(payload, config)


# ADD 2026-09-05: External engine I/O shape 변경을 fail-closed 처리한다.
def test_validate_container_build_payload_rejects_io_change() -> None:
    config = load_deepstream_tensorrt_config()
    payload = _valid_payload()

    io_tensors = payload["io_tensors"]
    assert isinstance(io_tensors, list)

    first = io_tensors[0]
    assert isinstance(first, dict)

    first["shape"] = [1, 3, 320, 320]

    with pytest.raises(
        ValueError,
        match="engine I/O contract changed",
    ):
        validate_container_build_payload(payload, config)


# ADD 2026-09-05: C5 accepted engine rebuild 허용으로 scope가 변경되는 것을 거부한다.
def test_scope_rejects_c5_engine_rebuild() -> None:
    config = load_deepstream_tensorrt_config()

    changed = replace(
        config,
        policy=replace(
            config.policy,
            c5_engine_rebuild_allowed=True,
        ),
    )

    with pytest.raises(
        ValueError,
        match="scope policy changed",
    ):
        changed.validate()
