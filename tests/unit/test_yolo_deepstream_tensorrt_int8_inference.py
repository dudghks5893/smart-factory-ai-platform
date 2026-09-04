"""Unit tests for C6-5C DeepStream TensorRT INT8 inference runtime."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from services.streaming.yolo_deepstream_tensorrt_int8 import (
    load_deepstream_tensorrt_config,
)
from services.streaming.yolo_deepstream_tensorrt_int8_inference import (
    CONTAINER_LABEL,
    CONTAINER_RESULT_PREFIX,
    EXPECTED_PLAN_SHA256,
    EXPECTED_REQUIRED_PLUGINS,
    INFERENCE_STATE,
    build_container_source,
    build_docker_command,
    build_nvinfer_config_text,
    load_deepstream_inference_config,
    parse_container_inference_payload,
    validate_container_inference_payload,
    validate_engine_file,
)


# ADD 2026-09-05: Frozen inference config의 plan/source/runtime identity를 검증한다.
def test_load_inference_config() -> None:
    config = load_deepstream_inference_config()

    assert config.inference_id == "c6_5c_deepstream_l4_tensorrt_int8_inference_v1"
    assert config.engine.sha256 == EXPECTED_PLAN_SHA256
    assert config.engine.size_bytes == 4_940_452
    assert config.pipeline.target_frames == 30
    assert config.pipeline.required_plugins == EXPECTED_REQUIRED_PLUGINS


# ADD 2026-09-05: C6-5C와 C6-5D scope가 decode/overlay boundary에서 분리되는지 검증한다.
def test_inference_scope_excludes_c6_5d() -> None:
    config = load_deepstream_inference_config()

    assert config.policy.application_inference is True
    assert config.policy.segmentation_decode_allowed is False
    assert config.policy.overlay_allowed is False
    assert config.policy.final_test_used is False


# ADD 2026-09-05: nvinfer config가 raw tensor meta와 sealed plan을 사용하도록 검증한다.
def test_nvinfer_config_text() -> None:
    config = load_deepstream_inference_config()

    text = build_nvinfer_config_text(config)

    assert "model-engine-file=/model/model.plan" in text
    assert "network-type=100" in text
    assert "output-tensor-meta=1" in text
    assert "maintain-aspect-ratio=1" in text
    assert "symmetric-padding=1" in text


# ADD 2026-09-05: Docker runtime이 no-network/read-only-plan exact digest를 사용하는지 검증한다.
def test_docker_command_contract(
    tmp_path: Path,
) -> None:
    config = load_deepstream_inference_config()
    build_config = load_deepstream_tensorrt_config()

    plan = tmp_path / "model.plan"

    command = build_docker_command(
        config,
        build_config,
        plan_path=plan,
    )

    assert command[:3] == (
        "sudo",
        "docker",
        "run",
    )
    assert "--network" in command
    assert "none" in command
    assert CONTAINER_LABEL in command
    assert build_config.runtime.repo_digest in command

    mount = f"{plan.resolve()}:/model/model.plan:ro"

    assert mount in command


# ADD 2026-09-05: Generated source의 30-frame nvinfer path를 검증한다.
def test_container_source_contract() -> None:
    config = load_deepstream_inference_config()
    build_config = load_deepstream_tensorrt_config()

    source = build_container_source(
        config,
        build_config,
    )

    assert EXPECTED_PLAN_SHA256 in source
    assert "nvstreammux" in source
    assert "nvv4l2decoder" in source
    assert "nvinfer" in source
    assert "eos-after=" in source
    assert "TARGET_FRAMES = 30" in source
    assert "video/x-raw(memory:NVMM)" in source
    assert CONTAINER_RESULT_PREFIX in source


# ADD 2026-09-05: Generated 함수의 typed/revision contract를 검증한다.
def test_generated_container_functions_are_typed() -> None:
    config = load_deepstream_inference_config()
    build_config = load_deepstream_tensorrt_config()

    source = build_container_source(
        config,
        build_config,
    )

    tree = ast.parse(source)
    lines = source.splitlines()

    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
    ]

    assert {node.name for node in functions} == {
        "run_command",
        "hash_file",
        "inspect_plugin",
        "main",
    }

    for node in functions:
        assert node.returns is not None

        arguments = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]

        assert all(argument.annotation is not None for argument in arguments)

        previous = lines[node.lineno - 2].strip()

        assert previous.startswith("# ADD 2026-09-05:")


# ADD 2026-09-05: Synthetic valid inference payload를 unit validation용으로 만든다.
def _valid_payload() -> dict[str, Any]:
    config = load_deepstream_inference_config()
    build_config = load_deepstream_tensorrt_config()

    return {
        "status": "passed",
        "tensorrt_version": (build_config.runtime.tensorrt_python_version),
        "gstreamer_version": config.gstreamer_version,
        "plan_sha256": config.engine.sha256,
        "plan_bytes": config.engine.size_bytes,
        "sample_sha256": config.source.sha256,
        "sample_bytes": config.source.size_bytes,
        "required_plugins": list(config.pipeline.required_plugins),
        "output_tensor_meta_property": True,
        "identity_eos_after_property": True,
        "pipeline_exit_code": 0,
        "eos_observed": True,
        "engine_deserialized": True,
        "engine_model_used": True,
        "nvmm_observed": True,
        "post_inference_eos_after": (config.pipeline.target_frames),
        "target_frames_reached": True,
        "application_inference_executed": True,
        "segmentation_decode_executed": False,
        "overlay_executed": False,
        "dataset_used": False,
        "validation_used": False,
        "test_used": False,
        "final_test_used": False,
    }


# ADD 2026-09-05: Valid payload가 strict inference contract를 통과하는지 검증한다.
def test_validate_container_payload() -> None:
    config = load_deepstream_inference_config()
    build_config = load_deepstream_tensorrt_config()

    payload = _valid_payload()

    validate_container_inference_payload(
        payload,
        config,
        build_config,
    )

    line = CONTAINER_RESULT_PREFIX + __import__("json").dumps(payload)

    parsed = parse_container_inference_payload(line)

    assert parsed["plan_sha256"] == config.engine.sha256


# ADD 2026-09-05: Plan identity 변경 payload를 fail-closed 처리하는지 검증한다.
def test_validate_payload_rejects_plan_change() -> None:
    config = load_deepstream_inference_config()
    build_config = load_deepstream_tensorrt_config()

    payload = _valid_payload()
    payload["plan_sha256"] = "a" * 64

    with pytest.raises(
        ValueError,
        match="result contract changed",
    ):
        validate_container_inference_payload(
            payload,
            config,
            build_config,
        )


# ADD 2026-09-05: EOS가 없는 inference result를 fail-closed 처리하는지 검증한다.
def test_validate_payload_rejects_missing_eos() -> None:
    config = load_deepstream_inference_config()
    build_config = load_deepstream_tensorrt_config()

    payload = _valid_payload()
    payload["eos_observed"] = False

    with pytest.raises(
        ValueError,
        match="result contract changed",
    ):
        validate_container_inference_payload(
            payload,
            config,
            build_config,
        )


# ADD 2026-09-05: Wrong local plan bytes를 canonical execution 전에 거부하는지 검증한다.
def test_validate_engine_file_rejects_change(
    tmp_path: Path,
) -> None:
    config = load_deepstream_inference_config()

    plan = tmp_path / "model.plan"
    plan.write_bytes(b"not-the-canonical-plan")

    with pytest.raises(
        ValueError,
        match="plan identity changed",
    ):
        validate_engine_file(
            plan,
            config,
        )


# ADD 2026-09-05: Canonical inference state 이름을 C6-5C에 고정한다.
def test_inference_state_contract() -> None:
    assert INFERENCE_STATE == "DEEPSTREAM_TENSORRT_INT8_INFERENCE_COMPLETED"
