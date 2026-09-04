"""Unit tests for the C6-5C DeepStream/L4 TensorRT INT8 build orchestrator."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

import pytest

from services.streaming.yolo_deepstream_tensorrt_int8 import (
    expected_engine_io,
    load_deepstream_tensorrt_config,
)
from services.streaming.yolo_deepstream_tensorrt_int8_build import (
    BUILD_METADATA_FILENAME,
    BUILD_STATE,
    BUILD_TIMEOUT_SECONDS,
    CONTAINER_LABEL,
    CONTAINER_RESULT_PREFIX,
    DockerImageIdentity,
    HostGpuIdentity,
    RepositoryIdentity,
    build_container_source,
    build_docker_command,
    build_metadata,
    validate_plan_file,
)
from shared.hashing import sha256_bytes


# ADD 2026-09-05: Generated TensorRT source가 frozen strongly-typed build만 수행하는지 검증한다.
def test_build_container_source_contract() -> None:
    config = load_deepstream_tensorrt_config()

    source = build_container_source(config)

    assert "NetworkDefinitionCreationFlag.STRONGLY_TYPED" in source
    assert "build_serialized_network" in source
    assert f"OUTPUT_FILENAME = {config.build.plan_filename!r}" in source
    assert "BuilderFlag.INT8" not in source
    assert "BuilderFlag.FP16" not in source
    assert "IInt8Calibrator" not in source
    assert "execute_async" not in source
    assert "execute_v2" not in source
    assert CONTAINER_RESULT_PREFIX in source


# ADD 2026-09-05: Generated container 함수의 typed/revision-comment contract를 검증한다.
def test_generated_container_functions_are_typed() -> None:
    config = load_deepstream_tensorrt_config()

    source = build_container_source(config)

    tree = ast.parse(source)

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
        "enum_name",
        "tensor_record",
        "main",
    }

    lines = source.splitlines()

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


# ADD 2026-09-05: Docker builder가 exact digest/no-network/GPU/temp-volume boundary를 유지한다.
def test_build_docker_command_contract() -> None:
    config = load_deepstream_tensorrt_config()

    command = build_docker_command(
        config,
        input_dir=Path("/tmp/c6-5c-input"),
        output_dir=Path("/tmp/c6-5c-output"),
    )

    assert command[:3] == (
        "sudo",
        "docker",
        "run",
    )
    assert "--rm" in command
    assert "--interactive" in command
    assert "--runtime=nvidia" in command
    assert "--network" in command
    assert "none" in command
    assert "--gpus" in command
    assert "all" in command
    assert CONTAINER_LABEL in command
    assert config.runtime.repo_digest in command
    assert command[-1] == "-"


# ADD 2026-09-05: Generated raw plan이 container-reported hash/bytes와 일치하는지 검증한다.
def test_validate_plan_file(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "model.plan"

    data = b"synthetic-c6-5c-plan"
    plan.write_bytes(data)

    payload: dict[str, object] = {
        "plan_sha256": sha256_bytes(data),
        "plan_bytes": len(data),
    }

    validate_plan_file(
        plan,
        payload,
    )


# ADD 2026-09-05: Plan hash mismatch를 canonical publication 전에 fail-closed 처리한다.
def test_validate_plan_file_rejects_hash_change(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "model.plan"
    plan.write_bytes(b"changed")

    payload: dict[str, object] = {
        "plan_sha256": "a" * 64,
        "plan_bytes": 7,
    }

    with pytest.raises(
        ValueError,
        match="does not match payload",
    ):
        validate_plan_file(
            plan,
            payload,
        )


# ADD 2026-09-05: Synthetic successful build payload를 metadata test용으로 만든다.
def _valid_payload() -> dict[str, object]:
    config = load_deepstream_tensorrt_config()

    io_tensors = [
        {
            "name": name,
            "mode": contract["mode"],
            "dtype": contract["dtype"],
            "shape": list(
                cast(
                    tuple[int, ...],
                    contract["shape"],
                )
            ),
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


# ADD 2026-09-05: Canonical metadata가 source/runtime/log/scope provenance를 보존하는지 검증한다.
def test_build_metadata_contract() -> None:
    config = load_deepstream_tensorrt_config()

    repository = RepositoryIdentity(
        git_commit="f" * 40,
        working_tree_dirty=False,
    )

    image = DockerImageIdentity(
        image_id=config.runtime.image_id,
        repo_digests=(config.runtime.repo_digest,),
    )

    gpu = HostGpuIdentity(
        gpu_name=config.runtime.gpu_name,
        driver_version=config.runtime.driver_version,
        compute_capability=config.runtime.gpu_compute_capability,
    )

    source_members = {
        filename: {
            "sha256": identity.sha256,
            "size_bytes": identity.size_bytes,
        }
        for filename, identity in config.source.files.items()
    }

    stdout = "synthetic stdout"
    stderr = "[TRT] [E] tactic skipped"

    metadata = build_metadata(
        config=config,
        repository=repository,
        image=image,
        gpu=gpu,
        source_members=source_members,
        payload=_valid_payload(),
        container_stdout=stdout,
        container_stderr=stderr,
    )

    assert metadata["state"] == BUILD_STATE
    assert metadata["stage"] == "C6-5C"

    artifact = metadata["artifact"]
    assert isinstance(
        artifact,
        dict,
    )

    assert artifact["metadata_filename"] == BUILD_METADATA_FILENAME

    diagnostics = metadata["diagnostics"]
    assert isinstance(
        diagnostics,
        dict,
    )

    assert diagnostics["container_stderr_sha256"] == sha256_bytes(stderr.encode("utf-8"))

    assert diagnostics["error_line_is_automatic_failure"] is False

    scope = metadata["scope"]
    assert isinstance(
        scope,
        dict,
    )

    assert scope["application_inference_executed"] is False
    assert scope["final_test_used"] is False
    assert scope["c5_engine_rebuild_allowed"] is False


# ADD 2026-09-05: Repository identity가 dirty canonical state를 거부하는지 검증한다.
def test_repository_identity_rejects_dirty_state() -> None:
    identity = RepositoryIdentity(
        git_commit="f" * 40,
        working_tree_dirty=True,
    )

    with pytest.raises(
        ValueError,
        match="clean repository",
    ):
        identity.validate()


# ADD 2026-09-05: Docker/GPU observations가 frozen runtime 변경을 fail-closed 처리하는지 검증한다.
def test_runtime_identity_rejects_change() -> None:
    config = load_deepstream_tensorrt_config()

    image = DockerImageIdentity(
        image_id="sha256:" + "0" * 64,
        repo_digests=(config.runtime.repo_digest,),
    )

    with pytest.raises(
        ValueError,
        match="image ID changed",
    ):
        image.validate(config)

    gpu = HostGpuIdentity(
        gpu_name="Different GPU",
        driver_version=config.runtime.driver_version,
        compute_capability=config.runtime.gpu_compute_capability,
    )

    with pytest.raises(
        ValueError,
        match="GPU runtime identity changed",
    ):
        gpu.validate(config)


# ADD 2026-09-05: Canonical build timeout을 충분한 bounded value로 고정한다.
def test_build_timeout_contract() -> None:
    assert BUILD_TIMEOUT_SECONDS == 300
