"""Unit tests for C6-5B DeepStream NVDEC/NVMM smoke contracts."""

from __future__ import annotations

import ast
import subprocess
from copy import deepcopy
from dataclasses import replace

import pytest

from services.streaming import (
    yolo_deepstream_nvmm_smoke as nvmm_module,
)
from services.streaming.yolo_deepstream_nvmm_smoke import (
    CONTAINER_RESULT_PREFIX,
    EXPECTED_REPO_DIGEST,
    EXPECTED_SAMPLE_SHA256,
    build_container_probe_source,
    build_docker_smoke_command,
    expected_gstreamer_pipeline,
    load_deepstream_nvmm_smoke_config,
    parse_container_smoke_payload,
    validate_container_smoke_payload,
)
from shared.hashing import sha256_bytes


# ADD 2026-09-05: Frozen config가 exact C6-5B runtime/sample/CDI contract를 로드하는지 검증한다.
# MODIFY 2026-09-05: CDI generated-file SHA 대신 stable path/required entry contract를 검증한다.
def test_load_deepstream_nvmm_smoke_config() -> None:
    config = load_deepstream_nvmm_smoke_config()

    assert config.smoke_id == "c6_5b_deepstream_nvdec_nvmm_v1"
    assert config.runtime.repo_digest == EXPECTED_REPO_DIGEST
    assert config.sample.sha256 == EXPECTED_SAMPLE_SHA256
    assert str(config.host_dependency.cdi_spec_path) == "/var/run/cdi/nvidia.yaml"
    assert config.host_dependency.cdi_required_entry == "libnvcuvid.so.595.84"
    assert not hasattr(config.host_dependency, "cdi_spec_sha256")
    assert config.host_dependency.cdi_refresh_required is True
    assert config.policy.inference_allowed is False
    assert config.policy.final_test_used is False


# ADD 2026-09-05: Exact gst-launch command가 NVDEC→NVMM→RGBA path를 유지하는지 검증한다.
def test_expected_gstreamer_pipeline() -> None:
    config = load_deepstream_nvmm_smoke_config()

    command = expected_gstreamer_pipeline(config)

    assert command[0:3] == ("gst-launch-1.0", "-e", "-v")
    assert "nvv4l2decoder" in command
    assert "video/x-raw(memory:NVMM)" in command
    assert "nvvideoconvert" in command
    assert "video/x-raw(memory:NVMM),format=RGBA" in command
    assert command[-3:] == ("fakesink", "sync=false", "async=false")


# ADD 2026-09-05: Generated container 함수가 revision과 typed contract를 유지하는지 검증한다.
def test_generated_container_functions_are_typed() -> None:
    config = load_deepstream_nvmm_smoke_config()
    source = build_container_probe_source(config)
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
        "run",
        "inspect_field",
        "hash_file",
        "emit",
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


# ADD 2026-09-05: Docker smoke가 exact digest, no-network, video capability와 stdin을 사용한다.
def test_build_docker_smoke_command() -> None:
    config = load_deepstream_nvmm_smoke_config()

    command = build_docker_smoke_command(config)

    assert "--rm" in command
    assert "--runtime=nvidia" in command
    assert "--network" in command
    assert "none" in command
    assert "--gpus" in command
    assert "all" in command
    assert "--interactive" in command
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video" in command
    assert config.runtime.repo_digest in command
    assert command[-1] == "-"


# ADD 2026-09-05: Container source의 exact sample/no-inference contract를 검증한다.
# MODIFY 2026-09-05: TensorRT policy key와 실제 TensorRT import를 구분해 검증한다.
def test_build_container_probe_source() -> None:
    config = load_deepstream_nvmm_smoke_config()

    source = build_container_probe_source(config)

    assert config.sample.path in source
    assert config.sample.sha256 in source
    assert "libnvcuvid.so.1" in source
    assert "nvv4l2decoder" in source
    assert "nvvideoconvert" in source
    assert '"tensorrt_engine_used": false' in source.lower()
    assert "model.engine" not in source

    tree = ast.parse(source)
    imported_modules: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")

    assert not any(name == "tensorrt" or name.startswith("tensorrt.") for name in imported_modules)


# ADD 2026-09-05: Successful synthetic container payload를 unit validation용으로 만든다.
def _valid_payload() -> dict[str, object]:
    config = load_deepstream_nvmm_smoke_config()
    log = "synthetic unit-test log"

    return {
        "status": "passed",
        "runtime": {
            "gpu_name": config.runtime.gpu_name,
            "driver_version": config.runtime.driver_version,
            "gpu_compute_capability": config.runtime.gpu_compute_capability,
            "deepstream_version": config.runtime.deepstream_version,
            "gstreamer_version": config.runtime.gstreamer_version,
            "driver_capabilities": "compute,utility,video",
            "visible_devices": "void",
        },
        "sample": {
            "path": config.sample.path,
            "sha256": config.sample.sha256,
            "size_bytes": config.sample.size_bytes,
        },
        "plugins": {
            "nvv4l2decoder": {
                "version": config.decoder_plugin.version,
                "filename": config.decoder_plugin.filename,
            },
            "nvvideoconvert": {
                "version": config.converter_plugin.version,
                "filename": config.converter_plugin.filename,
            },
        },
        "nvdec": {
            "dynamic_load": True,
        },
        "pipeline": {
            "command": list(expected_gstreamer_pipeline(config)),
            "exit_code": 0,
            "eos": True,
            "decoder_nvmm_caps": True,
            "decoder_format": "NV12",
            "converter_nvmm_caps": True,
            "converter_format": "RGBA",
            "fatal_patterns": [],
            "decoder_caps_lines": ["nvv4l2decoder GstPad:src video/x-raw(memory:NVMM) NV12"],
            "converter_caps_lines": ["nvvideoconvert GstPad:src video/x-raw(memory:NVMM) RGBA"],
            "log_sha256": sha256_bytes(log.encode("utf-8")),
            "log": log,
        },
        "scope": {
            "inference_executed": False,
            "tensorrt_engine_used": False,
            "dataset_used": False,
            "validation_used": False,
            "test_used": False,
            "final_test_used": False,
        },
    }


# ADD 2026-09-05: Frozen decoder plugin identity 변경을 fail-closed 처리하는지 검증한다.
def test_config_rejects_decoder_plugin_identity_change() -> None:
    config = load_deepstream_nvmm_smoke_config()

    changed = replace(
        config,
        decoder_plugin=replace(
            config.decoder_plugin,
            version="9.9.9",
        ),
    )

    with pytest.raises(
        ValueError,
        match="decoder plugin identity changed",
    ):
        changed.validate()


# ADD 2026-09-05: Labeled C6-5B container가 발견되면 강제 제거 command를 실행하는지 검증한다.
def test_cleanup_deepstream_nvmm_containers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    responses = iter(
        (
            subprocess.CompletedProcess(
                args=(),
                returncode=0,
                stdout="container-a\ncontainer-b\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=(),
                returncode=0,
                stdout="container-a\ncontainer-b\n",
                stderr="",
            ),
        )
    )

    # ADD 2026-09-05: Cleanup unit test에서 Docker subprocess 응답을 deterministic하게 대체한다.
    def fake_run(
        command: tuple[str, ...],
        **_: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return next(responses)

    monkeypatch.setattr(
        nvmm_module.subprocess,
        "run",
        fake_run,
    )

    nvmm_module.cleanup_deepstream_nvmm_containers()

    assert calls == [
        (
            "sudo",
            "docker",
            "ps",
            "-aq",
            "--filter",
            "label=c6_5b_smoke=1",
        ),
        (
            "sudo",
            "docker",
            "rm",
            "-f",
            "container-a",
            "container-b",
        ),
    ]


# ADD 2026-09-05: Marked result JSON이 strict mapping으로 복원되는지 검증한다.
def test_parse_container_smoke_payload() -> None:
    stdout = CONTAINER_RESULT_PREFIX + '{"status":"passed","value":1}\n'

    result = parse_container_smoke_payload(stdout)

    assert result == {
        "status": "passed",
        "value": 1,
    }


# ADD 2026-09-05: Valid NVDEC/NVMM payload가 frozen acceptance contract를 통과하는지 검증한다.
def test_validate_container_smoke_payload() -> None:
    config = load_deepstream_nvmm_smoke_config()

    validate_container_smoke_payload(
        _valid_payload(),
        config,
    )


# ADD 2026-09-05: Evidence log와 SHA-256 불일치를 fail-closed 처리하는지 검증한다.
def test_validate_container_smoke_payload_rejects_log_hash_mismatch() -> None:
    config = load_deepstream_nvmm_smoke_config()
    payload = deepcopy(_valid_payload())

    pipeline = payload["pipeline"]
    assert isinstance(pipeline, dict)
    pipeline["log"] = "tampered log"

    with pytest.raises(
        ValueError,
        match="does not match log content",
    ):
        validate_container_smoke_payload(
            payload,
            config,
        )


# ADD 2026-09-05: SIGSEGV/driver fatal diagnostics가 canonical payload로 허용되지 않는지 검증한다.
def test_validate_container_smoke_payload_rejects_fatal() -> None:
    config = load_deepstream_nvmm_smoke_config()
    payload = deepcopy(_valid_payload())

    pipeline = payload["pipeline"]
    assert isinstance(pipeline, dict)
    pipeline["fatal_patterns"] = ["Caught SIGSEGV"]

    with pytest.raises(
        ValueError,
        match="fatal runtime diagnostics",
    ):
        validate_container_smoke_payload(
            payload,
            config,
        )


# ADD 2026-09-05: NVMM converter contract가 사라진 payload를 fail-closed 처리하는지 검증한다.
def test_validate_container_smoke_payload_rejects_missing_nvmm() -> None:
    config = load_deepstream_nvmm_smoke_config()
    payload = deepcopy(_valid_payload())

    pipeline = payload["pipeline"]
    assert isinstance(pipeline, dict)
    pipeline["converter_nvmm_caps"] = False

    with pytest.raises(
        ValueError,
        match="converter did not preserve NVMM",
    ):
        validate_container_smoke_payload(
            payload,
            config,
        )
