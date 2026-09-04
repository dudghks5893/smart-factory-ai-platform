"""Unit tests for C6-5A DeepStream TensorRT compatibility contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.streaming.yolo_deepstream_compatibility import (
    COMPATIBLE_REASON,
    DEFAULT_DEEPSTREAM_COMPATIBILITY_CONFIG,
    EXPECTED_REPO_DIGEST,
    VERSION_INCOMPATIBLE_REASON,
    build_docker_probe_command,
    classify_compatibility_reason,
    expected_engine_header,
    load_deepstream_compatibility_config,
    parse_container_probe_payload,
    read_engine_container,
)


# ADD 2026-09-04: Frozen JSON config가 exact C6-5A source/runtime policy를 로드하는지 검증한다.
def test_load_deepstream_compatibility_config() -> None:
    config = load_deepstream_compatibility_config()

    assert config.probe_id == "c6_5a_c5_engine_deepstream_compatibility_v1"
    assert config.runtime.repo_digest == EXPECTED_REPO_DIGEST
    assert config.source_bundle.rebuild_allowed is False
    assert config.policy.mode == "deserialize_only"
    assert config.policy.inference_allowed is False
    assert config.policy.final_test_used is False


# ADD 2026-09-04: Synthetic C5 wrapper에서 JSON header와 TensorRT plan을 분리한다.
def test_read_engine_container(
    tmp_path: Path,
) -> None:
    header = expected_engine_header()
    header_bytes = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    expected_size = 7_607_228
    serialized = b"x" * expected_size

    path = tmp_path / "model.engine"
    path.write_bytes(
        len(header_bytes).to_bytes(
            4,
            byteorder="little",
            signed=True,
        )
        + header_bytes
        + serialized
    )

    observation = read_engine_container(path)

    assert observation.header == header
    assert observation.header_length == 155
    assert len(observation.serialized_engine) == expected_size


# ADD 2026-09-04: TensorRT version-plan mismatch logger를 stable reason으로 분류한다.
def test_classify_tensorrt_version_incompatibility() -> None:
    diagnostic = (
        "The engine plan file is not compatible "
        "with this version of TensorRT. "
        "In checkEngineVersionCompatible"
    )

    result = classify_compatibility_reason(
        status="incompatible",
        diagnostic_text=diagnostic,
    )

    assert result == VERSION_INCOMPATIBLE_REASON


# ADD 2026-09-04: Successful TensorRT deserialize를 compatible reason으로 분류한다.
def test_classify_compatible_deserialization() -> None:
    result = classify_compatibility_reason(
        status="compatible",
        diagnostic_text="",
    )

    assert result == COMPATIBLE_REASON


# ADD 2026-09-04: Invalid compatibility status는 fail-closed 처리하는지 검증한다.
def test_classify_invalid_status_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="compatibility status is invalid",
    ):
        classify_compatibility_reason(
            status="unknown",
            diagnostic_text="",
        )


# ADD 2026-09-04: Docker probe의 격리 및 exact artifact mount 계약을 검증한다.
def test_build_docker_probe_command() -> None:
    config = load_deepstream_compatibility_config(DEFAULT_DEEPSTREAM_COMPATIBILITY_CONFIG)
    input_dir = Path("/tmp/c6-5-inputs")

    command = build_docker_probe_command(
        input_dir=input_dir,
        config=config,
    )

    assert "--rm" in command
    assert "--interactive" in command
    assert "--network" in command
    assert "none" in command
    assert config.runtime.repo_digest in command
    assert f"{input_dir.resolve()}:/c6-5-inputs:ro" in command
    assert command[-1] == "-"


# ADD 2026-09-04: Marked container result JSON이 strict mapping으로 복원되는지 검증한다.
def test_parse_container_probe_payload() -> None:
    stdout = 'DeepStream startup text\nC6_5A_RESULT_JSON={"status":"incompatible","value":1}\n'

    result = parse_container_probe_payload(stdout)

    assert result == {
        "status": "incompatible",
        "value": 1,
    }


# ADD 2026-09-04: Marker가 없는 container stdout은 evidence로 허용하지 않는다.
def test_parse_container_probe_payload_requires_marker() -> None:
    with pytest.raises(
        RuntimeError,
        match="did not emit its result marker",
    ):
        parse_container_probe_payload("ordinary output only\n")
