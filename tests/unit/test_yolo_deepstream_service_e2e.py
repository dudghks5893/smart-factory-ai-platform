"""Unit contracts for the C6-6 DeepStream service E2E stdout bridge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.streaming.yolo_deepstream_segmentation import (
    EXPECTED_DECODER_ID,
    load_deepstream_segmentation_config,
)
from services.streaming.yolo_deepstream_service_e2e import (
    EXPECTED_CONTAINER_LABEL,
    EXPECTED_E2E_ADAPTER_ID,
    EXPECTED_FRAME_PREFIX,
    EXPECTED_IMAGE_HEIGHT,
    EXPECTED_IMAGE_WIDTH,
    EXPECTED_RESULT_PREFIX,
    build_service_e2e_docker_command,
    build_service_e2e_probe_source,
    build_service_e2e_shell_payload,
    parse_service_e2e_frame_line,
    parse_service_e2e_result,
    validate_service_e2e_result,
)
from services.streaming.yolo_deepstream_service_integration import (
    load_service_integration_config,
)

SEGMENTATION_CONFIG = Path("configs/streaming/yolo_deepstream_segmentation.json")
SERVICE_CONFIG = Path("configs/streaming/yolo_deepstream_service_integration.json")


def _frame_line() -> str:
    return EXPECTED_FRAME_PREFIX + json.dumps(
        {
            "frame_number": 7,
            "pts_ns": 233_333_333,
            "image_width": EXPECTED_IMAGE_WIDTH,
            "image_height": EXPECTED_IMAGE_HEIGHT,
            "instances": [
                {
                    "class_id": 2,
                    "confidence": 0.91,
                    "left": 10.0,
                    "top": 20.0,
                    "width": 100.0,
                    "height": 80.0,
                    "mask_width": 25,
                    "mask_height": 20,
                    "mask_positive_sample_count": 250,
                }
            ],
        },
        separators=(",", ":"),
    )


def _result_line() -> str:
    return EXPECTED_RESULT_PREFIX + json.dumps(
        {
            "status": "passed",
            "frames": 30,
            "events": 12,
            "instances": 19,
            "eos_observed": True,
            "batch_meta_observed": True,
            "target_frames_reached": True,
            "events_observed": True,
            "instances_observed": True,
            "metadata_valid": True,
            "invalid_reason": "",
            "deepstream_gpu_runtime_executed": True,
            "segmentation_decode_executed": True,
            "compact_metadata_emitted": True,
            "network_used": False,
            "raw_frame_emitted": False,
            "raw_mask_emitted": False,
            "dataset_used": False,
            "validation_used": False,
            "test_used": False,
            "final_test_used": False,
        },
        separators=(",", ":"),
    )


# ADD 2026-09-07: actual metadata 검증 → MODIFY 2026-09-07: mask grid summary 검증
def test_probe_source_uses_actual_deepstream_metadata_without_raw_media() -> None:
    config = load_deepstream_segmentation_config(SEGMENTATION_CONFIG)
    source = build_service_e2e_probe_source(config)

    assert config.decoder_id == EXPECTED_DECODER_ID
    assert EXPECTED_E2E_ADAPTER_ID == "c6_6_deepstream_stdout_service_e2e_v1"
    assert "gst_buffer_get_nvds_batch_meta" in source
    assert "NvDsObjectMeta" in source
    assert "object_meta->mask_params" in source
    assert "mask.data[index]" in source
    assert "mask.width" in source
    assert "mask.height" in source
    assert "mask_positive_sample_count" in source
    assert "mask_pixel_count" not in source
    assert EXPECTED_FRAME_PREFIX in source
    assert EXPECTED_RESULT_PREFIX in source
    assert "nvinfer name=primary" in source
    assert "nvv4l2decoder" in source
    assert "pyds" not in source


# ADD 2026-09-07: Sample identity와 networkless GPU boundary를 검증한다.
def test_shell_and_docker_boundary_are_fail_closed() -> None:
    config = load_deepstream_segmentation_config(SEGMENTATION_CONFIG)
    shell = build_service_e2e_shell_payload(config)
    command = build_service_e2e_docker_command(Path("/tmp/model.plan"))

    assert "sha256sum" in shell
    assert "g++ -std=c++17" in shell
    assert "rm -f /tmp/c6_6_service_e2e_probe.cpp" in shell
    assert command[:4] == ("sudo", "docker", "run", "--rm")
    assert "--runtime=nvidia" in command
    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    assert EXPECTED_CONTAINER_LABEL in command
    assert "/tmp/model.plan:/model/model.plan:ro" in command


# ADD 2026-09-07: C++ compact line의 publisher snapshot 변환을 검증한다.
def test_frame_line_parses_to_publisher_snapshot() -> None:
    snapshot = parse_service_e2e_frame_line(_frame_line())
    service_config = load_service_integration_config(SERVICE_CONFIG)
    snapshot.validate(service_config)

    assert snapshot.frame_number == 7
    assert snapshot.pts_ns == 233_333_333
    assert snapshot.image_width == EXPECTED_IMAGE_WIDTH
    assert snapshot.image_height == EXPECTED_IMAGE_HEIGHT
    assert len(snapshot.instances) == 1
    assert snapshot.instances[0].class_id == 2
    assert snapshot.instances[0].mask_pixel_count == 4000


# ADD 2026-09-07: bbox-local positive sample count가 mask grid를 넘으면 거부한다.
def test_frame_line_rejects_positive_samples_beyond_mask_grid() -> None:
    payload = json.loads(_frame_line()[len(EXPECTED_FRAME_PREFIX) :])
    payload["instances"][0]["mask_positive_sample_count"] = 501

    with pytest.raises(ValueError):
        parse_service_e2e_frame_line(
            EXPECTED_FRAME_PREFIX + json.dumps(payload, separators=(",", ":"))
        )


# ADD 2026-09-07: Raw mask 같은 추가 payload field는 compact process boundary에서 거부한다.
def test_frame_line_rejects_extra_raw_mask_field() -> None:
    payload = json.loads(_frame_line()[len(EXPECTED_FRAME_PREFIX) :])
    payload["raw_mask"] = [0.0, 1.0]

    with pytest.raises(ValueError):
        parse_service_e2e_frame_line(
            EXPECTED_FRAME_PREFIX + json.dumps(payload, separators=(",", ":"))
        )


# ADD 2026-09-07: Frozen class mapping 밖의 object metadata는 fail closed한다.
def test_frame_line_rejects_unknown_class() -> None:
    payload = json.loads(_frame_line()[len(EXPECTED_FRAME_PREFIX) :])
    payload["instances"][0]["class_id"] = 9

    with pytest.raises(ValueError):
        parse_service_e2e_frame_line(
            EXPECTED_FRAME_PREFIX + json.dumps(payload, separators=(",", ":"))
        )


# ADD 2026-09-07: GPU compact-metadata summary의 exact acceptance scope를 검증한다.
def test_result_parser_and_validator_accept_exact_runtime_summary() -> None:
    payload = parse_service_e2e_result("noise\n" + _result_line() + "\n")
    validate_service_e2e_result(payload)

    assert payload["status"] == "passed"
    assert payload["network_used"] is False
    assert payload["raw_frame_emitted"] is False
    assert payload["raw_mask_emitted"] is False
    assert payload["final_test_used"] is False


# ADD 2026-09-07: GPU summary가 network/raw-media/final-test scope를 넘으면 거부한다.
@pytest.mark.parametrize(
    "field",
    ("network_used", "raw_frame_emitted", "raw_mask_emitted", "final_test_used"),
)
def test_result_validator_rejects_forbidden_scope(field: str) -> None:
    payload = parse_service_e2e_result(_result_line())
    payload[field] = True

    with pytest.raises(ValueError):
        validate_service_e2e_result(payload)
