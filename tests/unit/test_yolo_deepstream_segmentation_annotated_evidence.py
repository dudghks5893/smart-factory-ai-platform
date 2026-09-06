"""Unit tests for C6-5D annotated DeepStream video evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.streaming.yolo_deepstream_segmentation import (
    load_deepstream_segmentation_config,
)
from services.streaming.yolo_deepstream_segmentation_annotated_evidence import (
    EXPECTED_CONTAINER_LABEL,
    EXPECTED_RESULT_PREFIX,
    EXPECTED_RUNTIME_IMAGE,
    EXPECTED_VIDEO_NAME,
    build_annotated_video_docker_command,
    build_annotated_video_probe_source,
    build_annotated_video_shell_payload,
    parse_annotated_video_payload,
    validate_annotated_video_payload,
)


# ADD 2026-09-06: MP4 encoder 연결 검증 → MODIFY 2026-09-06: MJPEG/AVI encoder 연결 검증
def test_probe_source_writes_mask_overlay_avi() -> None:
    source = build_annotated_video_probe_source(load_deepstream_segmentation_config())

    required = (
        "nvdsosd name=osd display-mask=1 display-bbox=0 display-text=0",
        "jpegenc ! avimux",
        f"filesink location=/evidence/{EXPECTED_VIDEO_NAME}",
        'annotated_video_written ? "true" : "false"',
        '"final_test_used\\":false}',
    )
    assert all(fragment in source for fragment in required)


# ADD 2026-09-06: MP4 decode 검증 → MODIFY 2026-09-06: MJPEG/AVI 재디코드 검증
def test_shell_payload_pins_sample_and_decodes_output() -> None:
    payload = build_annotated_video_shell_payload(load_deepstream_segmentation_config())

    assert "sha256sum" in payload
    assert "gst-inspect-1.0 jpegenc" in payload
    assert "gst-inspect-1.0 avimux" in payload
    assert "avidemux ! jpegdec" in payload
    assert f"test -s /evidence/{EXPECTED_VIDEO_NAME}" in payload


# ADD 2026-09-06: Docker command가 evidence dir만 writable로 mount하는지 검증한다.
def test_docker_command_uses_bounded_mounts(tmp_path: Path) -> None:
    plan = tmp_path / "model.plan"
    evidence = tmp_path / "evidence"
    command = build_annotated_video_docker_command(plan, evidence)

    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    assert f"{plan.resolve()}:/model/model.plan:ro" in command
    assert f"{evidence.resolve()}:/evidence:rw" in command
    assert EXPECTED_CONTAINER_LABEL in command
    assert EXPECTED_RUNTIME_IMAGE in command


# ADD 2026-09-06: Complete annotated-video payload가 strict validator를 통과한다.
def test_annotated_payload_accepts_complete_result() -> None:
    payload = {
        "status": "passed",
        "pre_osd_frames": 35,
        "post_osd_frames": 35,
        "objects": 54,
        "masks": 54,
        "mask_elements": 147930,
        "eos_observed": True,
        "batch_meta_observed": True,
        "display_mask_enabled": True,
        "pre_osd_frames_reached": True,
        "post_osd_frames_reached": True,
        "masks_cover_objects": True,
        "metadata_valid": True,
        "invalid_reason": "",
        "segmentation_decode_executed": True,
        "instance_metadata_executed": True,
        "mask_rendering_executed": True,
        "overlay_executed": True,
        "annotated_video_written": True,
        "dataset_used": False,
        "validation_used": False,
        "test_used": False,
        "final_test_used": False,
    }

    parsed = parse_annotated_video_payload(EXPECTED_RESULT_PREFIX + json.dumps(payload))
    validate_annotated_video_payload(parsed)


# ADD 2026-09-06: final-test activation은 annotated sample evidence에서 거부한다.
def test_annotated_payload_rejects_final_test() -> None:
    payload = {
        "status": "passed",
        "pre_osd_frames": 35,
        "post_osd_frames": 35,
        "objects": 54,
        "masks": 54,
        "mask_elements": 147930,
        "eos_observed": True,
        "batch_meta_observed": True,
        "display_mask_enabled": True,
        "pre_osd_frames_reached": True,
        "post_osd_frames_reached": True,
        "masks_cover_objects": True,
        "metadata_valid": True,
        "invalid_reason": "",
        "segmentation_decode_executed": True,
        "instance_metadata_executed": True,
        "mask_rendering_executed": True,
        "overlay_executed": True,
        "annotated_video_written": True,
        "dataset_used": False,
        "validation_used": False,
        "test_used": False,
        "final_test_used": True,
    }

    with pytest.raises(
        ValueError,
        match="annotated-video runtime acceptance failed",
    ):
        validate_annotated_video_payload(payload)
