"""Unit tests for C6-5D DeepStream mask rendering runtime contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.streaming.yolo_deepstream_segmentation import (
    load_deepstream_segmentation_config,
)
from services.streaming.yolo_deepstream_segmentation_render import (
    EXPECTED_RENDER_CONTAINER_LABEL,
    EXPECTED_RENDER_RESULT_PREFIX,
    EXPECTED_RENDER_RUNTIME_IMAGE,
    build_mask_render_docker_command,
    build_mask_render_probe_source,
    build_mask_render_shell_payload,
    parse_mask_render_payload,
    validate_mask_render_payload,
)


# ADD 2026-09-06: Runtime probe가 nvdsosd mask-only overlay 경계를 사용하는지 검증한다.
def test_probe_source_uses_nvdsosd_mask_rendering_without_video_output() -> None:
    source = build_mask_render_probe_source(load_deepstream_segmentation_config())

    required = (
        "nvvideoconvert",
        "video/x-raw(memory:NVMM),format=RGBA",
        "nvdsosd name=osd display-mask=1 display-bbox=0 display-text=0",
        "inspect_pre_osd",
        "inspect_post_osd",
        '"mask_rendering_executed\\":true"',
        '"overlay_executed\\":true"',
        '"annotated_video_written\\":false"',
    )
    assert all(fragment in source for fragment in required)
    assert "filesink" not in source
    assert "nvv4l2h264enc" not in source


# ADD 2026-09-06: Shell payload가 exact sample과 required rendering plugins를 고정한다.
def test_shell_payload_pins_sample_and_render_plugins() -> None:
    payload = build_mask_render_shell_payload(load_deepstream_segmentation_config())

    assert "sha256sum" in payload
    assert "stat -c %s" in payload
    assert "gst-inspect-1.0 nvvideoconvert" in payload
    assert "gst-inspect-1.0 nvdsosd" in payload
    assert "rm -f /tmp/c6_5d_mask_render_probe.cpp" in payload


# ADD 2026-09-06: Docker command가 no-network/read-only-plan rendering 경계를 사용하는지 검증한다.
def test_render_docker_command_is_networkless_and_read_only(tmp_path: Path) -> None:
    plan = tmp_path / "model.plan"
    command = build_mask_render_docker_command(plan)

    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--rm" in command
    assert f"{plan.resolve()}:/model/model.plan:ro" in command
    assert EXPECTED_RENDER_CONTAINER_LABEL in command
    assert EXPECTED_RENDER_RUNTIME_IMAGE in command


# ADD 2026-09-06: Complete rendering payload가 strict validator를 통과하는지 검증한다.
def test_render_payload_accepts_complete_result() -> None:
    payload = {
        "status": "passed",
        "pre_osd_frames": 34,
        "post_osd_frames": 34,
        "objects": 54,
        "masks": 54,
        "mask_elements": 147930,
        "eos_observed": True,
        "batch_meta_observed": True,
        "display_mask_enabled": True,
        "pre_osd_frames_reached": True,
        "post_osd_frames_reached": True,
        "mask_metadata_observed": True,
        "masks_cover_objects": True,
        "metadata_valid": True,
        "invalid_reason": "",
        "segmentation_decode_executed": True,
        "instance_metadata_executed": True,
        "mask_rendering_executed": True,
        "overlay_executed": True,
        "annotated_video_written": False,
        "dataset_used": False,
        "validation_used": False,
        "test_used": False,
        "final_test_used": False,
    }

    text = EXPECTED_RENDER_RESULT_PREFIX + json.dumps(payload)
    parsed = parse_mask_render_payload(text)
    validate_mask_render_payload(parsed)


# ADD 2026-09-06: Missing mask rendering evidence가 acceptance에서 거부되는지 검증한다.
def test_render_payload_rejects_missing_overlay() -> None:
    payload = {
        "status": "passed",
        "pre_osd_frames": 34,
        "post_osd_frames": 34,
        "objects": 54,
        "masks": 54,
        "mask_elements": 147930,
        "eos_observed": True,
        "batch_meta_observed": True,
        "display_mask_enabled": True,
        "pre_osd_frames_reached": True,
        "post_osd_frames_reached": True,
        "mask_metadata_observed": True,
        "masks_cover_objects": True,
        "metadata_valid": True,
        "invalid_reason": "",
        "segmentation_decode_executed": True,
        "instance_metadata_executed": True,
        "mask_rendering_executed": False,
        "overlay_executed": False,
        "annotated_video_written": False,
        "dataset_used": False,
        "validation_used": False,
        "test_used": False,
        "final_test_used": False,
    }

    with pytest.raises(ValueError, match="mask rendering runtime acceptance failed"):
        validate_mask_render_payload(payload)
