"""Unit tests for C6-5D instance-mask metadata runtime contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.streaming.yolo_deepstream_segmentation import load_deepstream_segmentation_config
from services.streaming.yolo_deepstream_segmentation_metadata import (
    EXPECTED_CONTAINER_LABEL,
    EXPECTED_RESULT_PREFIX,
    EXPECTED_RUNTIME_IMAGE,
    build_instance_metadata_docker_command,
    build_instance_metadata_probe_source,
    build_instance_metadata_shell_payload,
    parse_instance_metadata_payload,
    validate_instance_metadata_payload,
)


# ADD 2026-09-06: Generated probe가 NvDsObjectMeta mask_params와 tensor-meta를 함께 읽는지 검증한다.
def test_probe_source_reads_instance_mask_metadata_without_overlay() -> None:
    source = build_instance_metadata_probe_source(load_deepstream_segmentation_config())

    required = (
        "gst_buffer_get_nvds_batch_meta",
        "frame_meta->obj_meta_list",
        "object_meta->mask_params",
        "NVDSINFER_TENSOR_OUTPUT_META",
        "mask.size != expected_bytes",
        "mask probability outside [0,1]",
        'segmentation_decode_executed\\":true',
        'instance_metadata_executed\\":true',
        'overlay_executed\\":false',
        'final_test_used\\":false',
    )
    assert all(fragment in source for fragment in required)
    assert "nvdsosd" not in source
    assert "filesink" not in source


# ADD 2026-09-06: Exact sample identity와 local compile cleanup을 검증한다.
def test_shell_payload_is_sample_pinned_and_ephemeral() -> None:
    payload = build_instance_metadata_shell_payload(load_deepstream_segmentation_config())

    assert "5f29353a6ec4727bd49fb523efc207d643e6638f4e5c56f060e1b61291aa6ea2" in payload
    assert "14759548" in payload
    assert "-lnvdsgst_meta -lnvds_meta" in payload
    assert "rm -f /tmp/c6_5d_instance_metadata_probe.cpp" in payload


# ADD 2026-09-06: Runtime container의 no-network/read-only-plan/ephemeral 경계를 검증한다.
def test_docker_command_is_networkless_and_read_only(tmp_path: Path) -> None:
    plan = tmp_path / "model.plan"
    command = build_instance_metadata_docker_command(plan)

    assert command[:3] == ("sudo", "docker", "run")
    assert "--rm" in command
    network_index = command.index("--network")
    assert command[network_index + 1] == "none"
    assert EXPECTED_CONTAINER_LABEL in command
    assert EXPECTED_RUNTIME_IMAGE in command
    mount = command[command.index("-v") + 1]
    assert mount.endswith(":/model/model.plan:ro")


# ADD 2026-09-06: Strict runtime payload parser/validator가 accepted metadata result를 허용한다.
def test_runtime_payload_accepts_complete_instance_metadata_result() -> None:
    payload = {
        "status": "passed",
        "frames": 30,
        "objects": 3,
        "masks": 3,
        "mask_elements": 99,
        "tensor_meta_frames": 30,
        "class_0": 1,
        "class_1": 1,
        "class_2": 1,
        "eos_observed": True,
        "batch_meta_observed": True,
        "target_frames_reached": True,
        "tensor_meta_observed": True,
        "object_metadata_observed": True,
        "instance_masks_observed": True,
        "masks_cover_objects": True,
        "metadata_valid": True,
        "invalid_reason": "",
        "segmentation_decode_executed": True,
        "instance_metadata_executed": True,
        "overlay_executed": False,
        "annotated_video_written": False,
        "dataset_used": False,
        "validation_used": False,
        "test_used": False,
        "final_test_used": False,
    }
    text = EXPECTED_RESULT_PREFIX + json.dumps(payload, separators=(",", ":"))

    parsed = parse_instance_metadata_payload(text)
    validate_instance_metadata_payload(parsed)


# ADD 2026-09-06: Object/mask 부재가 metadata acceptance에서 거부되는지 검증한다.
def test_runtime_payload_rejects_zero_object_characterization() -> None:
    payload = {
        "status": "passed",
        "frames": 30,
        "objects": 0,
        "masks": 0,
        "mask_elements": 0,
        "tensor_meta_frames": 30,
        "class_0": 0,
        "class_1": 0,
        "class_2": 0,
        "eos_observed": True,
        "batch_meta_observed": True,
        "target_frames_reached": True,
        "tensor_meta_observed": True,
        "object_metadata_observed": False,
        "instance_masks_observed": False,
        "masks_cover_objects": False,
        "metadata_valid": True,
        "invalid_reason": "",
        "segmentation_decode_executed": True,
        "instance_metadata_executed": True,
        "overlay_executed": False,
        "annotated_video_written": False,
        "dataset_used": False,
        "validation_used": False,
        "test_used": False,
        "final_test_used": False,
    }

    with pytest.raises(ValueError, match="runtime acceptance failed"):
        validate_instance_metadata_payload(payload)
