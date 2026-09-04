from __future__ import annotations

from dataclasses import replace

import pytest

from services.streaming.yolo_rtsp_fault_injection_smoke import (
    DEFAULT_RTSP_FAULT_INJECTION_CONFIG,
    StreamHealthTracker,
    build_fixture_server_command,
    load_rtsp_fault_injection_config,
)
from services.streaming.yolo_rtsp_fixture_server import build_fixture_launch


def test_repository_fault_injection_config_loads() -> None:
    config = load_rtsp_fault_injection_config(DEFAULT_RTSP_FAULT_INJECTION_CONFIG)

    assert config.smoke_id == "c6_4b_rtsp_fault_injection_smoke_v1"
    assert config.required_c6_4a_commit == "47d782e0f33f218f5bd10508840c2126837f1ca5"
    assert config.runtime.initial_healthy_frames == 10
    assert config.runtime.recovery_healthy_frames == 30
    assert config.scope.external_camera_used is False
    assert config.scope.final_test_used is False


def test_fault_runtime_rejects_boundary_mutation() -> None:
    config = load_rtsp_fault_injection_config(DEFAULT_RTSP_FAULT_INJECTION_CONFIG)
    mutated = replace(config.runtime, recovery_healthy_frames=29)

    with pytest.raises(ValueError, match="recovery_healthy_frames"):
        mutated.validate()


def test_fixture_launch_is_h264_low_latency_rtsp_payload() -> None:
    launch = build_fixture_launch(
        pattern="ball",
        width=320,
        height=240,
        framerate=30,
        bitrate_kbps=500,
        key_int_max=30,
    )

    assert "videotestsrc is-live=true" in launch
    assert "pattern=ball" in launch
    assert "width=320,height=240,framerate=30/1" in launch
    assert "x264enc tune=zerolatency" in launch
    assert "key-int-max=30" in launch
    assert "rtph264pay name=pay0 pt=96 config-interval=1" in launch


def test_fixture_server_command_is_local_and_deterministic() -> None:
    config = load_rtsp_fault_injection_config(DEFAULT_RTSP_FAULT_INJECTION_CONFIG)
    command = build_fixture_server_command(config, port=18554)

    joined = " ".join(command)
    assert "services.streaming.yolo_rtsp_fixture_server" in joined
    assert "--host 127.0.0.1" in joined
    assert "--port 18554" in joined
    assert "--mount /inspection" in joined
    assert "--pattern ball" in joined


def test_health_tracker_records_required_state_sequence() -> None:
    tracker = StreamHealthTracker()
    transitions = ["DISCONNECTED"]

    tracker.transition("CONNECTING", transitions)
    tracker.transition("STREAMING", transitions)
    tracker.transition("RECONNECTING", transitions)
    tracker.transition("STREAMING", transitions)

    assert transitions == [
        "DISCONNECTED",
        "CONNECTING",
        "STREAMING",
        "RECONNECTING",
        "STREAMING",
    ]
    assert tracker.state == "STREAMING"


def test_scope_rejects_external_camera_or_final_test() -> None:
    config = load_rtsp_fault_injection_config(DEFAULT_RTSP_FAULT_INJECTION_CONFIG)

    with pytest.raises(ValueError, match="scope changed"):
        replace(config.scope, external_camera_used=True).validate()

    with pytest.raises(ValueError, match="scope changed"):
        replace(config.scope, final_test_used=True).validate()


def test_health_tracker_exposes_last_frame_gauge() -> None:
    tracker = StreamHealthTracker()
    tracker.record_frame(now_monotonic_s=10.0)

    assert tracker.seconds_since_last_frame(now_monotonic_s=10.25) == pytest.approx(0.25)

    with pytest.raises(ValueError, match="backwards"):
        tracker.seconds_since_last_frame(now_monotonic_s=9.0)


def test_reconnect_budget_reset_requires_healthy_boundary() -> None:
    tracker = StreamHealthTracker()
    tracker.begin_reconnect()

    for index in range(29):
        tracker.record_frame(now_monotonic_s=float(index))

    assert tracker.reset_reconnect_budget_if_healthy(required_frames=30) is False
    assert tracker.reconnect_attempts_since_healthy_reset == 1

    tracker.record_frame(now_monotonic_s=29.0)

    assert tracker.reset_reconnect_budget_if_healthy(required_frames=30) is True
    assert tracker.reconnect_attempts_since_healthy_reset == 0
    assert tracker.healthy_frames_since_reconnect == 0
