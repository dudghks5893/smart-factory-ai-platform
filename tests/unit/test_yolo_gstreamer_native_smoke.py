"""Unit tests for the C6-2 native GStreamer smoke contract."""

from __future__ import annotations

from pathlib import Path

from services.streaming.yolo_gstreamer_native_smoke import (
    DEFAULT_NATIVE_SMOKE_CONFIG,
    build_file_decode_smoke_command,
    build_fixture_generation_command,
    build_synthetic_smoke_command,
    load_native_gstreamer_smoke_config,
)


# ADD 2026-09-04: Default C6-2 smoke config가 required plugins와 short fixture contract를 고정한다.
def test_default_native_smoke_config() -> None:
    config = load_native_gstreamer_smoke_config(DEFAULT_NATIVE_SMOKE_CONFIG)

    assert config.smoke_id == "c6_2_yolo_gstreamer_native_smoke_v1"
    assert config.synthetic.num_buffers == 30
    assert config.synthetic.pattern == "ball"
    assert config.fixture.width == 320
    assert config.fixture.height == 240
    assert config.fixture.framerate == 30
    assert config.fixture.encoder.name == "x264enc"
    assert set(config.required_plugins) == {
        "videotestsrc",
        "videoconvert",
        "queue",
        "appsink",
        "uridecodebin",
        "x264enc",
        "h264parse",
        "mp4mux",
    }


# ADD 2026-09-04: Synthetic command가 C6-1 BGR/appsink contract를 보존하는지 검증한다.
def test_synthetic_command_preserves_c6_1_contract() -> None:
    config = load_native_gstreamer_smoke_config(DEFAULT_NATIVE_SMOKE_CONFIG)
    command = build_synthetic_smoke_command("/gst-launch-1.0", config)
    joined = " ".join(command)

    assert "videotestsrc" in command
    assert "num-buffers=30" in command
    assert "video/x-raw,format=BGR" in command
    assert "max-size-buffers=1" in command
    assert "leaky=downstream" in command
    assert "appsink" in command
    assert "max-buffers=1" in command
    assert "drop=true" in command
    assert "sync=false" in command
    assert "wait-on-eos=false" in joined


# ADD 2026-09-04: Fixture command가 short H264/MP4 asset를 생성하는지 검증한다.
def test_fixture_generation_command(tmp_path: Path) -> None:
    config = load_native_gstreamer_smoke_config(DEFAULT_NATIVE_SMOKE_CONFIG)
    fixture = tmp_path / "fixture.mp4"
    command = build_fixture_generation_command("/gst-launch-1.0", config, fixture)

    assert "video/x-raw,width=320,height=240,framerate=30/1" in command
    assert "x264enc" in command
    assert "tune=zerolatency" in command
    assert "speed-preset=ultrafast" in command
    assert "h264parse" in command
    assert "mp4mux" in command
    assert f"location={fixture}" in command


# ADD 2026-09-04: Local-file decode가 동일 BGR/appsink boundary를 쓰는지 검증한다.
def test_file_decode_command(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture with space.mp4"
    command = build_file_decode_smoke_command("/gst-launch-1.0", fixture)

    assert "uridecodebin" in command
    assert f"uri={fixture.resolve().as_uri()}" in command
    assert "video/x-raw,format=BGR" in command
    assert "appsink" in command
    assert "drop=true" in command
    assert "wait-on-eos=false" in command
