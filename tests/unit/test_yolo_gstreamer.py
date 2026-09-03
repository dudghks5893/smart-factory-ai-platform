"""Unit tests for the C6-1 GStreamer ingress contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from services.streaming.yolo_gstreamer import (
    DEFAULT_GSTREAMER_CONFIG,
    EXPECTED_ENGINE_SHA256,
    EXPECTED_POLICY_SHA256,
    build_yolo_gstreamer_pipeline,
    load_yolo_gstreamer_ingress_config,
)


# ADD 2026-09-04: Default config가 accepted backend와 latest-frame policy를 고정하는지 검증한다.
def test_default_gstreamer_contract_is_frozen() -> None:
    config = load_yolo_gstreamer_ingress_config(DEFAULT_GSTREAMER_CONFIG)

    assert config.accepted_backend.engine_sha256 == EXPECTED_ENGINE_SHA256
    assert config.accepted_backend.acceptance_policy_sha256 == EXPECTED_POLICY_SHA256
    assert config.accepted_backend.acceptance_state == "TENSORRT_INT8_PARITY_ACCEPTED"
    assert config.source.kind == "test"
    assert config.frame_contract.pixel_format == "BGR"
    assert config.latency_policy.mode == "latest_frame_wins"
    assert config.latency_policy.queue_max_buffers == 1
    assert config.latency_policy.appsink_max_buffers == 1
    assert config.latency_policy.appsink_drop is True
    assert config.latency_policy.appsink_sync is False


# ADD 2026-09-04: Synthetic live pipeline의 bounded/drop appsink contract를 검증한다.
def test_build_test_source_pipeline() -> None:
    config = load_yolo_gstreamer_ingress_config(DEFAULT_GSTREAMER_CONFIG)
    pipeline = build_yolo_gstreamer_pipeline(config)

    assert pipeline.startswith("videotestsrc is-live=true do-timestamp=true pattern=ball")
    assert "video/x-raw,format=BGR" in pipeline
    assert "max-size-buffers=1" in pipeline
    assert "leaky=downstream" in pipeline
    assert "appsink name=framesink" in pipeline
    assert "max-buffers=1" in pipeline
    assert "drop=true" in pipeline
    assert "sync=false" in pipeline


# ADD 2026-09-04: File source는 explicit path를 file URI로 변환해 pipeline에 주입하는지 검증한다.
def test_build_file_source_pipeline(tmp_path: Path) -> None:
    config = load_yolo_gstreamer_ingress_config(DEFAULT_GSTREAMER_CONFIG)
    file_source = replace(config.source, kind="file", location=None, is_live=False)
    config = replace(config, source=file_source)
    video = tmp_path / "sample video.mp4"

    pipeline = build_yolo_gstreamer_pipeline(config, source_override=str(video))

    assert "uridecodebin uri=" in pipeline
    assert video.resolve().as_uri() in pipeline


# ADD 2026-09-04: File source location이 없으면 fail closed하는지 검증한다.
def test_file_source_requires_location() -> None:
    config = load_yolo_gstreamer_ingress_config(DEFAULT_GSTREAMER_CONFIG)
    file_source = replace(config.source, kind="file", location=None, is_live=False)
    config = replace(config, source=file_source)

    with pytest.raises(ValueError, match="requires source_override"):
        build_yolo_gstreamer_pipeline(config)


# ADD 2026-09-04: Accepted INT8 engine identity가 바뀌면 C6 ingress가 시작되지 않게 한다.
def test_rejects_different_int8_engine() -> None:
    config = load_yolo_gstreamer_ingress_config(DEFAULT_GSTREAMER_CONFIG)
    bad_backend = replace(config.accepted_backend, engine_sha256="a" * 64)
    bad_config = replace(config, accepted_backend=bad_backend)

    with pytest.raises(ValueError, match="backend identity changed"):
        bad_config.validate()


# ADD 2026-09-04: Backlog을 허용하는 queue 설정은 real-time contract에서 거부한다.
def test_rejects_unbounded_backpressure_policy() -> None:
    config = load_yolo_gstreamer_ingress_config(DEFAULT_GSTREAMER_CONFIG)
    bad_latency = replace(config.latency_policy, queue_max_buffers=8)
    bad_config = replace(config, latency_policy=bad_latency)

    with pytest.raises(ValueError, match="latency/backpressure"):
        bad_config.validate()
