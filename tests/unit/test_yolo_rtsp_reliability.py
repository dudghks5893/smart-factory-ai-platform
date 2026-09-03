from __future__ import annotations

from dataclasses import replace

import pytest

from services.streaming.yolo_rtsp_reliability import (
    DEFAULT_RTSP_RELIABILITY_CONFIG,
    EXPECTED_COUNTERS,
    EXPECTED_GAUGES,
    EXPECTED_STATES,
    build_rtsp_pipeline,
    is_frame_stale,
    load_rtsp_reliability_config,
    reconnect_backoff_schedule_ms,
    redact_rtsp_uri,
    validate_rtsp_uri,
)


def test_repository_rtsp_contract_loads() -> None:
    config = load_rtsp_reliability_config(DEFAULT_RTSP_RELIABILITY_CONFIG)

    assert config.contract_id == "c6_4a_yolo_rtsp_reliability_v1"
    assert config.c6_3_identity.acceptance_state == "TENSORRT_INT8_STREAMING_ACCEPTED"
    assert config.source.transport == "tcp"
    assert config.backpressure.mode == "latest_frame_wins"
    assert config.foundation.actual_rtsp_used is False


def test_rtsp_source_policy_rejects_mutation() -> None:
    config = load_rtsp_reliability_config(DEFAULT_RTSP_RELIABILITY_CONFIG)
    mutated = replace(config.source, frame_stale_after_ms=9999)

    with pytest.raises(ValueError, match="source policy changed"):
        mutated.validate()


def test_pipeline_preserves_rtsp_and_latest_frame_wins_contract() -> None:
    config = load_rtsp_reliability_config(DEFAULT_RTSP_RELIABILITY_CONFIG)
    uri = "rtsp://127.0.0.1:8554/inspection"

    pipeline = build_rtsp_pipeline(config, uri=uri)

    assert "rtspsrc" in pipeline
    assert "protocols=tcp" in pipeline
    assert "latency=200" in pipeline
    assert "tcp-timeout=5000000" in pipeline
    assert "rtph264depay" in pipeline
    assert "video/x-raw,format=BGR" in pipeline
    assert "max-size-buffers=1" in pipeline
    assert "leaky=downstream" in pipeline
    assert "appsink name=framesink" in pipeline
    assert "max-buffers=1 drop=true sync=false" in pipeline


def test_rtsp_uri_validation_and_credential_redaction() -> None:
    uri = "rtsp://user:secret@camera.local:8554/live"

    validate_rtsp_uri(uri)
    redacted = redact_rtsp_uri(uri)

    assert redacted == "rtsp://***:***@camera.local:8554/live"
    assert "user" not in redacted
    assert "secret" not in redacted

    with pytest.raises(ValueError, match="rtsp://"):
        validate_rtsp_uri("https://camera.local/live")


def test_reconnect_schedule_is_bounded_exponential_backoff() -> None:
    config = load_rtsp_reliability_config(DEFAULT_RTSP_RELIABILITY_CONFIG)

    assert reconnect_backoff_schedule_ms(config.reconnect) == (
        500,
        1000,
        2000,
        4000,
        8000,
    )


def test_frame_stale_boundary_is_deterministic() -> None:
    assert not is_frame_stale(
        last_frame_monotonic_s=10.0,
        now_monotonic_s=11.499,
        stale_after_ms=1500,
    )
    assert is_frame_stale(
        last_frame_monotonic_s=10.0,
        now_monotonic_s=11.5,
        stale_after_ms=1500,
    )

    with pytest.raises(ValueError, match="backwards"):
        is_frame_stale(
            last_frame_monotonic_s=10.0,
            now_monotonic_s=9.0,
            stale_after_ms=1500,
        )


def test_observability_surface_is_frozen() -> None:
    config = load_rtsp_reliability_config(DEFAULT_RTSP_RELIABILITY_CONFIG)

    assert config.observability.states == EXPECTED_STATES
    assert config.observability.counters == EXPECTED_COUNTERS
    assert config.observability.gauges == EXPECTED_GAUGES
    assert config.observability.redact_credentials is True
