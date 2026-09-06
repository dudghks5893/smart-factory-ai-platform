"""Unit contracts for C6-6A DeepStream-to-service integration."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from services.streaming.yolo_deepstream_service_integration import (
    DEFAULT_SERVICE_INTEGRATION_CONFIG,
    EXPECTED_C6_5_CLOSURE_COMMIT,
    EXPECTED_C6_5_FINAL_STATE,
    EXPECTED_CLASSES,
    EXPECTED_DECODER_ID,
    EXPECTED_ENGINE_SHA256,
    EXPECTED_EVENT_TYPE,
    EXPECTED_INTERNAL_ENDPOINT,
    EXPECTED_LABELS_SHA256,
    EXPECTED_PARSER_SHA256,
    EXPECTED_WEBSOCKET_ENDPOINT,
    StreamingDefectInstance,
    StreamingKnownDefectObservation,
    load_service_integration_config,
)


def _instance() -> StreamingDefectInstance:
    return StreamingDefectInstance(
        class_id=1,
        class_name="color",
        confidence=0.91,
        box_xyxy=(1.0, 2.0, 18.0, 9.0),
        mask_pixel_count=100,
        mask_area_ratio=0.5,
    )


def _observation(
    *,
    source_id: str = "line-1-camera-1",
    observed_at: datetime | None = None,
    instances: tuple[StreamingDefectInstance, ...] | None = None,
) -> StreamingKnownDefectObservation:
    return StreamingKnownDefectObservation(
        observation_id=uuid4(),
        source_id=source_id,
        stream_session_id=uuid4(),
        frame_number=42,
        pts_ns=1_400_000_000,
        observed_at=observed_at or datetime(2026, 9, 6, 1, 30, tzinfo=UTC),
        image_width=20,
        image_height=10,
        decoder_id=EXPECTED_DECODER_ID,
        engine_sha256=EXPECTED_ENGINE_SHA256,
        parser_sha256=EXPECTED_PARSER_SHA256,
        labels_sha256=EXPECTED_LABELS_SHA256,
        diagnostic_confidence=0.25,
        instances=instances if instances is not None else (_instance(),),
    )


# ADD 2026-09-06: C6-6A YAML이 exact sealed identity와 typed contract로 로드되는지 검증한다.
def test_config_loads_exact_contract() -> None:
    config = load_service_integration_config()

    assert config.config_path == DEFAULT_SERVICE_INTEGRATION_CONFIG
    assert config.c6_5_identity.closure_commit == EXPECTED_C6_5_CLOSURE_COMMIT
    assert config.c6_5_identity.final_state == EXPECTED_C6_5_FINAL_STATE
    assert config.c6_5_identity.engine_sha256 == EXPECTED_ENGINE_SHA256
    assert config.c6_5_identity.parser_sha256 == EXPECTED_PARSER_SHA256
    assert config.c6_5_identity.labels_sha256 == EXPECTED_LABELS_SHA256
    assert config.event.classes == EXPECTED_CLASSES


# ADD 2026-09-06: Upload YOLO와 streaming event namespace 분리를 검증한다.
def test_process_boundary_is_separate_from_existing_image_upload_contract() -> None:
    boundary = load_service_integration_config().process_boundary

    assert boundary.internal_endpoint == EXPECTED_INTERNAL_ENDPOINT
    assert boundary.websocket_endpoint == EXPECTED_WEBSOCKET_ENDPOINT
    assert boundary.internal_endpoint != "/v1/known-defects"
    assert boundary.websocket_endpoint != "/v1/ws/known-defects"
    assert boundary.reuse_image_upload_endpoint is False
    assert boundary.reuse_persisted_known_defect_event is False


# ADD 2026-09-06: Live event namespace가 durable known_defect.created와 구분되는지 검증한다.
def test_streaming_event_namespace_is_observation_only() -> None:
    event = load_service_integration_config().event

    assert event.event_type == EXPECTED_EVENT_TYPE
    assert event.event_type != "known_defect.created"
    assert event.include_raw_frame is False
    assert event.include_raw_mask is False
    assert event.include_rtsp_uri is False


# ADD 2026-09-06: API outage가 GPU pipeline을 막지 않는 bounded latest-event-wins policy를 검증한다.
def test_delivery_policy_is_bounded_best_effort() -> None:
    delivery = load_service_integration_config().delivery

    assert delivery.semantics == "best_effort_live_observation"
    assert delivery.queue_mode == "latest_event_wins"
    assert delivery.queue_max_events == 1
    assert delivery.request_timeout_ms == 1000
    assert delivery.max_retries == 3
    assert delivery.initial_backoff_ms == 100
    assert delivery.multiplier == 2.0
    assert delivery.max_backoff_ms == 1000
    assert delivery.fail_pipeline_on_delivery_exhaustion is False


# ADD 2026-09-06: C6-6A foundation의 runtime/persistence 금지를 검증한다.
def test_foundation_scope_is_contract_only() -> None:
    foundation = load_service_integration_config().foundation

    assert foundation.contract_only is True
    assert foundation.service_endpoint_implemented is False
    assert foundation.worker_publisher_implemented is False
    assert foundation.websocket_channel_implemented is False
    assert foundation.actual_rtsp_used is False
    assert foundation.deepstream_runtime_used is False
    assert foundation.persistence_used is False
    assert foundation.network_used is False
    assert foundation.final_test_used is False


# ADD 2026-09-06: Valid observation이 compact JSON payload로 직렬화되는지 검증한다.
def test_valid_observation_builds_compact_payload() -> None:
    config = load_service_integration_config()
    payload = _observation().to_payload(config)
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["schema_version"] == "1"
    assert payload["type"] == EXPECTED_EVENT_TYPE
    assert "raw_frame" not in serialized
    assert "raw_mask" not in serialized
    assert "rtsp://" not in serialized
    assert "://" not in serialized
    assert "password" not in serialized.lower()


# ADD 2026-09-06: source_id에 URI나 credential 형태가 들어오는 것을 거부하는지 검증한다.
def test_observation_rejects_source_uri_or_credentials() -> None:
    config = load_service_integration_config()

    with pytest.raises(ValueError, match="opaque non-secret"):
        _observation(source_id="rtsp://user:secret@camera/stream").validate(config)


# ADD 2026-09-06: DeepStream class ID/name mapping이 bent/color/scratch 계약과 다르면 거부한다.
def test_observation_rejects_wrong_class_mapping() -> None:
    config = load_service_integration_config()
    invalid = replace(_instance(), class_name="scratch")

    with pytest.raises(ValueError, match="class mapping"):
        _observation(instances=(invalid,)).validate(config)


# ADD 2026-09-06: Streaming bbox가 source image bounds를 벗어나면 거부한다.
def test_observation_rejects_out_of_bounds_bbox() -> None:
    config = load_service_integration_config()
    invalid = replace(_instance(), box_xyxy=(1.0, 2.0, 21.0, 9.0))

    with pytest.raises(ValueError, match="outside the image bounds"):
        _observation(instances=(invalid,)).validate(config)


# ADD 2026-09-06: Compact mask ratio가 pixel count/image area와 다르면 거부한다.
def test_observation_rejects_inconsistent_mask_ratio() -> None:
    config = load_service_integration_config()
    invalid = replace(_instance(), mask_area_ratio=0.25)

    with pytest.raises(ValueError, match="does not match"):
        _observation(instances=(invalid,)).validate(config)


# ADD 2026-09-06: Cross-process event timestamp를 timezone-aware UTC로 제한한다.
def test_observation_rejects_non_utc_timestamp() -> None:
    config = load_service_integration_config()
    kst = timezone(timedelta(hours=9))

    with pytest.raises(ValueError, match="timezone-aware UTC"):
        _observation(observed_at=datetime(2026, 9, 6, 10, 30, tzinfo=kst)).validate(config)


# ADD 2026-09-06: 한 frame event의 instance 수를 decoder max_detections와 같은 300으로 제한한다.
def test_observation_rejects_more_than_300_instances() -> None:
    config = load_service_integration_config()
    repeated = tuple(_instance() for _ in range(301))

    with pytest.raises(ValueError, match="max_instances"):
        _observation(instances=repeated).validate(config)


# ADD 2026-09-06: Runtime provenance가 sealed C6-5 identity와 다르면 event를 거부한다.
def test_observation_rejects_runtime_identity_change() -> None:
    config = load_service_integration_config()
    invalid = replace(_observation(), engine_sha256="0" * 64)

    with pytest.raises(ValueError, match="runtime identity"):
        invalid.validate(config)


# ADD 2026-09-06: JSON에 임의 field가 추가되면 strict loader가 fail-closed하는지 검증한다.
def test_config_rejects_extra_top_level_field(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_SERVICE_INTEGRATION_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    modified = copy.deepcopy(raw)
    modified["unexpected"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(modified), encoding="utf-8")

    with pytest.raises(ValueError, match="fields do not match schema"):
        load_service_integration_config(path)
