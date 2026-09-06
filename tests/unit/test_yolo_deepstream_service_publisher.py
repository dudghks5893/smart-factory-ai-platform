"""Unit contracts for the C6-6C latest-event worker publisher."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

from services.streaming.yolo_deepstream_service_integration import (
    EXPECTED_COUNTERS,
    EXPECTED_GAUGES,
    StreamingKnownDefectObservation,
    load_service_integration_config,
)
from services.streaming.yolo_deepstream_service_publisher import (
    DeepStreamFrameSnapshot,
    DeepStreamInstanceSnapshot,
    LatestEventWinsPublisher,
    build_streaming_observation,
)

CONFIG = Path("configs/streaming/yolo_deepstream_service_integration.json")
SOURCE_ID = "line-01-camera-01"
SESSION_ID = UUID("11111111-1111-4111-8111-111111111111")
OBSERVATION_ID = UUID("22222222-2222-4222-8222-222222222222")
OBSERVED_AT = datetime(2026, 9, 6, 3, 0, tzinfo=UTC)


class _ScriptedTransport:
    def __init__(self, outcomes: list[int | Exception]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[str, Mapping[str, object], float]] = []

    def post_json(
        self,
        *,
        url: str,
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> int:
        self.calls.append((url, payload, timeout_seconds))
        if not self._outcomes:
            raise AssertionError("Unexpected transport call.")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _frame(*, frame_number: int = 42) -> DeepStreamFrameSnapshot:
    return DeepStreamFrameSnapshot(
        frame_number=frame_number,
        pts_ns=1_400_000_000 + frame_number,
        image_width=640,
        image_height=640,
        instances=(
            DeepStreamInstanceSnapshot(
                class_id=2,
                confidence=0.91,
                left=10.0,
                top=20.0,
                width=100.0,
                height=80.0,
                mask_pixel_count=1000,
            ),
        ),
    )


def _observation(*, frame_number: int = 42) -> StreamingKnownDefectObservation:
    config = load_service_integration_config(CONFIG)
    return build_streaming_observation(
        _frame(frame_number=frame_number),
        source_id=SOURCE_ID,
        stream_session_id=SESSION_ID,
        config=config,
        observed_at=OBSERVED_AT,
        observation_id=OBSERVATION_ID,
    )


# ADD 2026-09-06: Snapshot이 sealed identity와 compact mask event로 변환되는지 검증한다.
def test_build_streaming_observation_is_compact_and_identity_preserving() -> None:
    config = load_service_integration_config(CONFIG)
    payload = _observation().to_payload(config)

    observation = cast(dict[str, object], payload["observation"])
    runtime = cast(dict[str, object], observation["runtime"])
    instances = cast(list[dict[str, object]], observation["instances"])
    instance = instances[0]
    mask = cast(dict[str, object], instance["mask"])

    assert payload["type"] == "streaming_known_defect.observed"
    assert observation["source_id"] == SOURCE_ID
    assert observation["stream_session_id"] == str(SESSION_ID)
    assert observation["frame_number"] == 42
    assert runtime["engine_sha256"] == config.c6_5_identity.engine_sha256
    assert instance["class_id"] == 2
    assert instance["class_name"] == "scratch"
    assert mask["pixel_count"] == 1000
    assert mask["area_ratio"] == 1000 / (640 * 640)

    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in ("raw_frame", "raw_mask", "rtsp_uri", "image_sha256"):
        assert forbidden not in serialized


# ADD 2026-09-06: Queue max 1에서 newest event가 pending을 대체하는지 검증한다.
def test_latest_event_wins_queue_replaces_pending_event() -> None:
    config = load_service_integration_config(CONFIG)
    transport = _ScriptedTransport([202])
    publisher = LatestEventWinsPublisher(
        service_base_url="http://127.0.0.1:8000",
        config=config,
        transport=transport,
    )

    first = _observation(frame_number=1)
    second = _observation(frame_number=2)
    publisher.submit(first)
    publisher.submit(second)

    before = publisher.metrics_snapshot()
    assert before.generated_total == 2
    assert before.dropped_total == 1
    assert before.pending_events == 1

    publisher.start()
    publisher.close()

    after = publisher.metrics_snapshot()
    assert after.delivered_total == 1
    assert after.dropped_total == 1
    assert len(transport.calls) == 1
    delivered = cast(dict[str, object], transport.calls[0][1]["observation"])
    assert delivered["frame_number"] == 2


# ADD 2026-09-06: Delivery exhaustion이 bounded retry/drop으로 격리되는지 검증한다.
def test_delivery_exhaustion_is_failure_isolated() -> None:
    config = load_service_integration_config(CONFIG)
    transport = _ScriptedTransport([500, 500, 500, 500])
    sleeps: list[float] = []
    publisher = LatestEventWinsPublisher(
        service_base_url="http://127.0.0.1:8000",
        config=config,
        transport=transport,
        sleep=sleeps.append,
    )

    publisher.start()
    publisher.submit(_observation())
    publisher.close()

    metrics = publisher.metrics_snapshot()
    assert metrics.generated_total == 1
    assert metrics.delivered_total == 0
    assert metrics.dropped_total == 1
    assert metrics.delivery_errors_total == 4
    assert metrics.delivery_retries_total == 3
    assert metrics.bridge_up == 0
    assert sleeps == [0.1, 0.2, 0.4]
    assert len(transport.calls) == 4


# ADD 2026-09-06: Transient failure 뒤 202에서 bridge 상태가 회복되는지 검증한다.
def test_transient_delivery_failure_recovers_on_accepted_response() -> None:
    config = load_service_integration_config(CONFIG)
    transport = _ScriptedTransport([503, 503, 202])
    sleeps: list[float] = []
    clock = iter((10.0, 10.25))
    publisher = LatestEventWinsPublisher(
        service_base_url="http://127.0.0.1:8000",
        config=config,
        transport=transport,
        sleep=sleeps.append,
        monotonic=lambda: next(clock),
    )

    publisher.start()
    publisher.submit(_observation())
    publisher.close()

    metrics = publisher.metrics_snapshot()
    assert metrics.generated_total == 1
    assert metrics.delivered_total == 1
    assert metrics.dropped_total == 0
    assert metrics.delivery_errors_total == 2
    assert metrics.delivery_retries_total == 2
    assert metrics.bridge_up == 1
    assert metrics.seconds_since_delivery == 0.25
    assert sleeps == [0.1, 0.2]


# ADD 2026-09-06: Observability가 C6-6A exact metric names를 유지하는지 검증한다.
def test_metrics_snapshot_matches_frozen_observability_names() -> None:
    config = load_service_integration_config(CONFIG)
    publisher = LatestEventWinsPublisher(
        service_base_url="http://127.0.0.1:8000",
        config=config,
        transport=_ScriptedTransport([]),
    )

    metrics = publisher.metrics_snapshot().as_contract_metrics()

    assert tuple(metrics) == EXPECTED_COUNTERS + EXPECTED_GAUGES
    assert metrics["stream_service_pending_events"] == 0
    publisher.close()


# ADD 2026-09-06: Internal bridge URL의 잘못된 scheme/credential/path를 거부한다.
@pytest.mark.parametrize(
    "url",
    (
        "https://127.0.0.1:8000",
        "http://user:pass@127.0.0.1:8000",
        "http://127.0.0.1:8000/api",
        "rtsp://127.0.0.1:8554",
    ),
)
def test_service_base_url_rejects_non_internal_http_shapes(url: str) -> None:
    config = load_service_integration_config(CONFIG)

    with pytest.raises(ValueError):
        LatestEventWinsPublisher(
            service_base_url=url,
            config=config,
            transport=_ScriptedTransport([]),
        )


# ADD 2026-09-06: Worker boundary에서도 RTSP URI를 source ID로 위장할 수 없음을 검증한다.
def test_observation_builder_rejects_credential_like_source_id() -> None:
    config = load_service_integration_config(CONFIG)

    with pytest.raises(ValueError):
        build_streaming_observation(
            _frame(),
            source_id="rtsp://user:pass@camera/stream",
            stream_session_id=SESSION_ID,
            config=config,
            observed_at=OBSERVED_AT,
            observation_id=OBSERVATION_ID,
        )
