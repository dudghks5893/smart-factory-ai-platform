"""C6-6C DeepStream worker-side latest-event publisher."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import UUID, uuid4

from services.streaming.yolo_deepstream_service_integration import (
    EXPECTED_CLASSES,
    EXPECTED_COUNTERS,
    EXPECTED_GAUGES,
    StreamingDefectInstance,
    StreamingKnownDefectObservation,
    YoloDeepStreamServiceIntegrationConfig,
)

EXPECTED_PUBLISHER_ID = "c6_6c_worker_publisher_v1"
EXPECTED_ACCEPTED_STATUS = 202


class JsonPostTransport(Protocol):
    """Minimal injectable HTTP JSON transport for the worker publisher."""

    def post_json(
        self,
        *,
        url: str,
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> int:
        """POST one JSON payload and return the HTTP status code."""


class StdlibJsonPostTransport:
    """Dependency-light internal HTTP transport."""

    # ADD 2026-09-06: Worker publisher가 stdlib HTTP POST로 internal bridge에 전달하게 한다.
    def post_json(
        self,
        *,
        url: str,
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> int:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout_seconds) as response:
            response.read()
            return int(response.status)


@dataclass(frozen=True)
class DeepStreamInstanceSnapshot:
    """Compact instance metadata extracted from NvDsObjectMeta/NvOSD_MaskParams."""

    class_id: int
    confidence: float
    left: float
    top: float
    width: float
    height: float
    mask_pixel_count: int

    # ADD 2026-09-06: DeepStream instance metadata를 compact event로 변환한다.
    def to_streaming_instance(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> StreamingDefectInstance:
        if self.class_id not in EXPECTED_CLASSES:
            raise ValueError("DeepStream instance class_id is outside the sealed mapping.")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("DeepStream instance rectangle must be positive.")
        image_area = image_width * image_height
        if not 0 < self.mask_pixel_count <= image_area:
            raise ValueError("DeepStream mask pixel count is outside the source image.")

        instance = StreamingDefectInstance(
            class_id=self.class_id,
            class_name=EXPECTED_CLASSES[self.class_id],
            confidence=self.confidence,
            box_xyxy=(
                self.left,
                self.top,
                self.left + self.width,
                self.top + self.height,
            ),
            mask_pixel_count=self.mask_pixel_count,
            mask_area_ratio=self.mask_pixel_count / image_area,
        )
        instance.validate(image_width=image_width, image_height=image_height)
        return instance


@dataclass(frozen=True)
class DeepStreamFrameSnapshot:
    """One compact post-inference frame snapshot handed to the publisher."""

    frame_number: int
    pts_ns: int
    image_width: int
    image_height: int
    instances: tuple[DeepStreamInstanceSnapshot, ...]

    # ADD 2026-09-06: Defect가 관측된 frame만 bounded live observation으로 변환 가능하게 검증한다.
    def validate(self, config: YoloDeepStreamServiceIntegrationConfig) -> None:
        config.validate()
        if self.frame_number < 0 or self.pts_ns < 0:
            raise ValueError("DeepStream frame number and PTS must be non-negative.")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("DeepStream frame dimensions must be positive.")
        if not self.instances:
            raise ValueError("Streaming known-defect publisher requires at least one instance.")
        if len(self.instances) > config.event.max_instances:
            raise ValueError("DeepStream frame exceeds the sealed max_instances.")
        for instance in self.instances:
            instance.to_streaming_instance(
                image_width=self.image_width,
                image_height=self.image_height,
            )


# ADD 2026-09-06: DeepStream frame에 sealed runtime identity를 붙여 event를 생성한다.
def build_streaming_observation(
    snapshot: DeepStreamFrameSnapshot,
    *,
    source_id: str,
    stream_session_id: UUID,
    config: YoloDeepStreamServiceIntegrationConfig,
    observed_at: datetime | None = None,
    observation_id: UUID | None = None,
) -> StreamingKnownDefectObservation:
    snapshot.validate(config)
    instances = tuple(
        instance.to_streaming_instance(
            image_width=snapshot.image_width,
            image_height=snapshot.image_height,
        )
        for instance in snapshot.instances
    )
    observation = StreamingKnownDefectObservation(
        observation_id=observation_id or uuid4(),
        source_id=source_id,
        stream_session_id=stream_session_id,
        frame_number=snapshot.frame_number,
        pts_ns=snapshot.pts_ns,
        observed_at=observed_at or datetime.now(UTC),
        image_width=snapshot.image_width,
        image_height=snapshot.image_height,
        decoder_id=config.c6_5_identity.decoder_id,
        engine_sha256=config.c6_5_identity.engine_sha256,
        parser_sha256=config.c6_5_identity.parser_sha256,
        labels_sha256=config.c6_5_identity.labels_sha256,
        diagnostic_confidence=config.event.diagnostic_confidence,
        instances=instances,
    )
    observation.validate(config)
    return observation


@dataclass(frozen=True)
class PublisherMetricsSnapshot:
    """Thread-safe publisher metrics projected onto the C6-6A observability contract."""

    generated_total: int
    delivered_total: int
    dropped_total: int
    delivery_errors_total: int
    delivery_retries_total: int
    bridge_up: int
    pending_events: int
    seconds_since_delivery: float | None

    # ADD 2026-09-06: Publisher state를 frozen C6-6A counter/gauge 이름으로 노출한다.
    def as_contract_metrics(self) -> dict[str, int | float | None]:
        metrics: dict[str, int | float | None] = {
            "stream_service_events_generated_total": self.generated_total,
            "stream_service_events_delivered_total": self.delivered_total,
            "stream_service_events_dropped_total": self.dropped_total,
            "stream_service_delivery_errors_total": self.delivery_errors_total,
            "stream_service_delivery_retries_total": self.delivery_retries_total,
            "stream_service_bridge_up": self.bridge_up,
            "stream_service_pending_events": self.pending_events,
            "stream_service_seconds_since_delivery": self.seconds_since_delivery,
        }
        if tuple(metrics) != EXPECTED_COUNTERS + EXPECTED_GAUGES:
            raise RuntimeError("Publisher observability metric order/names changed.")
        return metrics


class LatestEventWinsPublisher:
    """Non-blocking queue-size-one HTTP publisher for live observations."""

    def __init__(
        self,
        *,
        service_base_url: str,
        config: YoloDeepStreamServiceIntegrationConfig,
        transport: JsonPostTransport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        config.validate()
        self._validate_service_base_url(service_base_url)
        self._config = config
        self._endpoint_url = (
            service_base_url.rstrip("/") + config.process_boundary.internal_endpoint
        )
        self._transport = transport or StdlibJsonPostTransport()
        self._monotonic = monotonic
        self._sleep = sleep
        self._condition = threading.Condition()
        self._pending: dict[str, object] | None = None
        self._delivery_active = False
        self._stopping = False
        self._closed = False
        self._thread: threading.Thread | None = None
        self._generated_total = 0
        self._delivered_total = 0
        self._dropped_total = 0
        self._delivery_errors_total = 0
        self._delivery_retries_total = 0
        self._bridge_up = 0
        self._last_delivery_monotonic: float | None = None

    @staticmethod
    def _validate_service_base_url(service_base_url: str) -> None:
        parsed = urlsplit(service_base_url)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError(
                "service_base_url must be credential-free internal http://host[:port]."
            )

    # ADD 2026-09-06: Delivery thread로 GPU producer와 network I/O를 분리한다.
    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("Publisher is already closed.")
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="c6-6c-stream-service-publisher",
                daemon=True,
            )
            self._thread.start()

    # ADD 2026-09-06: submit은 queue size 1에서 newest event로 pending을 대체한다.
    def submit(self, observation: StreamingKnownDefectObservation) -> None:
        payload = observation.to_payload(self._config)
        with self._condition:
            if self._closed or self._stopping:
                raise RuntimeError("Publisher is stopping or closed.")
            self._generated_total += 1
            if self._pending is not None:
                self._dropped_total += 1
            self._pending = payload
            self._condition.notify_all()

    # ADD 2026-09-06: Shutdown에서 pending event까지 전달하고 worker thread를 종료한다.
    def close(self, *, timeout_seconds: float = 5.0) -> None:
        with self._condition:
            if self._closed:
                return
            self._stopping = True
            self._condition.notify_all()
            thread = self._thread

        if thread is not None:
            thread.join(timeout=timeout_seconds)
            if thread.is_alive():
                raise TimeoutError("Publisher thread did not stop within timeout.")

        with self._condition:
            if thread is None and self._pending is not None:
                self._dropped_total += 1
                self._pending = None
            self._closed = True
            self._condition.notify_all()

    # ADD 2026-09-06: Queue와 active delivery가 비었는지 bounded wait한다.
    def wait_until_idle(self, *, timeout_seconds: float) -> bool:
        deadline = self._monotonic() + timeout_seconds
        with self._condition:
            while self._pending is not None or self._delivery_active:
                remaining = deadline - self._monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    # ADD 2026-09-06: Counter/gauge를 lock 아래 일관된 snapshot으로 읽는다.
    def metrics_snapshot(self) -> PublisherMetricsSnapshot:
        with self._condition:
            if self._last_delivery_monotonic is None:
                seconds_since_delivery = None
            else:
                seconds_since_delivery = max(
                    0.0,
                    self._monotonic() - self._last_delivery_monotonic,
                )
            return PublisherMetricsSnapshot(
                generated_total=self._generated_total,
                delivered_total=self._delivered_total,
                dropped_total=self._dropped_total,
                delivery_errors_total=self._delivery_errors_total,
                delivery_retries_total=self._delivery_retries_total,
                bridge_up=self._bridge_up,
                pending_events=int(self._pending is not None),
                seconds_since_delivery=seconds_since_delivery,
            )

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._pending is None and self._stopping:
                    return
                payload = self._pending
                self._pending = None
                self._delivery_active = True
                self._condition.notify_all()

            assert payload is not None
            self._deliver_with_retries(payload)

            with self._condition:
                self._delivery_active = False
                self._condition.notify_all()

    def _deliver_with_retries(self, payload: Mapping[str, object]) -> None:
        policy = self._config.delivery
        timeout_seconds = policy.request_timeout_ms / 1000.0
        backoff_ms = policy.initial_backoff_ms

        for attempt in range(policy.max_retries + 1):
            try:
                status = self._transport.post_json(
                    url=self._endpoint_url,
                    payload=payload,
                    timeout_seconds=timeout_seconds,
                )
                if status != EXPECTED_ACCEPTED_STATUS:
                    raise RuntimeError(f"Streaming bridge returned HTTP {status}.")
            except Exception:
                with self._condition:
                    self._delivery_errors_total += 1
                    self._bridge_up = 0
                if attempt >= policy.max_retries:
                    with self._condition:
                        self._dropped_total += 1
                    return

                with self._condition:
                    self._delivery_retries_total += 1
                self._sleep(backoff_ms / 1000.0)
                backoff_ms = min(
                    int(backoff_ms * policy.multiplier),
                    policy.max_backoff_ms,
                )
                continue

            with self._condition:
                self._delivered_total += 1
                self._bridge_up = 1
                self._last_delivery_monotonic = self._monotonic()
            return
