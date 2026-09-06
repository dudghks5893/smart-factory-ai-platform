"""C6-6A contract between the DeepStream worker and FastAPI service boundary."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from shared.hashing import is_sha256_digest

DEFAULT_SERVICE_INTEGRATION_CONFIG = Path(
    "configs/streaming/yolo_deepstream_service_integration.json"
)

EXPECTED_CONTRACT_ID = "c6_6a_deepstream_service_integration_v1"
EXPECTED_C6_5_CLOSURE_COMMIT = "3491730c3854997bbe6640ac58b3ab1daf971e85"
EXPECTED_C6_5_FINAL_STATE = "DEEPSTREAM_GPU_NVMM_SEGMENTATION_ACCEPTED"
EXPECTED_ENGINE_SHA256 = "97acd724809f4817ad4a95525a1bafae6294b1a7c99e04c12d451eeda878866e"
EXPECTED_PARSER_SHA256 = "5cc5f9accc465b1c8dc5b8dd59a5983db85bbd39da89dd1322a7bcd910ad2728"
EXPECTED_LABELS_SHA256 = "8b305d45726151909e68c165f5e29321e50bcb2e700ae034780c51b9d16c1559"
EXPECTED_DECODER_ID = "c6_5d_yolo11n_seg_deepstream_decoder_v1"
EXPECTED_CLASSES = {0: "bent", 1: "color", 2: "scratch"}

EXPECTED_INTERNAL_ENDPOINT = "/internal/v1/streaming/known-defects"
EXPECTED_WEBSOCKET_ENDPOINT = "/v1/ws/streaming-known-defects"
EXPECTED_EVENT_TYPE = "streaming_known_defect.observed"
EXPECTED_SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

EXPECTED_COUNTERS = (
    "stream_service_events_generated_total",
    "stream_service_events_delivered_total",
    "stream_service_events_dropped_total",
    "stream_service_delivery_errors_total",
    "stream_service_delivery_retries_total",
)
EXPECTED_GAUGES = (
    "stream_service_bridge_up",
    "stream_service_pending_events",
    "stream_service_seconds_since_delivery",
)


@dataclass(frozen=True)
class C65Identity:
    """Sealed C6-5 identity inherited by service integration."""

    closure_commit: str
    final_state: str
    engine_sha256: str
    parser_sha256: str
    labels_sha256: str
    decoder_id: str

    # ADD 2026-09-06: C6-6가 sealed C6-5 DeepStream segmentation identity를 상속하게 한다.
    def validate(self) -> None:
        if (
            self.closure_commit != EXPECTED_C6_5_CLOSURE_COMMIT
            or self.final_state != EXPECTED_C6_5_FINAL_STATE
            or self.engine_sha256 != EXPECTED_ENGINE_SHA256
            or self.parser_sha256 != EXPECTED_PARSER_SHA256
            or self.labels_sha256 != EXPECTED_LABELS_SHA256
            or self.decoder_id != EXPECTED_DECODER_ID
        ):
            raise ValueError("C6-6A inherited C6-5 identity changed.")
        for digest in (self.engine_sha256, self.parser_sha256, self.labels_sha256):
            if not is_sha256_digest(digest):
                raise ValueError("C6-6A inherited identity contains invalid SHA-256.")


@dataclass(frozen=True)
class ProcessBoundaryPolicy:
    """One-way worker-to-service process boundary."""

    deepstream_worker: str
    service_process: str
    transport: str
    direction: str
    internal_endpoint: str
    websocket_endpoint: str
    reuse_image_upload_endpoint: bool
    reuse_persisted_known_defect_event: bool

    # ADD 2026-09-06: DeepStream worker와 FastAPI를 별도 process 및 event namespace로 분리한다.
    def validate(self) -> None:
        if (
            self.deepstream_worker != "separate_gpu_process"
            or self.service_process != "fastapi"
            or self.transport != "http_json"
            or self.direction != "deepstream_to_service"
            or self.internal_endpoint != EXPECTED_INTERNAL_ENDPOINT
            or self.websocket_endpoint != EXPECTED_WEBSOCKET_ENDPOINT
            or self.reuse_image_upload_endpoint is not False
            or self.reuse_persisted_known_defect_event is not False
        ):
            raise ValueError("C6-6A service process boundary changed without review.")
        if self.internal_endpoint == "/v1/known-defects":
            raise ValueError("Streaming ingest must not reuse the image-upload YOLO endpoint.")
        if self.websocket_endpoint == "/v1/ws/known-defects":
            raise ValueError("Streaming observation must not reuse persisted known-defect events.")


@dataclass(frozen=True)
class EventPolicy:
    """Compact live observation payload policy."""

    schema_version: str
    event_type: str
    classes: dict[int, str]
    diagnostic_confidence: float
    max_instances: int
    include_raw_frame: bool
    include_raw_mask: bool
    include_rtsp_uri: bool

    # ADD 2026-09-06: Live event를 compact metadata로 제한하고 raw media/RTSP URI를 금지한다.
    def validate(self) -> None:
        if (
            self.schema_version != "1"
            or self.event_type != EXPECTED_EVENT_TYPE
            or self.classes != EXPECTED_CLASSES
            or self.diagnostic_confidence != 0.25
            or type(self.max_instances) is not int
            or self.max_instances != 300
            or self.include_raw_frame is not False
            or self.include_raw_mask is not False
            or self.include_rtsp_uri is not False
        ):
            raise ValueError("C6-6A streaming event policy changed without review.")


@dataclass(frozen=True)
class DeliveryPolicy:
    """Non-blocking live-observation delivery boundary."""

    semantics: str
    queue_mode: str
    queue_max_events: int
    request_timeout_ms: int
    max_retries: int
    initial_backoff_ms: int
    multiplier: float
    max_backoff_ms: int
    fail_pipeline_on_delivery_exhaustion: bool

    # ADD 2026-09-06: API/UI 장애가 GPU inference를 막지 않도록 bounded best-effort 전달을 고정한다.
    def validate(self) -> None:
        if (
            self.semantics != "best_effort_live_observation"
            or self.queue_mode != "latest_event_wins"
            or type(self.queue_max_events) is not int
            or self.queue_max_events != 1
            or type(self.request_timeout_ms) is not int
            or self.request_timeout_ms != 1000
            or type(self.max_retries) is not int
            or self.max_retries != 3
            or type(self.initial_backoff_ms) is not int
            or self.initial_backoff_ms != 100
            or self.multiplier != 2.0
            or type(self.max_backoff_ms) is not int
            or self.max_backoff_ms != 1000
            or self.fail_pipeline_on_delivery_exhaustion is not False
        ):
            raise ValueError("C6-6A delivery policy changed without review.")


@dataclass(frozen=True)
class ObservabilityPolicy:
    """Minimum service-bridge metrics contract."""

    counters: tuple[str, ...]
    gauges: tuple[str, ...]
    redact_source_credentials: bool

    # ADD 2026-09-06: Bridge delivery/drop 상태와 credential redaction을 관측 계약으로 고정한다.
    def validate(self) -> None:
        if (
            self.counters != EXPECTED_COUNTERS
            or self.gauges != EXPECTED_GAUGES
            or self.redact_source_credentials is not True
        ):
            raise ValueError("C6-6A observability policy changed without review.")


@dataclass(frozen=True)
class SecurityPolicy:
    """Internal-only C6 bridge security boundary."""

    internal_only: bool
    send_rtsp_uri: bool
    send_credentials: bool
    authentication: str

    # ADD 2026-09-06: C6 내부 bridge payload에서 RTSP URI와 credential 전달을 금지한다.
    def validate(self) -> None:
        if (
            self.internal_only is not True
            or self.send_rtsp_uri is not False
            or self.send_credentials is not False
            or self.authentication != "none_internal_network_only"
        ):
            raise ValueError("C6-6A security policy changed without review.")


@dataclass(frozen=True)
class C66AFoundationScope:
    """C6-6A freezes contracts without starting service or streaming runtime."""

    contract_only: bool
    service_endpoint_implemented: bool
    worker_publisher_implemented: bool
    websocket_channel_implemented: bool
    actual_rtsp_used: bool
    deepstream_runtime_used: bool
    persistence_used: bool
    network_used: bool
    final_test_used: bool

    # ADD 2026-09-06: C6-6A에서 runtime, persistence, network와 final-test 사용을 금지한다.
    def validate(self) -> None:
        if (
            self.contract_only is not True
            or self.service_endpoint_implemented is not False
            or self.worker_publisher_implemented is not False
            or self.websocket_channel_implemented is not False
            or self.actual_rtsp_used is not False
            or self.deepstream_runtime_used is not False
            or self.persistence_used is not False
            or self.network_used is not False
            or self.final_test_used is not False
        ):
            raise ValueError("C6-6A foundation scope changed.")


@dataclass(frozen=True)
class YoloDeepStreamServiceIntegrationConfig:
    """Top-level C6-6A service integration contract."""

    schema_version: int
    contract_id: str
    c6_5_identity: C65Identity
    process_boundary: ProcessBoundaryPolicy
    event: EventPolicy
    delivery: DeliveryPolicy
    observability: ObservabilityPolicy
    security: SecurityPolicy
    foundation: C66AFoundationScope
    config_path: Path

    # ADD 2026-09-06: C6-6A service integration config 전체를 strict하게 검증한다.
    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported C6-6A schema.")
        if self.contract_id != EXPECTED_CONTRACT_ID:
            raise ValueError("Unexpected C6-6A contract_id.")
        self.c6_5_identity.validate()
        self.process_boundary.validate()
        self.event.validate()
        self.delivery.validate()
        self.observability.validate()
        self.security.validate()
        self.foundation.validate()


@dataclass(frozen=True)
class StreamingDefectInstance:
    """Compact DeepStream instance summary without raw mask pixels."""

    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    mask_pixel_count: int
    mask_area_ratio: float

    # ADD 2026-09-06: Streaming instance class, bbox, confidence와 compact mask area를 검증한다.
    def validate(self, *, image_width: int, image_height: int) -> None:
        if EXPECTED_CLASSES.get(self.class_id) != self.class_name:
            raise ValueError("Streaming instance class mapping is invalid.")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Streaming instance confidence must be finite and in [0, 1].")
        if len(self.box_xyxy) != 4 or not all(math.isfinite(value) for value in self.box_xyxy):
            raise ValueError("Streaming instance bbox must contain four finite coordinates.")
        x1, y1, x2, y2 = self.box_xyxy
        if not (0.0 <= x1 <= x2 <= image_width and 0.0 <= y1 <= y2 <= image_height):
            raise ValueError("Streaming instance bbox is outside the image bounds.")
        image_area = image_width * image_height
        if not 0 < self.mask_pixel_count <= image_area:
            raise ValueError("Streaming mask pixel count is outside the image area.")
        if not math.isfinite(self.mask_area_ratio) or not 0.0 < self.mask_area_ratio <= 1.0:
            raise ValueError("Streaming mask area ratio must be finite and in (0, 1].")
        expected_ratio = self.mask_pixel_count / image_area
        if not math.isclose(self.mask_area_ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Streaming mask area ratio does not match its pixel count.")

    def to_payload(self) -> dict[str, object]:
        """Return the compact JSON-compatible instance payload."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "box": {
                "x_min": self.box_xyxy[0],
                "y_min": self.box_xyxy[1],
                "x_max": self.box_xyxy[2],
                "y_max": self.box_xyxy[3],
            },
            "mask": {
                "pixel_count": self.mask_pixel_count,
                "area_ratio": self.mask_area_ratio,
            },
        }


@dataclass(frozen=True)
class StreamingKnownDefectObservation:
    """One versioned live DeepStream observation crossing the service bridge."""

    observation_id: UUID
    source_id: str
    stream_session_id: UUID
    frame_number: int
    pts_ns: int
    observed_at: datetime
    image_width: int
    image_height: int
    decoder_id: str
    engine_sha256: str
    parser_sha256: str
    labels_sha256: str
    diagnostic_confidence: float
    instances: tuple[StreamingDefectInstance, ...]

    # ADD 2026-09-06: Worker event identity와 runtime provenance를 검증한다.
    def validate(self, config: YoloDeepStreamServiceIntegrationConfig) -> None:
        config.validate()
        if not isinstance(self.observation_id, UUID) or not isinstance(
            self.stream_session_id, UUID
        ):
            raise ValueError("Streaming observation IDs must be UUID values.")
        if not EXPECTED_SOURCE_ID_PATTERN.fullmatch(self.source_id):
            raise ValueError("source_id must be an opaque non-secret identifier.")
        if type(self.frame_number) is not int or self.frame_number < 0:
            raise ValueError("frame_number must be a non-negative integer.")
        if type(self.pts_ns) is not int or self.pts_ns < 0:
            raise ValueError("pts_ns must be a non-negative integer.")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() != timedelta(0):
            raise ValueError("observed_at must be timezone-aware UTC.")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Streaming image dimensions must be positive.")
        if (
            self.decoder_id != config.c6_5_identity.decoder_id
            or self.engine_sha256 != config.c6_5_identity.engine_sha256
            or self.parser_sha256 != config.c6_5_identity.parser_sha256
            or self.labels_sha256 != config.c6_5_identity.labels_sha256
        ):
            raise ValueError("Streaming observation runtime identity changed.")
        if self.diagnostic_confidence != config.event.diagnostic_confidence:
            raise ValueError("Streaming observation confidence contract changed.")
        if len(self.instances) > config.event.max_instances:
            raise ValueError("Streaming observation exceeds max_instances.")
        for instance in self.instances:
            instance.validate(image_width=self.image_width, image_height=self.image_height)

    def to_payload(self, config: YoloDeepStreamServiceIntegrationConfig) -> dict[str, object]:
        """Validate and return the compact JSON-compatible live event."""
        self.validate(config)
        return {
            "schema_version": config.event.schema_version,
            "type": config.event.event_type,
            "observation": {
                "observation_id": str(self.observation_id),
                "source_id": self.source_id,
                "stream_session_id": str(self.stream_session_id),
                "frame_number": self.frame_number,
                "pts_ns": self.pts_ns,
                "observed_at": self.observed_at.isoformat().replace("+00:00", "Z"),
                "image": {"width": self.image_width, "height": self.image_height},
                "runtime": {
                    "decoder_id": self.decoder_id,
                    "engine_sha256": self.engine_sha256,
                    "parser_sha256": self.parser_sha256,
                    "labels_sha256": self.labels_sha256,
                },
                "diagnostic_confidence": self.diagnostic_confidence,
                "instances": [instance.to_payload() for instance in self.instances],
            },
        }


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return cast(dict[str, Any], value)


def _require_fields(raw: dict[str, Any], cls: type[Any], *, label: str) -> None:
    expected = {field.name for field in fields(cls)}
    if set(raw) != expected:
        raise ValueError(f"{label} fields do not match schema.")


def _tuple_strings(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a string list.")
    return tuple(value)


def _classes(value: object) -> dict[int, str]:
    if not isinstance(value, dict):
        raise ValueError("event.classes must be a mapping.")
    normalized: dict[int, str] = {}
    for key, class_name in value.items():
        if not isinstance(key, str) or not key.isdigit() or not isinstance(class_name, str):
            raise ValueError("event.classes must map JSON integer-string IDs to strings.")
        normalized[int(key)] = class_name
    return normalized


# ADD 2026-09-06: JSON을 typed C6-6A service integration config로 로드한다.
def load_service_integration_config(
    path: Path = DEFAULT_SERVICE_INTEGRATION_CONFIG,
) -> YoloDeepStreamServiceIntegrationConfig:
    raw_obj: object = json.loads(path.read_text(encoding="utf-8"))
    raw = _mapping(raw_obj, label="C6-6A config")
    expected_top = {
        "schema_version",
        "contract_id",
        "c6_5_identity",
        "process_boundary",
        "event",
        "delivery",
        "observability",
        "security",
        "foundation",
    }
    if set(raw) != expected_top:
        raise ValueError("C6-6A config fields do not match schema.")

    identity_raw = _mapping(raw["c6_5_identity"], label="c6_5_identity")
    boundary_raw = _mapping(raw["process_boundary"], label="process_boundary")
    event_raw = _mapping(raw["event"], label="event")
    delivery_raw = _mapping(raw["delivery"], label="delivery")
    observability_raw = _mapping(raw["observability"], label="observability")
    security_raw = _mapping(raw["security"], label="security")
    foundation_raw = _mapping(raw["foundation"], label="foundation")

    _require_fields(identity_raw, C65Identity, label="c6_5_identity")
    _require_fields(boundary_raw, ProcessBoundaryPolicy, label="process_boundary")
    _require_fields(event_raw, EventPolicy, label="event")
    _require_fields(delivery_raw, DeliveryPolicy, label="delivery")
    _require_fields(observability_raw, ObservabilityPolicy, label="observability")
    _require_fields(security_raw, SecurityPolicy, label="security")
    _require_fields(foundation_raw, C66AFoundationScope, label="foundation")

    event_values = dict(event_raw)
    event_values["classes"] = _classes(event_raw["classes"])
    observability_values = dict(observability_raw)
    observability_values["counters"] = _tuple_strings(
        observability_raw["counters"], label="observability.counters"
    )
    observability_values["gauges"] = _tuple_strings(
        observability_raw["gauges"], label="observability.gauges"
    )

    config = YoloDeepStreamServiceIntegrationConfig(
        schema_version=raw["schema_version"],
        contract_id=raw["contract_id"],
        c6_5_identity=C65Identity(**identity_raw),
        process_boundary=ProcessBoundaryPolicy(**boundary_raw),
        event=EventPolicy(**event_values),
        delivery=DeliveryPolicy(**delivery_raw),
        observability=ObservabilityPolicy(**observability_values),
        security=SecurityPolicy(**security_raw),
        foundation=C66AFoundationScope(**foundation_raw),
        config_path=path,
    )
    config.validate()
    return config
