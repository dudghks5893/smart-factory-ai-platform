"""C6-4A RTSP reconnect, backpressure, and observability contract."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast
from urllib.parse import SplitResult, urlsplit, urlunsplit

import yaml

from shared.hashing import is_sha256_digest

DEFAULT_RTSP_RELIABILITY_CONFIG = Path("configs/streaming/yolo_rtsp_reliability.yaml")

EXPECTED_CONTRACT_ID = "c6_4a_yolo_rtsp_reliability_v1"
EXPECTED_C6_3_CLOSURE_COMMIT = "2b23c1993e5a3c71567d7ea1a7c381ffa8754117"
EXPECTED_C6_3_ACCEPTANCE_STATE = "TENSORRT_INT8_STREAMING_ACCEPTED"
EXPECTED_C6_3_ACCEPTANCE_SHA256 = "23b0717b114a579290de56babc5afdd09f6e71c3873b32e1547511c6e251a35e"
EXPECTED_ENGINE_SHA256 = "4f397d59741f4efb7832087030b890a0fe059a657d074a3b07cdeb54493e8971"

EXPECTED_RETRYABLE_EVENTS = ("gst_error", "eos", "frame_timeout")
EXPECTED_STATES = (
    "DISCONNECTED",
    "CONNECTING",
    "STREAMING",
    "STALE",
    "RECONNECTING",
    "FAILED",
)
EXPECTED_COUNTERS = (
    "rtsp_connection_attempts_total",
    "rtsp_reconnects_total",
    "rtsp_frames_received_total",
    "rtsp_frames_processed_total",
    "rtsp_frames_dropped_total",
    "rtsp_errors_total",
    "rtsp_eos_total",
    "rtsp_stale_events_total",
)
EXPECTED_GAUGES = (
    "rtsp_stream_up",
    "rtsp_seconds_since_last_frame",
    "rtsp_current_backoff_seconds",
)


@dataclass(frozen=True)
class C63Identity:
    """Accepted C6-3 streaming backend inherited by RTSP work."""

    closure_commit: str
    acceptance_state: str
    acceptance_sha256: str
    engine_sha256: str

    # ADD 2026-09-04: C6-4가 accepted C6-3 streaming backend identity를 상속하게 한다.
    def validate(self) -> None:
        if (
            self.closure_commit != EXPECTED_C6_3_CLOSURE_COMMIT
            or self.acceptance_state != EXPECTED_C6_3_ACCEPTANCE_STATE
            or self.acceptance_sha256 != EXPECTED_C6_3_ACCEPTANCE_SHA256
            or self.engine_sha256 != EXPECTED_ENGINE_SHA256
        ):
            raise ValueError("C6-4A inherited C6-3 identity changed.")
        for digest in (self.acceptance_sha256, self.engine_sha256):
            if not is_sha256_digest(digest):
                raise ValueError("C6-4A inherited identity contains invalid SHA-256.")


@dataclass(frozen=True)
class RtspSourcePolicy:
    """RTSP source transport and stale-frame boundary."""

    scheme: str
    codec: str
    transport: str
    location_env: str
    latency_ms: int
    connect_timeout_ms: int
    frame_stale_after_ms: int
    drop_on_latency: bool

    # ADD 2026-09-04: 첫 RTSP source를 H264/TCP와 bounded latency로 고정한다.
    def validate(self) -> None:
        if (
            self.scheme != "rtsp"
            or self.codec != "H264"
            or self.transport != "tcp"
            or self.location_env != "SMART_FACTORY_RTSP_URL"
            or type(self.latency_ms) is not int
            or self.latency_ms != 200
            or type(self.connect_timeout_ms) is not int
            or self.connect_timeout_ms != 5000
            or type(self.frame_stale_after_ms) is not int
            or self.frame_stale_after_ms != 1500
            or self.drop_on_latency is not True
        ):
            raise ValueError("C6-4A RTSP source policy changed without review.")


@dataclass(frozen=True)
class RtspBackpressurePolicy:
    """Latest-frame-wins queue inherited from the accepted C6-3 path."""

    mode: str
    queue_max_buffers: int
    queue_leaky: str
    appsink_name: str
    appsink_max_buffers: int
    appsink_drop: bool
    appsink_sync: bool

    # ADD 2026-09-04: RTSP reconnect 중에도 C6-3 latest-frame-wins policy를 유지한다.
    def validate(self) -> None:
        if (
            self.mode != "latest_frame_wins"
            or type(self.queue_max_buffers) is not int
            or self.queue_max_buffers != 1
            or self.queue_leaky != "downstream"
            or self.appsink_name != "framesink"
            or type(self.appsink_max_buffers) is not int
            or self.appsink_max_buffers != 1
            or self.appsink_drop is not True
            or self.appsink_sync is not False
        ):
            raise ValueError("C6-4A backpressure policy changed without review.")


@dataclass(frozen=True)
class RtspReconnectPolicy:
    """Application-level reconnect schedule for RTSP interruptions."""

    retryable_events: tuple[str, ...]
    max_reconnect_attempts: int
    initial_backoff_ms: int
    multiplier: float
    max_backoff_ms: int
    reset_after_healthy_frames: int
    fail_closed_after_exhaustion: bool

    # ADD 2026-09-04: reconnect event와 bounded exponential backoff를 고정한다.
    def validate(self) -> None:
        if (
            self.retryable_events != EXPECTED_RETRYABLE_EVENTS
            or type(self.max_reconnect_attempts) is not int
            or self.max_reconnect_attempts != 5
            or type(self.initial_backoff_ms) is not int
            or self.initial_backoff_ms != 500
            or self.multiplier != 2.0
            or type(self.max_backoff_ms) is not int
            or self.max_backoff_ms != 8000
            or type(self.reset_after_healthy_frames) is not int
            or self.reset_after_healthy_frames != 30
            or self.fail_closed_after_exhaustion is not True
        ):
            raise ValueError("C6-4A reconnect policy changed without review.")


@dataclass(frozen=True)
class RtspObservabilityPolicy:
    """Required health states and metrics for the RTSP lifecycle."""

    states: tuple[str, ...]
    counters: tuple[str, ...]
    gauges: tuple[str, ...]
    redact_credentials: bool

    # ADD 2026-09-04: RTSP health state와 최소 metric surface를 contract로 고정한다.
    def validate(self) -> None:
        if (
            self.states != EXPECTED_STATES
            or self.counters != EXPECTED_COUNTERS
            or self.gauges != EXPECTED_GAUGES
            or self.redact_credentials is not True
        ):
            raise ValueError("C6-4A observability policy changed without review.")


@dataclass(frozen=True)
class RtspFoundationScope:
    """C6-4A is contract-only and must not consume sealed model data."""

    contract_only: bool
    actual_rtsp_used: bool
    tensorrt_inference_used: bool
    deepstream_used: bool
    final_test_used: bool

    # ADD 2026-09-04: C6-4A foundation에서 runtime/model execution을 명시적으로 금지한다.
    def validate(self) -> None:
        if (
            self.contract_only is not True
            or self.actual_rtsp_used is not False
            or self.tensorrt_inference_used is not False
            or self.deepstream_used is not False
            or self.final_test_used is not False
        ):
            raise ValueError("C6-4A foundation scope changed.")


@dataclass(frozen=True)
class YoloRtspReliabilityConfig:
    """Top-level C6-4A reliability contract."""

    schema_version: int
    contract_id: str
    c6_3_identity: C63Identity
    source: RtspSourcePolicy
    backpressure: RtspBackpressurePolicy
    reconnect: RtspReconnectPolicy
    observability: RtspObservabilityPolicy
    foundation: RtspFoundationScope
    config_path: Path

    # ADD 2026-09-04: C6-4A RTSP reliability config 전체를 strict하게 검증한다.
    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported C6-4A schema.")
        if self.contract_id != EXPECTED_CONTRACT_ID:
            raise ValueError("Unexpected C6-4A contract_id.")
        self.c6_3_identity.validate()
        self.source.validate()
        self.backpressure.validate()
        self.reconnect.validate()
        self.observability.validate()
        self.foundation.validate()


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


# ADD 2026-09-04: YAML을 typed C6-4A RTSP reliability config로 로드한다.
def load_rtsp_reliability_config(path: Path) -> YoloRtspReliabilityConfig:
    raw_obj: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = _mapping(raw_obj, label="C6-4A config")
    expected_top = {
        "schema_version",
        "contract_id",
        "c6_3_identity",
        "source",
        "backpressure",
        "reconnect",
        "observability",
        "foundation",
    }
    if set(raw) != expected_top:
        raise ValueError("C6-4A config fields do not match schema.")

    identity_raw = _mapping(raw["c6_3_identity"], label="c6_3_identity")
    source_raw = _mapping(raw["source"], label="source")
    backpressure_raw = _mapping(raw["backpressure"], label="backpressure")
    reconnect_raw = _mapping(raw["reconnect"], label="reconnect")
    observability_raw = _mapping(raw["observability"], label="observability")
    foundation_raw = _mapping(raw["foundation"], label="foundation")

    for nested, cls, label in (
        (identity_raw, C63Identity, "c6_3_identity"),
        (source_raw, RtspSourcePolicy, "source"),
        (backpressure_raw, RtspBackpressurePolicy, "backpressure"),
        (foundation_raw, RtspFoundationScope, "foundation"),
    ):
        _require_fields(nested, cls, label=label)

    _require_fields(reconnect_raw, RtspReconnectPolicy, label="reconnect")
    reconnect_raw["retryable_events"] = _tuple_strings(
        reconnect_raw["retryable_events"],
        label="reconnect.retryable_events",
    )

    _require_fields(observability_raw, RtspObservabilityPolicy, label="observability")
    for name in ("states", "counters", "gauges"):
        observability_raw[name] = _tuple_strings(
            observability_raw[name],
            label=f"observability.{name}",
        )

    config = YoloRtspReliabilityConfig(
        schema_version=raw["schema_version"],
        contract_id=str(raw["contract_id"]),
        c6_3_identity=C63Identity(**identity_raw),
        source=RtspSourcePolicy(**source_raw),
        backpressure=RtspBackpressurePolicy(**backpressure_raw),
        reconnect=RtspReconnectPolicy(**reconnect_raw),
        observability=RtspObservabilityPolicy(**observability_raw),
        foundation=RtspFoundationScope(**foundation_raw),
        config_path=path.resolve(),
    )
    config.validate()
    return config


# ADD 2026-09-04: RTSP URI가 contract scheme을 따르는지 검증한다.
def validate_rtsp_uri(uri: str) -> None:
    parsed = urlsplit(uri)
    if parsed.scheme != "rtsp":
        raise ValueError("C6-4A requires an rtsp:// URI.")
    if not parsed.hostname:
        raise ValueError("C6-4A RTSP URI requires a hostname.")


# ADD 2026-09-04: 로그와 evidence에서 RTSP credentials가 노출되지 않게 한다.
def redact_rtsp_uri(uri: str) -> str:
    validate_rtsp_uri(uri)
    parsed = urlsplit(uri)
    if parsed.username is None:
        return uri

    host = parsed.hostname or ""
    host_text = f"[{host}]" if ":" in host else host
    port_text = f":{parsed.port}" if parsed.port is not None else ""
    netloc = f"***:***@{host_text}{port_text}"
    safe = SplitResult(
        scheme=parsed.scheme,
        netloc=netloc,
        path=parsed.path,
        query=parsed.query,
        fragment=parsed.fragment,
    )
    return urlunsplit(safe)


def _quote_gst(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


# ADD 2026-09-04: H264/TCP RTSP를 accepted BGR/appsink boundary에 연결한다.
def build_rtsp_pipeline(config: YoloRtspReliabilityConfig, *, uri: str) -> str:
    config.validate()
    validate_rtsp_uri(uri)

    source = config.source
    backpressure = config.backpressure
    return " ! ".join(
        (
            (
                f"rtspsrc location={_quote_gst(uri)} protocols={source.transport} "
                f"latency={source.latency_ms} "
                f"drop-on-latency={'true' if source.drop_on_latency else 'false'} "
                f"tcp-timeout={source.connect_timeout_ms * 1000} name=rtsp_source"
            ),
            "application/x-rtp,media=video,encoding-name=H264",
            "rtph264depay",
            "h264parse",
            "decodebin",
            "videoconvert",
            "video/x-raw,format=BGR",
            (
                f"queue max-size-buffers={backpressure.queue_max_buffers} "
                "max-size-bytes=0 max-size-time=0 "
                f"leaky={backpressure.queue_leaky}"
            ),
            (
                f"appsink name={backpressure.appsink_name} emit-signals=false "
                f"max-buffers={backpressure.appsink_max_buffers} "
                f"drop={'true' if backpressure.appsink_drop else 'false'} "
                f"sync={'true' if backpressure.appsink_sync else 'false'} "
                "wait-on-eos=false"
            ),
        )
    )


# ADD 2026-09-04: reconnect attempt별 bounded exponential backoff를 계산한다.
def reconnect_backoff_schedule_ms(policy: RtspReconnectPolicy) -> tuple[int, ...]:
    policy.validate()
    delays: list[int] = []
    for attempt in range(policy.max_reconnect_attempts):
        delay = policy.initial_backoff_ms * (policy.multiplier**attempt)
        delays.append(min(int(delay), policy.max_backoff_ms))
    return tuple(delays)


# ADD 2026-09-04: 마지막 frame 이후 timeout을 stale event로 판정한다.
def is_frame_stale(
    *,
    last_frame_monotonic_s: float,
    now_monotonic_s: float,
    stale_after_ms: int,
) -> bool:
    if now_monotonic_s < last_frame_monotonic_s:
        raise ValueError("Monotonic time cannot move backwards.")
    if stale_after_ms <= 0:
        raise ValueError("stale_after_ms must be positive.")
    elapsed_ms = (now_monotonic_s - last_frame_monotonic_s) * 1000.0
    return elapsed_ms >= stale_after_ms
