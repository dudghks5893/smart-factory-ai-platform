"""C6-1 GStreamer ingress contract for real-time YOLO deployment."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

import yaml

from ml.training.yolo_segmentation import validate_artifact_id
from shared.hashing import is_sha256_digest

DEFAULT_GSTREAMER_CONFIG = Path("configs/streaming/yolo_gstreamer_ingress.yaml")

EXPECTED_ENGINE_SHA256 = "4f397d59741f4efb7832087030b890a0fe059a657d074a3b07cdeb54493e8971"
EXPECTED_POLICY_SHA256 = "938c06a099b681de9ac48d95132f423f5255ba4527f05d3f27f75d9eae5ad56c"
EXPECTED_C5_4_CLOSURE_COMMIT = "88e9b0b2440e99b6dfd2594bdc9a4947eff75187"
EXPECTED_ACCEPTANCE_STATE = "TENSORRT_INT8_PARITY_ACCEPTED"
EXPECTED_PIPELINE_ID = "c6_1_yolo_gstreamer_ingress_v1"

SourceKind = Literal["test", "file"]


@dataclass(frozen=True)
class AcceptedInt8Backend:
    """Exact C5-4 INT8 backend allowed to enter C6 streaming work."""

    precision: str
    engine_sha256: str
    acceptance_policy_sha256: str
    acceptance_state: str
    c5_4_closure_commit: str

    # ADD 2026-09-04: C6가 C5-4에서 accepted 된 exact INT8 backend만 사용하게 한다.
    def validate(self) -> None:
        if (
            self.precision != "int8"
            or self.engine_sha256 != EXPECTED_ENGINE_SHA256
            or self.acceptance_policy_sha256 != EXPECTED_POLICY_SHA256
            or self.acceptance_state != EXPECTED_ACCEPTANCE_STATE
            or self.c5_4_closure_commit != EXPECTED_C5_4_CLOSURE_COMMIT
        ):
            raise ValueError("C6-1 accepted TensorRT INT8 backend identity changed.")
        if not is_sha256_digest(self.engine_sha256):
            raise ValueError("C6-1 engine SHA-256 is invalid.")
        if not is_sha256_digest(self.acceptance_policy_sha256):
            raise ValueError("C6-1 acceptance policy SHA-256 is invalid.")


@dataclass(frozen=True)
class StreamSource:
    """Source boundary supported by the first GStreamer stage."""

    kind: SourceKind
    location: str | None
    test_pattern: str
    is_live: bool

    # ADD 2026-09-04: C6-1은 deterministic test source와 local file source만 허용한다.
    def validate(self) -> None:
        if self.kind not in {"test", "file"}:
            raise ValueError("C6-1 supports only test or file source.")
        if type(self.is_live) is not bool:
            raise TypeError("C6-1 source is_live must be boolean.")
        if not self.test_pattern:
            raise ValueError("C6-1 test_pattern must be non-empty.")
        if self.kind == "test":
            if self.location is not None:
                raise ValueError("C6-1 test source must not define location.")
            if self.is_live is not True:
                raise ValueError("C6-1 test source must behave as a live source.")
        elif self.location is not None and not self.location.strip():
            raise ValueError("C6-1 file source location must be null or non-empty.")


@dataclass(frozen=True)
class FrameContract:
    """CPU frame representation delivered by appsink in C6-1."""

    pixel_format: str
    dtype: str
    layout: str
    contiguous: bool

    # ADD 2026-09-04: Streaming adapter 앞 raw-frame contract를 BGR uint8 HWC로 고정한다.
    def validate(self) -> None:
        if (
            self.pixel_format != "BGR"
            or self.dtype != "uint8"
            or self.layout != "HWC"
            or self.contiguous is not True
        ):
            raise ValueError("C6-1 frame contract changed without review.")


@dataclass(frozen=True)
class LatencyPolicy:
    """Backpressure rule for live inspection: bounded queue, newest frame wins."""

    mode: str
    queue_max_buffers: int
    queue_leaky: str
    appsink_max_buffers: int
    appsink_drop: bool
    appsink_sync: bool

    # ADD 2026-09-04: Live frame backlog 방지를 위해 bounded-drop policy를 고정한다.
    def validate(self) -> None:
        if (
            self.mode != "latest_frame_wins"
            or type(self.queue_max_buffers) is not int
            or self.queue_max_buffers != 1
            or self.queue_leaky != "downstream"
            or type(self.appsink_max_buffers) is not int
            or self.appsink_max_buffers != 1
            or self.appsink_drop is not True
            or self.appsink_sync is not False
        ):
            raise ValueError("C6-1 latency/backpressure policy changed without review.")


@dataclass(frozen=True)
class AppSinkContract:
    """Named appsink consumed by the future Python frame adapter."""

    name: str
    emit_signals: bool

    # ADD 2026-09-04: Pull-based appsink surface를 하나의 stable sink name으로 고정한다.
    def validate(self) -> None:
        if self.name != "framesink" or self.emit_signals is not False:
            raise ValueError("C6-1 appsink contract changed without review.")


@dataclass(frozen=True)
class YoloGStreamerIngressConfig:
    """Top-level C6-1 repository contract."""

    schema_version: int
    pipeline_id: str
    accepted_backend: AcceptedInt8Backend
    source: StreamSource
    frame_contract: FrameContract
    latency_policy: LatencyPolicy
    sink: AppSinkContract
    config_path: Path

    # ADD 2026-09-04: C6-1 GStreamer ingress config의 full repository contract를 검증한다.
    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported C6-1 GStreamer config schema.")
        validate_artifact_id(self.pipeline_id)
        if self.pipeline_id != EXPECTED_PIPELINE_ID:
            raise ValueError("C6-1 pipeline_id changed without review.")
        self.accepted_backend.validate()
        self.source.validate()
        self.frame_contract.validate()
        self.latency_policy.validate()
        self.sink.validate()


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return cast(dict[str, Any], value)


def _require_fields(raw: dict[str, Any], cls: type[Any], *, label: str) -> None:
    expected = {field.name for field in fields(cls)}
    if set(raw) != expected:
        raise ValueError(f"{label} fields do not match schema.")


# ADD 2026-09-04: YAML을 strict typed C6-1 ingress config로 로드한다.
def load_yolo_gstreamer_ingress_config(path: Path) -> YoloGStreamerIngressConfig:
    try:
        raw_obj: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("Cannot read C6-1 GStreamer ingress config.") from exc

    raw = _mapping(raw_obj, label="C6-1 config")
    expected_top = {
        "schema_version",
        "pipeline_id",
        "accepted_backend",
        "source",
        "frame_contract",
        "latency_policy",
        "sink",
    }
    if set(raw) != expected_top:
        raise ValueError("C6-1 config fields do not match schema.")

    backend_raw = _mapping(raw["accepted_backend"], label="accepted_backend")
    source_raw = _mapping(raw["source"], label="source")
    frame_raw = _mapping(raw["frame_contract"], label="frame_contract")
    latency_raw = _mapping(raw["latency_policy"], label="latency_policy")
    sink_raw = _mapping(raw["sink"], label="sink")

    for nested, cls, label in (
        (backend_raw, AcceptedInt8Backend, "accepted_backend"),
        (source_raw, StreamSource, "source"),
        (frame_raw, FrameContract, "frame_contract"),
        (latency_raw, LatencyPolicy, "latency_policy"),
        (sink_raw, AppSinkContract, "sink"),
    ):
        _require_fields(nested, cls, label=label)

    config = YoloGStreamerIngressConfig(
        schema_version=raw["schema_version"],
        pipeline_id=str(raw["pipeline_id"]),
        accepted_backend=AcceptedInt8Backend(**backend_raw),
        source=StreamSource(**source_raw),
        frame_contract=FrameContract(**frame_raw),
        latency_policy=LatencyPolicy(**latency_raw),
        sink=AppSinkContract(**sink_raw),
        config_path=path.resolve(),
    )
    config.validate()
    return config


def _quote_gst(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _file_uri(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme:
        if parsed.scheme != "file":
            raise ValueError("C6-1 file source accepts only local path or file:// URI.")
        return value
    return Path(value).expanduser().resolve().as_uri()


# ADD 2026-09-04: Native Gst import 없이 review 가능한 deterministic pipeline description을 만든다.
def build_yolo_gstreamer_pipeline(
    config: YoloGStreamerIngressConfig,
    *,
    source_override: str | None = None,
) -> str:
    config.validate()
    source = config.source

    if source.kind == "test":
        head = f"videotestsrc is-live=true do-timestamp=true pattern={source.test_pattern}"
    else:
        location = source_override if source_override is not None else source.location
        if location is None:
            raise ValueError("C6-1 file source requires source_override or configured location.")
        head = f"uridecodebin uri={_quote_gst(_file_uri(location))}"

    latency = config.latency_policy
    sink = config.sink
    return " ! ".join(
        (
            head,
            "videoconvert",
            f"video/x-raw,format={config.frame_contract.pixel_format}",
            (
                "queue "
                f"max-size-buffers={latency.queue_max_buffers} "
                "max-size-bytes=0 max-size-time=0 "
                f"leaky={latency.queue_leaky}"
            ),
            (
                f"appsink name={sink.name} "
                f"emit-signals={'true' if sink.emit_signals else 'false'} "
                f"max-buffers={latency.appsink_max_buffers} "
                f"drop={'true' if latency.appsink_drop else 'false'} "
                f"sync={'true' if latency.appsink_sync else 'false'}"
            ),
        )
    )


# ADD 2026-09-04: 로컬 machine에 GStreamer launcher가 있는지만 side-effect 없이 확인한다.
def detect_gstreamer_launcher() -> str | None:
    return shutil.which("gst-launch-1.0")
