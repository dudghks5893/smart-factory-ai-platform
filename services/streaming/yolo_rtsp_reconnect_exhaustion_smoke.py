"""Canonical C6-4C RTSP reconnect-exhaustion / fail-closed smoke runtime."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import yaml

from services.streaming import yolo_rtsp_fault_injection_smoke as recovery_smoke
from services.streaming.gstreamer_frames import load_gstreamer_modules
from services.streaming.yolo_rtsp_reliability import (
    DEFAULT_RTSP_RELIABILITY_CONFIG,
    build_rtsp_pipeline,
    load_rtsp_reliability_config,
    reconnect_backoff_schedule_ms,
    redact_rtsp_uri,
)
from shared.hashing import is_sha256_digest

DEFAULT_RTSP_RECONNECT_EXHAUSTION_CONFIG = Path(
    "configs/streaming/yolo_rtsp_reconnect_exhaustion_smoke.yaml"
)
EXPECTED_SMOKE_ID = "c6_4c_rtsp_reconnect_exhaustion_v1"
EXPECTED_C6_4B_ACCEPTANCE_COMMIT = "028c86264aef6859ca4b68cf5d561f26fa341f95"
EXPECTED_C6_4B_SMOKE_SHA256 = "f6f140bfee3ac6cf71ab71f5e03c8043ad539f9a775cce6146f593f6b209683e"
EXPECTED_C6_4B_ARCHIVE_SHA256 = "a6497cf91438981c9fbf23f65e391f030d1b19cfc09815d918e32dc163b1ab17"
EXPECTED_BACKOFF_MS = (500, 1000, 2000, 4000, 8000)
SMOKE_STATE = "RTSP_RECONNECT_EXHAUSTION_COMPLETED"


@dataclass(frozen=True)
class ExhaustionRuntimeConfig:
    """Frozen C6-4C reconnect-exhaustion boundaries."""

    initial_healthy_frames: int
    expected_max_reconnect_attempts: int
    expected_backoff_ms: tuple[int, ...]
    connection_failure_timeout_ms: int
    fault_kind: str
    keep_fixture_offline_after_fault: bool

    # ADD 2026-09-04: C6-4C exhaustion attempt 수와 backoff schedule을 exact하게 고정한다.
    def validate(self) -> None:
        if type(self.initial_healthy_frames) is not int or self.initial_healthy_frames != 10:
            raise ValueError("C6-4C initial healthy frame boundary changed.")
        if (
            type(self.expected_max_reconnect_attempts) is not int
            or self.expected_max_reconnect_attempts != 5
        ):
            raise ValueError("C6-4C reconnect attempt boundary changed.")
        if self.expected_backoff_ms != EXPECTED_BACKOFF_MS:
            raise ValueError("C6-4C expected backoff schedule changed.")
        if (
            type(self.connection_failure_timeout_ms) is not int
            or self.connection_failure_timeout_ms != 5000
        ):
            raise ValueError("C6-4C connection failure timeout changed.")
        if self.fault_kind != "terminate_fixture_server_process_and_keep_offline":
            raise ValueError("C6-4C fault kind changed.")
        if self.keep_fixture_offline_after_fault is not True:
            raise ValueError("C6-4C fixture must remain offline after fault.")


@dataclass(frozen=True)
class ExhaustionScopeConfig:
    """C6-4C stays inside localhost RTSP reliability validation."""

    localhost_rtsp_used: bool
    external_camera_used: bool
    tensorrt_inference_used: bool
    deepstream_used: bool
    final_test_used: bool

    # ADD 2026-09-04: C6-4C가 model/DeepStream/final-test boundary를 넘지 않게 한다.
    def validate(self) -> None:
        if (
            self.localhost_rtsp_used is not True
            or self.external_camera_used is not False
            or self.tensorrt_inference_used is not False
            or self.deepstream_used is not False
            or self.final_test_used is not False
        ):
            raise ValueError("C6-4C smoke scope changed.")


@dataclass(frozen=True)
class RtspReconnectExhaustionConfig:
    """Top-level canonical C6-4C configuration."""

    schema_version: int
    smoke_id: str
    required_c6_4b_acceptance_commit: str
    required_c6_4b_smoke_sha256: str
    required_c6_4b_archive_sha256: str
    runtime: ExhaustionRuntimeConfig
    output_root: Path
    scope: ExhaustionScopeConfig
    config_path: Path

    # ADD 2026-09-04: C6-4C config가 accepted C6-4B lineage와 frozen policy를 상속하게 한다.
    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported C6-4C schema.")
        if self.smoke_id != EXPECTED_SMOKE_ID:
            raise ValueError("Unexpected C6-4C smoke_id.")
        if self.required_c6_4b_acceptance_commit != EXPECTED_C6_4B_ACCEPTANCE_COMMIT:
            raise ValueError("C6-4C required C6-4B acceptance commit changed.")
        if self.required_c6_4b_smoke_sha256 != EXPECTED_C6_4B_SMOKE_SHA256:
            raise ValueError("C6-4C required C6-4B smoke SHA changed.")
        if self.required_c6_4b_archive_sha256 != EXPECTED_C6_4B_ARCHIVE_SHA256:
            raise ValueError("C6-4C required C6-4B archive SHA changed.")
        for digest in (
            self.required_c6_4b_smoke_sha256,
            self.required_c6_4b_archive_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C6-4C inherited evidence identity contains invalid SHA-256.")
        self.runtime.validate()
        self.scope.validate()
        if not self.output_root.parts:
            raise ValueError("C6-4C output_root must be non-empty.")


@dataclass(frozen=True)
class ReconnectAttemptEvidence:
    """One failed reconnect attempt after the fixture is intentionally kept offline."""

    attempt: int
    requested_backoff_ms: int
    actual_backoff_ms: float
    failure_event: str
    failure_detection_ms: float


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return cast(dict[str, Any], value)


def _require_fields(raw: dict[str, Any], cls: type[Any], *, label: str) -> None:
    expected = {field.name for field in fields(cls)}
    if set(raw) != expected:
        raise ValueError(f"{label} fields do not match schema.")


def _tuple_ints(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty integer list.")
    if not all(type(item) is int for item in value):
        raise ValueError(f"{label} must contain only integers.")
    return tuple(cast(list[int], value))


# ADD 2026-09-04: YAML을 typed C6-4C reconnect-exhaustion config로 로드한다.
def load_rtsp_reconnect_exhaustion_config(path: Path) -> RtspReconnectExhaustionConfig:
    raw_obj: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = _mapping(raw_obj, label="C6-4C config")
    if set(raw) != {
        "schema_version",
        "smoke_id",
        "required_c6_4b_acceptance_commit",
        "required_c6_4b_smoke_sha256",
        "required_c6_4b_archive_sha256",
        "runtime",
        "output_root",
        "scope",
    }:
        raise ValueError("C6-4C config fields do not match schema.")

    runtime_raw = _mapping(raw["runtime"], label="runtime")
    scope_raw = _mapping(raw["scope"], label="scope")
    _require_fields(runtime_raw, ExhaustionRuntimeConfig, label="runtime")
    _require_fields(scope_raw, ExhaustionScopeConfig, label="scope")
    runtime_raw["expected_backoff_ms"] = _tuple_ints(
        runtime_raw["expected_backoff_ms"],
        label="runtime.expected_backoff_ms",
    )

    config = RtspReconnectExhaustionConfig(
        schema_version=raw["schema_version"],
        smoke_id=str(raw["smoke_id"]),
        required_c6_4b_acceptance_commit=str(raw["required_c6_4b_acceptance_commit"]),
        required_c6_4b_smoke_sha256=str(raw["required_c6_4b_smoke_sha256"]),
        required_c6_4b_archive_sha256=str(raw["required_c6_4b_archive_sha256"]),
        runtime=ExhaustionRuntimeConfig(**runtime_raw),
        output_root=Path(str(raw["output_root"])),
        scope=ExhaustionScopeConfig(**scope_raw),
        config_path=path.resolve(),
    )
    config.validate()
    return config


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=repo, text=True).strip()


# ADD 2026-09-04: C6-4B acceptance commit이 canonical C6-4C run의 ancestor인지 검증한다.
def _require_c6_4b_acceptance_ancestor(repo: Path, required_commit: str) -> None:
    completed = subprocess.run(
        ("git", "merge-base", "--is-ancestor", required_commit, "HEAD"),
        cwd=repo,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("C6-4C requires the accepted C6-4B lineage.")


# ADD 2026-09-04: Frozen reconnect policy를 ordered attempt/backoff plan으로 만든다.
def build_reconnect_exhaustion_plan() -> tuple[tuple[int, int], ...]:
    reliability = load_rtsp_reliability_config(DEFAULT_RTSP_RELIABILITY_CONFIG)
    schedule = reconnect_backoff_schedule_ms(reliability.reconnect)
    if schedule != EXPECTED_BACKOFF_MS:
        raise RuntimeError("C6-4C inherited reconnect schedule changed.")
    return tuple(enumerate(schedule, start=1))


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.1)
        return probe.connect_ex((host, port)) == 0


# ADD 2026-09-04: Offline RTSP endpoint에 실제 GStreamer reconnect를 시도하고 실패 event를 수집한다.
def _attempt_offline_rtsp_connection(
    *,
    uri: str,
    timeout_ms: int,
) -> tuple[str, float]:
    gst, _ = load_gstreamer_modules()
    gst.init(None)
    reliability = load_rtsp_reliability_config(DEFAULT_RTSP_RELIABILITY_CONFIG)
    pipeline = gst.parse_launch(build_rtsp_pipeline(reliability, uri=uri))
    bus = pipeline.get_bus()
    started = time.monotonic()
    try:
        pipeline.set_state(gst.State.PLAYING)

        # MODIFY 2026-09-04: synchronous state-change FAILURE도 bus ERROR/EOS로
        # 정규화해 frozen retryable event surface로 판정한다.
        message = bus.timed_pop_filtered(
            timeout_ms * gst.MSECOND,
            gst.MessageType.ERROR | gst.MessageType.EOS,
        )
        if message is None:
            raise RuntimeError("C6-4C offline reconnect attempt did not fail within timeout.")

        elapsed_ms = (time.monotonic() - started) * 1000.0
        if message.type == gst.MessageType.ERROR:
            return "gst_error", elapsed_ms
        if message.type == gst.MessageType.EOS:
            return "eos", elapsed_ms
        raise RuntimeError("C6-4C received an unexpected GStreamer message.")
    finally:
        pipeline.set_state(gst.State.NULL)


def _record_failure_event(
    tracker: recovery_smoke.StreamHealthTracker,
    event: str,
) -> None:
    if event == "gst_error":
        tracker.errors_total += 1
    elif event == "eos":
        tracker.eos_total += 1
    else:
        raise RuntimeError(f"C6-4C unexpected reconnect failure event: {event}")


# ADD 2026-09-04: 5회 reconnect exhaustion 후 FAILED fail-closed evidence를 생성한다.
def run_rtsp_reconnect_exhaustion_smoke(
    config: RtspReconnectExhaustionConfig,
    *,
    repo: Path,
) -> Path:
    config.validate()
    commit = _git_output(repo, "rev-parse", "HEAD")
    if _git_output(repo, "status", "--porcelain"):
        raise RuntimeError("C6-4C canonical smoke requires a clean working tree.")
    _require_c6_4b_acceptance_ancestor(repo, config.required_c6_4b_acceptance_commit)

    reliability = load_rtsp_reliability_config(DEFAULT_RTSP_RELIABILITY_CONFIG)
    if (
        reliability.reconnect.max_reconnect_attempts
        != config.runtime.expected_max_reconnect_attempts
    ):
        raise RuntimeError("C6-4C reconnect attempt budget no longer matches C6-4A.")
    if reliability.reconnect.fail_closed_after_exhaustion is not True:
        raise RuntimeError("C6-4C requires fail_closed_after_exhaustion=true.")

    recovery_config = recovery_smoke.load_rtsp_fault_injection_config(
        recovery_smoke.DEFAULT_RTSP_FAULT_INJECTION_CONFIG
    )
    if recovery_config.runtime.initial_healthy_frames != config.runtime.initial_healthy_frames:
        raise RuntimeError("C6-4C initial healthy phase no longer matches accepted C6-4B.")
    if recovery_config.fixture_server.host != "127.0.0.1":
        raise RuntimeError("C6-4C fixture must remain localhost.")

    plan = build_reconnect_exhaustion_plan()
    fixture = recovery_config.fixture_server
    port = recovery_smoke._find_free_local_port(fixture.host)
    uri = f"rtsp://{fixture.host}:{port}{fixture.mount}"

    tracker = recovery_smoke.StreamHealthTracker()
    transitions = ["DISCONNECTED"]
    initial_hashes: list[str] = []
    injected_fault_event = ""
    injected_fault_detection_ms = 0.0
    attempts: list[ReconnectAttemptEvidence] = []

    pipeline: Any | None = None
    server: subprocess.Popen[str] | None = None
    server_log: Any | None = None

    with tempfile.TemporaryDirectory(prefix="c6_4c_rtsp_") as tmp:
        tmp_path = Path(tmp)
        server_log_path = tmp_path / "fixture_server.log"

        try:
            tracker.transition("CONNECTING", transitions)
            tracker.connection_attempts_total += 1
            server, server_log = recovery_smoke._start_fixture_server(
                recovery_config,
                port=port,
                log_path=server_log_path,
            )
            gst, pipeline, appsink = recovery_smoke._create_client(
                recovery_config,
                uri=uri,
            )
            initial_hashes, last_frame_monotonic = recovery_smoke._receive_frames(
                gst=gst,
                appsink=appsink,
                count=config.runtime.initial_healthy_frames,
                timeout_ms=recovery_config.runtime.sample_timeout_ms,
                config=recovery_config,
                tracker=tracker,
            )
            tracker.transition("STREAMING", transitions)

            recovery_smoke._stop_fixture_server(
                server,
                server_log,
                host=fixture.host,
                port=port,
                timeout_ms=recovery_config.runtime.server_ready_timeout_ms,
            )
            server = None
            server_log = None
            if _port_is_open(fixture.host, port):
                raise RuntimeError("C6-4C fixture port remained open after injected fault.")

            injected_fault_event, injected_fault_detection_ms = recovery_smoke._detect_interruption(
                gst=gst,
                pipeline=pipeline,
                appsink=appsink,
                last_frame_monotonic=last_frame_monotonic,
                config=recovery_config,
                tracker=tracker,
            )
            if injected_fault_event == "frame_timeout":
                tracker.transition("STALE", transitions)

            pipeline.set_state(gst.State.NULL)
            pipeline = None
            tracker.transition("RECONNECTING", transitions)

            for attempt_number, requested_backoff_ms in plan:
                if _port_is_open(fixture.host, port):
                    raise RuntimeError("C6-4C fixture unexpectedly restarted during exhaustion.")

                tracker.begin_reconnect()
                tracker.connection_attempts_total += 1
                tracker.current_backoff_seconds = requested_backoff_ms / 1000.0

                backoff_started = time.monotonic()
                time.sleep(requested_backoff_ms / 1000.0)
                actual_backoff_ms = (time.monotonic() - backoff_started) * 1000.0
                if actual_backoff_ms < requested_backoff_ms:
                    raise RuntimeError("C6-4C reconnect backoff was shorter than frozen policy.")

                failure_event, failure_detection_ms = _attempt_offline_rtsp_connection(
                    uri=uri,
                    timeout_ms=config.runtime.connection_failure_timeout_ms,
                )
                _record_failure_event(tracker, failure_event)
                attempts.append(
                    ReconnectAttemptEvidence(
                        attempt=attempt_number,
                        requested_backoff_ms=requested_backoff_ms,
                        actual_backoff_ms=actual_backoff_ms,
                        failure_event=failure_event,
                        failure_detection_ms=failure_detection_ms,
                    )
                )

            observed_attempts = tuple(item.attempt for item in attempts)
            observed_requested_backoff_ms = tuple(item.requested_backoff_ms for item in attempts)
            if observed_attempts != tuple(
                range(1, reliability.reconnect.max_reconnect_attempts + 1)
            ):
                raise RuntimeError("C6-4C reconnect attempt sequence changed.")
            if observed_requested_backoff_ms != EXPECTED_BACKOFF_MS:
                raise RuntimeError("C6-4C reconnect backoff sequence changed.")
            if any(item.actual_backoff_ms < item.requested_backoff_ms for item in attempts):
                raise RuntimeError("C6-4C observed a shorter-than-policy reconnect backoff.")
            if any(
                item.failure_event not in reliability.reconnect.retryable_events
                for item in attempts
            ):
                raise RuntimeError("C6-4C observed a non-retryable reconnect failure event.")
            if len(attempts) != reliability.reconnect.max_reconnect_attempts:
                raise RuntimeError("C6-4C did not consume the full reconnect budget.")
            if (
                tracker.reconnect_attempts_since_healthy_reset
                != reliability.reconnect.max_reconnect_attempts
            ):
                raise RuntimeError("C6-4C reconnect budget counter did not reach exhaustion.")

            tracker.current_backoff_seconds = 0.0
            tracker.transition("FAILED", transitions)

        finally:
            if pipeline is not None:
                try:
                    cleanup_gst, _ = load_gstreamer_modules()
                    pipeline.set_state(cleanup_gst.State.NULL)
                except Exception:
                    pass
            if server is not None and server_log is not None:
                try:
                    recovery_smoke._stop_fixture_server(
                        server,
                        server_log,
                        host=fixture.host,
                        port=port,
                        timeout_ms=recovery_config.runtime.server_ready_timeout_ms,
                    )
                except Exception:
                    pass

    if tracker.state != "FAILED":
        raise RuntimeError("C6-4C did not fail closed to FAILED.")
    expected_connection_attempts = 1 + reliability.reconnect.max_reconnect_attempts
    if tracker.connection_attempts_total != expected_connection_attempts:
        raise RuntimeError("C6-4C connection attempt counter changed.")
    if tracker.reconnects_total != reliability.reconnect.max_reconnect_attempts:
        raise RuntimeError("C6-4C reconnect counter changed.")
    if tracker.healthy_frames_since_reconnect != 0:
        raise RuntimeError("C6-4C must not observe healthy frames during exhaustion.")
    if tracker.frames_received_total < config.runtime.initial_healthy_frames:
        raise RuntimeError("C6-4C initial healthy frame boundary was not reached.")
    if tracker.frames_processed_total != tracker.frames_received_total:
        raise RuntimeError("C6-4C processed/received frame counters diverged.")
    if tracker.frames_dropped_total != 0:
        raise RuntimeError("C6-4C observed dropped frames before fail-closed.")

    fixture_port_open = _port_is_open(fixture.host, port)
    client_pipeline_active = pipeline is not None
    fixture_server_active = server is not None and server.poll() is None
    additional_attempts_after_exhaustion = max(
        0,
        tracker.reconnects_total - reliability.reconnect.max_reconnect_attempts,
    )
    rtsp_stream_up = 1 if tracker.state == "STREAMING" else 0
    budget_exhausted = (
        len(attempts) == reliability.reconnect.max_reconnect_attempts
        and tracker.reconnect_attempts_since_healthy_reset
        == reliability.reconnect.max_reconnect_attempts
    )

    if fixture_port_open:
        raise RuntimeError("C6-4C fail-closed endpoint must remain offline.")
    if client_pipeline_active:
        raise RuntimeError("C6-4C fail-closed client pipeline remained active.")
    if fixture_server_active:
        raise RuntimeError("C6-4C fail-closed fixture server remained active.")
    if additional_attempts_after_exhaustion != 0:
        raise RuntimeError("C6-4C performed reconnect attempts after exhaustion.")
    if rtsp_stream_up != 0:
        raise RuntimeError("C6-4C FAILED state must report rtsp_stream_up=0.")
    if not budget_exhausted:
        raise RuntimeError("C6-4C reconnect budget did not reach runtime exhaustion.")

    seconds_since_last_frame = tracker.seconds_since_last_frame(now_monotonic_s=time.monotonic())
    output_dir = repo / config.output_root / config.smoke_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "smoke.json"

    payload = {
        "schema_version": 1,
        "stage": "C6-4C",
        "state": SMOKE_STATE,
        "created_at": datetime.now(UTC).isoformat(),
        "repository": {
            "git_commit": commit,
            "working_tree_dirty_before_run": False,
            "required_c6_4b_acceptance_commit": config.required_c6_4b_acceptance_commit,
            "required_c6_4b_smoke_sha256": config.required_c6_4b_smoke_sha256,
            "required_c6_4b_archive_sha256": config.required_c6_4b_archive_sha256,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "python_packages": {
            "numpy": version("numpy"),
            "pygobject": version("PyGObject"),
            "pycairo": version("pycairo"),
        },
        "gstreamer": {
            "version": load_gstreamer_modules()[0].version_string(),
            "transport": reliability.source.transport,
            "codec": reliability.source.codec,
            "backpressure": reliability.backpressure.mode,
        },
        "fixture": {
            "type": "localhost_gst_rtsp_server",
            "uri_redacted": redact_rtsp_uri(uri),
            "pattern": fixture.pattern,
            "width": fixture.width,
            "height": fixture.height,
            "framerate": fixture.framerate,
            "fault_kind": config.runtime.fault_kind,
            "kept_offline_after_fault": True,
        },
        "normal_phase": {
            "healthy_frames_required": config.runtime.initial_healthy_frames,
            "unique_frame_sha256_count": len(set(initial_hashes)),
        },
        "injected_fault": {
            "detected_event": injected_fault_event,
            "detection_ms": injected_fault_detection_ms,
        },
        "reconnect_exhaustion": {
            "max_reconnect_attempts": reliability.reconnect.max_reconnect_attempts,
            "expected_backoff_ms": list(EXPECTED_BACKOFF_MS),
            "attempts": [
                {
                    "attempt": item.attempt,
                    "requested_backoff_ms": item.requested_backoff_ms,
                    "actual_backoff_ms": item.actual_backoff_ms,
                    "failure_event": item.failure_event,
                    "failure_detection_ms": item.failure_detection_ms,
                }
                for item in attempts
            ],
            "budget_exhausted": budget_exhausted,
            "reconnect_attempts_since_healthy_reset": (
                tracker.reconnect_attempts_since_healthy_reset
            ),
            "healthy_frames_since_reconnect": tracker.healthy_frames_since_reconnect,
        },
        "fail_closed": {
            "policy_enabled": reliability.reconnect.fail_closed_after_exhaustion,
            "final_state": tracker.state,
            "rtsp_stream_up": rtsp_stream_up,
            "client_pipeline_active": client_pipeline_active,
            "fixture_server_active": fixture_server_active,
            "fixture_port_open": fixture_port_open,
            "additional_attempts_after_exhaustion": additional_attempts_after_exhaustion,
        },
        "observability": {
            "state_transitions": transitions,
            "final_state": tracker.state,
            "rtsp_connection_attempts_total": tracker.connection_attempts_total,
            "rtsp_reconnects_total": tracker.reconnects_total,
            "rtsp_frames_received_total": tracker.frames_received_total,
            "rtsp_frames_processed_total": tracker.frames_processed_total,
            "rtsp_frames_dropped_total": tracker.frames_dropped_total,
            "rtsp_errors_total": tracker.errors_total,
            "rtsp_eos_total": tracker.eos_total,
            "rtsp_stale_events_total": tracker.stale_events_total,
            "rtsp_stream_up": rtsp_stream_up,
            "rtsp_seconds_since_last_frame": seconds_since_last_frame,
            "rtsp_current_backoff_seconds": tracker.current_backoff_seconds,
        },
        "scope": {
            "localhost_rtsp_used": True,
            "external_camera_used": False,
            "tensorrt_inference_used": False,
            "deepstream_used": False,
            "final_test_used": False,
        },
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
