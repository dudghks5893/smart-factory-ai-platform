"""Canonical C6-4B localhost RTSP fault-injection smoke runtime."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from services.streaming.gstreamer_frames import (
    gst_sample_to_bgr_numpy,
    load_gstreamer_modules,
)
from services.streaming.yolo_rtsp_reliability import (
    DEFAULT_RTSP_RELIABILITY_CONFIG,
    build_rtsp_pipeline,
    load_rtsp_reliability_config,
    reconnect_backoff_schedule_ms,
)

DEFAULT_RTSP_FAULT_INJECTION_CONFIG = Path("configs/streaming/yolo_rtsp_fault_injection_smoke.yaml")
EXPECTED_SMOKE_ID = "c6_4b_rtsp_fault_injection_smoke_v1"
EXPECTED_C6_4A_COMMIT = "47d782e0f33f218f5bd10508840c2126837f1ca5"
SMOKE_STATE = "RTSP_FAULT_INJECTION_SMOKE_COMPLETED"


@dataclass(frozen=True)
class FixtureServerConfig:
    """Synthetic localhost RTSP fixture settings."""

    host: str
    mount: str
    pattern: str
    width: int
    height: int
    framerate: int
    encoder: str
    bitrate_kbps: int
    key_int_max: int

    # ADD 2026-09-04: C6-4B localhost H264 fixture server boundary를 고정한다.
    def validate(self) -> None:
        if self.host != "127.0.0.1":
            raise ValueError("C6-4B fixture host must remain localhost.")
        if self.mount != "/inspection":
            raise ValueError("C6-4B fixture mount changed without review.")
        if self.pattern != "ball":
            raise ValueError("C6-4B fixture pattern must remain ball.")
        if self.encoder != "x264enc":
            raise ValueError("C6-4B fixture encoder must remain x264enc.")
        for label, value in (
            ("width", self.width),
            ("height", self.height),
            ("framerate", self.framerate),
            ("bitrate_kbps", self.bitrate_kbps),
            ("key_int_max", self.key_int_max),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"fixture_server.{label} must be a positive integer.")
        if (self.width, self.height, self.framerate) != (320, 240, 30):
            raise ValueError("C6-4B fixture dimensions/framerate changed without review.")


@dataclass(frozen=True)
class FaultRuntimeConfig:
    """Canonical fault-injection phase boundaries."""

    initial_healthy_frames: int
    recovery_healthy_frames: int
    server_ready_timeout_ms: int
    sample_timeout_ms: int
    fault_detection_timeout_ms: int
    fault_kind: str

    # ADD 2026-09-04: Normal/fault/recovery phase size와 timeout을 deterministic하게 고정한다.
    def validate(self) -> None:
        expected = {
            "initial_healthy_frames": 10,
            "recovery_healthy_frames": 30,
            "server_ready_timeout_ms": 5000,
            "sample_timeout_ms": 5000,
            "fault_detection_timeout_ms": 5000,
        }
        for label, expected_value in expected.items():
            value = getattr(self, label)
            if type(value) is not int or value != expected_value:
                raise ValueError(f"C6-4B runtime.{label} changed without review.")
        if self.fault_kind != "terminate_fixture_server_process":
            raise ValueError("C6-4B fault kind changed without review.")


@dataclass(frozen=True)
class FaultScopeConfig:
    """C6-4B must remain a localhost streaming reliability smoke."""

    localhost_rtsp_used: bool
    external_camera_used: bool
    tensorrt_inference_used: bool
    deepstream_used: bool
    final_test_used: bool

    # ADD 2026-09-04: C6-4B smoke scope가 model/test boundary를 넘지 않게 한다.
    def validate(self) -> None:
        if (
            self.localhost_rtsp_used is not True
            or self.external_camera_used is not False
            or self.tensorrt_inference_used is not False
            or self.deepstream_used is not False
            or self.final_test_used is not False
        ):
            raise ValueError("C6-4B smoke scope changed.")


@dataclass(frozen=True)
class RtspFaultInjectionSmokeConfig:
    """Top-level canonical C6-4B smoke configuration."""

    schema_version: int
    smoke_id: str
    required_c6_4a_commit: str
    fixture_server: FixtureServerConfig
    runtime: FaultRuntimeConfig
    output_root: Path
    scope: FaultScopeConfig
    config_path: Path

    # ADD 2026-09-04: C6-4B canonical config 전체를 strict하게 검증한다.
    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported C6-4B schema.")
        if self.smoke_id != EXPECTED_SMOKE_ID:
            raise ValueError("Unexpected C6-4B smoke_id.")
        if self.required_c6_4a_commit != EXPECTED_C6_4A_COMMIT:
            raise ValueError("C6-4B required C6-4A commit changed.")
        self.fixture_server.validate()
        self.runtime.validate()
        self.scope.validate()
        if not self.output_root.parts:
            raise ValueError("C6-4B output_root must be non-empty.")


@dataclass
class StreamHealthTracker:
    """Minimal in-process counter/state surface required by C6-4A."""

    state: str = "DISCONNECTED"
    connection_attempts_total: int = 0
    reconnects_total: int = 0
    frames_received_total: int = 0
    frames_processed_total: int = 0
    frames_dropped_total: int = 0
    errors_total: int = 0
    eos_total: int = 0
    stale_events_total: int = 0
    current_backoff_seconds: float = 0.0
    last_frame_monotonic_s: float | None = None
    reconnect_attempts_since_healthy_reset: int = 0
    healthy_frames_since_reconnect: int = 0

    # ADD 2026-09-04: Runtime state transition을 evidence용 ordered sequence로 기록한다.
    def transition(self, state: str, transitions: list[str]) -> None:
        self.state = state
        transitions.append(state)

    # ADD 2026-09-04: Frame counter와 last-frame gauge source를 한 번에 갱신한다.
    def record_frame(self, *, now_monotonic_s: float) -> None:
        self.frames_received_total += 1
        self.frames_processed_total += 1
        self.last_frame_monotonic_s = now_monotonic_s
        if self.reconnect_attempts_since_healthy_reset > 0:
            self.healthy_frames_since_reconnect += 1

    # ADD 2026-09-04: Reconnect budget 사용을 실제 runtime state로 기록한다.
    def begin_reconnect(self) -> None:
        self.reconnects_total += 1
        self.reconnect_attempts_since_healthy_reset += 1
        self.healthy_frames_since_reconnect = 0

    # ADD 2026-09-04: Frozen healthy-frame boundary 충족 시 reconnect budget을 reset한다.
    def reset_reconnect_budget_if_healthy(self, *, required_frames: int) -> bool:
        if self.healthy_frames_since_reconnect < required_frames:
            return False
        self.reconnect_attempts_since_healthy_reset = 0
        self.healthy_frames_since_reconnect = 0
        return True

    # ADD 2026-09-04: C6-4A last-frame gauge 값을 monotonic clock으로 계산한다.
    def seconds_since_last_frame(self, *, now_monotonic_s: float) -> float:
        if self.last_frame_monotonic_s is None:
            raise RuntimeError("No RTSP frame has been recorded.")
        if now_monotonic_s < self.last_frame_monotonic_s:
            raise ValueError("Monotonic time cannot move backwards.")
        return now_monotonic_s - self.last_frame_monotonic_s


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return cast(dict[str, Any], value)


def _require_fields(raw: dict[str, Any], cls: type[Any], *, label: str) -> None:
    expected = {field.name for field in fields(cls)}
    if set(raw) != expected:
        raise ValueError(f"{label} fields do not match schema.")


# ADD 2026-09-04: YAML을 typed C6-4B canonical smoke config로 로드한다.
def load_rtsp_fault_injection_config(path: Path) -> RtspFaultInjectionSmokeConfig:
    raw_obj: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = _mapping(raw_obj, label="C6-4B config")
    if set(raw) != {
        "schema_version",
        "smoke_id",
        "required_c6_4a_commit",
        "fixture_server",
        "runtime",
        "output_root",
        "scope",
    }:
        raise ValueError("C6-4B config fields do not match schema.")

    fixture_raw = _mapping(raw["fixture_server"], label="fixture_server")
    runtime_raw = _mapping(raw["runtime"], label="runtime")
    scope_raw = _mapping(raw["scope"], label="scope")

    _require_fields(fixture_raw, FixtureServerConfig, label="fixture_server")
    _require_fields(runtime_raw, FaultRuntimeConfig, label="runtime")
    _require_fields(scope_raw, FaultScopeConfig, label="scope")

    config = RtspFaultInjectionSmokeConfig(
        schema_version=raw["schema_version"],
        smoke_id=str(raw["smoke_id"]),
        required_c6_4a_commit=str(raw["required_c6_4a_commit"]),
        fixture_server=FixtureServerConfig(**fixture_raw),
        runtime=FaultRuntimeConfig(**runtime_raw),
        output_root=Path(str(raw["output_root"])),
        scope=FaultScopeConfig(**scope_raw),
        config_path=path.resolve(),
    )
    config.validate()
    return config


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=repo, text=True).strip()


# ADD 2026-09-04: C6-4A contract commit이 canonical run commit의 ancestor인지 검증한다.
def _require_c6_4a_ancestor(repo: Path, required_commit: str) -> None:
    completed = subprocess.run(
        ("git", "merge-base", "--is-ancestor", required_commit, "HEAD"),
        cwd=repo,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("C6-4B requires the committed C6-4A reliability contract.")


def _find_free_local_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def _wait_for_port(host: str, port: int, *, timeout_ms: int, expect_open: bool) -> None:
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.1)
            open_now = probe.connect_ex((host, port)) == 0
        if open_now is expect_open:
            return
        time.sleep(0.05)
    expectation = "open" if expect_open else "closed"
    raise RuntimeError(f"C6-4B fixture port did not become {expectation}.")


# ADD 2026-09-04: Fixture RTSP server를 child process로 띄워 TCP fault를 주입한다.
def build_fixture_server_command(
    config: RtspFaultInjectionSmokeConfig,
    *,
    port: int,
) -> tuple[str, ...]:
    fixture = config.fixture_server
    return (
        sys.executable,
        "-u",
        "-m",
        "services.streaming.yolo_rtsp_fixture_server",
        "--host",
        fixture.host,
        "--port",
        str(port),
        "--mount",
        fixture.mount,
        "--pattern",
        fixture.pattern,
        "--width",
        str(fixture.width),
        "--height",
        str(fixture.height),
        "--framerate",
        str(fixture.framerate),
        "--bitrate-kbps",
        str(fixture.bitrate_kbps),
        "--key-int-max",
        str(fixture.key_int_max),
    )


def _start_fixture_server(
    config: RtspFaultInjectionSmokeConfig,
    *,
    port: int,
    log_path: Path,
) -> tuple[subprocess.Popen[str], Any]:
    log_handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        build_fixture_server_command(config, port=port),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        _wait_for_port(
            config.fixture_server.host,
            port,
            timeout_ms=config.runtime.server_ready_timeout_ms,
            expect_open=True,
        )
    except Exception:
        process.terminate()
        process.wait(timeout=2)
        log_handle.close()
        detail = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"C6-4B fixture server failed to start:\n{detail}") from None
    return process, log_handle


def _stop_fixture_server(
    process: subprocess.Popen[str],
    log_handle: Any,
    *,
    host: str,
    port: int,
    timeout_ms: int,
) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    log_handle.close()
    _wait_for_port(host, port, timeout_ms=timeout_ms, expect_open=False)


def _create_client(config: RtspFaultInjectionSmokeConfig, *, uri: str) -> tuple[Any, Any, Any]:
    gst, _ = load_gstreamer_modules()
    gst.init(None)
    reliability = load_rtsp_reliability_config(DEFAULT_RTSP_RELIABILITY_CONFIG)
    pipeline = gst.parse_launch(build_rtsp_pipeline(reliability, uri=uri))
    appsink = pipeline.get_by_name("framesink")
    if appsink is None:
        raise RuntimeError("C6-4B client could not find framesink.")
    result = pipeline.set_state(gst.State.PLAYING)
    if result == gst.StateChangeReturn.FAILURE:
        raise RuntimeError("C6-4B client pipeline failed to start.")
    return gst, pipeline, appsink


def _validate_frame(frame: np.ndarray, config: RtspFaultInjectionSmokeConfig) -> None:
    fixture = config.fixture_server
    if frame.shape != (fixture.height, fixture.width, 3):
        raise RuntimeError(f"C6-4B frame shape changed: {frame.shape}.")
    if frame.dtype != np.uint8:
        raise RuntimeError(f"C6-4B frame dtype changed: {frame.dtype}.")
    if not frame.flags.c_contiguous or not frame.flags.owndata:
        raise RuntimeError("C6-4B frame ownership/contiguity contract changed.")


def _receive_frames(
    *,
    gst: Any,
    appsink: Any,
    count: int,
    timeout_ms: int,
    config: RtspFaultInjectionSmokeConfig,
    tracker: StreamHealthTracker,
) -> tuple[list[str], float]:
    hashes: list[str] = []
    last_frame_monotonic = time.monotonic()
    for index in range(count):
        sample = appsink.emit("try-pull-sample", timeout_ms * gst.MSECOND)
        if sample is None:
            raise RuntimeError(f"C6-4B received no sample at frame {index}.")
        frame = gst_sample_to_bgr_numpy(sample)
        _validate_frame(frame, config)
        now_monotonic = time.monotonic()
        tracker.record_frame(now_monotonic_s=now_monotonic)
        last_frame_monotonic = now_monotonic
        hashes.append(hashlib.sha256(frame.tobytes(order="C")).hexdigest())
    return hashes, last_frame_monotonic


def _detect_interruption(
    *,
    gst: Any,
    pipeline: Any,
    appsink: Any,
    last_frame_monotonic: float,
    config: RtspFaultInjectionSmokeConfig,
    tracker: StreamHealthTracker,
) -> tuple[str, float]:
    reliability = load_rtsp_reliability_config(DEFAULT_RTSP_RELIABILITY_CONFIG)
    bus = pipeline.get_bus()
    start = time.monotonic()
    deadline = start + config.runtime.fault_detection_timeout_ms / 1000.0

    while time.monotonic() < deadline:
        sample = appsink.emit("try-pull-sample", 100 * gst.MSECOND)
        if sample is not None:
            frame = gst_sample_to_bgr_numpy(sample)
            _validate_frame(frame, config)
            now_monotonic = time.monotonic()
            tracker.record_frame(now_monotonic_s=now_monotonic)
            last_frame_monotonic = now_monotonic

        message = bus.pop_filtered(gst.MessageType.ERROR | gst.MessageType.EOS)
        if message is not None:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            if message.type == gst.MessageType.ERROR:
                tracker.errors_total += 1
                return "gst_error", elapsed_ms
            tracker.eos_total += 1
            return "eos", elapsed_ms

        elapsed_since_frame_ms = (time.monotonic() - last_frame_monotonic) * 1000.0
        if elapsed_since_frame_ms >= reliability.source.frame_stale_after_ms:
            tracker.stale_events_total += 1
            return "frame_timeout", (time.monotonic() - start) * 1000.0

    raise RuntimeError("C6-4B did not detect the injected RTSP interruption in time.")


# ADD 2026-09-04: Clean commit에서 RTSP fault/reconnect recovery evidence를 만든다.
def run_rtsp_fault_injection_smoke(
    config: RtspFaultInjectionSmokeConfig,
    *,
    repo: Path,
) -> Path:
    config.validate()

    commit = _git_output(repo, "rev-parse", "HEAD")
    if _git_output(repo, "status", "--porcelain"):
        raise RuntimeError("C6-4B canonical smoke requires a clean working tree.")
    _require_c6_4a_ancestor(repo, config.required_c6_4a_commit)

    reliability = load_rtsp_reliability_config(DEFAULT_RTSP_RELIABILITY_CONFIG)
    backoffs = reconnect_backoff_schedule_ms(reliability.reconnect)
    if not backoffs or backoffs[0] != 500:
        raise RuntimeError("C6-4B expected first reconnect backoff of 500 ms.")
    if config.runtime.recovery_healthy_frames != reliability.reconnect.reset_after_healthy_frames:
        raise RuntimeError("C6-4B recovery boundary must match the frozen reconnect reset policy.")

    fixture = config.fixture_server
    runtime = config.runtime
    port = _find_free_local_port(fixture.host)
    uri = f"rtsp://{fixture.host}:{port}{fixture.mount}"

    tracker = StreamHealthTracker()
    transitions = ["DISCONNECTED"]
    interruption_event = ""
    interruption_detection_ms = 0.0
    actual_backoff_ms = 0.0
    reconnect_budget_reset = False
    final_seconds_since_last_frame = 0.0
    initial_hashes: list[str] = []
    recovery_hashes: list[str] = []

    pipeline: Any | None = None
    server: subprocess.Popen[str] | None = None
    server_log: Any | None = None

    with tempfile.TemporaryDirectory(prefix="c6_4b_rtsp_") as tmp:
        tmp_path = Path(tmp)
        server_log_path = tmp_path / "fixture_server.log"

        try:
            tracker.transition("CONNECTING", transitions)
            tracker.connection_attempts_total += 1
            server, server_log = _start_fixture_server(
                config,
                port=port,
                log_path=server_log_path,
            )
            gst, pipeline, appsink = _create_client(config, uri=uri)
            initial_hashes, last_frame_monotonic = _receive_frames(
                gst=gst,
                appsink=appsink,
                count=runtime.initial_healthy_frames,
                timeout_ms=runtime.sample_timeout_ms,
                config=config,
                tracker=tracker,
            )
            tracker.transition("STREAMING", transitions)

            _stop_fixture_server(
                server,
                server_log,
                host=fixture.host,
                port=port,
                timeout_ms=runtime.server_ready_timeout_ms,
            )
            server = None
            server_log = None

            interruption_event, interruption_detection_ms = _detect_interruption(
                gst=gst,
                pipeline=pipeline,
                appsink=appsink,
                last_frame_monotonic=last_frame_monotonic,
                config=config,
                tracker=tracker,
            )
            if interruption_event == "frame_timeout":
                tracker.transition("STALE", transitions)

            tracker.transition("RECONNECTING", transitions)
            tracker.begin_reconnect()
            pipeline.set_state(gst.State.NULL)
            pipeline = None

            requested_backoff_ms = backoffs[0]
            tracker.current_backoff_seconds = requested_backoff_ms / 1000.0
            backoff_started = time.monotonic()
            time.sleep(requested_backoff_ms / 1000.0)
            actual_backoff_ms = (time.monotonic() - backoff_started) * 1000.0

            tracker.connection_attempts_total += 1
            server, server_log = _start_fixture_server(
                config,
                port=port,
                log_path=server_log_path,
            )
            gst, pipeline, appsink = _create_client(config, uri=uri)
            recovery_hashes, _ = _receive_frames(
                gst=gst,
                appsink=appsink,
                count=runtime.recovery_healthy_frames,
                timeout_ms=runtime.sample_timeout_ms,
                config=config,
                tracker=tracker,
            )
            reconnect_budget_reset = tracker.reset_reconnect_budget_if_healthy(
                required_frames=reliability.reconnect.reset_after_healthy_frames
            )
            if not reconnect_budget_reset:
                raise RuntimeError("C6-4B reconnect budget did not reset after healthy recovery.")
            tracker.current_backoff_seconds = 0.0
            tracker.transition("STREAMING", transitions)
            final_seconds_since_last_frame = tracker.seconds_since_last_frame(
                now_monotonic_s=time.monotonic()
            )

        finally:
            if pipeline is not None:
                try:
                    gst, _ = load_gstreamer_modules()
                    pipeline.set_state(gst.State.NULL)
                except Exception:
                    pass
            if server is not None and server_log is not None:
                try:
                    _stop_fixture_server(
                        server,
                        server_log,
                        host=fixture.host,
                        port=port,
                        timeout_ms=runtime.server_ready_timeout_ms,
                    )
                except Exception:
                    pass

    if tracker.connection_attempts_total != 2 or tracker.reconnects_total != 1:
        raise RuntimeError("C6-4B connection/reconnect counters changed.")
    if len(recovery_hashes) != runtime.recovery_healthy_frames:
        raise RuntimeError("C6-4B recovery did not reach the 30-frame reset boundary.")
    if transitions[-1] != "STREAMING":
        raise RuntimeError("C6-4B did not recover to STREAMING.")
    if actual_backoff_ms < 500.0:
        raise RuntimeError("C6-4B reconnect backoff was shorter than the frozen policy.")

    output_dir = repo / config.output_root / config.smoke_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "smoke.json"

    payload = {
        "schema_version": 1,
        "stage": "C6-4B",
        "state": SMOKE_STATE,
        "created_at": datetime.now(UTC).isoformat(),
        "repository": {
            "git_commit": commit,
            "working_tree_dirty_before_run": False,
            "required_c6_4a_commit": config.required_c6_4a_commit,
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
            "frame_stale_after_ms": reliability.source.frame_stale_after_ms,
        },
        "fixture": {
            "type": "localhost_gst_rtsp_server",
            "uri_redacted": uri,
            "pattern": fixture.pattern,
            "width": fixture.width,
            "height": fixture.height,
            "framerate": fixture.framerate,
            "fault_kind": runtime.fault_kind,
        },
        "normal_phase": {
            "healthy_frames_required": runtime.initial_healthy_frames,
            "unique_frame_sha256_count": len(set(initial_hashes)),
        },
        "fault_phase": {
            "detected_event": interruption_event,
            "detection_ms": interruption_detection_ms,
        },
        "recovery_phase": {
            "reconnect_attempt": 1,
            "requested_backoff_ms": backoffs[0],
            "actual_backoff_ms": actual_backoff_ms,
            "healthy_frames_required_for_reset": runtime.recovery_healthy_frames,
            "healthy_frames_received": len(recovery_hashes),
            "unique_frame_sha256_count": len(set(recovery_hashes)),
            "reconnect_budget_reset": reconnect_budget_reset,
            "reconnect_attempts_since_healthy_reset": (
                tracker.reconnect_attempts_since_healthy_reset
            ),
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
            "rtsp_stream_up": int(tracker.state == "STREAMING"),
            "rtsp_seconds_since_last_frame": final_seconds_since_last_frame,
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
