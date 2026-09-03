"""C6-2 native GStreamer smoke runner."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from ml.training.yolo_segmentation import validate_artifact_id

DEFAULT_NATIVE_SMOKE_CONFIG = Path("configs/streaming/yolo_gstreamer_native_smoke.yaml")
EXPECTED_C6_1_COMMIT = "6c817de0a519dca7f1eaf813c4892ec7a3e921ac"
SMOKE_STATE = "NATIVE_GSTREAMER_SMOKE_COMPLETED"


@dataclass(frozen=True)
class SyntheticSmokeConfig:
    """Synthetic live-like source settings."""

    num_buffers: int
    pattern: str

    # ADD 2026-09-04: C6-2 synthetic source가 짧고 deterministic하게 끝나도록 검증한다.
    def validate(self) -> None:
        if type(self.num_buffers) is not int or self.num_buffers <= 0:
            raise ValueError("synthetic.num_buffers must be a positive integer.")
        if not self.pattern:
            raise ValueError("synthetic.pattern must be non-empty.")


@dataclass(frozen=True)
class FixtureEncoderConfig:
    """Temporary MP4 encoder settings used only by the smoke test."""

    name: str
    tune: str
    speed_preset: str
    key_int_max: int

    # ADD 2026-09-04: Local-file decode smoke용 fixture encoder contract를 고정한다.
    def validate(self) -> None:
        if self.name != "x264enc":
            raise ValueError("C6-2 fixture encoder must remain x264enc.")
        if self.tune != "zerolatency" or self.speed_preset != "ultrafast":
            raise ValueError("C6-2 fixture encoder latency settings changed.")
        if type(self.key_int_max) is not int or self.key_int_max <= 0:
            raise ValueError("fixture.encoder.key_int_max must be positive.")


@dataclass(frozen=True)
class FixtureSmokeConfig:
    """Deterministic local video fixture settings."""

    num_buffers: int
    width: int
    height: int
    framerate: int
    encoder: FixtureEncoderConfig

    # ADD 2026-09-04: Temporary local MP4 fixture dimensions와 duration boundary를 검증한다.
    def validate(self) -> None:
        for label, value in (
            ("num_buffers", self.num_buffers),
            ("width", self.width),
            ("height", self.height),
            ("framerate", self.framerate),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"fixture.{label} must be a positive integer.")
        self.encoder.validate()


@dataclass(frozen=True)
class NativeGStreamerSmokeConfig:
    """Top-level C6-2 smoke configuration."""

    schema_version: int
    smoke_id: str
    required_plugins: tuple[str, ...]
    synthetic: SyntheticSmokeConfig
    fixture: FixtureSmokeConfig
    output_root: Path
    config_path: Path

    # ADD 2026-09-04: C6-2 native smoke config를 strict repository contract로 검증한다.
    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported C6-2 smoke config schema.")
        validate_artifact_id(self.smoke_id)
        if len(self.required_plugins) != len(set(self.required_plugins)):
            raise ValueError("C6-2 required_plugins contains duplicates.")
        if not self.required_plugins:
            raise ValueError("C6-2 required_plugins must not be empty.")
        self.synthetic.validate()
        self.fixture.validate()
        if not self.output_root.parts:
            raise ValueError("C6-2 output_root must be non-empty.")


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return cast(dict[str, Any], value)


# ADD 2026-09-04: YAML smoke config를 typed C6-2 config로 로드한다.
def load_native_gstreamer_smoke_config(path: Path) -> NativeGStreamerSmokeConfig:
    raw_obj: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = _mapping(raw_obj, label="C6-2 config")
    if set(raw) != {
        "schema_version",
        "smoke_id",
        "required_plugins",
        "synthetic",
        "fixture",
        "output_root",
    }:
        raise ValueError("C6-2 config fields do not match schema.")

    synthetic_raw = _mapping(raw["synthetic"], label="synthetic")
    fixture_raw = _mapping(raw["fixture"], label="fixture")
    encoder_raw = _mapping(fixture_raw["encoder"], label="fixture.encoder")

    if set(synthetic_raw) != {"num_buffers", "pattern"}:
        raise ValueError("C6-2 synthetic fields do not match schema.")
    if set(fixture_raw) != {"num_buffers", "width", "height", "framerate", "encoder"}:
        raise ValueError("C6-2 fixture fields do not match schema.")
    if set(encoder_raw) != {"name", "tune", "speed_preset", "key_int_max"}:
        raise ValueError("C6-2 fixture encoder fields do not match schema.")

    plugins = raw["required_plugins"]
    if not isinstance(plugins, list) or not all(isinstance(item, str) for item in plugins):
        raise ValueError("C6-2 required_plugins must be a list of strings.")

    config = NativeGStreamerSmokeConfig(
        schema_version=raw["schema_version"],
        smoke_id=str(raw["smoke_id"]),
        required_plugins=tuple(plugins),
        synthetic=SyntheticSmokeConfig(**synthetic_raw),
        fixture=FixtureSmokeConfig(
            num_buffers=fixture_raw["num_buffers"],
            width=fixture_raw["width"],
            height=fixture_raw["height"],
            framerate=fixture_raw["framerate"],
            encoder=FixtureEncoderConfig(**encoder_raw),
        ),
        output_root=Path(str(raw["output_root"])),
        config_path=path.resolve(),
    )
    config.validate()
    return config


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
    )


def _first_version_line(executable: str) -> str:
    completed = _run((executable, "--version"))
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{executable} returned no version output.")
    return lines[0]


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=repo, text=True).strip()


# ADD 2026-09-04: C6-1 bounded BGR/appsink contract와 동일한 synthetic native command를 만든다.
def build_synthetic_smoke_command(
    gst_launch: str,
    config: NativeGStreamerSmokeConfig,
) -> tuple[str, ...]:
    synthetic = config.synthetic
    return (
        gst_launch,
        "-q",
        "videotestsrc",
        f"num-buffers={synthetic.num_buffers}",
        "is-live=true",
        "do-timestamp=true",
        f"pattern={synthetic.pattern}",
        "!",
        "videoconvert",
        "!",
        "video/x-raw,format=BGR",
        "!",
        "queue",
        "max-size-buffers=1",
        "max-size-bytes=0",
        "max-size-time=0",
        "leaky=downstream",
        "!",
        "appsink",
        "name=framesink",
        "emit-signals=false",
        "max-buffers=1",
        "drop=true",
        "sync=false",
        "wait-on-eos=false",
    )


# ADD 2026-09-04: Local-file decode 검증을 위한 temporary MP4 fixture command를 만든다.
def build_fixture_generation_command(
    gst_launch: str,
    config: NativeGStreamerSmokeConfig,
    fixture_path: Path,
) -> tuple[str, ...]:
    fixture = config.fixture
    encoder = fixture.encoder
    return (
        gst_launch,
        "-q",
        "videotestsrc",
        f"num-buffers={fixture.num_buffers}",
        "pattern=ball",
        "!",
        f"video/x-raw,width={fixture.width},height={fixture.height},framerate={fixture.framerate}/1",
        "!",
        "videoconvert",
        "!",
        encoder.name,
        f"tune={encoder.tune}",
        f"speed-preset={encoder.speed_preset}",
        f"key-int-max={encoder.key_int_max}",
        "!",
        "h264parse",
        "!",
        "mp4mux",
        "!",
        "filesink",
        f"location={fixture_path}",
    )


# ADD 2026-09-04: Generated local MP4를 C6-1 BGR/appsink boundary로 decode하는 command를 만든다.
def build_file_decode_smoke_command(
    gst_launch: str,
    fixture_path: Path,
) -> tuple[str, ...]:
    return (
        gst_launch,
        "-q",
        "uridecodebin",
        f"uri={fixture_path.resolve().as_uri()}",
        "!",
        "videoconvert",
        "!",
        "video/x-raw,format=BGR",
        "!",
        "queue",
        "max-size-buffers=1",
        "max-size-bytes=0",
        "max-size-time=0",
        "leaky=downstream",
        "!",
        "appsink",
        "name=framesink",
        "emit-signals=false",
        "max-buffers=1",
        "drop=true",
        "sync=false",
        "wait-on-eos=false",
    )


# ADD 2026-09-04: Clean commit에서 native smoke를 실행하고 JSON evidence를 만든다.
def run_native_gstreamer_smoke(
    config: NativeGStreamerSmokeConfig,
    *,
    repo: Path,
) -> Path:
    config.validate()

    commit = _git_output(repo, "rev-parse", "HEAD")
    dirty = bool(_git_output(repo, "status", "--porcelain"))
    if dirty:
        raise RuntimeError("C6-2 canonical smoke requires a clean working tree.")

    gst_launch = shutil.which("gst-launch-1.0")
    gst_inspect = shutil.which("gst-inspect-1.0")
    if gst_launch is None or gst_inspect is None:
        raise RuntimeError("C6-2 requires gst-launch-1.0 and gst-inspect-1.0.")

    plugin_checks: dict[str, bool] = {}
    for plugin in config.required_plugins:
        completed = subprocess.run(
            (gst_inspect, plugin),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        plugin_checks[plugin] = completed.returncode == 0
    missing = [name for name, passed in plugin_checks.items() if not passed]
    if missing:
        raise RuntimeError(f"C6-2 missing required GStreamer plugins: {missing}")

    with tempfile.TemporaryDirectory(prefix="c6_2_gstreamer_") as tmp:
        fixture = Path(tmp) / "c6_2_fixture.mp4"

        _run(build_synthetic_smoke_command(gst_launch, config))
        _run(build_fixture_generation_command(gst_launch, config, fixture))
        if not fixture.is_file() or fixture.stat().st_size <= 0:
            raise RuntimeError("C6-2 fixture generation produced no MP4 bytes.")
        fixture_size = fixture.stat().st_size
        _run(build_file_decode_smoke_command(gst_launch, fixture))

    output_dir = repo / config.output_root / config.smoke_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "smoke.json"

    payload = {
        "schema_version": 1,
        "stage": "C6-2",
        "state": SMOKE_STATE,
        "created_at": datetime.now(UTC).isoformat(),
        "repository": {
            "git_commit": commit,
            "working_tree_dirty_before_run": False,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "gstreamer": {
            "gst_launch_path": gst_launch,
            "gst_inspect_path": gst_inspect,
            "gst_launch_version": _first_version_line(gst_launch),
            "gst_inspect_version": _first_version_line(gst_inspect),
            "required_plugins": plugin_checks,
        },
        "synthetic": {
            "num_buffers": config.synthetic.num_buffers,
            "pattern": config.synthetic.pattern,
            "exit_code": 0,
            "frame_contract": "BGR/uint8/HWC",
            "backpressure": "latest_frame_wins",
        },
        "local_file": {
            "fixture_bytes": fixture_size,
            "exit_code": 0,
            "decoder": "uridecodebin",
            "frame_contract": "BGR/uint8/HWC",
        },
        "test_used": False,
        "tensorrt_inference_used": False,
        "deepstream_used": False,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
