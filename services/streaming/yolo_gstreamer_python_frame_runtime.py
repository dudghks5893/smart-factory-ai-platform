"""Canonical C6-3 Python GStreamer frame validation runtime."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
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

DEFAULT_PYTHON_FRAME_CONFIG = Path("configs/streaming/yolo_gstreamer_python_frame.yaml")
PYTHON_FRAME_STATE = "PYTHON_GSTREAMER_FRAME_ADAPTER_COMPLETED"
EXPECTED_VALIDATION_ID = "c6_3_python_gstreamer_frame_v1"


@dataclass(frozen=True)
class PythonFrameSourceConfig:
    """Synthetic source settings for the canonical Python frame validation."""

    num_buffers: int
    pattern: str
    width: int
    height: int

    # ADD 2026-09-04: Canonical source dimensions와 deterministic one-frame boundary를 검증한다.
    def validate(self) -> None:
        if type(self.num_buffers) is not int or self.num_buffers != 1:
            raise ValueError("C6-3 Python frame validation must use exactly one source buffer.")
        if self.pattern != "ball":
            raise ValueError("C6-3 Python frame validation pattern must remain ball.")
        for label, value in (("width", self.width), ("height", self.height)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"source.{label} must be a positive integer.")


@dataclass(frozen=True)
class PythonFrameValidationConfig:
    """Top-level canonical Python GStreamer frame validation configuration."""

    schema_version: int
    validation_id: str
    source: PythonFrameSourceConfig
    timeout_seconds: int
    output_root: Path
    config_path: Path

    # ADD 2026-09-04: C6-3 Python frame validation config를 strict schema로 고정한다.
    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported C6-3 Python frame config schema.")
        if self.validation_id != EXPECTED_VALIDATION_ID:
            raise ValueError("Unexpected C6-3 Python frame validation_id.")
        self.source.validate()
        if type(self.timeout_seconds) is not int or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer.")
        if not self.output_root.parts:
            raise ValueError("output_root must be non-empty.")


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return cast(dict[str, Any], value)


# ADD 2026-09-04: YAML contract를 typed canonical Python frame validation config로 로드한다.
def load_python_frame_validation_config(path: Path) -> PythonFrameValidationConfig:
    raw_obj: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = _mapping(raw_obj, label="C6-3 Python frame config")
    if set(raw) != {
        "schema_version",
        "validation_id",
        "source",
        "timeout_seconds",
        "output_root",
    }:
        raise ValueError("C6-3 Python frame config fields do not match schema.")

    source_raw = _mapping(raw["source"], label="source")
    if set(source_raw) != {"num_buffers", "pattern", "width", "height"}:
        raise ValueError("C6-3 Python frame source fields do not match schema.")

    config = PythonFrameValidationConfig(
        schema_version=raw["schema_version"],
        validation_id=str(raw["validation_id"]),
        source=PythonFrameSourceConfig(
            num_buffers=source_raw["num_buffers"],
            pattern=str(source_raw["pattern"]),
            width=source_raw["width"],
            height=source_raw["height"],
        ),
        timeout_seconds=raw["timeout_seconds"],
        output_root=Path(str(raw["output_root"])),
        config_path=path.resolve(),
    )
    config.validate()
    return config


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=repo, text=True).strip()


# ADD 2026-09-04: C6-1 latest-frame-wins contract와 동일한 Python appsink pipeline을 만든다.
def build_python_frame_pipeline(config: PythonFrameValidationConfig) -> str:
    source = config.source
    return (
        f"videotestsrc num-buffers={source.num_buffers} pattern={source.pattern} "
        "! videoconvert "
        f"! video/x-raw,format=BGR,width={source.width},height={source.height} "
        "! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream "
        "! appsink name=framesink emit-signals=false "
        "max-buffers=1 drop=true sync=false wait-on-eos=false"
    )


# ADD 2026-09-04: Clean commit에서 real PyGObject appsink frame을 검증하고 JSON evidence를 기록한다.
def run_python_frame_validation(
    config: PythonFrameValidationConfig,
    *,
    repo: Path,
) -> Path:
    config.validate()

    commit = _git_output(repo, "rev-parse", "HEAD")
    if _git_output(repo, "status", "--porcelain"):
        raise RuntimeError("C6-3 canonical Python frame validation requires a clean working tree.")

    gst, gst_video = load_gstreamer_modules()
    gst.init(None)

    pipeline = gst.parse_launch(build_python_frame_pipeline(config))
    appsink = pipeline.get_by_name("framesink")
    if appsink is None:
        raise RuntimeError("C6-3 Python frame validation could not find framesink.")

    state_result = pipeline.set_state(gst.State.PLAYING)
    if state_result == gst.StateChangeReturn.FAILURE:
        raise RuntimeError("C6-3 Python frame validation pipeline failed to start.")

    try:
        sample = appsink.emit("try-pull-sample", config.timeout_seconds * gst.SECOND)
        if sample is None:
            raise RuntimeError("C6-3 Python frame validation received no sample.")

        caps = sample.get_caps()
        if caps is None or caps.get_size() != 1:
            raise RuntimeError("C6-3 Python frame validation requires exactly one caps structure.")

        structure = caps.get_structure(0)
        caps_format = structure.get_string("format")
        caps_width = int(structure.get_value("width"))
        caps_height = int(structure.get_value("height"))

        video_info = gst_video.VideoInfo.new_from_caps(caps)
        if video_info is None:
            raise RuntimeError("Cannot derive GstVideo.VideoInfo from canonical sample.")
        stride = int(video_info.stride[0])

        frame = gst_sample_to_bgr_numpy(sample)
    finally:
        pipeline.set_state(gst.State.NULL)

    source = config.source
    expected_shape = (source.height, source.width, 3)
    if caps_format != "BGR":
        raise RuntimeError(f"Canonical caps format changed: {caps_format!r}.")
    if (caps_width, caps_height) != (source.width, source.height):
        raise RuntimeError("Canonical caps dimensions changed.")
    if frame.shape != expected_shape:
        raise RuntimeError(f"Canonical frame shape changed: {frame.shape}.")
    if frame.dtype != np.uint8:
        raise RuntimeError(f"Canonical frame dtype changed: {frame.dtype}.")
    if not frame.flags.c_contiguous:
        raise RuntimeError("Canonical frame is not C-contiguous.")
    if not frame.flags.owndata:
        raise RuntimeError("Canonical frame must own its NumPy memory.")

    output_dir = repo / config.output_root / config.validation_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "validation.json"

    payload = {
        "schema_version": 1,
        "stage": "C6-3",
        "state": PYTHON_FRAME_STATE,
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
        "python_packages": {
            "numpy": version("numpy"),
            "pygobject": version("PyGObject"),
            "pycairo": version("pycairo"),
        },
        "gstreamer": {
            "version": gst.version_string(),
            "caps_format": caps_format,
            "caps_width": caps_width,
            "caps_height": caps_height,
            "stride_bytes": stride,
            "appsink_name": "framesink",
            "backpressure": "latest_frame_wins",
        },
        "source": {
            "type": "test",
            "num_buffers": source.num_buffers,
            "pattern": source.pattern,
            "width": source.width,
            "height": source.height,
        },
        "frame": {
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "bytes": int(frame.nbytes),
            "c_contiguous": bool(frame.flags.c_contiguous),
            "owned": bool(frame.flags.owndata),
            "sha256": hashlib.sha256(frame.tobytes(order="C")).hexdigest(),
            "contract": "BGR/uint8/HWC/C-contiguous/owned",
        },
        "test_used": False,
        "final_test_used": False,
        "tensorrt_inference_used": False,
        "deepstream_used": False,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
