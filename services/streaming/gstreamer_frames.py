"""Python frame adapter for the C6 GStreamer appsink boundary."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

BGR_CHANNELS = 3


@dataclass(frozen=True)
class BgrFrameSpec:
    """Memory layout required by the C6 CPU frame boundary."""

    width: int
    height: int
    stride: int

    # ADD 2026-09-04: GstBuffer row layout가 BGR frame을 안전하게 포함하는지 검증한다.
    def validate(self) -> None:
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("BGR frame width must be a positive integer.")
        if type(self.height) is not int or self.height <= 0:
            raise ValueError("BGR frame height must be a positive integer.")
        if type(self.stride) is not int or self.stride < self.row_bytes:
            raise ValueError("BGR frame stride is smaller than one packed row.")

    @property
    def row_bytes(self) -> int:
        return self.width * BGR_CHANNELS

    @property
    def required_bytes(self) -> int:
        return self.stride * self.height


# ADD 2026-09-04: Native buffer bytes를 owned C-contiguous uint8 HWC BGR frame으로 변환한다.
def bgr_buffer_to_numpy(
    data: bytes | bytearray | memoryview,
    *,
    width: int,
    height: int,
    stride: int | None = None,
) -> NDArray[np.uint8]:
    resolved_stride = width * BGR_CHANNELS if stride is None else stride
    spec = BgrFrameSpec(width=width, height=height, stride=resolved_stride)
    spec.validate()

    view = memoryview(data)
    if view.nbytes < spec.required_bytes:
        raise ValueError(
            "GStreamer buffer is smaller than the declared BGR frame layout: "
            f"{view.nbytes} < {spec.required_bytes}."
        )

    flat = np.frombuffer(view[: spec.required_bytes], dtype=np.uint8)
    rows = flat.reshape(spec.height, spec.stride)
    packed_rows = rows[:, : spec.row_bytes]
    frame = packed_rows.reshape(spec.height, spec.width, BGR_CHANNELS).copy(order="C")
    validate_bgr_numpy_frame(frame, width=spec.width, height=spec.height)
    return frame


# ADD 2026-09-04: Downstream TensorRT adapter 앞 NumPy frame contract를 fail-closed로 검증한다.
def validate_bgr_numpy_frame(
    frame: NDArray[np.uint8],
    *,
    width: int,
    height: int,
) -> None:
    if frame.dtype != np.uint8:
        raise ValueError(f"Expected uint8 frame, got {frame.dtype}.")
    if frame.shape != (height, width, BGR_CHANNELS):
        raise ValueError(
            f"Expected HWC BGR frame shape {(height, width, BGR_CHANNELS)}, got {frame.shape}."
        )
    if not frame.flags.c_contiguous:
        raise ValueError("BGR frame must be C-contiguous.")


# ADD 2026-09-04: PyGObject import를 streaming runtime 경계까지 지연해 기본 repo test와 분리한다.
def load_gstreamer_modules() -> tuple[Any, Any]:
    try:
        gi = importlib.import_module("gi")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyGObject is unavailable. Run through scripts/run_streaming_uv.sh."
        ) from exc

    gi.require_version("Gst", "1.0")
    gi.require_version("GstVideo", "1.0")

    gst = importlib.import_module("gi.repository.Gst")
    gst_video = importlib.import_module("gi.repository.GstVideo")
    return gst, gst_video


# ADD 2026-09-04: GstSample을 map한 뒤 lifetime에서 분리된 NumPy BGR frame copy를 반환한다.
def gst_sample_to_bgr_numpy(sample: Any) -> NDArray[np.uint8]:
    gst, gst_video = load_gstreamer_modules()

    caps = sample.get_caps()
    if caps is None or caps.get_size() != 1:
        raise RuntimeError("C6 GStreamer sample must expose exactly one caps structure.")

    structure = caps.get_structure(0)
    pixel_format = structure.get_string("format")
    if pixel_format != "BGR":
        raise RuntimeError(f"Expected GStreamer BGR sample, got {pixel_format!r}.")

    width = int(structure.get_value("width"))
    height = int(structure.get_value("height"))

    video_info = gst_video.VideoInfo.new_from_caps(caps)
    if video_info is None:
        raise RuntimeError("Cannot derive GstVideo.VideoInfo from sample caps.")
    stride = int(video_info.stride[0])

    buffer = sample.get_buffer()
    if buffer is None:
        raise RuntimeError("GStreamer sample does not contain a GstBuffer.")

    mapped, map_info = buffer.map(gst.MapFlags.READ)
    if not mapped:
        raise RuntimeError("Cannot map GStreamer frame buffer.")

    try:
        return bgr_buffer_to_numpy(
            map_info.data,
            width=width,
            height=height,
            stride=stride,
        )
    finally:
        buffer.unmap(map_info)
