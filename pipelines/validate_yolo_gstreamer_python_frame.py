"""Validate the C6 Python GStreamer appsink-to-NumPy frame boundary."""

from __future__ import annotations

from services.streaming.gstreamer_frames import (
    gst_sample_to_bgr_numpy,
    load_gstreamer_modules,
)


# ADD 2026-09-04: Real PyGObject appsink에서 NumPy BGR frame 하나를 canonical하게 pull한다.
def main() -> None:
    gst, _ = load_gstreamer_modules()
    gst.init(None)

    pipeline = gst.parse_launch(
        "videotestsrc num-buffers=1 pattern=ball "
        "! videoconvert "
        "! video/x-raw,format=BGR,width=320,height=240 "
        "! queue max-size-buffers=1 max-size-bytes=0 "
        "max-size-time=0 leaky=downstream "
        "! appsink name=framesink emit-signals=false "
        "max-buffers=1 drop=true sync=false wait-on-eos=false"
    )

    appsink = pipeline.get_by_name("framesink")
    if appsink is None:
        raise RuntimeError("C6 Python frame validation could not find framesink.")

    state_result = pipeline.set_state(gst.State.PLAYING)
    if state_result == gst.StateChangeReturn.FAILURE:
        raise RuntimeError("C6 Python frame validation pipeline failed to start.")

    try:
        sample = appsink.emit("try-pull-sample", 5 * gst.SECOND)
        if sample is None:
            raise RuntimeError("C6 Python frame validation received no sample.")

        frame = gst_sample_to_bgr_numpy(sample)

        print("C6-3 Python GStreamer frame adapter: PASS")
        print(f"GStreamer: {gst.version_string()}")
        print(f"shape: {frame.shape}")
        print(f"dtype: {frame.dtype}")
        print(f"C-contiguous: {frame.flags.c_contiguous}")
        print(f"owned: {frame.flags.owndata}")
        print("frame_contract: BGR uint8 HWC contiguous")
        print("TensorRT inference: NOT STARTED")
        print("final test used: false")
    finally:
        pipeline.set_state(gst.State.NULL)


if __name__ == "__main__":
    main()
