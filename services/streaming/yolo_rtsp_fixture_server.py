"""Deterministic localhost RTSP fixture server for C6-4B."""

from __future__ import annotations

import argparse
import importlib


# ADD 2026-09-04: C6-4B fault injection에 사용할 deterministic H264 RTSP fixture launch를 만든다.
def build_fixture_launch(
    *,
    pattern: str,
    width: int,
    height: int,
    framerate: int,
    bitrate_kbps: int,
    key_int_max: int,
) -> str:
    return (
        "("
        f" videotestsrc is-live=true do-timestamp=true pattern={pattern}"
        f" ! video/x-raw,width={width},height={height},framerate={framerate}/1"
        " ! videoconvert"
        f" ! x264enc tune=zerolatency speed-preset=ultrafast key-int-max={key_int_max}"
        f" bitrate={bitrate_kbps}"
        " ! rtph264pay name=pay0 pt=96 config-interval=1"
        " )"
    )


# ADD 2026-09-04: GstRtspServer를 child process로 격리해 fault injection을 허용한다.
def run_fixture_server(
    *,
    host: str,
    port: int,
    mount: str,
    pattern: str,
    width: int,
    height: int,
    framerate: int,
    bitrate_kbps: int,
    key_int_max: int,
) -> None:
    gi = importlib.import_module("gi")
    gi.require_version("Gst", "1.0")
    gi.require_version("GstRtspServer", "1.0")
    glib = importlib.import_module("gi.repository.GLib")
    gst = importlib.import_module("gi.repository.Gst")
    gst_rtsp_server = importlib.import_module("gi.repository.GstRtspServer")

    gst.init(None)

    server = gst_rtsp_server.RTSPServer()
    server.set_address(host)
    server.set_service(str(port))

    factory = gst_rtsp_server.RTSPMediaFactory()
    factory.set_shared(True)
    factory.set_launch(
        build_fixture_launch(
            pattern=pattern,
            width=width,
            height=height,
            framerate=framerate,
            bitrate_kbps=bitrate_kbps,
            key_int_max=key_int_max,
        )
    )

    mounts = server.get_mount_points()
    mounts.add_factory(mount, factory)

    source_id = server.attach(None)
    if source_id == 0:
        raise RuntimeError("Could not attach C6-4B localhost RTSP fixture server.")

    print(f"READY rtsp://{host}:{port}{mount}", flush=True)
    glib.MainLoop().run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mount", required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--framerate", type=int, required=True)
    parser.add_argument("--bitrate-kbps", type=int, required=True)
    parser.add_argument("--key-int-max", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_fixture_server(
        host=args.host,
        port=args.port,
        mount=args.mount,
        pattern=args.pattern,
        width=args.width,
        height=args.height,
        framerate=args.framerate,
        bitrate_kbps=args.bitrate_kbps,
        key_int_max=args.key_int_max,
    )


if __name__ == "__main__":
    main()
