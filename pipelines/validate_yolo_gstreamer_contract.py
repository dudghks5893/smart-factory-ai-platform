"""Validate and print the C6-1 GStreamer ingress contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from services.streaming.yolo_gstreamer import (
    DEFAULT_GSTREAMER_CONFIG,
    build_yolo_gstreamer_pipeline,
    detect_gstreamer_launcher,
    load_yolo_gstreamer_ingress_config,
)


# ADD 2026-09-04: C6-1 contract validation CLI arguments를 정의한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate C6-1 YOLO GStreamer ingress contract.")
    parser.add_argument("--config", type=Path, default=DEFAULT_GSTREAMER_CONFIG)
    parser.add_argument("--source", help="Optional local file path for source.kind=file")
    parser.add_argument(
        "--require-gstreamer",
        action="store_true",
        help="Fail if gst-launch-1.0 is unavailable on this host.",
    )
    return parser


# ADD 2026-09-04: Native runtime 실행 전 repository contract와 pipeline string을 검증한다.
def main() -> None:
    args = build_parser().parse_args()
    config = load_yolo_gstreamer_ingress_config(args.config)
    pipeline = build_yolo_gstreamer_pipeline(config, source_override=args.source)
    launcher = detect_gstreamer_launcher()

    if args.require_gstreamer and launcher is None:
        raise RuntimeError("gst-launch-1.0 is required but not installed.")

    print("C6-1 GStreamer ingress contract: PASS")
    print(f"Pipeline ID: {config.pipeline_id}")
    print(f"Source kind: {config.source.kind}")
    print(f"Frame contract: {config.frame_contract.pixel_format} uint8 HWC")
    print(f"Backpressure: {config.latency_policy.mode}")
    print(f"GStreamer launcher: {launcher or '<not installed>'}")
    print("Pipeline:")
    print(pipeline)
    print("C6-2 native GStreamer runtime smoke test has NOT started.")


if __name__ == "__main__":
    main()
