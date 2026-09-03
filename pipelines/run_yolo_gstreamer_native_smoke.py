"""Run the canonical C6-2 native GStreamer smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

from services.streaming.yolo_gstreamer_native_smoke import (
    DEFAULT_NATIVE_SMOKE_CONFIG,
    load_native_gstreamer_smoke_config,
    run_native_gstreamer_smoke,
)


# ADD 2026-09-04: Canonical C6-2 native smoke CLI arguments를 정의한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run C6-2 native GStreamer synthetic/file smoke test."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_NATIVE_SMOKE_CONFIG)
    return parser


# ADD 2026-09-04: Clean repository에서 C6-2 native smoke를 실행하고 evidence path를 출력한다.
def main() -> None:
    args = build_parser().parse_args()
    repo = Path.cwd().resolve()
    config = load_native_gstreamer_smoke_config(args.config)
    output_path = run_native_gstreamer_smoke(config, repo=repo)

    print("C6-2 native GStreamer smoke: PASS")
    print(f"Evidence: {output_path}")
    print("State: NATIVE_GSTREAMER_SMOKE_COMPLETED")
    print("TensorRT inference: NOT STARTED")
    print("DeepStream integration: NOT STARTED")


if __name__ == "__main__":
    main()
