"""Run the canonical C6-3 Python GStreamer frame validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from services.streaming.yolo_gstreamer_python_frame_runtime import (
    DEFAULT_PYTHON_FRAME_CONFIG,
    PYTHON_FRAME_STATE,
    load_python_frame_validation_config,
    run_python_frame_validation,
)


# ADD 2026-09-04: Canonical C6-3 Python frame validation CLI arguments를 정의한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run canonical C6-3 Python GStreamer appsink frame validation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_PYTHON_FRAME_CONFIG)
    return parser


# ADD 2026-09-04: Clean repository에서 Python appsink frame evidence를 생성한다.
def main() -> None:
    args = build_parser().parse_args()
    repo = Path.cwd().resolve()
    config = load_python_frame_validation_config(args.config)
    output_path = run_python_frame_validation(config, repo=repo)

    print("C6-3 canonical Python GStreamer frame validation: PASS")
    print(f"Evidence: {output_path}")
    print(f"State: {PYTHON_FRAME_STATE}")
    print("TensorRT inference: NOT STARTED")
    print("DeepStream integration: NOT STARTED")
    print("final test used: false")


if __name__ == "__main__":
    main()
