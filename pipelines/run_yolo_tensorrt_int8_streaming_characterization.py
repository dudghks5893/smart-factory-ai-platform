"""Run C6-3B TensorRT INT8 streaming characterization."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.deployment.yolo_tensorrt_int8_engine import (
    DEFAULT_TENSORRT_INT8_ENGINE_CONFIG,
    load_yolo_tensorrt_int8_engine_config,
)
from services.streaming.yolo_tensorrt_int8_streaming import (
    DEFAULT_STREAMING_CHARACTERIZATION_CONFIG,
    STREAMING_CHARACTERIZATION_STATE,
    load_streaming_characterization_config,
    run_streaming_characterization,
)


# ADD 2026-09-04: C6-3B canonical runner arguments를 정의한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run threshold-free C6-3B TensorRT INT8 streaming characterization."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_STREAMING_CHARACTERIZATION_CONFIG,
    )
    parser.add_argument(
        "--engine-artifact-dir",
        type=Path,
        default=None,
        help="Exact restored C5-4B2 artifact directory. Defaults to repository canonical path.",
    )
    return parser


# ADD 2026-09-04: Exact engine을 rebuild 없이 사용해 streaming metrics evidence를 생성한다.
def main() -> None:
    args = build_parser().parse_args()
    repo = Path.cwd().resolve()
    config = load_streaming_characterization_config(args.config)

    if args.engine_artifact_dir is None:
        engine_config = load_yolo_tensorrt_int8_engine_config(
            repo / DEFAULT_TENSORRT_INT8_ENGINE_CONFIG
        )
        engine_artifact_dir = (repo / engine_config.output_root / engine_config.export_id).resolve()
    else:
        engine_artifact_dir = args.engine_artifact_dir.resolve()

    output_path = run_streaming_characterization(
        config=config,
        repo=repo,
        engine_artifact_dir=engine_artifact_dir,
    )

    print("C6-3B TensorRT INT8 streaming characterization: PASS")
    print(f"Evidence: {output_path}")
    print(f"State: {STREAMING_CHARACTERIZATION_STATE}")
    print("Acceptance: PENDING_TENSORRT_STREAMING_TOLERANCE_APPROVAL")
    print("Engine rebuilt: false")
    print("Dataset used: false")
    print("Final test used: false")
    print("DeepStream used: false")


if __name__ == "__main__":
    main()
