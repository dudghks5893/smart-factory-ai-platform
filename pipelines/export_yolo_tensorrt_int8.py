"""CLI for C5-4B2 TensorRT INT8 engine build."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.deployment.yolo_tensorrt_int8_engine import (
    DEFAULT_TENSORRT_INT8_ENGINE_CONFIG,
    export_tensorrt_int8_engine,
    load_yolo_tensorrt_int8_engine_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--qdq-artifact", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_TENSORRT_INT8_ENGINE_CONFIG)
    parser.add_argument("--created-at", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repository_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_yolo_tensorrt_int8_engine_config(config_path)
    artifacts = export_tensorrt_int8_engine(
        repository_root=root,
        qdq_artifact_dir=args.qdq_artifact.resolve(),
        config=config,
        created_at=args.created_at,
    )
    print(f"TensorRT INT8 engine: {artifacts.engine_path}")
    print(f"TensorRT INT8 metadata: {artifacts.metadata_path}")
    print(f"Engine SHA-256: {artifacts.metadata.engine_sha256}")
    print(f"State: {artifacts.metadata.state}")
    print("Explicit Q/DQ: true")
    print("Strongly typed network: true")
    print("Builder INT8 flag: false")
    print("Builder FP16 flag: false")
    print("Legacy calibrator: false")
    print("Validation used: false")
    print("Test split used: false")


if __name__ == "__main__":
    main()
