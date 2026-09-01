"""Build TensorRT FP16 engine from the exact accepted C5-1 ONNX artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.deployment.yolo_tensorrt import (
    DEFAULT_TENSORRT_EXPORT_CONFIG,
    export_frozen_yolo_tensorrt,
    load_yolo_tensorrt_export_config,
)


# ADD 2026-09-02: C5-3A CLI를 exact ONNX artifact와 repository-owned FP16 config로 제한한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build static TensorRT FP16 engine from the exact accepted YOLO ONNX."
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--onnx-artifact", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_TENSORRT_EXPORT_CONFIG)
    parser.add_argument(
        "--created-at",
        required=True,
        help="Timezone-aware ISO-8601 timestamp for deterministic engine metadata.",
    )
    return parser


# ADD 2026-09-02: Exact ONNX identity 검증 뒤 ignored namespace에 TensorRT engine을 생성한다.
def main() -> None:
    args = build_parser().parse_args()
    repository_root = args.repository_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repository_root / args.config).resolve()
    )
    config = load_yolo_tensorrt_export_config(config_path)
    result = export_frozen_yolo_tensorrt(
        repository_root=repository_root,
        onnx_artifact_dir=args.onnx_artifact.resolve(),
        config=config,
        created_at=args.created_at,
    )
    print(f"TensorRT engine: {result.engine_path}")
    print(f"TensorRT metadata: {result.metadata_path}")
    print(f"Engine SHA-256: {result.metadata.engine_sha256}")
    print("C5-3A state: TENSORRT_FP16_ENGINE_BUILT")
    print("Test split used: false")


if __name__ == "__main__":
    main()
