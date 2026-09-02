"""Quantize the exact accepted YOLO ONNX into an explicit INT8 Q/DQ ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.deployment.yolo_tensorrt_int8 import (
    DEFAULT_TENSORRT_INT8_CONFIG,
    load_yolo_tensorrt_int8_config,
)
from ml.deployment.yolo_tensorrt_int8_quantization import export_int8_qdq_onnx


# ADD 2026-09-02: C5-4B1 CLI를 exact ONNX, frozen dataset, committed INT8 contract로 제한한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantize the accepted YOLO FP32 ONNX into a ModelOpt INT8 Q/DQ ONNX."
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--onnx-artifact", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_TENSORRT_INT8_CONFIG)
    parser.add_argument(
        "--created-at",
        required=True,
        help="Timezone-aware ISO-8601 timestamp for deterministic quantization metadata.",
    )
    return parser


# ADD 2026-09-02: Train-only calibration으로 Q/DQ ONNX를 생성하고 provenance identity를 출력한다.
def main() -> None:
    args = build_parser().parse_args()
    repository_root = args.repository_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repository_root / args.config).resolve()
    )
    config = load_yolo_tensorrt_int8_config(config_path)
    result = export_int8_qdq_onnx(
        repository_root=repository_root,
        onnx_artifact_dir=args.onnx_artifact.resolve(),
        dataset_root=args.dataset_root.resolve(),
        config=config,
        created_at=args.created_at,
    )
    print(f"INT8 Q/DQ ONNX: {result.onnx_path}")
    print(f"INT8 Q/DQ metadata: {result.metadata_path}")
    print(f"Q/DQ ONNX SHA-256: {result.metadata.quantized_onnx_sha256}")
    print(f"QuantizeLinear nodes: {result.metadata.quantize_linear_count}")
    print(f"DequantizeLinear nodes: {result.metadata.dequantize_linear_count}")
    print("C5-4B1 state: INT8_QDQ_ONNX_QUANTIZED")
    print("Calibration split: train")
    print("Validation used: false")
    print("Test split used: false")


if __name__ == "__main__":
    main()
