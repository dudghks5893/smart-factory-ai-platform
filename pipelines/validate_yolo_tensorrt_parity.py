"""Collect validation-only PyTorch FP32 versus TensorRT FP16 characterization metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.deployment.yolo_onnx import (
    DEFAULT_FROZEN_MANIFEST,
    prepare_frozen_yolo_source,
)
from ml.deployment.yolo_tensorrt import (
    DEFAULT_TENSORRT_EXPORT_CONFIG,
    load_yolo_tensorrt_export_config,
)
from ml.deployment.yolo_tensorrt_parity import evaluate_frozen_yolo_tensorrt_parity


# ADD 2026-09-02: C5-3B CLI를 exact artifacts와 validation-only dataset으로 제한한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Characterize frozen PyTorch FP32 versus TensorRT FP16 on validation only."
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--frozen-manifest", type=Path, default=DEFAULT_FROZEN_MANIFEST)
    parser.add_argument("--official-package", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_TENSORRT_EXPORT_CONFIG)
    parser.add_argument("--onnx-artifact", type=Path, required=True)
    parser.add_argument("--tensorrt-artifact", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--parity-id", required=True)
    parser.add_argument(
        "--created-at",
        required=True,
        help="Timezone-aware ISO-8601 timestamp for deterministic characterization evidence.",
    )
    return parser


# ADD 2026-09-02: Frozen model과 exact engine을 같은 GPU에서 validation-only로 비교한다.
def main() -> None:
    args = build_parser().parse_args()
    repository_root = args.repository_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repository_root / args.config).resolve()
    )
    config = load_yolo_tensorrt_export_config(config_path)
    source = prepare_frozen_yolo_source(
        repository_root=repository_root,
        manifest_path=args.frozen_manifest,
        package_path=args.official_package,
    )
    result_path = evaluate_frozen_yolo_tensorrt_parity(
        source=source,
        config=config,
        onnx_artifact_dir=args.onnx_artifact.resolve(),
        tensorrt_artifact_dir=args.tensorrt_artifact.resolve(),
        dataset_root=args.dataset_root.resolve(),
        parity_id=args.parity_id,
        created_at=args.created_at,
    )
    print(f"TensorRT parity evidence: {result_path}")
    print("C5-3B state: TENSORRT_FP16_METRICS_COLLECTED_ACCEPTANCE_PENDING")
    print("Split: val")
    print("Test split used: false")


if __name__ == "__main__":
    main()
