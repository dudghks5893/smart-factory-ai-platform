"""Collect validation-only PyTorch and ONNX Runtime parity evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.deployment.yolo_onnx import (
    DEFAULT_EXPORT_CONFIG,
    DEFAULT_FROZEN_MANIFEST,
    load_yolo_onnx_export_config,
    prepare_frozen_yolo_source,
)
from ml.deployment.yolo_onnx_parity import evaluate_frozen_yolo_onnx_parity


# ADD 2026-09-02: C5-2 CLI를 validation-only dataset과 exact artifacts로 제한한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect frozen PyTorch versus ONNX Runtime validation parity metrics."
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--frozen-manifest", type=Path, default=DEFAULT_FROZEN_MANIFEST)
    parser.add_argument("--official-package", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPORT_CONFIG)
    parser.add_argument("--onnx-artifact", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--parity-id", required=True)
    parser.add_argument(
        "--created-at",
        required=True,
        help="Timezone-aware ISO-8601 timestamp for deterministic parity evidence.",
    )
    return parser


# ADD 2026-09-02: Test를 배제한 validation records에서 두 backend evidence를 생성한다.
def main() -> None:
    args = build_parser().parse_args()
    repository_root = args.repository_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repository_root / args.config).resolve()
    )

    # Frozen PyTorch와 exported ONNX identity를 검증한 뒤 validation content만 연다.
    config = load_yolo_onnx_export_config(config_path)
    source = prepare_frozen_yolo_source(
        repository_root=repository_root,
        manifest_path=args.frozen_manifest,
        package_path=args.official_package,
    )
    result_path = evaluate_frozen_yolo_onnx_parity(
        source=source,
        config=config,
        onnx_artifact_dir=args.onnx_artifact.resolve(),
        dataset_root=args.dataset_root.resolve(),
        parity_id=args.parity_id,
        created_at=args.created_at,
    )
    print(f"Parity evidence: {result_path}")
    print("C5-2 state: METRICS_COLLECTED_ACCEPTANCE_PENDING")
    print("Split: val")
    print("Test split used: false")


if __name__ == "__main__":
    main()
