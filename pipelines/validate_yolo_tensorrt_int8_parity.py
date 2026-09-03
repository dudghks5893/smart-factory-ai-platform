"""CLI for C5-4C validation-only TensorRT INT8 characterization."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.deployment.yolo_onnx import DEFAULT_FROZEN_MANIFEST, prepare_frozen_yolo_source
from ml.deployment.yolo_tensorrt_int8 import (
    DEFAULT_TENSORRT_INT8_CONFIG,
    load_yolo_tensorrt_int8_config,
)
from ml.deployment.yolo_tensorrt_int8_engine import (
    DEFAULT_TENSORRT_INT8_ENGINE_CONFIG,
    load_yolo_tensorrt_int8_engine_config,
)
from ml.deployment.yolo_tensorrt_int8_parity import (
    DEFAULT_TENSORRT_INT8_CHARACTERIZATION_CONFIG,
    evaluate_frozen_yolo_tensorrt_int8_characterization,
    load_yolo_tensorrt_int8_characterization_config,
)


# ADD 2026-09-04: C5-4C CLI를 exact engine과 validation-only dataset boundary로 제한한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Characterize frozen PyTorch FP32 versus exact TensorRT INT8 on validation only."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--frozen-manifest", type=Path, default=DEFAULT_FROZEN_MANIFEST)
    parser.add_argument("--official-package", type=Path, required=True)
    parser.add_argument("--int8-config", type=Path, default=DEFAULT_TENSORRT_INT8_CONFIG)
    parser.add_argument(
        "--engine-config",
        type=Path,
        default=DEFAULT_TENSORRT_INT8_ENGINE_CONFIG,
    )
    parser.add_argument(
        "--characterization-config",
        type=Path,
        default=DEFAULT_TENSORRT_INT8_CHARACTERIZATION_CONFIG,
    )
    parser.add_argument("--int8-engine-artifact", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


# ADD 2026-09-04: Exact B2 engine을 rebuild 없이 복원해 C5-4C metrics-only evidence를 만든다.
def main() -> None:
    args = build_parser().parse_args()
    root = args.repository_root.resolve()
    int8_config = load_yolo_tensorrt_int8_config(_resolve(root, args.int8_config))
    engine_config = load_yolo_tensorrt_int8_engine_config(_resolve(root, args.engine_config))
    characterization_config = load_yolo_tensorrt_int8_characterization_config(
        _resolve(root, args.characterization_config)
    )
    source = prepare_frozen_yolo_source(
        repository_root=root,
        manifest_path=_resolve(root, args.frozen_manifest),
        package_path=args.official_package.resolve(),
    )
    result_path = evaluate_frozen_yolo_tensorrt_int8_characterization(
        source=source,
        int8_config=int8_config,
        engine_config=engine_config,
        characterization_config=characterization_config,
        int8_engine_artifact_dir=args.int8_engine_artifact.resolve(),
        dataset_root=args.dataset_root.resolve(),
        created_at=args.created_at,
    )
    print(f"TensorRT INT8 characterization evidence: {result_path}")
    print("C5-4C state: TENSORRT_INT8_METRICS_COLLECTED_ACCEPTANCE_PENDING")
    print("Split: val")
    print("Validation samples: 28")
    print("Numeric thresholds: none")
    print("Test split used: false")
    print("C5-4D has NOT started.")


if __name__ == "__main__":
    main()
