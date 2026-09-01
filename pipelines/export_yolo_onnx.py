"""Export the exact C4-3 frozen YOLO candidate to conservative FP32 ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.deployment.yolo_onnx import (
    DEFAULT_EXPORT_CONFIG,
    DEFAULT_FROZEN_MANIFEST,
    export_frozen_yolo_onnx,
    load_yolo_onnx_export_config,
    prepare_frozen_yolo_source,
)


# ADD 2026-09-02: C5-1 CLI를 frozen source와 repository-owned export config로 제한한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the frozen C4-3 YOLO candidate to static FP32 ONNX."
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--frozen-manifest", type=Path, default=DEFAULT_FROZEN_MANIFEST)
    parser.add_argument("--official-package", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPORT_CONFIG)
    parser.add_argument(
        "--created-at",
        required=True,
        help="Timezone-aware ISO-8601 timestamp for deterministic export metadata.",
    )
    return parser


# ADD 2026-09-02: Frozen identity 검증 뒤 ignored namespace에 ONNX artifact를 생성한다.
def main() -> None:
    args = build_parser().parse_args()
    repository_root = args.repository_root.resolve()

    # Repository config와 C4-3 pointer, Official package identity를 먼저 검증한다.
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repository_root / args.config).resolve()
    )
    config = load_yolo_onnx_export_config(config_path)
    source = prepare_frozen_yolo_source(
        repository_root=repository_root,
        manifest_path=args.frozen_manifest,
        package_path=args.official_package,
    )

    # Exact checkpoint를 static FP32 graph로 export하고 hash/provenance metadata를 저장한다.
    result = export_frozen_yolo_onnx(
        source=source,
        config=config,
        created_at=args.created_at,
    )
    print(f"ONNX model: {result.model_path}")
    print(f"ONNX metadata: {result.metadata_path}")
    print(f"ONNX SHA-256: {result.metadata.onnx_sha256}")
    print("C5-1 state: ONNX_EXPORT_COMPLETED")
    print("Test split used: false")


if __name__ == "__main__":
    main()
