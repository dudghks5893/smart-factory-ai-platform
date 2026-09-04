"""Run the canonical C6-5C DeepStream/L4 TensorRT INT8 raw-plan build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.streaming.yolo_deepstream_tensorrt_int8 import (
    DEFAULT_DEEPSTREAM_TENSORRT_CONFIG,
)
from services.streaming.yolo_deepstream_tensorrt_int8_build import (
    write_deepstream_tensorrt_build_evidence,
)


# ADD 2026-09-05: Canonical C6-5C source archive와 evidence destination CLI를 정의한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen DeepStream/L4 TensorRT INT8 raw plan "
            "from the accepted C5-4B1 Q/DQ ONNX."
        )
    )

    parser.add_argument(
        "--source-archive",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--evidence-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DEEPSTREAM_TENSORRT_CONFIG,
    )

    return parser


# ADD 2026-09-05: Clean foundation에서 canonical C6-5C plan build를 실행한다.
def main() -> None:
    args = build_parser().parse_args()

    output = write_deepstream_tensorrt_build_evidence(
        source_archive=args.source_archive,
        evidence_output=args.evidence_output,
        repo=Path.cwd(),
        config_path=args.config,
    )

    raw: object = json.loads(output.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise RuntimeError("C6-5C evidence root must be JSON object.")

    artifact = raw["artifact"]

    if not isinstance(
        artifact,
        dict,
    ):
        raise RuntimeError("C6-5C evidence artifact must be JSON object.")

    print("C6-5C DeepStream/L4 TensorRT INT8 build: COMPLETE")
    print(f"Evidence: {output}")
    print(f"State: {raw['state']}")
    print("Plan SHA-256: " + str(artifact["plan_sha256"]))
    print("Plan bytes: " + str(artifact["plan_bytes"]))
    print("Application inference executed: false")
    print("Final test used: false")


if __name__ == "__main__":
    main()
