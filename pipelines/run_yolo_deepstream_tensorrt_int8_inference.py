"""Run canonical C6-5C DeepStream TensorRT INT8 inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.streaming.yolo_deepstream_tensorrt_int8_inference import (
    DEFAULT_DEEPSTREAM_INFERENCE_CONFIG,
    write_deepstream_tensorrt_inference_evidence,
)


# ADD 2026-09-05: Canonical C6-5C inference evidence CLI arguments를 정의한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run the frozen DeepStream/L4 TensorRT INT8 30-frame inference path.")
    )

    parser.add_argument(
        "--evidence-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DEEPSTREAM_INFERENCE_CONFIG,
    )

    return parser


# ADD 2026-09-05: Canonical C6-5C inference를 실행하고 identity를 출력한다.
def main() -> None:
    args = build_parser().parse_args()

    output = write_deepstream_tensorrt_inference_evidence(
        evidence_output=args.evidence_output,
        repo=Path.cwd(),
        inference_config_path=args.config,
    )

    raw: object = json.loads(output.read_text(encoding="utf-8"))

    if not isinstance(
        raw,
        dict,
    ):
        raise RuntimeError("C6-5C inference evidence root must be object.")

    result = raw["result"]

    if not isinstance(
        result,
        dict,
    ):
        raise RuntimeError("C6-5C inference result must be object.")

    print("C6-5C DeepStream TensorRT INT8 inference: COMPLETE")
    print(f"Evidence: {output}")
    print(f"State: {raw['state']}")
    print("Plan SHA-256: " + str(result["plan_sha256"]))
    print("Post-inference frames: " + str(result["post_inference_eos_after"]))
    print("Application inference executed: true")
    print("Segmentation decode executed: false")
    print("Final test used: false")


if __name__ == "__main__":
    main()
