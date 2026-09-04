"""Run the canonical C6-5B DeepStream NVDEC/NVMM smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.streaming.yolo_deepstream_nvmm_smoke import (
    DEFAULT_DEEPSTREAM_NVMM_CONFIG,
    write_deepstream_nvmm_evidence,
)


# ADD 2026-09-05: C6-5B config와 external evidence destination CLI를 정의한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen DeepStream NVDEC to NVMM GPU-path smoke without TensorRT inference."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DEEPSTREAM_NVMM_CONFIG,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser


# ADD 2026-09-05: Clean foundation commit에서 C6-5B canonical evidence를 생성한다.
def main() -> None:
    args = build_parser().parse_args()

    output = write_deepstream_nvmm_evidence(
        output_path=args.output,
        repo=Path.cwd(),
        config_path=args.config,
    )

    raw: object = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("C6-5B evidence root must be a JSON object.")

    pipeline = raw["pipeline"]

    print("C6-5B DeepStream NVDEC/NVMM smoke: COMPLETE")
    print(f"Evidence: {output}")
    print(f"State: {raw['state']}")
    print(f"Decoder: {pipeline['decoder_format']} / NVMM")
    print(f"Converter: {pipeline['converter_format']} / NVMM")
    print("Inference executed: false")
    print("TensorRT engine used: false")
    print("Final test used: false")


if __name__ == "__main__":
    main()
