"""Run the canonical C6-5A DeepStream TensorRT compatibility probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.streaming.yolo_deepstream_compatibility import (
    DEFAULT_DEEPSTREAM_COMPATIBILITY_CONFIG,
    write_deepstream_compatibility_evidence,
)


# ADD 2026-09-04: C6-5A source bundle, config, evidence destination을 CLI로 정의한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe the exact accepted C5 TensorRT engine "
            "inside the frozen C6-5A DeepStream runtime."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DEEPSTREAM_COMPATIBILITY_CONFIG,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser


# ADD 2026-09-04: Clean commit에서 deserialize-only observation을 canonical evidence로 기록한다.
def main() -> None:
    args = build_parser().parse_args()

    output_path = write_deepstream_compatibility_evidence(
        input_dir=args.input_dir.resolve(),
        output_path=args.output.resolve(),
        repo=Path.cwd().resolve(),
        config_path=args.config,
    )

    raw: object = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("C6-5A evidence root must be a JSON object.")

    print("C6-5A DeepStream compatibility observation: COMPLETE")
    print(f"Evidence: {output_path}")
    print(f"State: {raw['state']}")
    print(f"Compatibility: {raw['compatibility']}")
    print(f"Reason: {raw['reason']}")
    print("Inference executed: false")
    print("Engine rebuilt: false")
    print("Final test used: false")


if __name__ == "__main__":
    main()
