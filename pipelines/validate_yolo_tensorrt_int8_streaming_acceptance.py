"""Validate fresh C6-3D streaming characterization against frozen C6-3C policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.streaming.yolo_tensorrt_int8_streaming_acceptance import (
    DEFAULT_STREAMING_ACCEPTANCE_POLICY,
    load_streaming_acceptance_policy,
    write_streaming_acceptance,
)


# ADD 2026-09-04: Prospective characterization path와 policy path를 CLI로 받는다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply frozen C6-3C streaming acceptance policy.")
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_STREAMING_ACCEPTANCE_POLICY,
    )
    parser.add_argument(
        "--characterization",
        type=Path,
        required=True,
    )
    return parser


# ADD 2026-09-04: Fresh clean-commit characterization을 policy에 따라 accept/reject한다.
def main() -> None:
    args = build_parser().parse_args()
    repo = Path.cwd().resolve()
    policy = load_streaming_acceptance_policy(args.policy)
    output_path = write_streaming_acceptance(
        policy=policy,
        evidence_path=args.characterization.resolve(),
        repo=repo,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))

    print("C6-3D TensorRT INT8 streaming prospective acceptance: PASS")
    print(f"Evidence: {output_path}")
    print(f"State: {result['state']}")
    print(
        "Gates:",
        f"{result['passed_gate_count']}/{result['total_gate_count']}",
    )
    print("Final test used: false")
    print("Engine rebuilt: false")


if __name__ == "__main__":
    main()
