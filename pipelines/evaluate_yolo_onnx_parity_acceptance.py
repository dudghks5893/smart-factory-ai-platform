"""Apply the frozen C5-2 ONNX FP32 parity acceptance policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.deployment.yolo_onnx_parity_acceptance import (
    DEFAULT_ACCEPTANCE_POLICY,
    evaluate_yolo_onnx_parity_acceptance,
)


# ADD 2026-09-02: Acceptance CLI는 saved parity JSON만 읽고 inference/dataset path를 받지 않는다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate saved YOLO ONNX FP32 parity evidence against frozen policy v1."
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--parity-evidence", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_ACCEPTANCE_POLICY)
    parser.add_argument("--output-dir", type=Path)
    return parser


# ADD 2026-09-02: Dataset/model을 열지 않고 committed policy와 parity evidence만 판정한다.
def main() -> None:
    args = build_parser().parse_args()
    result_path = evaluate_yolo_onnx_parity_acceptance(
        repository_root=args.repository_root,
        parity_evidence_path=args.parity_evidence,
        policy_path=args.policy,
        output_dir=args.output_dir,
    )
    print(f"Parity acceptance: {result_path}")


if __name__ == "__main__":
    main()
