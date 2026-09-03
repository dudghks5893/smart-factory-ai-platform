"""Apply the frozen C5-4D TensorRT INT8 parity acceptance policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.deployment.yolo_tensorrt_int8_parity_acceptance import (
    DEFAULT_ACCEPTANCE_POLICY,
    evaluate_yolo_tensorrt_int8_parity_acceptance,
)


# ADD 2026-09-04: Acceptance CLI를 fresh policy-commit INT8 evidence와 frozen policy로 제한한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fresh YOLO TensorRT INT8 validation evidence against frozen C5-4D policy v1."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--characterization-evidence", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_ACCEPTANCE_POLICY)
    parser.add_argument("--output-dir", type=Path)
    return parser


# ADD 2026-09-04: Inference/dataset을 열지 않고 committed policy와 saved evidence만 판정한다.
def main() -> None:
    args = build_parser().parse_args()
    result_path = evaluate_yolo_tensorrt_int8_parity_acceptance(
        repository_root=args.repository_root,
        characterization_evidence_path=args.characterization_evidence,
        policy_path=args.policy,
        output_dir=args.output_dir,
    )
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    print(f"TensorRT INT8 parity acceptance: {result_path}")
    print(f"State: {raw['state']}")
    print(f"Accepted: {str(raw['accepted']).lower()}")
    print(
        f"Checks: {sum(bool(item['passed']) for item in raw['checks'])}/{len(raw['checks'])} PASS"
    )


if __name__ == "__main__":
    main()
