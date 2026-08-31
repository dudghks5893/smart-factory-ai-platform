"""Preflight and explicitly execute the C4-4 one-time YOLO final test."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.evaluation.yolo_final_test import (
    FINAL_TEST_OUTPUT_ROOT,
    execute_yolo_final_test,
    prepare_yolo_final_test,
)
from ml.training.device import SUPPORTED_DEVICES

DEFAULT_FROZEN_MANIFEST = Path("configs/model/yolo_segmentation_final_candidate.json")


# ADD 2026-09-01: C4-4 preflight inputs와 explicit final-test unlock flag를 정의한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--frozen-candidate", type=Path, default=DEFAULT_FROZEN_MANIFEST)
    parser.add_argument("--official-package", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=FINAL_TEST_OUTPUT_ROOT)
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default="auto")
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Explicitly unlock the one-time derived-test evaluator after preflight.",
    )
    return parser


# ADD 2026-09-01: Preflight를 항상 수행하고 explicit unlock이 없으면 sealed state로 종료한다.
def main() -> int:
    args = build_parser().parse_args()
    preflight = prepare_yolo_final_test(
        frozen_manifest_path=args.frozen_candidate,
        official_package_path=args.official_package,
        dataset_root=args.dataset,
        repository_root=args.repository_root,
        output_root=args.output_root,
        requested_device=args.device,
    )
    print("C4-4 preflight: READY_FOR_FINAL_TEST")
    print(f"Frozen candidate: {preflight.candidate.selected_experiment_id}")
    print(f"Execution commit: {preflight.repository_provenance.git_commit}")
    if not args.confirm_final_test:
        print("FINAL TEST = SEALED_NOT_USED")
        return 0

    artifacts = execute_yolo_final_test(
        preflight,
        confirm_final_test=args.confirm_final_test,
    )
    if artifacts is None:
        raise RuntimeError("Explicit final-test confirmation was not preserved.")
    print("FINAL TEST = FINAL_TEST_COMPLETED")
    print(f"Result: {artifacts.result_path}")
    print(f"Evidence package: {artifacts.package_path}")
    print(f"Evidence package SHA-256: {artifacts.package_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
