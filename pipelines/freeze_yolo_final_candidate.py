"""Freeze one Official validation-confirmed YOLO candidate before final-test access."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.experiments.yolo_final_candidate import (
    build_final_candidate_manifest,
    load_official_candidate_evidence,
    write_final_candidate_manifest,
)

DEFAULT_OUTPUT_PATH = Path("configs/model/yolo_segmentation_final_candidate.json")


# ADD 2026-09-01: Official package를 검증하고 validation-only final candidate pointer를 고정한다.
def freeze_yolo_final_candidate(
    *,
    official_package_path: Path,
    expected_package_sha256: str,
    output_path: Path,
    selected_at: str,
) -> Path:
    # Official package bytes와 experiment/model/metadata/config provenance를 먼저 검증한다.
    evidence = load_official_candidate_evidence(
        official_package_path,
        expected_package_sha256=expected_package_sha256,
    )

    # Derived test가 봉인된 confirmed candidate만 immutable repository pointer로 저장한다.
    manifest = build_final_candidate_manifest(evidence, selected_at=selected_at)
    return write_final_candidate_manifest(manifest, output_path)


# ADD 2026-09-01: C4-3 freeze CLI argument를 정의한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze one Official validation-confirmed YOLO candidate without test access."
    )
    parser.add_argument("--official-package", type=Path, required=True)
    parser.add_argument("--expected-package-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--selected-at",
        required=True,
        help="Timezone-aware ISO-8601 evidence timestamp required for deterministic regeneration.",
    )
    return parser


# ADD 2026-09-01: C4-3 freeze를 실행하고 final-test seal을 명시한다.
def main() -> None:
    args = build_parser().parse_args()
    output_path = freeze_yolo_final_candidate(
        official_package_path=args.official_package,
        expected_package_sha256=args.expected_package_sha256,
        output_path=args.output,
        selected_at=args.selected_at,
    )
    print(f"Final candidate manifest: {output_path}")
    print("Selection state: FINAL_CANDIDATE_FROZEN")
    print("FINAL TEST = SEALED_NOT_USED")


if __name__ == "__main__":
    main()
