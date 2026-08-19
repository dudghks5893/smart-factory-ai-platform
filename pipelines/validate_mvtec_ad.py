"""CLI for validating an MVTec AD category."""

import argparse
from pathlib import Path

from ml.datasets.constants import MVTEC_AD_CATEGORIES
from ml.datasets.validation import DatasetValidationReport, validate_mvtec_category


# ADD 2026-08-18: CLI 입력 인자를 정의하고 파싱한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an MVTec AD category.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/mvtec_ad"),
        help="Path containing the MVTec AD category directories.",
    )
    parser.add_argument(
        "--category",
        choices=MVTEC_AD_CATEGORIES,
        default="metal_nut",
        help="MVTec AD category to validate.",
    )
    return parser.parse_args()


# ADD 2026-08-18: Dataset validation report를 CLI 형식으로 출력한다.
def _print_report(report: DatasetValidationReport) -> None:
    print(f"Dataset validation: {'PASS' if report.is_valid else 'FAIL'}")
    print(f"Category: {report.category}")
    print(f"Train good: {report.train_good_count}")
    print(f"Test good: {report.test_good_count}")
    print(f"Test anomaly: {report.test_anomaly_count}")
    print(f"Ground-truth masks: {report.mask_count}")

    print("Defect types:")
    for defect_type, count in sorted(report.defect_counts.items()):
        print(f"  - {defect_type}: {count}")

    print(f"Corrupted files: {len(report.corrupted_files)}")
    print(f"Missing masks: {len(report.missing_masks)}")
    print(f"Unexpected masks: {len(report.unexpected_masks)}")

    issues = [
        *report.errors,
        *report.corrupted_files,
        *(f"missing mask: {path}" for path in report.missing_masks),
        *(f"unexpected mask: {path}" for path in report.unexpected_masks),
    ]
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")


# ADD 2026-08-18: CLI 작업 흐름을 조정하고 종료 코드를 반환한다.
def main() -> int:
    # 지정한 MVTec category를 검증하고 CLI report를 출력한다.
    args = _parse_args()
    report = validate_mvtec_category(args.dataset_root, args.category)
    _print_report(report)
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
