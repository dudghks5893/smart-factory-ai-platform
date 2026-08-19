"""Validate MVTec AD and generate deterministic dataset artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from ml.datasets.manifest import ManifestRecord, build_mvtec_manifest, write_manifest_csv
from ml.datasets.manifest_validation import validate_manifest_records
from ml.datasets.validation import validate_mvtec_category


@dataclass(frozen=True)
class PreparationConfig:
    dataset_root: Path
    category: str
    validation_ratio: float
    random_seed: int
    manifest_path: Path
    summary_path: Path


@dataclass(frozen=True)
class PreparationSummary:
    category: str
    random_seed: int
    validation_ratio: float
    train_count: int
    validation_count: int
    test_good_count: int
    test_anomaly_count: int
    manifest_count: int
    defect_counts: dict[str, int]
    image_size_counts: dict[str, int]


# ADD 2026-08-18: YAML config root를 mapping으로 로드한다.
def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return raw


# ADD 2026-08-18: Load and validate the MVTec AD preparation configuration.
def load_preparation_config(path: Path) -> PreparationConfig:
    """Load and validate the MVTec AD preparation configuration."""
    raw = _load_yaml(path)

    try:
        dataset = raw["dataset"]
        split = raw["split"]
        output = raw["output"]

        return PreparationConfig(
            dataset_root=Path(dataset["root"]),
            category=str(dataset["category"]),
            validation_ratio=float(split["validation_ratio"]),
            random_seed=int(split["random_seed"]),
            manifest_path=Path(output["manifest_path"]),
            summary_path=Path(output["summary_path"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid preparation config: {path}") from exc


# ADD 2026-08-18: Manifest record에서 dataset preparation summary를 생성한다.
def _build_summary(
    records: list[ManifestRecord],
    config: PreparationConfig,
) -> PreparationSummary:
    test_records = [record for record in records if record.split == "test"]
    defect_counts = Counter(record.defect_type for record in test_records)
    image_size_counts = Counter(f"{record.width}x{record.height}" for record in records)

    return PreparationSummary(
        category=config.category,
        random_seed=config.random_seed,
        validation_ratio=config.validation_ratio,
        train_count=sum(record.split == "train" for record in records),
        validation_count=sum(record.split == "validation" for record in records),
        test_good_count=sum(record.split == "test" and record.label == 0 for record in records),
        test_anomaly_count=sum(record.split == "test" and record.label == 1 for record in records),
        manifest_count=len(records),
        defect_counts=dict(sorted(defect_counts.items())),
        image_size_counts=dict(sorted(image_size_counts.items())),
    )


# ADD 2026-08-18: Dataset preparation summary를 JSON artifact로 저장한다.
def _write_summary(summary: PreparationSummary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ADD 2026-08-18: Validate raw data and create manifest/summary artifacts.
def prepare_mvtec_dataset(config: PreparationConfig) -> PreparationSummary:
    """Validate raw data and create manifest/summary artifacts."""
    # Manifest 생성 전에 raw MVTec category 구조와 image/mask integrity를 검증한다.
    validation_report = validate_mvtec_category(
        config.dataset_root,
        config.category,
    )
    if not validation_report.is_valid:
        issues = [
            *validation_report.errors,
            *validation_report.corrupted_files,
            *(f"missing mask: {path}" for path in validation_report.missing_masks),
            *(f"unexpected mask: {path}" for path in validation_report.unexpected_masks),
        ]
        raise ValueError("Dataset validation failed:\n" + "\n".join(issues))

    # Official test를 보존하면서 train good만 deterministic train/validation으로 분할한다.
    records = build_mvtec_manifest(
        dataset_root=config.dataset_root,
        category=config.category,
        validation_ratio=config.validation_ratio,
        random_seed=config.random_seed,
    )

    # Artifact 저장 전에 record semantics, 중복, path와 dimension을 재검증한다.
    manifest_report = validate_manifest_records(records, config.dataset_root)
    if not manifest_report.is_valid:
        raise ValueError(
            "Manifest integrity validation failed:\n" + "\n".join(manifest_report.errors)
        )

    # 검증된 manifest와 재현성 summary만 interim artifact로 저장한다.
    write_manifest_csv(records, config.manifest_path)

    summary = _build_summary(records, config)
    _write_summary(summary, config.summary_path)
    return summary


# ADD 2026-08-18: CLI 입력 인자를 정의하고 파싱한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an MVTec AD category for downstream ML stages."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/mvtec_ad.yaml"),
        help="Path to the dataset preparation YAML configuration.",
    )
    return parser.parse_args()


# ADD 2026-08-18: CLI 작업 흐름을 조정하고 종료 코드를 반환한다.
def main() -> int:
    # Dataset preparation config를 로드하고 전체 pipeline을 실행한다.
    args = _parse_args()
    config = load_preparation_config(args.config)
    summary = prepare_mvtec_dataset(config)

    print("MVTec AD preparation: PASS")
    print(f"Category: {summary.category}")
    print(f"Train: {summary.train_count}")
    print(f"Validation: {summary.validation_count}")
    print(f"Test good: {summary.test_good_count}")
    print(f"Test anomaly: {summary.test_anomaly_count}")
    print(f"Manifest rows: {summary.manifest_count}")
    print(f"Defect counts: {summary.defect_counts}")
    print(f"Image sizes: {summary.image_size_counts}")
    print(f"Manifest: {config.manifest_path}")
    print(f"Summary: {config.summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
