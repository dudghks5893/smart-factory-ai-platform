"""Manifest generation and loading utilities for MVTec AD."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from ml.datasets.constants import GOOD_DIR_NAME, MASK_SUFFIX
from ml.datasets.splits import deterministic_train_validation_split


@dataclass(frozen=True)
class ManifestRecord:
    """One logical dataset sample recorded in the generated manifest."""

    sample_id: str
    category: str
    source_split: str
    split: str
    defect_type: str
    label: int
    image_path: str
    mask_path: str
    width: int
    height: int


MANIFEST_FIELDS = tuple(ManifestRecord.__dataclass_fields__)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _relative(path: Path, dataset_root: Path) -> str:
    return str(path.relative_to(dataset_root))


def _build_record(
    *,
    dataset_root: Path,
    category: str,
    image_path: Path,
    source_split: str,
    split: str,
    defect_type: str,
    label: int,
    mask_path: Path | None,
) -> ManifestRecord:
    width, height = _image_size(image_path)
    relative_image_path = _relative(image_path, dataset_root)
    relative_mask_path = "" if mask_path is None else _relative(mask_path, dataset_root)

    sample_id = f"{category}_{source_split}_{defect_type}_{image_path.stem}"

    return ManifestRecord(
        sample_id=sample_id,
        category=category,
        source_split=source_split,
        split=split,
        defect_type=defect_type,
        label=label,
        image_path=relative_image_path,
        mask_path=relative_mask_path,
        width=width,
        height=height,
    )


def build_mvtec_manifest(
    dataset_root: Path,
    category: str,
    validation_ratio: float,
    random_seed: int,
) -> list[ManifestRecord]:
    """Build a manifest while preserving the official MVTec AD test split."""
    category_root = dataset_root / category
    train_good_root = category_root / "train" / GOOD_DIR_NAME
    test_root = category_root / "test"
    ground_truth_root = category_root / "ground_truth"

    train_good_images = sorted(train_good_root.glob("*.png"))
    train_images, validation_images = deterministic_train_validation_split(
        train_good_images,
        validation_ratio=validation_ratio,
        random_seed=random_seed,
    )

    records: list[ManifestRecord] = []

    for image_path in train_images:
        records.append(
            _build_record(
                dataset_root=dataset_root,
                category=category,
                image_path=image_path,
                source_split="train",
                split="train",
                defect_type=GOOD_DIR_NAME,
                label=0,
                mask_path=None,
            )
        )

    for image_path in validation_images:
        records.append(
            _build_record(
                dataset_root=dataset_root,
                category=category,
                image_path=image_path,
                source_split="train",
                split="validation",
                defect_type=GOOD_DIR_NAME,
                label=0,
                mask_path=None,
            )
        )

    for defect_directory in sorted(path for path in test_root.iterdir() if path.is_dir()):
        defect_type = defect_directory.name
        label = 0 if defect_type == GOOD_DIR_NAME else 1

        for image_path in sorted(defect_directory.glob("*.png")):
            mask_path = None
            if label == 1:
                mask_path = ground_truth_root / defect_type / f"{image_path.stem}{MASK_SUFFIX}"

            records.append(
                _build_record(
                    dataset_root=dataset_root,
                    category=category,
                    image_path=image_path,
                    source_split="test",
                    split="test",
                    defect_type=defect_type,
                    label=label,
                    mask_path=mask_path,
                )
            )

    return records


def write_manifest_csv(records: list[ManifestRecord], output_path: Path) -> None:
    """Write manifest records to CSV in a stable field order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _required_csv_value(row: dict[str, str | None], field: str) -> str:
    value = row.get(field)
    if value is None:
        raise ValueError(f"Manifest field is missing: {field}")
    return value


def read_manifest_csv(manifest_path: Path) -> list[ManifestRecord]:
    """Read manifest records from CSV with explicit type conversion."""
    records: list[ManifestRecord] = []

    with manifest_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(MANIFEST_FIELDS):
            raise ValueError(
                f"Unexpected manifest columns: {reader.fieldnames}; "
                f"expected: {list(MANIFEST_FIELDS)}"
            )

        for row in reader:
            records.append(
                ManifestRecord(
                    sample_id=_required_csv_value(row, "sample_id"),
                    category=_required_csv_value(row, "category"),
                    source_split=_required_csv_value(row, "source_split"),
                    split=_required_csv_value(row, "split"),
                    defect_type=_required_csv_value(row, "defect_type"),
                    label=int(_required_csv_value(row, "label")),
                    image_path=_required_csv_value(row, "image_path"),
                    mask_path=_required_csv_value(row, "mask_path"),
                    width=int(_required_csv_value(row, "width")),
                    height=int(_required_csv_value(row, "height")),
                )
            )

    return records
