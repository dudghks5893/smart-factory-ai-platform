"""Tests for generated manifest integrity validation."""

from pathlib import Path

from PIL import Image

from ml.datasets.manifest import ManifestRecord
from ml.datasets.manifest_validation import validate_manifest_records


def _write_png(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)


def _valid_record(dataset_root: Path) -> ManifestRecord:
    image_path = dataset_root / "metal_nut/train/good/000.png"
    _write_png(image_path)

    return ManifestRecord(
        sample_id="metal_nut_train_good_000",
        category="metal_nut",
        source_split="train",
        split="train",
        defect_type="good",
        label=0,
        image_path="metal_nut/train/good/000.png",
        mask_path="",
        width=8,
        height=8,
    )


def test_manifest_integrity_accepts_valid_record(tmp_path: Path) -> None:
    dataset_root = tmp_path / "mvtec_ad"
    record = _valid_record(dataset_root)

    report = validate_manifest_records([record], dataset_root)

    assert report.is_valid
    assert report.record_count == 1


def test_manifest_integrity_detects_duplicate_sample_id(tmp_path: Path) -> None:
    dataset_root = tmp_path / "mvtec_ad"
    record = _valid_record(dataset_root)

    report = validate_manifest_records([record, record], dataset_root)

    assert not report.is_valid
    assert "duplicate sample_id: metal_nut_train_good_000" in report.errors
