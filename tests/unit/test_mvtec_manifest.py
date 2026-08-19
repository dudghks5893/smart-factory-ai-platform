"""Tests for MVTec AD manifest generation."""

from pathlib import Path

from PIL import Image

from ml.datasets.manifest import build_mvtec_manifest


# ADD 2026-08-18: 테스트 fixture 파일을 생성한다.
def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8)).save(path)


# ADD 2026-08-18: 테스트에 필요한 dataset fixture를 구성한다.
def _build_dataset(root: Path) -> Path:
    dataset_root = root / "mvtec_ad"

    for index in range(10):
        _write_png(dataset_root / f"metal_nut/train/good/{index:03d}.png")

    _write_png(dataset_root / "metal_nut/test/good/000.png")
    _write_png(dataset_root / "metal_nut/test/bent/000.png")
    _write_png(dataset_root / "metal_nut/ground_truth/bent/000_mask.png")

    return dataset_root


# ADD 2026-08-18: manifest preserves source split and internal split 테스트 시나리오를 검증한다.
def test_manifest_preserves_source_split_and_internal_split(tmp_path: Path) -> None:
    dataset_root = _build_dataset(tmp_path)

    records = build_mvtec_manifest(
        dataset_root=dataset_root,
        category="metal_nut",
        validation_ratio=0.2,
        random_seed=42,
    )

    train_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "validation"]
    test_records = [record for record in records if record.split == "test"]

    assert len(train_records) == 8
    assert len(validation_records) == 2
    assert len(test_records) == 2

    assert all(record.source_split == "train" for record in train_records)
    assert all(record.source_split == "train" for record in validation_records)
    assert all(record.source_split == "test" for record in test_records)


# ADD 2026-08-18: manifest records anomaly mask and metadata 테스트 시나리오를 검증한다.
def test_manifest_records_anomaly_mask_and_metadata(tmp_path: Path) -> None:
    dataset_root = _build_dataset(tmp_path)

    records = build_mvtec_manifest(
        dataset_root=dataset_root,
        category="metal_nut",
        validation_ratio=0.2,
        random_seed=42,
    )

    anomaly_record = next(record for record in records if record.label == 1)

    assert anomaly_record.defect_type == "bent"
    assert anomaly_record.mask_path == "metal_nut/ground_truth/bent/000_mask.png"
    assert anomaly_record.width == 12
    assert anomaly_record.height == 8
