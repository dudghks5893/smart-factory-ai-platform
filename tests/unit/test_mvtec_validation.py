"""Tests for MVTec AD dataset validation."""

from pathlib import Path

from PIL import Image

from ml.datasets.validation import validate_mvtec_category


# ADD 2026-08-18: 테스트 fixture 파일을 생성한다.
def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (8, 8))
    image.save(path)


# ADD 2026-08-18: 테스트에 필요한 dataset fixture를 구성한다.
def _build_valid_metal_nut_dataset(root: Path) -> Path:
    dataset_root = root / "mvtec_ad"

    _write_png(dataset_root / "metal_nut/train/good/000.png")
    _write_png(dataset_root / "metal_nut/train/good/001.png")
    _write_png(dataset_root / "metal_nut/test/good/000.png")

    _write_png(dataset_root / "metal_nut/test/bent/000.png")
    _write_png(dataset_root / "metal_nut/test/bent/001.png")
    _write_png(dataset_root / "metal_nut/ground_truth/bent/000_mask.png")
    _write_png(dataset_root / "metal_nut/ground_truth/bent/001_mask.png")

    return dataset_root


# ADD 2026-08-18: validate mvtec category accepts valid dataset 테스트 시나리오를 검증한다.
def test_validate_mvtec_category_accepts_valid_dataset(tmp_path: Path) -> None:
    dataset_root = _build_valid_metal_nut_dataset(tmp_path)

    report = validate_mvtec_category(dataset_root, "metal_nut")

    assert report.is_valid
    assert report.train_good_count == 2
    assert report.test_good_count == 1
    assert report.test_anomaly_count == 2
    assert report.mask_count == 2
    assert report.defect_counts == {"bent": 2}


# ADD 2026-08-18: validate mvtec category detects missing mask 테스트 시나리오를 검증한다.
def test_validate_mvtec_category_detects_missing_mask(tmp_path: Path) -> None:
    dataset_root = _build_valid_metal_nut_dataset(tmp_path)
    missing_mask = dataset_root / "metal_nut/ground_truth/bent/001_mask.png"
    missing_mask.unlink()

    report = validate_mvtec_category(dataset_root, "metal_nut")

    assert not report.is_valid
    assert report.missing_masks == ["metal_nut/ground_truth/bent/001_mask.png"]


# ADD 2026-08-18: validate mvtec category detects corrupted image 테스트 시나리오를 검증한다.
def test_validate_mvtec_category_detects_corrupted_image(tmp_path: Path) -> None:
    dataset_root = _build_valid_metal_nut_dataset(tmp_path)
    corrupted_image = dataset_root / "metal_nut/test/good/000.png"
    corrupted_image.write_bytes(b"not-a-valid-image")

    report = validate_mvtec_category(dataset_root, "metal_nut")

    assert not report.is_valid
    assert len(report.corrupted_files) == 1
