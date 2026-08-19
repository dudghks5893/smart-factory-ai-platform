"""Tests for the manifest-backed PyTorch dataset."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from ml.datasets.dataset import MVTecManifestDataset
from ml.datasets.manifest import ManifestRecord, write_manifest_csv


# ADD 2026-08-18: 테스트 fixture 파일을 생성한다.
def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 128, 0)).save(path)


# ADD 2026-08-18: 테스트 fixture 파일을 생성한다.
def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), color=255).save(path)


# ADD 2026-08-18: 테스트에 필요한 dataset fixture를 구성한다.
def _build_manifest(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "mvtec_ad"
    train_image = dataset_root / "metal_nut/train/good/000.png"
    anomaly_image = dataset_root / "metal_nut/test/bent/000.png"
    anomaly_mask = dataset_root / "metal_nut/ground_truth/bent/000_mask.png"

    _write_rgb(train_image)
    _write_rgb(anomaly_image)
    _write_mask(anomaly_mask)

    records = [
        ManifestRecord(
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
        ),
        ManifestRecord(
            sample_id="metal_nut_test_bent_000",
            category="metal_nut",
            source_split="test",
            split="test",
            defect_type="bent",
            label=1,
            image_path="metal_nut/test/bent/000.png",
            mask_path="metal_nut/ground_truth/bent/000_mask.png",
            width=8,
            height=8,
        ),
    ]

    manifest_path = tmp_path / "manifest.csv"
    write_manifest_csv(records, manifest_path)
    return dataset_root, manifest_path


# ADD 2026-08-18: 정상 sample의 float image와 zero mask 반환을 검증한다.
def test_dataset_returns_float_image_and_zero_mask_for_normal_sample(
    tmp_path: Path,
) -> None:
    dataset_root, manifest_path = _build_manifest(tmp_path)

    dataset = MVTecManifestDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="train",
    )
    sample = dataset[0]

    image = sample["image"]
    mask = sample["mask"]

    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.shape == (3, 8, 8)
    assert image.dtype == torch.float32
    assert mask.shape == (1, 8, 8)
    assert torch.count_nonzero(mask) == 0


# ADD 2026-08-18: dataset loads anomaly mask 테스트 시나리오를 검증한다.
def test_dataset_loads_anomaly_mask(tmp_path: Path) -> None:
    dataset_root, manifest_path = _build_manifest(tmp_path)

    dataset = MVTecManifestDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="test",
    )
    sample = dataset[0]
    mask = sample["mask"]

    assert isinstance(mask, torch.Tensor)
    assert torch.all(mask == 1)


# ADD 2026-08-19: Image-only consumer가 anomaly mask disk I/O를 생략하는지 검증한다.
def test_dataset_can_skip_mask_loading(tmp_path: Path) -> None:
    dataset_root, manifest_path = _build_manifest(tmp_path)
    (dataset_root / "metal_nut/ground_truth/bent/000_mask.png").unlink()

    dataset = MVTecManifestDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="test",
        load_masks=False,
    )
    sample = dataset[0]

    assert isinstance(sample["image"], torch.Tensor)
    assert "mask" not in sample


# ADD 2026-08-18: dataloader collates manifest samples 테스트 시나리오를 검증한다.
def test_dataloader_collates_manifest_samples(tmp_path: Path) -> None:
    dataset_root, manifest_path = _build_manifest(tmp_path)

    dataset = MVTecManifestDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="train",
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    images = batch["image"]
    masks = batch["mask"]

    assert isinstance(images, torch.Tensor)
    assert isinstance(masks, torch.Tensor)
    assert images.shape == (1, 3, 8, 8)
    assert masks.shape == (1, 1, 8, 8)
