"""Tests for the stable actual-transform preview adapter contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.experiments.yolo_augmentation import (
    prepare_preview_dataset,
    preview_actual_training_augmentations,
)
from ml.training.yolo_segmentation import load_yolo_segmentation_config
from shared.hashing import sha256_file


class Format:
    """Transform-name fixture matching the stable adapter check."""


class FakeDataset:
    """Minimal configured dataset result without importing framework internals."""

    # ADD 2026-08-27: One sample and visible Format transform로 fixture를 만든다.
    def __init__(
        self,
        image_path: Path,
        imgsz: int,
        *,
        class_id: int | None = 0,
    ) -> None:
        self.im_files = [str(image_path)]
        self.transforms = Format()
        self._imgsz = imgsz
        self._class_id = class_id

    # ADD 2026-08-27: Aligned image/class/mask tensors를 actual adapter shape로 반환한다.
    def __getitem__(self, index: int) -> dict[str, Any]:
        assert index == 0
        mask = torch.zeros((1, self._imgsz // 4, self._imgsz // 4), dtype=torch.uint8)
        classes = torch.zeros((0, 1))
        if self._class_id is not None:
            mask[:, 2:5, 3:7] = 1
            classes = torch.tensor([[float(self._class_id)]])
        return {
            "img": torch.zeros((3, self._imgsz, self._imgsz), dtype=torch.uint8),
            "cls": classes,
            "masks": mask,
        }


# ADD 2026-08-27: Source mirror가 image/label bytes를 바꾸지 않는지 검증한다.
def test_prepare_preview_dataset_preserves_source_bytes(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    image_path = dataset / "images" / "train" / "sample.png"
    label_path = dataset / "labels" / "train" / "sample.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image-bytes")
    label_path.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
    record = DerivedManifestRecord(
        "fixture",
        "v1",
        "yolo_segmentation",
        "a" * 64,
        "test",
        "test",
        "source/sample.png",
        "source/sample.png",
        "metal_nut",
        "sample",
        "bent",
        "bent",
        "0",
        "train",
        False,
        32,
        32,
        "images/train/sample.png",
        "labels/train/sample.txt",
        "b" * 64,
        "c" * 64,
        1,
        1,
        0,
        3,
        "1",
        "1",
        "1",
    )
    before = (sha256_file(image_path), sha256_file(label_path))
    prepare_preview_dataset(
        dataset_root=dataset,
        preview_root=tmp_path / "preview",
        records=[record],
        split="train",
    )
    assert (sha256_file(image_path), sha256_file(label_path)) == before


# ADD 2026-08-27: Configured transform와 class/mask alignment, config immutability를 검증한다.
def test_augmentation_adapter_uses_configured_dataset_contract(tmp_path: Path) -> None:
    baseline = load_yolo_segmentation_config(Path("configs/model/yolo_segmentation_baseline.yaml"))
    observed: list[tuple[int, str]] = []

    def build_dataset(config: Any, preview_root: Path, split: str) -> FakeDataset:
        observed.append((config.training.imgsz, split))
        return FakeDataset(preview_root / "images" / split / "sample.png", config.training.imgsz)

    previews = preview_actual_training_augmentations(
        config=baseline,
        preview_root=tmp_path,
        sample_ids=["sample"],
        variants=3,
        dataset_builder=build_dataset,
    )
    assert observed == [(640, "train")]
    assert [item.variant for item in previews] == [1, 2, 3]
    assert all(len(item.component_masks) == len(item.class_ids) == 1 for item in previews)
    assert baseline.training.imgsz == 640


# ADD 2026-08-27: Legitimate empty-label output을 허용하고 malformed class는 계속 거부한다.
def test_augmentation_adapter_accepts_empty_labels_without_weakening_validation(
    tmp_path: Path,
) -> None:
    baseline = load_yolo_segmentation_config(Path("configs/model/yolo_segmentation_baseline.yaml"))

    def build_empty(config: Any, preview_root: Path, split: str) -> FakeDataset:
        return FakeDataset(
            preview_root / "images" / split / "good-negative.png",
            config.training.imgsz,
            class_id=None,
        )

    previews = preview_actual_training_augmentations(
        config=baseline,
        preview_root=tmp_path,
        sample_ids=["good-negative"],
        variants=1,
        dataset_builder=build_empty,
    )
    assert previews[0].class_ids == ()
    assert previews[0].component_masks == ()

    def build_invalid(config: Any, preview_root: Path, split: str) -> FakeDataset:
        return FakeDataset(
            preview_root / "images" / split / "invalid.png",
            config.training.imgsz,
            class_id=99,
        )

    with pytest.raises(ValueError, match="unsupported class"):
        preview_actual_training_augmentations(
            config=baseline,
            preview_root=tmp_path,
            sample_ids=["invalid"],
            variants=1,
            dataset_builder=build_invalid,
        )
