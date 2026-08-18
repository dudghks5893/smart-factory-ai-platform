"""PyTorch dataset backed by an MVTec AD manifest."""

from pathlib import Path
from typing import cast

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor  # type: ignore[import-untyped]

from ml.datasets.manifest import ManifestRecord, read_manifest_csv

type SampleValue = Tensor | int | str
type MVTecSample = dict[str, SampleValue]


def _image_to_float_tensor(image: Image.Image) -> Tensor:
    tensor = cast(Tensor, pil_to_tensor(image.convert("RGB")))
    return tensor.to(dtype=torch.float32).div_(255.0)


def _mask_to_float_tensor(mask: Image.Image) -> Tensor:
    tensor = cast(Tensor, pil_to_tensor(mask.convert("L")))
    return tensor.gt(0).to(dtype=torch.float32)


class MVTecManifestDataset(Dataset[MVTecSample]):
    """Load MVTec AD samples from a generated manifest without copying raw images."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        manifest_path: Path,
        split: str,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.dataset_root = dataset_root
        self.split = split
        self.records = [
            record for record in read_manifest_csv(manifest_path) if record.split == split
        ]

        if not self.records:
            raise ValueError(f"No manifest records found for split: {split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> MVTecSample:
        record = self.records[index]
        image_path = self.dataset_root / record.image_path

        with Image.open(image_path) as image:
            image_tensor = _image_to_float_tensor(image)

        mask_tensor = self._load_mask(record)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": record.label,
            "sample_id": record.sample_id,
            "category": record.category,
            "defect_type": record.defect_type,
            "image_path": record.image_path,
        }

    def _load_mask(self, record: ManifestRecord) -> Tensor:
        if record.label == 0:
            return torch.zeros(
                (1, record.height, record.width),
                dtype=torch.float32,
            )

        mask_path = self.dataset_root / record.mask_path
        with Image.open(mask_path) as mask:
            return _mask_to_float_tensor(mask)
