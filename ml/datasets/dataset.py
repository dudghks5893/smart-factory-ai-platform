"""PyTorch dataset backed by an MVTec AD manifest."""

from pathlib import Path
from typing import cast

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor  # type: ignore[import-untyped]

from ml.datasets.constants import MVTEC_SPLITS
from ml.datasets.manifest import ManifestRecord, read_manifest_csv

type SampleValue = Tensor | int | str
type MVTecSample = dict[str, SampleValue]


# ADD 2026-08-18: PIL image를 [0, 1] float RGB tensor로 변환한다.
def _image_to_float_tensor(image: Image.Image) -> Tensor:
    tensor = cast(Tensor, pil_to_tensor(image.convert("RGB")))
    return tensor.to(dtype=torch.float32).div_(255.0)


# ADD 2026-08-18: PIL mask를 binary float tensor로 변환한다.
def _mask_to_float_tensor(mask: Image.Image) -> Tensor:
    tensor = cast(Tensor, pil_to_tensor(mask.convert("L")))
    return tensor.gt(0).to(dtype=torch.float32)


class MVTecManifestDataset(Dataset[MVTecSample]):
    """Load MVTec AD samples from a generated manifest without copying raw images."""

    # ADD 2026-08-18: Manifest split을 검증하고 해당 record만 lazy dataset으로 초기화한다.
    # MODIFY 2026-08-19: 공통 split 사용 및 image-only consumer용 mask loading 선택을 지원한다.
    def __init__(
        self,
        *,
        dataset_root: Path,
        manifest_path: Path,
        split: str,
        load_masks: bool = True,
    ) -> None:
        if split not in MVTEC_SPLITS:
            raise ValueError(f"Unsupported split: {split}")

        self.dataset_root = dataset_root
        self.split = split
        self.load_masks = load_masks
        self.records = [
            record for record in read_manifest_csv(manifest_path) if record.split == split
        ]

        if not self.records:
            raise ValueError(f"No manifest records found for split: {split}")

    # ADD 2026-08-18: 선택된 manifest record 수를 반환한다.
    def __len__(self) -> int:
        return len(self.records)

    # ADD 2026-08-18: Manifest record의 image와 mask 및 metadata를 로드한다.
    # MODIFY 2026-08-19: Benchmark에서 불필요한 ground-truth mask I/O를 생략할 수 있게 했다.
    def __getitem__(self, index: int) -> MVTecSample:
        record = self.records[index]
        image_path = self.dataset_root / record.image_path

        # Raw image를 열어 RGB [0, 1] float tensor로 변환한다.
        with Image.open(image_path) as image:
            image_tensor = _image_to_float_tensor(image)

        sample: MVTecSample = {
            "image": image_tensor,
            "label": record.label,
            "sample_id": record.sample_id,
            "category": record.category,
            "defect_type": record.defect_type,
            "image_path": record.image_path,
        }
        if self.load_masks:
            # Pixel evaluation consumer에만 normal/anomaly 규칙에 맞는 mask tensor를 로드한다.
            sample["mask"] = self._load_mask(record)
        return sample

    # ADD 2026-08-18: 정상 sample의 zero mask 또는 anomaly mask를 로드한다.
    def _load_mask(self, record: ManifestRecord) -> Tensor:
        if record.label == 0:
            return torch.zeros(
                (1, record.height, record.width),
                dtype=torch.float32,
            )

        mask_path = self.dataset_root / record.mask_path
        with Image.open(mask_path) as mask:
            return _mask_to_float_tensor(mask)
