"""Ground-truth mask alignment for PatchCore pixel evaluation."""

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ml.datasets.dataset import MVTecManifestDataset
from ml.training.batches import require_batch_tensor
from ml.training.preprocessing import PatchCorePreprocessor


# ADD 2026-08-19: Dataset mask를 PatchCore geometry에 맞춰 contiguous tensor로 정렬한다.
def load_aligned_ground_truth_masks(
    *,
    dataset: MVTecManifestDataset,
    preprocessor: PatchCorePreprocessor,
    batch_size: int,
    num_workers: int = 0,
) -> Tensor:
    """Load and align ground-truth masks with the PatchCore anomaly-map geometry."""
    if batch_size <= 0:
        raise ValueError("Pixel evaluation batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("Pixel evaluation num_workers must be non-negative.")

    crop_height, crop_width = preprocessor.config.center_crop_size
    aligned_masks = torch.empty(
        (len(dataset), 1, crop_height, crop_width),
        dtype=torch.float32,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # 기존 image/mask preprocessing을 함께 호출해 prediction geometry와 동일하게 정렬한다.
    offset = 0
    for batch in loader:
        images = require_batch_tensor(batch, "image")
        masks = require_batch_tensor(batch, "mask")
        _, transformed_masks = preprocessor(images, masks)
        if transformed_masks is None:
            raise RuntimeError("PatchCore preprocessor did not return aligned masks.")
        next_offset = offset + transformed_masks.shape[0]
        aligned_masks[offset:next_offset].copy_(transformed_masks)
        offset = next_offset

    if offset != len(dataset):
        raise RuntimeError(f"Loaded {offset} ground-truth masks; expected {len(dataset)}.")
    return aligned_masks.contiguous()
