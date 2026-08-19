"""PatchCore-specific image and mask preprocessing."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode  # type: ignore[import-untyped]
from torchvision.transforms import functional as transform_functional


@dataclass(frozen=True)
class PatchCorePreprocessingConfig:
    """Geometric and image normalization settings for PatchCore."""

    resize_size: tuple[int, int]
    center_crop_size: tuple[int, int]
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]

    # ADD 2026-08-19: Resize, crop, normalization configuration을 검증한다.
    def validate(self) -> None:
        if any(dimension <= 0 for dimension in self.resize_size):
            raise ValueError("resize_size dimensions must be positive.")
        if any(dimension <= 0 for dimension in self.center_crop_size):
            raise ValueError("center_crop_size dimensions must be positive.")
        if any(
            crop > resized
            for crop, resized in zip(self.center_crop_size, self.resize_size, strict=True)
        ):
            raise ValueError("center_crop_size cannot be larger than resize_size.")
        if any(value <= 0 for value in self.image_std):
            raise ValueError("image_std values must be positive.")


class PatchCorePreprocessor(nn.Module):
    """Apply PatchCore geometry to images and masks and normalize images only."""

    # ADD 2026-08-19: 검증된 PatchCore preprocessing configuration을 보관한다.
    def __init__(self, config: PatchCorePreprocessingConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

    # ADD 2026-08-19: Image와 mask에 정렬된 geometry를 적용하고 image만 정규화한다.
    def forward(
        self,
        images: Tensor,
        masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        # 비용이 있는 resize 전에 image/mask shape와 dtype 계약을 확인한다.
        self._validate_inputs(images, masks)

        # Image에는 baseline geometry를 적용한 뒤 ImageNet normalization을 수행한다.
        resized_images = transform_functional.resize(
            images,
            list(self.config.resize_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        cropped_images = transform_functional.center_crop(
            resized_images,
            list(self.config.center_crop_size),
        )
        normalized_images = transform_functional.normalize(
            cropped_images,
            mean=list(self.config.image_mean),
            std=list(self.config.image_std),
        )

        if masks is None:
            return normalized_images, None

        # Mask에는 동일 geometry와 nearest interpolation만 적용해 binary alignment를 유지한다.
        resized_masks = transform_functional.resize(
            masks,
            list(self.config.resize_size),
            interpolation=InterpolationMode.NEAREST,
        )
        cropped_masks = transform_functional.center_crop(
            resized_masks,
            list(self.config.center_crop_size),
        )
        binary_masks = cropped_masks.gt(0.5).to(dtype=torch.float32)
        return normalized_images, binary_masks

    # ADD 2026-08-19: 전처리 전에 image와 mask tensor shape 및 dtype을 검증한다.
    @staticmethod
    def _validate_inputs(images: Tensor, masks: Tensor | None) -> None:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (batch, 3, height, width).")
        if not images.is_floating_point():
            raise TypeError("images must be floating-point tensors.")

        if masks is None:
            return

        if masks.ndim != 4 or masks.shape[1] != 1:
            raise ValueError("masks must have shape (batch, 1, height, width).")
        if masks.shape[0] != images.shape[0] or masks.shape[-2:] != images.shape[-2:]:
            raise ValueError("images and masks must have matching batch and spatial dimensions.")
