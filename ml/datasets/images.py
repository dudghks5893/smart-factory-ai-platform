"""Image tensor conversion shared by dataset and serving consumers."""

from typing import cast

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import pil_to_tensor  # type: ignore[import-untyped]


# ADD 2026-08-18: PIL image를 [0, 1] float RGB tensor로 변환한다.
# MODIFY 2026-08-19: Dataset private helper → serving도 재사용하는 public conversion으로 이동했다.
def image_to_float_tensor(image: Image.Image) -> Tensor:
    """Convert a PIL image to a three-channel float tensor in the [0, 1] range."""
    tensor = cast(Tensor, pil_to_tensor(image.convert("RGB")))
    return tensor.to(dtype=torch.float32).div_(255.0)
