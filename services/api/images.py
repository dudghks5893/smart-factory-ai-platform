"""Bounded in-memory image validation and decoding for inference requests."""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError
from torch import Tensor

from ml.datasets.images import image_to_float_tensor
from services.api.errors import ApiError

SUPPORTED_IMAGE_MEDIA_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
}


@dataclass(frozen=True)
class DecodedInferenceInputs:
    """One validated RGB decode materialized for both inference runtimes."""

    patchcore_tensor: Tensor
    yolo_rgb: NDArray[np.uint8]
    width: int
    height: int


# ADD 2026-08-26: One Pillow decode에서 PatchCore tensor와 YOLO array를 함께 생성한다.
def decode_uploaded_inference_inputs(
    content: bytes,
    *,
    content_type: str | None,
    max_upload_bytes: int,
) -> DecodedInferenceInputs:
    """Decode and validate an upload exactly once for combined inference."""
    image = _decode_uploaded_rgb_image(
        content,
        content_type=content_type,
        max_upload_bytes=max_upload_bytes,
    )
    width, height = image.size
    return DecodedInferenceInputs(
        patchcore_tensor=image_to_float_tensor(image).unsqueeze(0),
        yolo_rgb=np.asarray(image, dtype=np.uint8),
        width=width,
        height=height,
    )


# ADD 2026-08-19: Upload bytes의 size/media/format을 검증하고 RGB batch tensor로 변환한다.
# MODIFY 2026-08-26: Shared RGB decode contract를 재사용해 YOLO와 upload validation을 일치시킨다.
def decode_uploaded_image(
    content: bytes,
    *,
    content_type: str | None,
    max_upload_bytes: int,
) -> Tensor:
    """Decode one supported JPEG or PNG upload without application-side temp files."""
    image = _decode_uploaded_rgb_image(
        content,
        content_type=content_type,
        max_upload_bytes=max_upload_bytes,
    )
    return image_to_float_tensor(image).unsqueeze(0)


# ADD 2026-08-26: Validated upload를 YOLO runtime용 HWC uint8 RGB array로 변환한다.
def decode_uploaded_rgb_array(
    content: bytes,
    *,
    content_type: str | None,
    max_upload_bytes: int,
) -> NDArray[np.uint8]:
    """Decode one supported upload with the same policy as PatchCore requests."""
    image = _decode_uploaded_rgb_image(
        content,
        content_type=content_type,
        max_upload_bytes=max_upload_bytes,
    )
    return np.asarray(image, dtype=np.uint8)


# ADD 2026-08-26: Bounded media/format validation을 하나의 RGB Pillow decode 경계로 제공한다.
def _decode_uploaded_rgb_image(
    content: bytes,
    *,
    content_type: str | None,
    max_upload_bytes: int,
) -> Image.Image:
    """Return a detached RGB image after fully validating the uploaded payload."""
    normalized_media_type = (content_type or "").lower()
    expected_format = SUPPORTED_IMAGE_MEDIA_TYPES.get(normalized_media_type)
    if expected_format is None:
        raise ApiError(
            415,
            "unsupported_media_type",
            "Only image/jpeg and image/png uploads are supported.",
        )
    if not content:
        raise ApiError(400, "empty_image", "Uploaded image is empty.")
    if len(content) > max_upload_bytes:
        raise ApiError(413, "image_too_large", "Uploaded image exceeds MAX_UPLOAD_BYTES.")

    # Pillow가 전체 payload를 decode하도록 해 truncated/non-image input을 inference 전에 거부한다.
    try:
        with Image.open(BytesIO(content)) as image:
            image.load()
            if image.format != expected_format:
                raise ApiError(
                    415,
                    "unsupported_image_format",
                    "Uploaded content does not match its supported media type.",
                )
            return image.convert("RGB")
    except ApiError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ApiError(400, "invalid_image", "Uploaded content is not a valid image.") from exc
