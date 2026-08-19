"""Bounded in-memory image validation and decoding for inference requests."""

from io import BytesIO

from PIL import Image, UnidentifiedImageError
from torch import Tensor

from ml.datasets.images import image_to_float_tensor
from services.api.errors import ApiError

SUPPORTED_IMAGE_MEDIA_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
}


# ADD 2026-08-19: Upload bytes의 size/media/format을 검증하고 RGB batch tensor로 변환한다.
def decode_uploaded_image(
    content: bytes,
    *,
    content_type: str | None,
    max_upload_bytes: int,
) -> Tensor:
    """Decode one supported JPEG or PNG upload without application-side temp files."""
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
            return image_to_float_tensor(image).unsqueeze(0)
    except ApiError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ApiError(400, "invalid_image", "Uploaded content is not a valid image.") from exc
