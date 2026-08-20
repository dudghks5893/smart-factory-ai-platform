"""Shared HTTP image preparation and response validation for serving tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from services.api.images import SUPPORTED_IMAGE_MEDIA_TYPES
from services.api.schemas import InferenceResponse

_SUFFIX_MEDIA_TYPES = {
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
}


@dataclass(frozen=True)
class PreparedImageUpload:
    """One image held in memory before an HTTP request timer starts."""

    filename: str
    content_type: str
    content: bytes

    # ADD 2026-08-20: TestClient multipart request에 사용할 upload tuple을 생성한다.
    def as_multipart_file(self) -> tuple[str, bytes, str]:
        return self.filename, self.content, self.content_type


# ADD 2026-08-20: Disk I/O를 timing 전에 완료하고 supported upload metadata를 고정한다.
def prepare_image_upload(path: Path, *, max_upload_bytes: int) -> PreparedImageUpload:
    """Read and validate a supported image payload before HTTP execution."""
    if max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be positive.")
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    content_type = _SUFFIX_MEDIA_TYPES.get(path.suffix.lower())
    if content_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
        raise ValueError(f"Unsupported image filename extension: {path.suffix}")

    content = path.read_bytes()
    if not content:
        raise ValueError(f"Image file is empty: {path}")
    if len(content) > max_upload_bytes:
        raise ValueError(f"Image file exceeds max_upload_bytes: {path}")
    return PreparedImageUpload(
        filename=path.name,
        content_type=content_type,
        content=content,
    )


# ADD 2026-08-20: HTTP JSON을 public inference schema와 strict threshold 계약으로 검증한다.
# MODIFY 2026-08-20: Persisted inspection UUID를 포함한 response schema를 검증한다.
def validate_prediction_payload(
    payload: object,
    *,
    expected_is_anomaly: bool | None = None,
) -> InferenceResponse:
    """Validate response schema, finite numbers, and score-threshold consistency."""
    try:
        response = InferenceResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Prediction response does not match the API schema.") from exc

    expected_from_threshold = response.anomaly_score > response.threshold
    if response.is_anomaly is not expected_from_threshold:
        raise ValueError("Prediction response violates the strict score > threshold contract.")
    if expected_is_anomaly is not None and response.is_anomaly is not expected_is_anomaly:
        expected_label = "anomaly" if expected_is_anomaly else "normal"
        raise ValueError(f"Expected {expected_label} prediction from the supplied smoke image.")
    return response
