"""Pydantic response contracts for PatchCore serving endpoints."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Liveness response independent of model readiness."""

    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    """Loaded model identity returned only when inference is ready."""

    status: Literal["ready"]
    model_name: str
    category: str
    device: str


class InferenceResponse(BaseModel):
    """Image-level PatchCore result without defect class or anomaly-map payload."""

    model_config = ConfigDict(allow_inf_nan=False)

    inspection_id: UUID
    model_name: str
    category: str
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    comparison_operator: Literal[">"]


class KnownDefectModelIdentity(BaseModel):
    """Loaded known-defect segmentation model identity."""

    name: str
    task: Literal["segment"]
    category: str
    device: str


class KnownDefectImageSummary(BaseModel):
    """Original image dimensions used by normalized masks and boxes."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)


class KnownDefectBoundingBox(BaseModel):
    """Pixel-space bounding box clipped to the original image."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


class KnownDefectMaskSummary(BaseModel):
    """Compact mask area without raw 700x700 pixels."""

    pixel_count: int = Field(gt=0)
    area_ratio: float = Field(gt=0.0, le=1.0)


class KnownDefectInstanceResponse(BaseModel):
    """One normalized defect instance returned by the YOLO endpoint."""

    model_config = ConfigDict(allow_inf_nan=False)

    class_id: int = Field(ge=0)
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: KnownDefectBoundingBox
    mask: KnownDefectMaskSummary


class KnownDefectResponse(BaseModel):
    """Known-defect segmentation response without manufacturing decision or raw mask."""

    model_config = ConfigDict(allow_inf_nan=False)

    model: KnownDefectModelIdentity
    image: KnownDefectImageSummary
    diagnostic_confidence: float = Field(gt=0.0, lt=1.0)
    inference_ms: float = Field(ge=0.0)
    instances: list[KnownDefectInstanceResponse]


class InspectionResponse(BaseModel):
    """Persisted inspection with prediction and immutable provenance."""

    model_config = ConfigDict(allow_inf_nan=False)

    inspection_id: UUID
    created_at: datetime
    model_name: str
    category: str
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    comparison_operator: Literal[">"]
    image_sha256: str
    image_size_bytes: int
    content_type: str
    model_sha256: str
    artifact_metadata_sha256: str
    threshold_artifact_sha256: str
    manifest_sha256: str
    device: str


class InspectionCreatedPayload(BaseModel):
    """Compact persisted inspection fields sent in one live notification."""

    model_config = ConfigDict(allow_inf_nan=False)

    inspection_id: UUID
    model_name: str
    category: str
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    comparison_operator: Literal[">"]
    device: str
    created_at: datetime


class InspectionCreatedEvent(BaseModel):
    """Versioned best-effort WebSocket notification after durable persistence."""

    schema_version: Literal["1"] = "1"
    type: Literal["inspection.created"] = "inspection.created"
    inspection: InspectionCreatedPayload


class InspectionHistoryResponse(BaseModel):
    """Newest-first inspection page without an aggregate count query."""

    items: list[InspectionResponse]
    limit: int
    offset: int
    returned_count: int
    has_more: bool


class ErrorDetail(BaseModel):
    """Stable public error code and non-sensitive message."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Envelope shared by API-level error responses."""

    error: ErrorDetail
