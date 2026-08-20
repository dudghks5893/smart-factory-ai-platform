"""Pydantic response contracts for PatchCore serving endpoints."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


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
