"""Pydantic response contracts for PatchCore serving endpoints."""

from typing import Literal

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

    model_name: str
    category: str
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    comparison_operator: Literal[">"]


class ErrorDetail(BaseModel):
    """Stable public error code and non-sensitive message."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Envelope shared by API-level error responses."""

    error: ErrorDetail
