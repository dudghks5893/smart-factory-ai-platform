"""Pydantic contracts for C6-6B live DeepStream observations."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Literal, Self, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from services.streaming.yolo_deepstream_service_integration import (
    EXPECTED_CLASSES,
    EXPECTED_DECODER_ID,
    EXPECTED_ENGINE_SHA256,
    EXPECTED_EVENT_TYPE,
    EXPECTED_LABELS_SHA256,
    EXPECTED_PARSER_SHA256,
    EXPECTED_SOURCE_ID_PATTERN,
)


class StreamingDefectBoundingBox(BaseModel):
    """Pixel-space box emitted by the DeepStream segmentation worker."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    x_min: float = Field(ge=0.0)
    y_min: float = Field(ge=0.0)
    x_max: float = Field(ge=0.0)
    y_max: float = Field(ge=0.0)


class StreamingDefectMaskSummary(BaseModel):
    """Compact mask size without transferring raw mask pixels."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    pixel_count: int = Field(gt=0)
    area_ratio: float = Field(gt=0.0, le=1.0)


class StreamingDefectInstanceEvent(BaseModel):
    """One compact segmentation instance in a live observation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    class_id: int = Field(ge=0, le=2)
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: StreamingDefectBoundingBox
    mask: StreamingDefectMaskSummary

    # ADD 2026-09-06: Live instance class ID/name을 sealed class mapping으로 제한한다.
    @model_validator(mode="after")
    def validate_class_mapping(self) -> Self:
        if EXPECTED_CLASSES.get(self.class_id) != self.class_name:
            raise ValueError("Streaming defect class mapping is invalid.")
        return self


class StreamingObservationImage(BaseModel):
    """Source frame dimensions used for box and mask normalization."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)


class StreamingRuntimeIdentity(BaseModel):
    """Exact C6-5 runtime identity carried across the service bridge."""

    model_config = ConfigDict(extra="forbid")

    decoder_id: str
    engine_sha256: str
    parser_sha256: str
    labels_sha256: str

    # ADD 2026-09-06: Service bridge가 sealed C6-5 decoder/engine/parser/labels만 수용하게 한다.
    @model_validator(mode="after")
    def validate_runtime_identity(self) -> Self:
        if (
            self.decoder_id != EXPECTED_DECODER_ID
            or self.engine_sha256 != EXPECTED_ENGINE_SHA256
            or self.parser_sha256 != EXPECTED_PARSER_SHA256
            or self.labels_sha256 != EXPECTED_LABELS_SHA256
        ):
            raise ValueError("Streaming runtime identity does not match sealed C6-5.")
        return self


class StreamingKnownDefectObservationPayload(BaseModel):
    """Validated observation metadata accepted from one DeepStream worker."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    observation_id: UUID
    source_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=EXPECTED_SOURCE_ID_PATTERN.pattern,
    )
    stream_session_id: UUID
    frame_number: int = Field(ge=0)
    pts_ns: int = Field(ge=0)
    observed_at: datetime
    image: StreamingObservationImage
    runtime: StreamingRuntimeIdentity
    diagnostic_confidence: float = Field(gt=0.0, lt=1.0)
    instances: list[StreamingDefectInstanceEvent] = Field(max_length=300)

    # ADD 2026-09-06: UTC timestamp, confidence, bbox, mask geometry를 frame contract로 검증한다.
    @model_validator(mode="after")
    def validate_observation_geometry(self) -> Self:
        from datetime import datetime

        if not isinstance(self.observed_at, datetime):
            raise ValueError("observed_at must be an ISO-8601 datetime.")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() != timedelta(0):
            raise ValueError("observed_at must be timezone-aware UTC.")
        if self.diagnostic_confidence != 0.25:
            raise ValueError("diagnostic_confidence must match the sealed 0.25 contract.")

        image_area = self.image.width * self.image.height
        for instance in self.instances:
            box = instance.box
            if not (
                box.x_min <= box.x_max <= self.image.width
                and box.y_min <= box.y_max <= self.image.height
            ):
                raise ValueError("Streaming defect bbox is outside the source image.")
            if instance.mask.pixel_count > image_area:
                raise ValueError("Streaming mask pixel count exceeds the source image.")
            expected_ratio = instance.mask.pixel_count / image_area
            if not math.isclose(
                instance.mask.area_ratio,
                expected_ratio,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("Streaming mask area ratio does not match pixel count.")
        return self


class StreamingKnownDefectObservedEvent(BaseModel):
    """Versioned live observation delivered by the DeepStream worker."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    type: Literal["streaming_known_defect.observed"] = cast(
        Literal["streaming_known_defect.observed"],
        EXPECTED_EVENT_TYPE,
    )
    observation: StreamingKnownDefectObservationPayload


class StreamingObservationAcceptedResponse(BaseModel):
    """Minimal acknowledgement returned before best-effort WebSocket broadcast."""

    status: Literal["accepted"] = "accepted"
    observation_id: UUID
