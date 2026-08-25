"""SQLAlchemy models for durable inspection history."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative metadata root consumed by Alembic and isolated tests."""


class InspectionRecord(Base):
    """Persisted image-level PatchCore inspection and immutable provenance."""

    __tablename__ = "inspections"
    __table_args__ = (
        CheckConstraint("comparison_operator = '>'", name="ck_inspections_operator_gt"),
        CheckConstraint("image_size_bytes > 0", name="ck_inspections_image_size_positive"),
        CheckConstraint("length(image_sha256) = 64", name="ck_inspections_image_sha256_length"),
        CheckConstraint("length(model_sha256) = 64", name="ck_inspections_model_sha256_length"),
        CheckConstraint(
            "length(artifact_metadata_sha256) = 64",
            name="ck_inspections_metadata_sha256_length",
        ),
        CheckConstraint(
            "length(threshold_artifact_sha256) = 64",
            name="ck_inspections_threshold_sha256_length",
        ),
        CheckConstraint(
            "length(manifest_sha256) = 64",
            name="ck_inspections_manifest_sha256_length",
        ),
        Index("ix_inspections_created_at", "created_at"),
        Index("ix_inspections_category_created_at", "category", "created_at"),
        Index("ix_inspections_anomaly_created_at", "is_anomaly", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    comparison_operator: Mapped[str] = mapped_column(String(2), nullable=False)
    image_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    image_size_bytes: Mapped[int] = mapped_column(nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    model_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_metadata_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    threshold_artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    device: Mapped[str] = mapped_column(String(50), nullable=False)


class KnownDefectInspectionRecord(Base):
    """Persisted parent for one independent YOLO segmentation inference."""

    __tablename__ = "known_defect_inspections"
    __table_args__ = (
        CheckConstraint("task = 'segment'", name="ck_known_defect_inspections_task_segment"),
        CheckConstraint(
            "diagnostic_confidence > 0 AND diagnostic_confidence < 1",
            name="ck_known_defect_inspections_confidence_range",
        ),
        CheckConstraint(
            "inference_ms >= 0",
            name="ck_known_defect_inspections_inference_ms_nonnegative",
        ),
        CheckConstraint(
            "image_width > 0 AND image_height > 0",
            name="ck_known_defect_inspections_image_dimensions_positive",
        ),
        CheckConstraint(
            "instance_count >= 0",
            name="ck_known_defect_inspections_instance_count_nonnegative",
        ),
        CheckConstraint(
            "length(image_sha256) = 64",
            name="ck_known_defect_inspections_image_sha256_length",
        ),
        CheckConstraint(
            "length(model_sha256) = 64",
            name="ck_known_defect_inspections_model_sha256_length",
        ),
        CheckConstraint(
            "length(artifact_metadata_sha256) = 64",
            name="ck_known_defect_inspections_metadata_sha256_length",
        ),
        CheckConstraint(
            "length(dataset_manifest_sha256) = 64",
            name="ck_known_defect_inspections_manifest_sha256_length",
        ),
        CheckConstraint(
            "length(dataset_semantic_fingerprint_sha256) = 64",
            name="ck_known_defect_inspections_fingerprint_sha256_length",
        ),
        Index("ix_known_defect_inspections_created_at", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    task: Mapped[str] = mapped_column(String(50), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    device: Mapped[str] = mapped_column(String(50), nullable=False)
    diagnostic_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    inference_ms: Mapped[float] = mapped_column(Float, nullable=False)
    image_width: Mapped[int] = mapped_column(nullable=False)
    image_height: Mapped[int] = mapped_column(nullable=False)
    image_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    model_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_metadata_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_semantic_fingerprint_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    instance_count: Mapped[int] = mapped_column(nullable=False)


class KnownDefectInstanceRecord(Base):
    """Persisted compact instance ordered within one known-defect inspection."""

    __tablename__ = "known_defect_instances"
    __table_args__ = (
        UniqueConstraint(
            "inspection_id",
            "instance_index",
            name="uq_known_defect_instances_inspection_index",
        ),
        CheckConstraint(
            "instance_index >= 0",
            name="ck_known_defect_instances_index_nonnegative",
        ),
        CheckConstraint("class_id >= 0", name="ck_known_defect_instances_class_id_nonnegative"),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_known_defect_instances_confidence_range",
        ),
        CheckConstraint(
            "bbox_x_min >= 0 AND bbox_y_min >= 0 "
            "AND bbox_x_max >= bbox_x_min AND bbox_y_max >= bbox_y_min",
            name="ck_known_defect_instances_bbox_order",
        ),
        CheckConstraint(
            "mask_pixel_count > 0",
            name="ck_known_defect_instances_mask_pixels_positive",
        ),
        CheckConstraint(
            "mask_area_ratio > 0 AND mask_area_ratio <= 1",
            name="ck_known_defect_instances_mask_area_range",
        ),
        Index("ix_known_defect_instances_inspection_id", "inspection_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    inspection_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("known_defect_inspections.id", ondelete="CASCADE"),
        nullable=False,
    )
    instance_index: Mapped[int] = mapped_column(nullable=False)
    class_id: Mapped[int] = mapped_column(nullable=False)
    class_name: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x_min: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y_min: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x_max: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y_max: Mapped[float] = mapped_column(Float, nullable=False)
    mask_pixel_count: Mapped[int] = mapped_column(nullable=False)
    mask_area_ratio: Mapped[float] = mapped_column(Float, nullable=False)
