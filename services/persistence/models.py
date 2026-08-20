"""SQLAlchemy models for durable inspection history."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    Index,
    String,
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
