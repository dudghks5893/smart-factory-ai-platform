"""Create known-defect inspection parent and instance tables.

Revision ID: 20260826_01
Revises: 20260820_01
Create Date: 2026-08-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260826_01"
down_revision: str | None = "20260820_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ADD 2026-08-26: YOLO inference parent와 ordered compact instance schema를 생성한다.
def upgrade() -> None:
    op.create_table(
        "known_defect_inspections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("task", sa.String(length=50), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("device", sa.String(length=50), nullable=False),
        sa.Column("diagnostic_confidence", sa.Float(), nullable=False),
        sa.Column("inference_ms", sa.Float(), nullable=False),
        sa.Column("image_width", sa.Integer(), nullable=False),
        sa.Column("image_height", sa.Integer(), nullable=False),
        sa.Column("image_sha256", sa.String(length=64), nullable=False),
        sa.Column("model_sha256", sa.String(length=64), nullable=False),
        sa.Column("artifact_metadata_sha256", sa.String(length=64), nullable=False),
        sa.Column("dataset_manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "dataset_semantic_fingerprint_sha256",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column("instance_count", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "task = 'segment'",
            name="ck_known_defect_inspections_task_segment",
        ),
        sa.CheckConstraint(
            "diagnostic_confidence > 0 AND diagnostic_confidence < 1",
            name="ck_known_defect_inspections_confidence_range",
        ),
        sa.CheckConstraint(
            "inference_ms >= 0",
            name="ck_known_defect_inspections_inference_ms_nonnegative",
        ),
        sa.CheckConstraint(
            "image_width > 0 AND image_height > 0",
            name="ck_known_defect_inspections_image_dimensions_positive",
        ),
        sa.CheckConstraint(
            "instance_count >= 0",
            name="ck_known_defect_inspections_instance_count_nonnegative",
        ),
        sa.CheckConstraint(
            "length(image_sha256) = 64",
            name="ck_known_defect_inspections_image_sha256_length",
        ),
        sa.CheckConstraint(
            "length(model_sha256) = 64",
            name="ck_known_defect_inspections_model_sha256_length",
        ),
        sa.CheckConstraint(
            "length(artifact_metadata_sha256) = 64",
            name="ck_known_defect_inspections_metadata_sha256_length",
        ),
        sa.CheckConstraint(
            "length(dataset_manifest_sha256) = 64",
            name="ck_known_defect_inspections_manifest_sha256_length",
        ),
        sa.CheckConstraint(
            "length(dataset_semantic_fingerprint_sha256) = 64",
            name="ck_known_defect_inspections_fingerprint_sha256_length",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_known_defect_inspections_created_at",
        "known_defect_inspections",
        ["created_at"],
    )
    op.create_table(
        "known_defect_instances",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("inspection_id", sa.Uuid(), nullable=False),
        sa.Column("instance_index", sa.Integer(), nullable=False),
        sa.Column("class_id", sa.Integer(), nullable=False),
        sa.Column("class_name", sa.String(length=100), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("bbox_x_min", sa.Float(), nullable=False),
        sa.Column("bbox_y_min", sa.Float(), nullable=False),
        sa.Column("bbox_x_max", sa.Float(), nullable=False),
        sa.Column("bbox_y_max", sa.Float(), nullable=False),
        sa.Column("mask_pixel_count", sa.Integer(), nullable=False),
        sa.Column("mask_area_ratio", sa.Float(), nullable=False),
        sa.CheckConstraint(
            "instance_index >= 0",
            name="ck_known_defect_instances_index_nonnegative",
        ),
        sa.CheckConstraint(
            "class_id >= 0",
            name="ck_known_defect_instances_class_id_nonnegative",
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_known_defect_instances_confidence_range",
        ),
        sa.CheckConstraint(
            "bbox_x_min >= 0 AND bbox_y_min >= 0 "
            "AND bbox_x_max >= bbox_x_min AND bbox_y_max >= bbox_y_min",
            name="ck_known_defect_instances_bbox_order",
        ),
        sa.CheckConstraint(
            "mask_pixel_count > 0",
            name="ck_known_defect_instances_mask_pixels_positive",
        ),
        sa.CheckConstraint(
            "mask_area_ratio > 0 AND mask_area_ratio <= 1",
            name="ck_known_defect_instances_mask_area_range",
        ),
        sa.ForeignKeyConstraint(
            ["inspection_id"],
            ["known_defect_inspections.id"],
            name="fk_known_defect_instances_inspection_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "inspection_id",
            "instance_index",
            name="uq_known_defect_instances_inspection_index",
        ),
    )
    op.create_index(
        "ix_known_defect_instances_inspection_id",
        "known_defect_instances",
        ["inspection_id"],
    )


# ADD 2026-08-26: Child/FK를 먼저 제거한 뒤 known-defect parent schema를 제거한다.
def downgrade() -> None:
    op.drop_index(
        "ix_known_defect_instances_inspection_id",
        table_name="known_defect_instances",
    )
    op.drop_table("known_defect_instances")
    op.drop_index(
        "ix_known_defect_inspections_created_at",
        table_name="known_defect_inspections",
    )
    op.drop_table("known_defect_inspections")
