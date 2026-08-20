"""Create inspection history table.

Revision ID: 20260820_01
Revises: None
Create Date: 2026-08-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260820_01"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ADD 2026-08-20: Initial inspection history table, constraints와 조회 index를 생성한다.
def upgrade() -> None:
    op.create_table(
        "inspections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("is_anomaly", sa.Boolean(), nullable=False),
        sa.Column("anomaly_score", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("comparison_operator", sa.String(length=2), nullable=False),
        sa.Column("image_sha256", sa.String(length=64), nullable=False),
        sa.Column("image_size_bytes", sa.Integer(), nullable=False),
        sa.Column("content_type", sa.String(length=100), nullable=False),
        sa.Column("model_sha256", sa.String(length=64), nullable=False),
        sa.Column("artifact_metadata_sha256", sa.String(length=64), nullable=False),
        sa.Column("threshold_artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("device", sa.String(length=50), nullable=False),
        sa.CheckConstraint(
            "comparison_operator = '>'",
            name="ck_inspections_operator_gt",
        ),
        sa.CheckConstraint(
            "image_size_bytes > 0",
            name="ck_inspections_image_size_positive",
        ),
        sa.CheckConstraint(
            "length(image_sha256) = 64",
            name="ck_inspections_image_sha256_length",
        ),
        sa.CheckConstraint(
            "length(model_sha256) = 64",
            name="ck_inspections_model_sha256_length",
        ),
        sa.CheckConstraint(
            "length(artifact_metadata_sha256) = 64",
            name="ck_inspections_metadata_sha256_length",
        ),
        sa.CheckConstraint(
            "length(threshold_artifact_sha256) = 64",
            name="ck_inspections_threshold_sha256_length",
        ),
        sa.CheckConstraint(
            "length(manifest_sha256) = 64",
            name="ck_inspections_manifest_sha256_length",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_inspections_created_at", "inspections", ["created_at"])
    op.create_index(
        "ix_inspections_category_created_at",
        "inspections",
        ["category", "created_at"],
    )
    op.create_index(
        "ix_inspections_anomaly_created_at",
        "inspections",
        ["is_anomaly", "created_at"],
    )


# ADD 2026-08-20: Inspection history index와 table을 역순으로 제거한다.
def downgrade() -> None:
    op.drop_index("ix_inspections_anomaly_created_at", table_name="inspections")
    op.drop_index("ix_inspections_category_created_at", table_name="inspections")
    op.drop_index("ix_inspections_created_at", table_name="inspections")
    op.drop_table("inspections")
