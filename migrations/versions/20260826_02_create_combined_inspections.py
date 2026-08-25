"""Create atomic dual-model inspection correlation table.

Revision ID: 20260826_02
Revises: 20260826_01
Create Date: 2026-08-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260826_02"
down_revision: str | None = "20260826_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ADD 2026-08-26: Dual-model child row를 연결하는 non-null correlation schema를 생성한다.
def upgrade() -> None:
    op.create_table(
        "combined_inspections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("patchcore_inspection_id", sa.Uuid(), nullable=False),
        sa.Column("known_defect_inspection_id", sa.Uuid(), nullable=False),
        sa.Column("image_sha256", sa.String(length=64), nullable=False),
        sa.Column("image_width", sa.Integer(), nullable=False),
        sa.Column("image_height", sa.Integer(), nullable=False),
        sa.Column("image_size_bytes", sa.Integer(), nullable=False),
        sa.Column("content_type", sa.String(length=100), nullable=False),
        sa.Column("patchcore_inference_ms", sa.Float(), nullable=False),
        sa.Column("orchestration_ms", sa.Float(), nullable=False),
        sa.CheckConstraint(
            "image_width > 0 AND image_height > 0",
            name="ck_combined_inspections_image_dimensions_positive",
        ),
        sa.CheckConstraint(
            "image_size_bytes > 0",
            name="ck_combined_inspections_image_size_positive",
        ),
        sa.CheckConstraint(
            "length(image_sha256) = 64",
            name="ck_combined_inspections_image_sha256_length",
        ),
        sa.CheckConstraint(
            "patchcore_inference_ms >= 0 AND orchestration_ms >= 0",
            name="ck_combined_inspections_timings_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["patchcore_inspection_id"],
            ["inspections.id"],
            name="fk_combined_inspections_patchcore_id",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["known_defect_inspection_id"],
            ["known_defect_inspections.id"],
            name="fk_combined_inspections_known_defect_id",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "patchcore_inspection_id",
            name="uq_combined_inspections_patchcore_id",
        ),
        sa.UniqueConstraint(
            "known_defect_inspection_id",
            name="uq_combined_inspections_known_defect_id",
        ),
    )
    op.create_index(
        "ix_combined_inspections_created_at",
        "combined_inspections",
        ["created_at"],
    )


# ADD 2026-08-26: Child history를 보존하면서 correlation schema만 제거한다.
def downgrade() -> None:
    op.drop_index("ix_combined_inspections_created_at", table_name="combined_inspections")
    op.drop_table("combined_inspections")
