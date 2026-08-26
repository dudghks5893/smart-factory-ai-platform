"""Create versioned manufacturing decisions and backfill C3-1 correlations.

Revision ID: 20260826_03
Revises: 20260826_02
Create Date: 2026-08-26
"""

from collections.abc import Sequence
from uuid import UUID, uuid4

import sqlalchemy as sa
from alembic import op

revision: str = "20260826_03"
down_revision: str | None = "20260826_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ADD 2026-08-26: Versioned decision schema를 만들고 기존 combined rows에 v1 policy를 적용한다.
def upgrade() -> None:
    op.create_table(
        "inspection_decisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("combined_inspection_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("disposition", sa.String(length=20), nullable=False),
        sa.Column("policy_name", sa.String(length=100), nullable=False),
        sa.Column("policy_version", sa.String(length=50), nullable=False),
        sa.Column("reason_code", sa.String(length=100), nullable=False),
        sa.Column("patchcore_is_anomaly", sa.Boolean(), nullable=False),
        sa.Column("patchcore_score", sa.Float(), nullable=False),
        sa.Column("patchcore_threshold", sa.Float(), nullable=False),
        sa.Column("known_defect_instance_count", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "disposition IN ('PASS', 'REJECT', 'REVIEW')",
            name="ck_inspection_decisions_disposition",
        ),
        sa.CheckConstraint(
            "reason_code IN ('NO_ANOMALY_EVIDENCE', 'UNKNOWN_ANOMALY', "
            "'MODEL_DISAGREEMENT', 'CONFIRMED_KNOWN_DEFECT')",
            name="ck_inspection_decisions_reason_code",
        ),
        sa.CheckConstraint(
            "known_defect_instance_count >= 0",
            name="ck_inspection_decisions_instance_count_nonnegative",
        ),
        sa.CheckConstraint(
            "length(policy_name) > 0 AND length(policy_version) > 0",
            name="ck_inspection_decisions_policy_identity_nonempty",
        ),
        sa.ForeignKeyConstraint(
            ["combined_inspection_id"],
            ["combined_inspections.id"],
            name="fk_inspection_decisions_combined_id",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "combined_inspection_id",
            name="uq_inspection_decisions_combined_id",
        ),
    )
    op.create_index(
        "ix_inspection_decisions_created_at",
        "inspection_decisions",
        ["created_at"],
    )
    op.create_index(
        "ix_inspection_decisions_disposition_created_at",
        "inspection_decisions",
        ["disposition", "created_at"],
    )

    # C3-1 rows의 immutable child evidence에 동일한 v1 truth table을 적용해 recovery를 보존한다.
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            "SELECT c.id AS combined_id, p.is_anomaly, p.anomaly_score, p.threshold, "
            "k.instance_count FROM combined_inspections c "
            "JOIN inspections p ON p.id = c.patchcore_inspection_id "
            "JOIN known_defect_inspections k ON k.id = c.known_defect_inspection_id"
        )
    ).mappings()
    decision_table = sa.table(
        "inspection_decisions",
        sa.column("id", sa.Uuid()),
        sa.column("combined_inspection_id", sa.Uuid()),
        sa.column("disposition", sa.String()),
        sa.column("policy_name", sa.String()),
        sa.column("policy_version", sa.String()),
        sa.column("reason_code", sa.String()),
        sa.column("patchcore_is_anomaly", sa.Boolean()),
        sa.column("patchcore_score", sa.Float()),
        sa.column("patchcore_threshold", sa.Float()),
        sa.column("known_defect_instance_count", sa.Integer()),
    )
    for row in rows:
        is_anomaly = bool(row["is_anomaly"])
        has_known_defect = int(row["instance_count"]) > 0
        disposition, reason_code = _backfill_decision(is_anomaly, has_known_defect)
        bind.execute(
            decision_table.insert().values(
                id=uuid4(),
                combined_inspection_id=UUID(str(row["combined_id"])),
                disposition=disposition,
                policy_name="model_agreement",
                policy_version="1",
                reason_code=reason_code,
                patchcore_is_anomaly=is_anomaly,
                patchcore_score=float(row["anomaly_score"]),
                patchcore_threshold=float(row["threshold"]),
                known_defect_instance_count=int(row["instance_count"]),
            )
        )


# ADD 2026-08-26: Historical migration 안에서 self-contained v1 truth table을 제공한다.
def _backfill_decision(is_anomaly: bool, has_known_defect: bool) -> tuple[str, str]:
    if not is_anomaly and not has_known_defect:
        return "PASS", "NO_ANOMALY_EVIDENCE"
    if is_anomaly and not has_known_defect:
        return "REVIEW", "UNKNOWN_ANOMALY"
    if not is_anomaly and has_known_defect:
        return "REVIEW", "MODEL_DISAGREEMENT"
    return "REJECT", "CONFIRMED_KNOWN_DEFECT"


# ADD 2026-08-26: Combined/child history를 보존하면서 decision schema만 제거한다.
def downgrade() -> None:
    op.drop_index(
        "ix_inspection_decisions_disposition_created_at",
        table_name="inspection_decisions",
    )
    op.drop_index("ix_inspection_decisions_created_at", table_name="inspection_decisions")
    op.drop_table("inspection_decisions")
