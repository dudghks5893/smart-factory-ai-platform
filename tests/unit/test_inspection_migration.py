"""SQLite structural test for the initial Alembic inspection migration."""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


# ADD 2026-08-20: Initial migration upgrade/downgrade가 inspection schema를 왕복하는지 검증한다.
# MODIFY 2026-08-26: Decision schema/backfill, FK와 기존 child row 보존을 검증한다.
def test_initial_inspection_migration_upgrade_and_downgrade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "migration.db"
    database_url = f"sqlite+pysqlite:///{database_path}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = Config("alembic.ini")

    command.upgrade(config, "20260820_01")
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO inspections ("
                "id, model_name, category, is_anomaly, anomaly_score, threshold, "
                "comparison_operator, image_sha256, image_size_bytes, content_type, "
                "model_sha256, artifact_metadata_sha256, threshold_artifact_sha256, "
                "manifest_sha256, device"
                ") VALUES ("
                ":id, 'patchcore', 'metal_nut', 0, 30, 40, '>', :sha, 100, "
                "'image/png', :sha, :sha, :sha, :sha, 'cpu'"
                ")"
            ),
            {"id": "00000000-0000-4000-8000-000000000001", "sha": "a" * 64},
        )
    engine.dispose()

    command.upgrade(config, "20260826_02")
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO known_defect_inspections ("
                "id, model_name, task, category, device, diagnostic_confidence, "
                "inference_ms, image_width, image_height, image_sha256, model_sha256, "
                "artifact_metadata_sha256, dataset_manifest_sha256, "
                "dataset_semantic_fingerprint_sha256, instance_count"
                ") VALUES ("
                ":id, 'yolo11n-seg.pt', 'segment', 'metal_nut', 'cpu', 0.25, "
                "4, 8, 8, :sha, :sha, :sha, :sha, :sha, 0"
                ")"
            ),
            {"id": "00000000-0000-4000-8000-000000000002", "sha": "a" * 64},
        )
        connection.execute(
            text(
                "INSERT INTO combined_inspections ("
                "id, patchcore_inspection_id, known_defect_inspection_id, image_sha256, "
                "image_width, image_height, image_size_bytes, content_type, "
                "patchcore_inference_ms, orchestration_ms"
                ") VALUES ("
                ":id, :patch_id, :known_id, :sha, 8, 8, 100, 'image/png', 5, 7"
                ")"
            ),
            {
                "id": "00000000-0000-4000-8000-000000000003",
                "patch_id": "00000000-0000-4000-8000-000000000001",
                "known_id": "00000000-0000-4000-8000-000000000002",
                "sha": "a" * 64,
            },
        )
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    inspector = inspect(engine)
    assert {
        "inspections",
        "known_defect_inspections",
        "known_defect_instances",
        "combined_inspections",
        "inspection_decisions",
    } <= set(inspector.get_table_names())
    assert {index["name"] for index in inspector.get_indexes("inspections")} == {
        "ix_inspections_anomaly_created_at",
        "ix_inspections_category_created_at",
        "ix_inspections_created_at",
    }
    assert {index["name"] for index in inspector.get_indexes("known_defect_inspections")} == {
        "ix_known_defect_inspections_created_at"
    }
    assert {index["name"] for index in inspector.get_indexes("known_defect_instances")} == {
        "ix_known_defect_instances_inspection_id"
    }
    foreign_keys = inspector.get_foreign_keys("known_defect_instances")
    assert len(foreign_keys) == 1
    assert foreign_keys[0]["referred_table"] == "known_defect_inspections"
    combined_foreign_keys = {
        foreign_key["referred_table"]
        for foreign_key in inspector.get_foreign_keys("combined_inspections")
    }
    assert combined_foreign_keys == {"inspections", "known_defect_inspections"}
    assert {index["name"] for index in inspector.get_indexes("combined_inspections")} == {
        "ix_combined_inspections_created_at"
    }
    decision_foreign_keys = inspector.get_foreign_keys("inspection_decisions")
    assert len(decision_foreign_keys) == 1
    assert decision_foreign_keys[0]["referred_table"] == "combined_inspections"
    assert {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("inspection_decisions")
    } == {"uq_inspection_decisions_combined_id"}
    assert {index["name"] for index in inspector.get_indexes("inspection_decisions")} == {
        "ix_inspection_decisions_created_at",
        "ix_inspection_decisions_disposition_created_at",
    }
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM inspections")) == 1
        decision = connection.execute(
            text(
                "SELECT disposition, reason_code, policy_name, policy_version "
                "FROM inspection_decisions"
            )
        ).one()
        assert decision == ("PASS", "NO_ANOMALY_EVIDENCE", "model_agreement", "1")
    engine.dispose()

    command.downgrade(config, "20260820_01")
    engine = create_engine(database_url)
    table_names = set(inspect(engine).get_table_names())
    assert "inspections" in table_names
    assert "known_defect_inspections" not in table_names
    assert "known_defect_instances" not in table_names
    assert "combined_inspections" not in table_names
    assert "inspection_decisions" not in table_names
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM inspections")) == 1
    engine.dispose()

    command.downgrade(config, "base")
    engine = create_engine(database_url)
    assert "inspections" not in inspect(engine).get_table_names()
    engine.dispose()
