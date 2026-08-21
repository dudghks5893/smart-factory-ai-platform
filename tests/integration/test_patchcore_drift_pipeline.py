"""Integration test for persisted inspection query and PatchCore drift report output."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import update

from ml.drift.patchcore import (
    DriftLineage,
    DriftPolicy,
    DriftReference,
    anomaly_ratio,
    build_reference_bin_edges,
    histogram_counts,
    summarize_scores,
    write_drift_reference,
)
from pipelines.analyze_patchcore_drift import analyze_patchcore_drift
from services.persistence.database import create_database_manager
from services.persistence.drift import SqlAlchemyDriftWindowRepository
from services.persistence.inspections import InspectionCreate, SqlAlchemyInspectionRepository
from services.persistence.models import Base, InspectionRecord


# ADD 2026-08-21: Integration test용 internally consistent drift reference를 생성한다.
def _reference() -> DriftReference:
    scores = tuple(float(index) for index in range(30))
    edges = build_reference_bin_edges(scores, 10)
    return DriftReference(
        schema_version=1,
        reference_id="integration-reference",
        model_name="patchcore",
        category="metal_nut",
        lineage=DriftLineage(
            model_sha256="a" * 64,
            artifact_metadata_sha256="b" * 64,
            manifest_sha256="c" * 64,
            threshold_artifact_sha256="d" * 64,
        ),
        source_split="validation",
        source_label="normal",
        validation_predictions_sha256="e" * 64,
        sample_count=len(scores),
        score_values=scores,
        summary=summarize_scores(scores),
        image_threshold=29.0,
        comparison_operator=">",
        reference_anomaly_ratio=anomaly_ratio(scores, threshold=29.0),
        psi_bin_count_requested=10,
        psi_bin_edges=edges,
        reference_bin_counts=histogram_counts(scores, edges),
        psi_epsilon=1e-6,
        created_at="2026-08-21T00:00:00+00:00",
    )


# ADD 2026-08-21: Integration DB insert용 reference-compatible inspection 값을 생성한다.
def _inspection(score: float) -> InspectionCreate:
    return InspectionCreate(
        model_name="patchcore",
        category="metal_nut",
        is_anomaly=False,
        anomaly_score=score,
        threshold=29.0,
        comparison_operator=">",
        image_sha256=f"{int(score):064x}",
        image_size_bytes=100,
        content_type="image/png",
        model_sha256="a" * 64,
        artifact_metadata_sha256="b" * 64,
        threshold_artifact_sha256="d" * 64,
        manifest_sha256="c" * 64,
        device="cpu",
    )


# ADD 2026-08-21: SQLite inspection history에서 query, analysis와 immutable JSON 저장을 검증한다.
def test_persisted_inspection_window_produces_drift_report(tmp_path: Path) -> None:
    database = create_database_manager(f"sqlite+pysqlite:///{tmp_path / 'pipeline.db'}")
    Base.metadata.create_all(database.engine)
    writer = SqlAlchemyInspectionRepository(database.session_factory)
    since = datetime(2026, 8, 21, tzinfo=UTC)
    reference = _reference()
    reference_path = tmp_path / "reference.json"
    write_drift_reference(reference, reference_path)

    try:
        rows = [writer.create(_inspection(score)) for score in reference.score_values]
        with database.engine.begin() as connection:
            for index, row in enumerate(rows):
                connection.execute(
                    update(InspectionRecord)
                    .where(InspectionRecord.id == row.id)
                    .values(created_at=since + timedelta(seconds=index))
                )

        summary = analyze_patchcore_drift(
            repository=SqlAlchemyDriftWindowRepository(database.session_factory),
            reference_path=reference_path,
            output_dir=tmp_path / "drift-output",
            drift_id="integration-run",
            since=since,
            until=since + timedelta(hours=1),
            policy=DriftPolicy(),
            created_at="2026-08-21T01:00:00+00:00",
        )
    finally:
        database.dispose()

    persisted = json.loads(summary.report_path.read_text(encoding="utf-8"))
    assert persisted == summary.report
    assert persisted["status"] == "stable"
    assert persisted["current_window"]["sample_count"] == 30
    assert persisted["current_window"]["boundary"] == "since_inclusive_until_exclusive"
    assert persisted["statistics"]["psi"] == 0.0
    assert "database_url" not in persisted
