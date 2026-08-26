"""Atomic persistence tests for correlated dual-model inspections."""

from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from services.decision import (
    DecisionInput,
    KnownDefectDecisionEvidence,
    PatchCoreDecisionEvidence,
    decide_manufacturing_inspection,
)
from services.persistence.combined_inspections import (
    CombinedInspectionCreate,
    SqlAlchemyCombinedInspectionRepository,
)
from services.persistence.database import PersistenceError, create_database_manager
from services.persistence.inspections import InspectionCreate
from services.persistence.known_defects import KnownDefectCreate
from services.persistence.models import (
    CombinedInspectionRecord,
    InspectionDecisionRecord,
    InspectionRecord,
    KnownDefectInspectionRecord,
)
from tests.persistence_helpers import prepare_sqlite_database


def _values(*, combined_id: UUID | None = None) -> CombinedInspectionCreate:
    image_sha256 = "a" * 64
    return CombinedInspectionCreate(
        id=combined_id or uuid4(),
        image_sha256=image_sha256,
        image_width=8,
        image_height=8,
        image_size_bytes=100,
        content_type="image/png",
        patchcore_inference_ms=5.0,
        orchestration_ms=7.0,
        patchcore=InspectionCreate(
            model_name="patchcore",
            category="metal_nut",
            is_anomaly=False,
            anomaly_score=30.0,
            threshold=40.0,
            comparison_operator=">",
            image_sha256=image_sha256,
            image_size_bytes=100,
            content_type="image/png",
            model_sha256="b" * 64,
            artifact_metadata_sha256="c" * 64,
            threshold_artifact_sha256="d" * 64,
            manifest_sha256="e" * 64,
            device="cpu",
        ),
        known_defect=KnownDefectCreate(
            model_name="yolo11n-seg.pt",
            task="segment",
            category="metal_nut",
            device="cpu",
            diagnostic_confidence=0.25,
            inference_ms=4.0,
            image_width=8,
            image_height=8,
            image_sha256=image_sha256,
            model_sha256="1" * 64,
            artifact_metadata_sha256="2" * 64,
            dataset_manifest_sha256="3" * 64,
            dataset_semantic_fingerprint_sha256="4" * 64,
            instances=(),
        ),
        decision=decide_manufacturing_inspection(
            DecisionInput(
                patchcore=PatchCoreDecisionEvidence(False, 30.0, 40.0),
                known_defects=KnownDefectDecisionEvidence(0, ()),
            )
        ),
    )


# ADD 2026-08-26: Correlation constraint failure가 두 child insert까지 rollback함을 검증한다.
def test_combined_repository_rolls_back_both_children_on_link_failure(tmp_path: Path) -> None:
    database_url = prepare_sqlite_database(tmp_path)
    database = create_database_manager(database_url)
    repository = SqlAlchemyCombinedInspectionRepository(database.session_factory)
    combined_id = uuid4()
    first = repository.create(_values(combined_id=combined_id))

    with pytest.raises(PersistenceError, match="Combined inspection insert failed"):
        repository.create(_values(combined_id=combined_id))

    with database.session_factory() as session:
        patchcore_count = session.scalar(select(func.count()).select_from(InspectionRecord))
        yolo_count = session.scalar(select(func.count()).select_from(KnownDefectInspectionRecord))
        decision_count = session.scalar(select(func.count()).select_from(InspectionDecisionRecord))
    recovered = repository.get(combined_id)
    database.dispose()

    assert patchcore_count == yolo_count == 1
    assert decision_count == 1
    assert recovered == first


# ADD 2026-08-26: Decision insert failure가 앞선 combined flush 전체를 rollback함을 검증한다.
def test_decision_insert_failure_rolls_back_entire_combined_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = prepare_sqlite_database(tmp_path)
    database = create_database_manager(database_url)
    repository = SqlAlchemyCombinedInspectionRepository(database.session_factory)

    def fail_decision_insert(*args: object, **kwargs: object) -> None:
        raise SQLAlchemyError("decision insert failure")

    monkeypatch.setattr(
        "services.persistence.combined_inspections.add_inspection_decision",
        fail_decision_insert,
    )
    with pytest.raises(PersistenceError, match="Combined inspection insert failed"):
        repository.create(_values())

    with database.session_factory() as session:
        counts = tuple(
            session.scalar(select(func.count()).select_from(record))
            for record in (
                InspectionRecord,
                KnownDefectInspectionRecord,
                CombinedInspectionRecord,
                InspectionDecisionRecord,
            )
        )
    database.dispose()

    assert counts == (0, 0, 0, 0)
