"""Unit tests for deterministic PatchCore score drift statistics and artifacts."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ml.datasets.manifest import ManifestRecord
from ml.drift.patchcore import (
    REFERENCE_FILENAME,
    DriftLineage,
    DriftObservation,
    DriftPolicy,
    DriftReference,
    anomaly_ratio,
    build_drift_report,
    build_reference_bin_edges,
    histogram_counts,
    load_validation_normal_scores,
    read_drift_reference,
    summarize_scores,
    write_drift_reference,
)
from ml.evaluation.predictions import RawPredictionRecord

CREATED_AT = "2026-08-21T00:00:00+00:00"
SINCE = datetime(2026, 8, 21, tzinfo=UTC)
UNTIL = SINCE + timedelta(hours=1)
REFERENCE_SCORES = tuple(float(value) for value in range(30))


# ADD 2026-08-21: Drift unit test에 사용할 complete SHA lineage를 생성한다.
def _lineage(**overrides: str) -> DriftLineage:
    values = {
        "model_sha256": "a" * 64,
        "artifact_metadata_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "threshold_artifact_sha256": "d" * 64,
    }
    values.update(overrides)
    return DriftLineage(**values)


# ADD 2026-08-21: Synthetic score에서 internally consistent reference artifact를 생성한다.
def _reference(
    scores: tuple[float, ...] = REFERENCE_SCORES,
    *,
    threshold: float | None = None,
) -> DriftReference:
    image_threshold = max(scores) if threshold is None else threshold
    edges = build_reference_bin_edges(scores, 10)
    reference = DriftReference(
        schema_version=1,
        reference_id="reference-1",
        model_name="patchcore",
        category="metal_nut",
        lineage=_lineage(),
        source_split="validation",
        source_label="normal",
        validation_predictions_sha256="e" * 64,
        sample_count=len(scores),
        score_values=scores,
        summary=summarize_scores(scores),
        image_threshold=image_threshold,
        comparison_operator=">",
        reference_anomaly_ratio=anomaly_ratio(scores, threshold=image_threshold),
        psi_bin_count_requested=10,
        psi_bin_edges=edges,
        reference_bin_counts=histogram_counts(scores, edges),
        psi_epsilon=1e-6,
        created_at=CREATED_AT,
    )
    reference.validate()
    return reference


# ADD 2026-08-21: Synthetic production score를 reference-compatible observation으로 변환한다.
def _observations(
    scores: tuple[float, ...],
    *,
    reference: DriftReference,
    lineage: DriftLineage | None = None,
) -> tuple[DriftObservation, ...]:
    return tuple(
        DriftObservation(
            created_at=SINCE + timedelta(seconds=index),
            model_name=reference.model_name,
            category=reference.category,
            anomaly_score=score,
            is_anomaly=score > reference.image_threshold,
            lineage=lineage or reference.lineage,
        )
        for index, score in enumerate(scores)
    )


# ADD 2026-08-21: Nested JSON payload의 모든 numeric value가 finite인지 재귀 검증한다.
def _all_numbers_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_numbers_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(value)
    return True


# ADD 2026-08-21: 동일 score distribution이 zero PSI와 stable status를 만드는지 검증한다.
def test_identical_distribution_is_stable_with_zero_psi() -> None:
    reference = _reference()
    report = build_drift_report(
        drift_id="identical",
        reference=reference,
        observations=_observations(reference.score_values, reference=reference),
        since=SINCE,
        until=UNTIL,
        policy=DriftPolicy(),
        created_at=CREATED_AT,
    )

    assert report["status"] == "stable"
    assert report["statistics"]["psi"] == pytest.approx(0.0)
    assert report["statistics"]["anomaly_ratio"] == pytest.approx(0.0)


# ADD 2026-08-21: 충분히 이동한 score가 PSI와 anomaly ratio를 높여 drift가 되는지 검증한다.
def test_shifted_distribution_produces_larger_psi_and_drift() -> None:
    reference = _reference()
    identical = build_drift_report(
        drift_id="identical",
        reference=reference,
        observations=_observations(reference.score_values, reference=reference),
        since=SINCE,
        until=UNTIL,
        policy=DriftPolicy(),
        created_at=CREATED_AT,
    )
    shifted_scores = tuple(score + 100.0 for score in reference.score_values)
    shifted = build_drift_report(
        drift_id="shifted",
        reference=reference,
        observations=_observations(shifted_scores, reference=reference),
        since=SINCE,
        until=UNTIL,
        policy=DriftPolicy(),
        created_at=CREATED_AT,
    )

    assert shifted["statistics"]["psi"] > identical["statistics"]["psi"]
    assert shifted["statistics"]["anomaly_ratio"] == pytest.approx(1.0)
    assert shifted["status"] == "drift"


# ADD 2026-08-21: Minimum 미만 sample이 insufficient_data status가 되는지 검증한다.
def test_small_current_window_is_insufficient_data() -> None:
    reference = _reference()
    report = build_drift_report(
        drift_id="small-window",
        reference=reference,
        observations=_observations((100.0, 101.0, 102.0), reference=reference),
        since=SINCE,
        until=UNTIL,
        policy=DriftPolicy(minimum_sample_count=30),
        created_at=CREATED_AT,
    )

    assert report["status"] == "insufficient_data"
    assert report["current_window"]["sample_count"] == 3


# ADD 2026-08-21: Full lineage가 하나라도 다르면 mixed production window를 거부하는지 검증한다.
def test_drift_report_rejects_mixed_lineage() -> None:
    reference = _reference()
    mixed_lineage = _lineage(threshold_artifact_sha256="f" * 64)

    with pytest.raises(ValueError, match="Mixed or mismatched"):
        build_drift_report(
            drift_id="mixed-lineage",
            reference=reference,
            observations=_observations((1.0,), reference=reference, lineage=mixed_lineage),
            since=SINCE,
            until=UNTIL,
            policy=DriftPolicy(minimum_sample_count=1),
            created_at=CREATED_AT,
        )


# ADD 2026-08-21: Zero reference anomaly ratio를 division 없이 absolute delta로 처리하는지 검증한다.
def test_zero_reference_anomaly_ratio_is_safe_and_finite() -> None:
    reference = _reference()
    report = build_drift_report(
        drift_id="zero-reference-ratio",
        reference=reference,
        observations=_observations(tuple([100.0] * 30), reference=reference),
        since=SINCE,
        until=UNTIL,
        policy=DriftPolicy(),
        created_at=CREATED_AT,
    )

    assert report["statistics"]["reference_anomaly_ratio"] == 0.0
    assert report["statistics"]["anomaly_ratio_absolute_delta"] == 1.0
    assert _all_numbers_finite(report)


# ADD 2026-08-21: Input 순서와 무관하게 동일한 drift report가 재현되는지 검증한다.
def test_drift_report_is_deterministic_for_same_population() -> None:
    reference = _reference()
    observations = _observations(reference.score_values, reference=reference)
    first = build_drift_report(
        drift_id="deterministic",
        reference=reference,
        observations=observations,
        since=SINCE,
        until=UNTIL,
        policy=DriftPolicy(),
        created_at=CREATED_AT,
    )
    second = build_drift_report(
        drift_id="deterministic",
        reference=reference,
        observations=tuple(reversed(observations)),
        since=SINCE,
        until=UNTIL,
        policy=DriftPolicy(),
        created_at=CREATED_AT,
    )

    assert second == first


# ADD 2026-08-21: Non-finite reference score가 artifact 생성 전에 거부되는지 검증한다.
def test_reference_rejects_nonfinite_score() -> None:
    with pytest.raises(ValueError, match="finite"):
        _reference((0.0, float("nan")))


# ADD 2026-08-21: Persisted reference threshold가 source score maximum과 다르면 거부한다.
def test_reference_rejects_threshold_contract_tampering() -> None:
    raw = _reference().to_json_dict()
    raw["threshold"]["image_threshold"] = 28.0
    raw["reference_anomaly_ratio"] = 1 / 30

    with pytest.raises(ValueError, match="maximum score"):
        DriftReference.from_json_dict(raw)


# ADD 2026-08-21: Reference artifact round-trip과 existing path overwrite 거부를 검증한다.
def test_reference_round_trip_and_overwrite_rejection(tmp_path: Path) -> None:
    reference = _reference()
    path = tmp_path / REFERENCE_FILENAME

    write_drift_reference(reference, path)
    assert read_drift_reference(path) == reference
    with pytest.raises(FileExistsError, match="already exists"):
        write_drift_reference(reference, path)


# ADD 2026-08-21: Reference loader가 validation-normal metadata와 순서를 보존하는지 검증한다.
def test_reference_loader_accepts_validation_normal_only(tmp_path: Path) -> None:
    manifest_records = [_manifest_record("validation-2"), _manifest_record("validation-1")]
    prediction_records = [
        _prediction_record("validation-1", score=0.1),
        _prediction_record("validation-2", score=0.2),
    ]
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        "".join(json.dumps(record.to_json_dict()) + "\n" for record in prediction_records),
        encoding="utf-8",
    )

    scores = load_validation_normal_scores(
        path,
        expected_records=manifest_records,
        category="metal_nut",
    )

    assert scores == (0.2, 0.1)


# ADD 2026-08-21: Test split 또는 anomaly label reference 혼입이 거부되는지 검증한다.
@pytest.mark.parametrize(("split", "label"), [("test", 0), ("validation", 1)])
def test_reference_loader_rejects_non_validation_normal_source(
    tmp_path: Path, split: str, label: int
) -> None:
    manifest = _manifest_record("sample", split=split, label=label)
    prediction = _prediction_record("sample", split=split, label=label, score=0.1)
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(prediction.to_json_dict()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="validation-normal"):
        load_validation_normal_scores(
            path,
            expected_records=[manifest],
            category="metal_nut",
        )


# ADD 2026-08-21: Reference loader test용 manifest record를 생성한다.
def _manifest_record(
    sample_id: str,
    *,
    split: str = "validation",
    label: int = 0,
) -> ManifestRecord:
    return ManifestRecord(
        sample_id=sample_id,
        category="metal_nut",
        source_split="train" if split == "validation" else "test",
        split=split,
        defect_type="good",
        label=label,
        image_path=f"metal_nut/train/good/{sample_id}.png",
        mask_path="",
        width=8,
        height=8,
    )


# ADD 2026-08-21: Reference loader test용 raw prediction record를 생성한다.
def _prediction_record(
    sample_id: str,
    *,
    score: float,
    split: str = "validation",
    label: int = 0,
) -> RawPredictionRecord:
    return RawPredictionRecord(
        sample_id=sample_id,
        category="metal_nut",
        defect_type="good",
        label=label,
        split=split,
        raw_anomaly_score=score,
        anomaly_map_key=sample_id,
    )
