"""Unit tests for raw PatchCore prediction artifact validation."""

import json
from pathlib import Path

import pytest
import torch

from ml.datasets.manifest import ManifestRecord
from ml.evaluation.predictions import RawPredictionRecord, load_prediction_bundle


# ADD 2026-08-19: Prediction contract test용 manifest record를 생성한다.
def _manifest_record(sample_id: str = "sample-1", split: str = "test") -> ManifestRecord:
    return ManifestRecord(
        sample_id=sample_id,
        category="metal_nut",
        source_split="test" if split == "test" else "train",
        split=split,
        defect_type="good",
        label=0,
        image_path=f"metal_nut/{split}/good/{sample_id}.png",
        mask_path="",
        width=8,
        height=8,
    )


# ADD 2026-08-19: Prediction JSONL과 anomaly map fixture를 저장한다.
def _write_prediction_artifacts(
    tmp_path: Path,
    records: list[RawPredictionRecord],
    maps: dict[str, torch.Tensor],
) -> tuple[Path, Path]:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(record.to_json_dict()) + "\n" for record in records),
        encoding="utf-8",
    )
    anomaly_maps_path = tmp_path / "anomaly_maps.pt"
    torch.save(maps, anomaly_maps_path)
    return predictions_path, anomaly_maps_path


# ADD 2026-08-19: Duplicate sample_id가 prediction bundle에서 거부되는지 검증한다.
def test_prediction_bundle_rejects_duplicate_sample_id(tmp_path: Path) -> None:
    record = RawPredictionRecord(
        sample_id="sample-1",
        category="metal_nut",
        defect_type="good",
        label=0,
        split="test",
        raw_anomaly_score=0.1,
        anomaly_map_key="sample-1",
    )
    predictions_path, maps_path = _write_prediction_artifacts(
        tmp_path,
        [record, record],
        {"sample-1": torch.zeros(1, 8, 8)},
    )

    with pytest.raises(ValueError, match="Duplicate prediction sample_id"):
        load_prediction_bundle(
            predictions_path=predictions_path,
            anomaly_maps_path=maps_path,
            expected_records=[_manifest_record()],
            expected_split="test",
            expected_map_shape=(1, 8, 8),
        )


# ADD 2026-08-19: Missing 및 extra anomaly map key를 정확히 거부하는지 검증한다.
@pytest.mark.parametrize(
    "maps",
    [
        {},
        {
            "sample-1": torch.zeros(1, 8, 8),
            "extra": torch.zeros(1, 8, 8),
        },
    ],
)
def test_prediction_bundle_rejects_anomaly_map_key_mismatch(
    tmp_path: Path,
    maps: dict[str, torch.Tensor],
) -> None:
    record = RawPredictionRecord(
        sample_id="sample-1",
        category="metal_nut",
        defect_type="good",
        label=0,
        split="test",
        raw_anomaly_score=0.1,
        anomaly_map_key="sample-1",
    )
    predictions_path, maps_path = _write_prediction_artifacts(tmp_path, [record], maps)

    with pytest.raises(ValueError, match="Anomaly map key mismatch"):
        load_prediction_bundle(
            predictions_path=predictions_path,
            anomaly_maps_path=maps_path,
            expected_records=[_manifest_record()],
            expected_split="test",
            expected_map_shape=(1, 8, 8),
        )


# ADD 2026-08-19: Non-finite anomaly map과 shape mismatch를 fail-fast 검증한다.
@pytest.mark.parametrize(
    ("anomaly_map", "message"),
    [
        (torch.full((1, 8, 8), float("inf")), "non-finite"),
        (torch.zeros(1, 7, 8), "shape mismatch"),
    ],
)
def test_prediction_bundle_rejects_invalid_anomaly_map(
    tmp_path: Path,
    anomaly_map: torch.Tensor,
    message: str,
) -> None:
    record = RawPredictionRecord(
        sample_id="sample-1",
        category="metal_nut",
        defect_type="good",
        label=0,
        split="test",
        raw_anomaly_score=0.1,
        anomaly_map_key="sample-1",
    )
    predictions_path, maps_path = _write_prediction_artifacts(
        tmp_path,
        [record],
        {"sample-1": anomaly_map},
    )

    with pytest.raises(ValueError, match=message):
        load_prediction_bundle(
            predictions_path=predictions_path,
            anomaly_maps_path=maps_path,
            expected_records=[_manifest_record()],
            expected_split="test",
            expected_map_shape=(1, 8, 8),
        )


# ADD 2026-08-19: Non-finite raw score가 JSON contract에서 거부되는지 검증한다.
def test_raw_prediction_record_rejects_nonfinite_score() -> None:
    raw = RawPredictionRecord(
        sample_id="sample-1",
        category="metal_nut",
        defect_type="good",
        label=0,
        split="test",
        raw_anomaly_score=0.1,
        anomaly_map_key="sample-1",
    ).to_json_dict()
    raw["raw_anomaly_score"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        RawPredictionRecord.from_json_dict(raw)


# ADD 2026-08-19: Test prediction bundle에 validation split record가 섞이면 거부하는지 검증한다.
def test_test_prediction_bundle_rejects_validation_record(tmp_path: Path) -> None:
    record = RawPredictionRecord(
        sample_id="sample-1",
        category="metal_nut",
        defect_type="good",
        label=0,
        split="validation",
        raw_anomaly_score=0.1,
        anomaly_map_key="sample-1",
    )
    predictions_path, maps_path = _write_prediction_artifacts(
        tmp_path,
        [record],
        {"sample-1": torch.zeros(1, 8, 8)},
    )

    with pytest.raises(ValueError, match="Prediction split mismatch"):
        load_prediction_bundle(
            predictions_path=predictions_path,
            anomaly_maps_path=maps_path,
            expected_records=[_manifest_record()],
            expected_split="test",
            expected_map_shape=(1, 8, 8),
        )
