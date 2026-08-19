"""Raw PatchCore prediction artifact contract and validation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from ml.datasets.manifest import ManifestRecord

PREDICTIONS_FILENAME = "predictions.jsonl"
ANOMALY_MAPS_FILENAME = "anomaly_maps.pt"


@dataclass(frozen=True)
class RawPredictionRecord:
    """Threshold-free prediction metadata for one manifest sample."""

    sample_id: str
    category: str
    defect_type: str
    label: int
    split: str
    raw_anomaly_score: float
    anomaly_map_key: str
    anomaly_map_file: str = ANOMALY_MAPS_FILENAME

    # ADD 2026-08-19: Raw prediction record를 stable JSON mapping으로 변환한다.
    def to_json_dict(self) -> dict[str, Any]:
        """Convert this record to a stable JSON-compatible mapping."""
        return asdict(self)

    # ADD 2026-08-19: JSON record를 strict raw prediction contract로 검증해 복원한다.
    @classmethod
    def from_json_dict(cls, raw: object) -> RawPredictionRecord:
        """Validate and construct one raw prediction record from JSON."""
        if not isinstance(raw, dict):
            raise ValueError("Raw prediction record must be a JSON object.")

        expected_fields = {
            "sample_id",
            "category",
            "defect_type",
            "label",
            "split",
            "raw_anomaly_score",
            "anomaly_map_key",
            "anomaly_map_file",
        }
        if set(raw) != expected_fields:
            raise ValueError(
                f"Unexpected raw prediction fields: {sorted(raw)}; "
                f"expected: {sorted(expected_fields)}"
            )

        label = raw["label"]
        if type(label) is not int or label not in {0, 1}:
            raise ValueError("Raw prediction label must be integer 0 or 1.")

        score = raw["raw_anomaly_score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("Raw prediction score must be a finite number.")
        finite_score = float(score)
        if not math.isfinite(finite_score):
            raise ValueError("Raw prediction score must be a finite number.")

        record = cls(
            sample_id=_required_string(raw["sample_id"], "sample_id"),
            category=_required_string(raw["category"], "category"),
            defect_type=_required_string(raw["defect_type"], "defect_type"),
            label=label,
            split=_required_string(raw["split"], "split"),
            raw_anomaly_score=finite_score,
            anomaly_map_key=_required_string(raw["anomaly_map_key"], "anomaly_map_key"),
            anomaly_map_file=_required_string(raw["anomaly_map_file"], "anomaly_map_file"),
        )
        if record.anomaly_map_key != record.sample_id:
            raise ValueError("Raw prediction anomaly_map_key must equal sample_id.")
        if record.anomaly_map_file != ANOMALY_MAPS_FILENAME:
            raise ValueError(f"Raw prediction anomaly_map_file must be '{ANOMALY_MAPS_FILENAME}'.")
        return record


@dataclass(frozen=True)
class PredictionBundle:
    """Validated raw records and contiguous anomaly tensors in manifest order."""

    records: tuple[RawPredictionRecord, ...]
    scores: Tensor
    anomaly_maps: Tensor

    # ADD 2026-08-19: Prediction record label을 contiguous tensor로 반환한다.
    def labels(self) -> Tensor:
        """Return binary image labels in prediction order."""
        return torch.tensor([record.label for record in self.records], dtype=torch.int64)


# ADD 2026-08-19: Raw prediction JSONL과 anomaly map artifact를 manifest contract로 검증한다.
def load_prediction_bundle(
    *,
    predictions_path: Path,
    anomaly_maps_path: Path,
    expected_records: list[ManifestRecord],
    expected_split: str,
    expected_map_shape: tuple[int, int, int],
) -> PredictionBundle:
    """Load raw predictions and validate them against one manifest split."""
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Prediction JSONL not found: {predictions_path}")
    if not anomaly_maps_path.is_file():
        raise FileNotFoundError(f"Prediction anomaly maps not found: {anomaly_maps_path}")
    if not expected_records:
        raise ValueError(f"Expected manifest split is empty: {expected_split}")

    # JSONL record를 strict schema로 읽으며 split과 sample uniqueness를 검증한다.
    records_by_id: dict[str, RawPredictionRecord] = {}
    try:
        lines = predictions_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read prediction JSONL: {predictions_path}") from exc
    if not lines:
        raise ValueError("Prediction JSONL must not be empty.")

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"Prediction JSONL contains a blank line at {line_number}.")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid prediction JSON at line {line_number}.") from exc
        record = RawPredictionRecord.from_json_dict(raw)
        if record.split != expected_split:
            raise ValueError(
                f"Prediction split mismatch: {record.sample_id} -> {record.split}; "
                f"expected {expected_split}."
            )
        if record.sample_id in records_by_id:
            raise ValueError(f"Duplicate prediction sample_id: {record.sample_id}")
        records_by_id[record.sample_id] = record

    expected_by_id = {record.sample_id: record for record in expected_records}
    if len(expected_by_id) != len(expected_records):
        raise ValueError("Expected manifest records contain duplicate sample_id values.")
    if set(records_by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(records_by_id))
        extra = sorted(set(records_by_id) - set(expected_by_id))
        raise ValueError(f"Prediction sample_id mismatch: missing={missing}, extra={extra}")

    # Manifest order로 record를 정렬하면서 metadata가 원본 record와 일치하는지 확인한다.
    ordered_records: list[RawPredictionRecord] = []
    for expected in expected_records:
        record = records_by_id[expected.sample_id]
        if (
            record.category != expected.category
            or record.defect_type != expected.defect_type
            or record.label != expected.label
            or record.split != expected.split
        ):
            raise ValueError(f"Prediction metadata mismatch for sample_id: {record.sample_id}")
        ordered_records.append(record)

    # Tensor-only artifact를 안전하게 로드하고 key, shape, finite score 계약을 검증한다.
    loaded = torch.load(anomaly_maps_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in loaded.items()
    ):
        raise ValueError("Prediction anomaly_maps.pt must contain a string-to-tensor mapping.")
    anomaly_maps_by_id = cast(dict[str, Tensor], loaded)
    expected_map_keys = {record.sample_id for record in ordered_records}
    if set(anomaly_maps_by_id) != expected_map_keys:
        missing = sorted(expected_map_keys - set(anomaly_maps_by_id))
        extra = sorted(set(anomaly_maps_by_id) - expected_map_keys)
        raise ValueError(f"Anomaly map key mismatch: missing={missing}, extra={extra}")

    ordered_maps: list[Tensor] = []
    for record in ordered_records:
        anomaly_map = anomaly_maps_by_id[record.sample_id]
        if tuple(anomaly_map.shape) != expected_map_shape:
            raise ValueError(
                f"Anomaly map shape mismatch for {record.sample_id}: "
                f"actual={tuple(anomaly_map.shape)}, expected={expected_map_shape}"
            )
        if not anomaly_map.is_floating_point():
            raise TypeError(f"Anomaly map must be floating-point: {record.sample_id}")
        if not torch.isfinite(anomaly_map).all():
            raise ValueError(f"Anomaly map contains non-finite values: {record.sample_id}")
        ordered_maps.append(anomaly_map)

    scores = torch.tensor(
        [record.raw_anomaly_score for record in ordered_records],
        dtype=torch.float64,
    )
    return PredictionBundle(
        records=tuple(ordered_records),
        scores=scores.contiguous(),
        anomaly_maps=torch.stack(ordered_maps).contiguous(),
    )


# ADD 2026-08-19: Prediction JSON string field가 비어있지 않은지 검증한다.
def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Raw prediction field '{field}' must be a non-empty string.")
    return value
