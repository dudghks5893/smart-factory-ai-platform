"""Tests for controlled YOLO experiment identity and comparison contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.evaluation.yolo_segmentation_error_analysis import require_validation_records
from ml.experiments.yolo_segmentation import (
    build_experiment_metadata,
    load_yolo_experiment_config,
    recommend_experiment,
    validate_experiment_result,
)
from ml.training.yolo_segmentation import load_yolo_segmentation_config

CONFIG_PATH = Path("configs/experiments/yolo_segmentation/c4_2a_yolo11n_seg_imgsz1024_seed42.yaml")


# ADD 2026-08-27: Minimal derived record로 validation leakage boundary를 검증한다.
def _record(sample_id: str, split: str) -> DerivedManifestRecord:
    return DerivedManifestRecord(
        dataset_name="dataset",
        dataset_version="v1",
        derived_task="yolo_segmentation",
        source_manifest_sha256="a" * 64,
        source_split="test",
        source_manifest_split="test",
        source_image_path=f"source/{sample_id}.png",
        source_mask_path=f"source/{sample_id}_mask.png",
        category="metal_nut",
        sample_id=sample_id,
        defect_type="bent",
        target_class="bent",
        target_class_id="0",
        derived_split=split,
        is_negative=False,
        image_width=10,
        image_height=10,
        image_path=f"images/{split}/{sample_id}.png",
        label_path=f"labels/{split}/{sample_id}.txt",
        image_sha256="b" * 64,
        mask_sha256="c" * 64,
        polygon_count=1,
        component_count=1,
        hole_count=0,
        polygon_vertex_count=4,
        round_trip_iou="1.0",
        pixel_precision="1.0",
        pixel_recall="1.0",
    )


# ADD 2026-08-27: Dedicated config identity와 imgsz-only training change를 검증한다.
def test_experiment_config_preserves_baseline_constants() -> None:
    experiment = load_yolo_experiment_config(CONFIG_PATH)
    baseline = load_yolo_segmentation_config(experiment.baseline_config_path)
    candidate = experiment.training_config(baseline)

    assert experiment.experiment_id == "c4_2a_yolo11n_seg_imgsz1024_seed42"
    assert experiment.controlled_change.field == "training.imgsz"
    assert experiment.controlled_change.before == 640
    assert experiment.controlled_change.after == 1024
    assert experiment.candidate_identity["pretrained_checkpoint"] == "yolo11n-seg.pt"
    assert experiment.candidate_identity["imgsz"] == 1024
    assert experiment.baseline_evidence["training"] == {
        "configured_epochs": 100,
        "completed_epochs": 80,
        "early_stopping": True,
        "best_epoch": 60,
        "checkpoint_cumulative_epoch_time_seconds": 222.485,
        "exact_end_to_end_wall_clock_seconds": None,
    }
    assert experiment.baseline_evidence["validation_framework"]["mask"]["map50_95"] == 0.34359
    assert experiment.baseline_evidence["derived_test_metrics_used_for_selection"] is False
    assert candidate.training.imgsz == 1024
    baseline_training = asdict(baseline.training)
    candidate_training = asdict(candidate.training)
    assert {key: value for key, value in candidate_training.items() if key != "imgsz"} == {
        key: value for key, value in baseline_training.items() if key != "imgsz"
    }
    assert candidate.model == baseline.model
    assert candidate.dataset_contract == baseline.dataset_contract
    assert candidate.training.batch == 16


# ADD 2026-08-27: Config parser가 baseline constant drift와 second intervention을 거부한다.
def test_experiment_config_rejects_uncontrolled_change(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8").replace("after: 1024", "after: 1280")
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="640 -> 1024"):
        load_yolo_experiment_config(path)


# ADD 2026-08-27: YAML 문자열을 boolean decision flag로 허용하지 않는다.
def test_experiment_config_rejects_string_boolean(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "require_small_recall_improvement: true",
        'require_small_recall_improvement: "true"',
    )
    path = tmp_path / "invalid-boolean.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="must be a boolean"):
        load_yolo_experiment_config(path)


# ADD 2026-08-27: Historical Baseline reference가 derived-test selection을 허용하지 않는다.
def test_experiment_config_rejects_historical_test_selection(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "derived_test_metrics_used_for_selection: false",
        "derived_test_metrics_used_for_selection: true",
    )
    path = tmp_path / "invalid-test-selection.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Derived-test metrics"):
        load_yolo_experiment_config(path)


# ADD 2026-08-27: Val만 허용하고 train/test/duplicate identity를 fail-fast한다.
def test_validation_protocol_rejects_train_test_and_duplicates() -> None:
    assert require_validation_records([_record("val", "val")])[0].sample_id == "val"
    for split in ("train", "test"):
        with pytest.raises(ValueError, match="non-val"):
            require_validation_records([_record(split, split)])
    record = _record("duplicate", "val")
    with pytest.raises(ValueError, match="unique"):
        require_validation_records([record, record])


# ADD 2026-08-27: Predeclared decision branch용 nested quality fixture를 만든다.
def _quality(
    *,
    map50_95: float = 0.5,
    recall: float = 0.6,
    small: float = 0.25,
    multi: float = 0.5,
    good_fp_rate: float = 0.0,
) -> dict[str, object]:
    return {
        "ultralytics": {"mask": {"map50_95": map50_95}},
        "diagnostic": {"recall": recall},
        "failure_modes": {
            "small_recall": small,
            "multi_component_recall": multi,
            "good_negative_fp_image_rate": good_fp_rate,
        },
    }


# ADD 2026-08-27: Primary/failure/guardrail 조합의 ACCEPT/REJECT/PENDING을 검증한다.
def test_predeclared_recommendation_is_multi_metric() -> None:
    config = load_yolo_experiment_config(CONFIG_PATH)
    before = _quality()
    accepted = recommend_experiment(
        quality_before=before,
        quality_after=_quality(map50_95=0.51, recall=0.62, small=0.5, multi=0.6),
        policy=config.decision_policy,
    )
    assert accepted.decision == "ACCEPT"
    rejected = recommend_experiment(
        quality_before=before,
        quality_after=_quality(map50_95=0.49, recall=0.7, small=0.6, multi=0.7),
        policy=config.decision_policy,
    )
    assert rejected.decision == "REJECT"
    pending = recommend_experiment(
        quality_before=before,
        quality_after=_quality(map50_95=0.51, recall=0.62, small=0.25, multi=0.6),
        policy=config.decision_policy,
    )
    assert pending.decision == "PENDING"
    comparison = {
        "quality_before": before,
        "quality_after": _quality(map50_95=0.51, recall=0.62, small=0.25, multi=0.6),
        "resource_cost_after": {"training_wall_clock_seconds": 10.0},
        "recommendation": asdict(pending),
    }
    serialized = json.loads(json.dumps(comparison, allow_nan=False))
    assert serialized["recommendation"]["decision"] == "PENDING"


# ADD 2026-08-27: Pre-run metadata가 lineage를 보존하고 과거 값을 만들지 않는지 검증한다.
def test_experiment_metadata_has_no_fabricated_resource_defaults() -> None:
    config = load_yolo_experiment_config(CONFIG_PATH)
    payload = build_experiment_metadata(
        config,
        git_commit="d" * 40,
        manifest_sha256="e" * 64,
    )
    assert payload["decision"] == "PENDING"
    assert payload["validation_protocol"]["split"] == "val"
    assert payload["validation_protocol"]["test_split_used"] is False
    assert payload["historical_baseline_evidence"]["validation_framework"]["split"] == "val"
    assert "resource_metrics" not in payload
    assert json.loads(json.dumps(payload))["experiment_id"] == config.experiment_id


# ADD 2026-08-27: Final result required fields와 sealed-test/decision schema를 검증한다.
def test_experiment_result_serialization_contract() -> None:
    payload = {
        "experiment_id": "experiment",
        "hypothesis": "hypothesis",
        "controlled_change": {},
        "constants": {},
        "split": "val",
        "test_split_used": False,
        "quality_before": {},
        "quality_after": {},
        "resource_metrics": {},
        "failure_mode_metrics": {},
        "model_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "decision": "PENDING",
        "decision_reason": "awaiting validation evidence",
    }
    validate_experiment_result(payload)
    with pytest.raises(ValueError, match="validation-only"):
        validate_experiment_result({**payload, "split": "test", "test_split_used": True})
    with pytest.raises(ValueError, match="decision"):
        validate_experiment_result({**payload, "decision": "PROMOTE"})


# ADD 2026-08-27: C4-1 validation protocol constants가 config drift를 거부하는지 검증한다.
def test_validation_protocol_keeps_c4_1_boundaries() -> None:
    config = load_yolo_experiment_config(CONFIG_PATH)
    protocol = config.validation_protocol
    assert protocol.diagnostic_confidence == 0.25
    assert protocol.mask_iou_threshold == 0.5
    assert protocol.small_max_area_ratio == 0.015947619047619047
    assert protocol.medium_max_area_ratio == 0.02447142857142857
    with pytest.raises(ValueError, match="size bucket"):
        replace(protocol, small_max_area_ratio=0.02).validate()
