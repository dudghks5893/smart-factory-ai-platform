"""Unit tests for YOLO segmentation config and artifact lineage contracts."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from ml.training.yolo_segmentation import (
    ARTIFACT_SCHEMA_VERSION,
    YoloArtifactMetadata,
    load_yolo_segmentation_config,
    validate_artifact_id,
    validate_yolo_artifact,
    write_yolo_artifact,
)
from pipelines.train_yolo_segmentation import parse_args

CONFIG_PATH = Path("configs/model/yolo_segmentation_baseline.yaml")


# ADD 2026-08-25: Checked-in baseline config의 architecture, split과 fixed lineage를 검증한다.
def test_load_yolo_segmentation_config() -> None:
    config = load_yolo_segmentation_config(CONFIG_PATH)
    assert config.model.weights == "yolo11n-seg.pt"
    assert config.training.imgsz == 640
    assert config.training.optimizer == "auto"
    assert config.training.lr0 is None
    assert config.evaluation.split == "test"
    assert config.dataset_contract.classes == {0: "bent", 1: "color", 2: "scratch"}


# ADD 2026-08-25: Floating tuning과 flip taxonomy가 config validation을 통과하지 못하는지 확인한다.
def test_config_rejects_tuned_learning_rate_and_flip(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["training"]["lr0"] = 0.001
    invalid_lr = tmp_path / "invalid-lr.yaml"
    invalid_lr.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="optimizer/LR defaults"):
        load_yolo_segmentation_config(invalid_lr)

    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["dataset_contract"]["classes"][3] = "flip"
    invalid_class = tmp_path / "invalid-class.yaml"
    invalid_class.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly bent/color/scratch"):
        load_yolo_segmentation_config(invalid_class)


# ADD 2026-08-25: Artifact validation test가 공유할 valid metadata fixture를 생성한다.
def _metadata() -> YoloArtifactMetadata:
    config = load_yolo_segmentation_config(CONFIG_PATH)
    return YoloArtifactMetadata(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        model_name=config.model.weights,
        task=config.model.task,
        architecture=config.model.architecture,
        category=config.dataset_contract.category,
        classes=config.dataset_contract.classes,
        seed=config.training.seed,
        dataset_manifest_sha256=config.dataset_contract.manifest_sha256,
        dataset_semantic_fingerprint_sha256=(config.dataset_contract.semantic_fingerprint_sha256),
        training_config={"epochs": config.training.epochs},
        created_at="2026-08-25T00:00:00+00:00",
        framework="ultralytics",
        framework_version="8.4.128",
        torch_version="2.13.0",
        device="cuda:0",
        best_epoch=12,
        source_checkpoint="weights/best.pt",
        checkpoint_sha256="0" * 64,
    )


# ADD 2026-08-25: Artifact checkpoint copy, metadata round-trip과 SHA corruption을 검증한다.
def test_yolo_artifact_metadata_and_checkpoint_sha(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"portable-ultralytics-checkpoint")
    artifact_dir = tmp_path / "artifact"
    saved = write_yolo_artifact(
        source_checkpoint=checkpoint,
        artifact_dir=artifact_dir,
        metadata=_metadata(),
    )
    loaded = validate_yolo_artifact(artifact_dir)
    assert loaded == saved
    assert loaded.checkpoint_sha256 != "0" * 64

    (artifact_dir / "model.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint SHA"):
        validate_yolo_artifact(artifact_dir)


# ADD 2026-08-25: Artifact dataset manifest lineage mismatch가 evaluation 전에 거부되는지 검증한다.
def test_yolo_artifact_rejects_dataset_lineage_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    artifact_dir = tmp_path / "artifact"
    write_yolo_artifact(
        source_checkpoint=checkpoint,
        artifact_dir=artifact_dir,
        metadata=_metadata(),
    )
    config = load_yolo_segmentation_config(CONFIG_PATH)
    wrong_contract = replace(config.dataset_contract, manifest_sha256="a" * 64)
    with pytest.raises(ValueError, match="dataset lineage"):
        validate_yolo_artifact(artifact_dir, expected_contract=wrong_contract)


# ADD 2026-08-25: Artifact ID traversal과 unsupported CLI device가 parser에서 거부되는지 검증한다.
def test_artifact_id_and_cli_argument_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    validate_artifact_id("yolo11n-seg-c2-v1")
    with pytest.raises(ValueError, match="Artifact ID"):
        validate_artifact_id("../escape")

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_yolo_segmentation", "--artifact-id", "run", "--device", "tpu"],
    )
    with pytest.raises(SystemExit):
        parse_args()
