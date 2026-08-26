"""Tests for C4-2A runner serialization and artifact helpers."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from pipelines.run_yolo_segmentation_experiment import (
    _assert_baseline_immutable,
    build_experiment_package,
    create_diagnostic_runtime_bundle,
    read_training_progress,
)
from shared.hashing import sha256_file


# ADD 2026-08-27: Runtime bundle copy와 source baseline immutability를 검증한다.
def test_diagnostic_bundle_and_baseline_immutability(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    model_dir = baseline / "model"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "model.pt"
    metadata_path = model_dir / "metadata.json"
    model_path.write_bytes(b"baseline-model")
    metadata_path.write_text('{"baseline": true}\n', encoding="utf-8")
    model_sha = sha256_file(model_path)
    metadata_sha = sha256_file(metadata_path)
    bundle = create_diagnostic_runtime_bundle(model_dir, tmp_path / "bundle")
    assert sha256_file(bundle / "model" / "model.pt") == model_sha
    _assert_baseline_immutable(
        baseline,
        model_sha256=model_sha,
        metadata_sha256=metadata_sha,
    )
    model_path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Baseline"):
        _assert_baseline_immutable(
            baseline,
            model_sha256=model_sha,
            metadata_sha256=metadata_sha,
        )


# ADD 2026-08-27: Completed epoch count와 early-stop status를 results.csv에서 읽는다.
def test_read_training_progress(tmp_path: Path) -> None:
    (tmp_path / "results.csv").write_text("epoch,metric\n1,0.1\n2,0.2\n", encoding="utf-8")
    progress = read_training_progress(tmp_path, configured_epochs=100)
    assert progress == {"epochs_completed": 2, "early_stopping": True}


# ADD 2026-08-27: Evidence package가 raw dataset 없이 model/config/evidence와 SHA manifest를 담는다.
def test_experiment_package_contract(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    evidence_names = (
        "experiment_metadata.json",
        "training_metrics.json",
        "validation_metrics.json",
        "error_analysis_summary.json",
        "resource_telemetry.json",
        "comparison_to_baseline.json",
        "experiment_result.json",
        "environment.json",
    )
    for name in evidence_names:
        (experiment_dir / name).write_text(json.dumps({"name": name}), encoding="utf-8")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.pt").write_bytes(b"candidate")
    (artifact / "metadata.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "experiment.yaml"
    config.write_text("experiment: fixture\n", encoding="utf-8")
    package = build_experiment_package(
        experiment_dir=experiment_dir,
        candidate_artifact_dir=artifact,
        experiment_config_path=config,
        package_path=tmp_path / "package.zip",
    )
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
    assert "model/model.pt" in names
    assert "model/metadata.json" in names
    assert "SHA256SUMS.txt" in names
    assert not any(name.startswith("data/") for name in names)
