"""Tests for C4-2A runner serialization and artifact helpers."""

from __future__ import annotations

import inspect
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.experiments.yolo_sampling import (
    RuntimeTrainViewArtifact,
    TrainViewArtifact,
    TrainViewEvidence,
)
from ml.experiments.yolo_segmentation import load_yolo_experiment_config
from ml.training.yolo_segmentation import YoloOutputConfig, load_yolo_segmentation_config
from pipelines.run_yolo_segmentation_experiment import (
    _assert_baseline_immutable,
    build_experiment_package,
    create_diagnostic_runtime_bundle,
    prepare_experiment_training_dataset,
    read_training_progress,
    run_yolo_segmentation_experiment,
)
from pipelines.train_yolo_segmentation import BackendTrainingResult, train_yolo_segmentation
from shared.hashing import sha256_file

C4_2B_CONFIG_PATH = Path(
    "configs/experiments/yolo_segmentation/"
    "c4_2b_yolo11n_seg_component_aware_sampling_x2_seed42.yaml"
)
C4_2A_CONFIG_PATH = Path(
    "configs/experiments/yolo_segmentation/c4_2a_yolo11n_seg_imgsz1024_seed42.yaml"
)


# ADD 2026-08-28: Official C4-2B count arithmetic의 train-view evidence fixture를 만든다.
def _train_view_evidence() -> TrainViewEvidence:
    eligible_ids = tuple(f"train-{index:03d}" for index in range(19))
    return TrainViewEvidence(
        schema_version=1,
        experiment_id="c4_2b_yolo11n_seg_component_aware_sampling_x2_seed42",
        sampling_rule_version="component_aware_bottom_third_union_multi_x2_v1",
        canonical_manifest_sha256="a" * 64,
        unique_train_count=84,
        unique_positive_count=42,
        unique_good_negative_count=42,
        small_aware_count=14,
        multi_component_count=14,
        eligible_overlap_count=9,
        eligible_union_count=19,
        expanded_entry_count=103,
        expanded_positive_count=61,
        expanded_good_negative_count=42,
        expanded_good_negative_ratio=0.4077669902912621,
        small_fraction_rule="bottom_third",
        eligible_multiplicity=2,
        observed_train_small_cutoff=0.011273469387755102,
        eligible_sample_ids=eligible_ids,
        sample_multiplicity={f"train-{index:03d}": 2 if index < 19 else 1 for index in range(84)},
        train_list_sha256="b" * 64,
        train_list_path_base="canonical_dataset_root",
        ordering_policy=("canonical_sample_id_order_then_eligible_second_copy_in_sample_id_order"),
        validation_used_for_sampling=False,
        test_split_used=False,
    )


# ADD 2026-08-28: Runner wiring용 train/val manifest identity fixture를 만든다.
def _record(sample_id: str, split: str) -> DerivedManifestRecord:
    return DerivedManifestRecord(
        dataset_name="fixture",
        dataset_version="v1",
        derived_task="yolo_segmentation",
        source_manifest_sha256="a" * 64,
        source_split="train",
        source_manifest_split="train",
        source_image_path=f"source/{sample_id}.png",
        source_mask_path="",
        category="metal_nut",
        sample_id=sample_id,
        defect_type="good",
        target_class="",
        target_class_id="",
        derived_split=split,
        is_negative=True,
        image_width=32,
        image_height=32,
        image_path=f"images/{split}/{sample_id}.png",
        label_path=f"labels/{split}/{sample_id}.txt",
        image_sha256="c" * 64,
        mask_sha256="",
        polygon_count=0,
        component_count=0,
        hole_count=0,
        polygon_vertex_count=0,
        round_trip_iou="",
        pixel_precision="",
        pixel_recall="",
    )


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
        "epoch_metrics.jsonl",
        "visualization_manifest.json",
    )
    for name in evidence_names:
        (experiment_dir / name).write_text(json.dumps({"name": name}), encoding="utf-8")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.pt").write_bytes(b"candidate")
    (artifact / "metadata.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "experiment.yaml"
    config.write_text("experiment: fixture\n", encoding="utf-8")
    train_view_dir = experiment_dir / "train_view"
    train_view_dir.mkdir()
    train_list = train_view_dir / "train.txt"
    train_metadata = train_view_dir / "metadata.json"
    train_list.write_text("images/train/sample.png\n", encoding="utf-8")
    train_metadata.write_text("{}\n", encoding="utf-8")
    runtime_dir = experiment_dir / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / "train.runtime.txt").write_text(
        "/machine/dataset/images/train/sample.png\n",
        encoding="utf-8",
    )
    train_view = TrainViewArtifact(
        output_dir=train_view_dir,
        train_list_path=train_list,
        metadata_path=train_metadata,
        evidence=replace(
            _train_view_evidence(),
            train_list_sha256=sha256_file(train_list),
        ),
    )
    package = build_experiment_package(
        experiment_dir=experiment_dir,
        candidate_artifact_dir=artifact,
        experiment_config_path=config,
        package_path=tmp_path / "package.zip",
        train_view=train_view,
    )
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
    assert "model/model.pt" in names
    assert "model/metadata.json" in names
    assert "evidence/train_view.txt" in names
    assert "evidence/train_view_metadata.json" in names
    assert "SHA256SUMS.txt" in names
    assert not any(name.startswith("data/") for name in names)
    assert not any("runtime" in name for name in names)


# ADD 2026-08-28: C4-2B preparation이 strict records와 test-less runtime YAML만 쓰는지 검증한다.
def test_c4_2b_training_dataset_preparation_wires_strict_sampling_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_yolo_experiment_config(C4_2B_CONFIG_PATH)
    baseline = load_yolo_segmentation_config(experiment.baseline_config_path)
    repository_root = tmp_path / "repository"
    dataset_root = repository_root / "data/dataset"
    experiment_dir = (
        repository_root / "outputs/experiments/yolo_segmentation" / experiment.experiment_id
    )
    records = tuple(
        [*(_record(f"train-{index:03d}", "train") for index in range(84))]
        + [*(_record(f"val-{index:03d}", "val") for index in range(28))]
    )
    calls: dict[str, object] = {}

    def fake_strict_validator(
        validated_root: Path,
        contract: object,
    ) -> tuple[DerivedManifestRecord, ...]:
        calls["strict"] = (validated_root, contract)
        return records

    def fake_train_view_builder(**kwargs: object) -> TrainViewArtifact:
        train_records = kwargs["train_records"]
        assert isinstance(train_records, tuple)
        assert len(train_records) == 84
        assert all(record.derived_split == "train" for record in train_records)
        output_dir = experiment_dir / "train_view"
        output_dir.mkdir(parents=True)
        train_list = output_dir / "train.txt"
        metadata = output_dir / "metadata.json"
        train_list.write_text("images/train/sample.png\n" * 103, encoding="utf-8")
        metadata.write_text("{}\n", encoding="utf-8")
        calls["builder"] = kwargs
        return TrainViewArtifact(
            output_dir=output_dir,
            train_list_path=train_list,
            metadata_path=metadata,
            evidence=_train_view_evidence(),
        )

    def fake_runtime_adapter(**kwargs: object) -> RuntimeTrainViewArtifact:
        destination = kwargs["destination"]
        assert isinstance(destination, Path)
        destination.parent.mkdir(parents=True)
        destination.write_text(
            f"{(dataset_root / 'images/train/sample.png').resolve()}\n" * 103,
            encoding="utf-8",
        )
        calls["adapter"] = kwargs
        return RuntimeTrainViewArtifact(destination, "b" * 64, 103)

    monkeypatch.setattr(
        "pipelines.run_yolo_segmentation_experiment.validate_experiment_dataset",
        fake_strict_validator,
    )
    monkeypatch.setattr(
        "pipelines.run_yolo_segmentation_experiment.build_component_aware_train_view",
        fake_train_view_builder,
    )
    monkeypatch.setattr(
        "pipelines.run_yolo_segmentation_experiment.build_runtime_train_view_adapter",
        fake_runtime_adapter,
    )

    prepared = prepare_experiment_training_dataset(
        experiment_config=experiment,
        baseline_config=baseline,
        dataset_root=dataset_root,
        repository_root=repository_root,
        experiment_dir=experiment_dir,
    )

    assert calls.keys() == {"strict", "builder", "adapter"}
    assert len(prepared.validated_records) == 112
    assert prepared.train_view is not None
    assert prepared.train_view.evidence.expanded_entry_count == 103
    assert prepared.runtime_train_view is not None
    assert prepared.runtime_train_view.entry_count == 103
    assert prepared.runtime_dataset_yaml is not None
    runtime_yaml = yaml.safe_load(prepared.runtime_dataset_yaml.read_text(encoding="utf-8"))
    assert runtime_yaml == {
        "path": str(dataset_root.resolve()),
        "train": str((experiment_dir / "runtime/train.runtime.txt").resolve()),
        "val": "images/val",
        "names": {0: "bent", 1: "color", 2: "scratch"},
    }


# ADD 2026-08-28: C4-2A replay도 strict records와 canonical test-less YAML을 쓰는지 검증한다.
def test_c4_2a_training_dataset_preparation_uses_common_test_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_yolo_experiment_config(C4_2A_CONFIG_PATH)
    baseline = load_yolo_segmentation_config(experiment.baseline_config_path)
    repository_root = tmp_path / "repository"
    dataset_root = repository_root / "data/dataset"
    experiment_dir = (
        repository_root / "outputs/experiments/yolo_segmentation" / experiment.experiment_id
    )
    records = tuple(
        [*(_record(f"train-{index:03d}", "train") for index in range(84))]
        + [*(_record(f"val-{index:03d}", "val") for index in range(28))]
    )
    calls: list[tuple[Path, object]] = []

    def fake_strict_validator(
        validated_root: Path,
        contract: object,
    ) -> tuple[DerivedManifestRecord, ...]:
        calls.append((validated_root, contract))
        return records

    def reject_sampling(*args: object, **kwargs: object) -> None:
        raise AssertionError("C4-2A must not build a sampling train view.")

    monkeypatch.setattr(
        "pipelines.run_yolo_segmentation_experiment.validate_experiment_dataset",
        fake_strict_validator,
    )
    monkeypatch.setattr(
        "pipelines.run_yolo_segmentation_experiment.build_component_aware_train_view",
        reject_sampling,
    )

    prepared = prepare_experiment_training_dataset(
        experiment_config=experiment,
        baseline_config=baseline,
        dataset_root=dataset_root,
        repository_root=repository_root,
        experiment_dir=experiment_dir,
    )

    assert calls == [(dataset_root, baseline.dataset_contract)]
    assert len(prepared.validated_records) == 112
    assert prepared.train_view is None
    assert prepared.runtime_train_view is None
    assert prepared.runtime_dataset_yaml is not None
    runtime_yaml = yaml.safe_load(prepared.runtime_dataset_yaml.read_text(encoding="utf-8"))
    assert runtime_yaml == {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "bent", 1: "color", 2: "scratch"},
    }
    assert experiment.training_config(baseline).training.imgsz == 1024


# ADD 2026-08-28: Prepared YAML training이 fresh checkpoint와 fixed constants를 backend에 전달한다.
def test_prepared_c4_2b_training_keeps_fresh_model_and_fixed_constants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_yolo_experiment_config(C4_2B_CONFIG_PATH)
    baseline = load_yolo_segmentation_config(experiment.baseline_config_path)
    candidate = experiment.training_config(baseline)
    candidate = replace(
        candidate,
        output=YoloOutputConfig(
            artifact_root=tmp_path / "artifacts",
            training_runtime_root=tmp_path / "training",
            evaluation_root=tmp_path / "evaluation",
        ),
    )
    runtime_yaml = tmp_path / "prepared/dataset.runtime.yaml"
    runtime_yaml.parent.mkdir()
    runtime_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str((tmp_path / "dataset").resolve()),
                "train": str((tmp_path / "runtime/train.runtime.txt").resolve()),
                "val": "images/val",
                "names": {0: "bent", 1: "color", 2: "scratch"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    def reject_full_validator(*args: object, **kwargs: object) -> None:
        raise AssertionError("Prepared experiment training must not reopen sealed test content.")

    def fake_training_runner(
        runner_config: object,
        dataset_yaml: Path,
        runtime_root: Path,
        artifact_id: str,
        requested_device: str,
    ) -> BackendTrainingResult:
        assert runner_config == candidate
        assert candidate.model.weights == "yolo11n-seg.pt"
        assert candidate.training.imgsz == 640
        assert candidate.training.batch == 16
        assert candidate.training.epochs == 100
        assert candidate.training.workers == 2
        assert candidate.training.patience == 20
        assert candidate.training.seed == 42
        assert candidate.training.optimizer == "auto"
        assert candidate.training.deterministic is True
        assert candidate.training.amp is True
        assert dataset_yaml == runtime_yaml.resolve()
        assert "test" not in yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
        checkpoint = runtime_root / artifact_id / "weights/best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fresh-c4-2b-checkpoint")
        assert requested_device == "cuda"
        return BackendTrainingResult(
            best_checkpoint=checkpoint,
            best_epoch=1,
            actual_device="cuda:0",
            framework_version="8.4.128",
            source_checkpoint="weights/best.pt",
        )

    monkeypatch.setattr(
        "pipelines.train_yolo_segmentation.validate_training_dataset",
        reject_full_validator,
    )
    result = train_yolo_segmentation(
        config=candidate,
        dataset_root=tmp_path / "dataset",
        artifact_id=experiment.experiment_id,
        requested_device="cuda",
        training_runner=fake_training_runner,
        created_at="2026-08-28T00:00:00+00:00",
        prepared_dataset_yaml=runtime_yaml,
    )
    assert result.metadata.model_name == "yolo11n-seg.pt"


# ADD 2026-08-28: Baseline/candidate diagnostics가 공통 validated records를 받도록 고정한다.
def test_controlled_runner_passes_validated_records_to_both_diagnostics() -> None:
    source = inspect.getsource(run_yolo_segmentation_experiment)

    assert "validated_analysis_records = prepared_dataset.validated_records" in source
    assert source.count("validated_records=validated_analysis_records") == 2
    assert "prepared_dataset_yaml=prepared_dataset.runtime_dataset_yaml" in source
