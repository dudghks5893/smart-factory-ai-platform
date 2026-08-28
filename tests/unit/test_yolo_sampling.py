"""Tests for deterministic C4-2B train sampling and sealed content access."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from typing import Any, cast

import pytest
import yaml
from PIL import Image

from ml.datasets.yolo_segmentation_manifest import (
    DERIVED_MANIFEST_FIELDS,
    DerivedManifestRecord,
    read_derived_manifest,
    write_derived_manifest,
)
from ml.experiments.yolo_sampling import (
    TrainSampleProfile,
    build_component_aware_train_view,
    build_runtime_train_view_adapter,
    expand_component_aware_train_entries,
    profile_train_samples,
    select_component_aware_eligibility,
)
from ml.experiments.yolo_segmentation import load_yolo_experiment_config
from ml.training.yolo_segmentation import (
    YoloDatasetContract,
    load_yolo_segmentation_config,
    validate_experiment_dataset,
)
from pipelines.run_yolo_segmentation_experiment import prepare_experiment_training_dataset
from pipelines.train_yolo_segmentation import write_runtime_dataset_yaml
from shared.hashing import sha256_file

BASELINE_CONFIG_PATH = Path("configs/model/yolo_segmentation_baseline.yaml")
EXPERIMENT_ID = "c4_2b_yolo11n_seg_component_aware_sampling_x2_seed42"


# ADD 2026-08-28: Portable synthetic Manifest record를 sampling/seal fixture용으로 만든다.
def _record(
    sample_id: str,
    *,
    split: str,
    is_negative: bool,
    component_count: int,
    image_sha256: str = "a" * 64,
) -> DerivedManifestRecord:
    class_id = "" if is_negative else "0"
    return DerivedManifestRecord(
        dataset_name="fixture",
        dataset_version="v1",
        derived_task="yolo_segmentation",
        source_manifest_sha256="b" * 64,
        source_split="train" if is_negative else "test",
        source_manifest_split="train" if is_negative else "test",
        source_image_path=f"source/{sample_id}.png",
        source_mask_path="" if is_negative else f"source/{sample_id}_mask.png",
        category="metal_nut",
        sample_id=sample_id,
        defect_type="good" if is_negative else "bent",
        target_class="" if is_negative else "bent",
        target_class_id=class_id,
        derived_split=split,
        is_negative=is_negative,
        image_width=32,
        image_height=32,
        image_path=f"images/{split}/{sample_id}.png",
        label_path=f"labels/{split}/{sample_id}.txt",
        image_sha256=image_sha256,
        mask_sha256="" if is_negative else "c" * 64,
        polygon_count=component_count,
        component_count=component_count,
        hole_count=0,
        polygon_vertex_count=component_count * 4,
        round_trip_iou="1.0",
        pixel_precision="1.0",
        pixel_recall="1.0",
    )


# ADD 2026-08-28: Full split-count keys를 작은 fixture count로 구성한다.
def _counts(
    *,
    train_positive: int,
    train_negative: int,
    val_positive: int = 0,
    val_negative: int = 0,
    test_positive: int = 0,
    test_negative: int = 0,
) -> dict[str, int]:
    return {
        "train": train_positive + train_negative,
        "val": val_positive + val_negative,
        "test": test_positive + test_negative,
        "train_positive": train_positive,
        "train_negative": train_negative,
        "val_positive": val_positive,
        "val_negative": val_negative,
        "test_positive": test_positive,
        "test_negative": test_negative,
    }


# ADD 2026-08-28: Baseline taxonomy를 유지한 synthetic dataset contract를 만든다.
def _contract(manifest_path: Path, counts: dict[str, int]) -> YoloDatasetContract:
    baseline = load_yolo_segmentation_config(BASELINE_CONFIG_PATH)
    return replace(
        baseline.dataset_contract,
        manifest_sha256=sha256_file(manifest_path),
        semantic_fingerprint_sha256="d" * 64,
        sample_counts=counts,
    )


# ADD 2026-08-28: Area와 component count를 직접 지정하는 pure sampling profile을 만든다.
def _profile(
    sample_id: str,
    *,
    area_ratio: float | None,
    component_count: int,
) -> TrainSampleProfile:
    return TrainSampleProfile(
        sample_id=sample_id,
        image_path=f"images/train/{sample_id}.png",
        is_negative=area_ratio is None,
        component_count=component_count,
        image_min_component_area_ratio=area_ratio,
    )


# ADD 2026-08-28: Bottom-third ceil과 equal-area sample_id tie-break를 고정한다.
def test_sampling_eligibility_is_deterministic_for_ceil_and_equal_area_ties() -> None:
    profiles = [
        _profile("z-equal", area_ratio=0.1, component_count=1),
        _profile("a-equal", area_ratio=0.1, component_count=1),
        _profile("lower", area_ratio=0.01, component_count=1),
        _profile("multi", area_ratio=0.2, component_count=2),
        _profile("good", area_ratio=None, component_count=0),
    ]

    first = select_component_aware_eligibility(profiles)
    second = select_component_aware_eligibility(list(reversed(profiles)))

    assert first == second
    assert first.small_aware_sample_ids == ("a-equal", "lower")
    assert first.multi_component_sample_ids == ("multi",)
    assert first.eligible_sample_ids == ("a-equal", "lower", "multi")
    assert first.observed_train_small_cutoff == 0.1


# ADD 2026-08-28: Current policy snapshot의 union, x2 multiplicity와 exposure arithmetic을 검증한다.
def test_sampling_policy_calculates_expected_current_exposure_snapshot() -> None:
    positives = [
        _profile(
            f"positive-{index:02d}",
            area_ratio=(index + 1) / 1000,
            component_count=2 if index in {*range(9), *range(20, 25)} else 1,
        )
        for index in range(42)
    ]
    negatives = [
        _profile(f"negative-{index:02d}", area_ratio=None, component_count=0) for index in range(42)
    ]
    profiles = [*positives, *negatives]

    eligibility = select_component_aware_eligibility(profiles)
    entries, multiplicity = expand_component_aware_train_entries(profiles, eligibility)

    assert len(eligibility.small_aware_sample_ids) == 14
    assert len(eligibility.multi_component_sample_ids) == 14
    assert (
        len(
            set(eligibility.small_aware_sample_ids).intersection(
                eligibility.multi_component_sample_ids
            )
        )
        == 9
    )
    assert len(eligibility.eligible_sample_ids) == 19
    assert len(entries) == 103
    assert sum(value == 2 for value in multiplicity.values()) == 19
    assert all(
        multiplicity[profile.sample_id]
        == (2 if profile.sample_id in eligibility.eligible_sample_ids else 1)
        for profile in profiles
    )
    assert 42 / len(entries) == 0.4077669902912621


# ADD 2026-08-28: Small train package와 deterministic label geometry를 disk에 만든다.
def _write_sampling_package(
    repository_root: Path,
) -> tuple[Path, YoloDatasetContract, list[DerivedManifestRecord]]:
    dataset_root = repository_root / "data" / "sampling-fixture"
    records = [
        _record("a-small", split="train", is_negative=False, component_count=1),
        _record("b-multi", split="train", is_negative=False, component_count=2),
        _record("c-large", split="train", is_negative=False, component_count=1),
        _record("n-good", split="train", is_negative=True, component_count=0),
    ]
    labels = {
        "a-small": "0 0.05 0.05 0.15 0.05 0.15 0.15 0.05 0.15\n",
        "b-multi": (
            "0 0.20 0.20 0.40 0.20 0.40 0.40 0.20 0.40\n0 0.60 0.60 0.75 0.60 0.75 0.75 0.60 0.75\n"
        ),
        "c-large": "0 0.10 0.10 0.80 0.10 0.80 0.80 0.10 0.80\n",
        "n-good": "",
    }
    for record in records:
        image_path = dataset_root / record.image_path
        label_path = dataset_root / record.label_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"sampling-profile-does-not-decode-images")
        label_path.write_text(labels[record.sample_id], encoding="utf-8")
    manifest_path = dataset_root / "manifest.csv"
    write_derived_manifest(records, manifest_path)
    return (
        dataset_root,
        _contract(manifest_path, _counts(train_positive=3, train_negative=1)),
        records,
    )


# ADD 2026-08-28: Same package가 byte-identical portable list와 metadata를 만드는지 검증한다.
def test_train_view_is_portable_deterministic_and_preserves_canonical_order(tmp_path: Path) -> None:
    roots = [tmp_path / "repository-a", tmp_path / "repository-b"]
    artifacts = []
    for repository_root in roots:
        dataset_root, contract, records = _write_sampling_package(repository_root)
        artifacts.append(
            build_component_aware_train_view(
                repository_root=repository_root,
                dataset_root=dataset_root,
                train_records=list(reversed(records)),
                contract=contract,
                experiment_id=EXPERIMENT_ID,
            )
        )

    first, second = artifacts
    assert first.train_list_path.read_bytes() == second.train_list_path.read_bytes()
    assert first.metadata_path.read_bytes() == second.metadata_path.read_bytes()
    assert first.evidence.train_list_sha256 == second.evidence.train_list_sha256
    lines = first.train_list_path.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "images/train/a-small.png",
        "images/train/b-multi.png",
        "images/train/c-large.png",
        "images/train/n-good.png",
        "images/train/a-small.png",
        "images/train/b-multi.png",
    ]
    assert all(not PurePosixPath(line).is_absolute() for line in lines)
    assert first.output_dir.relative_to(roots[0]) == (
        Path("outputs/experiments/yolo_segmentation") / EXPERIMENT_ID / "train_view"
    )
    evidence = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    assert evidence["validation_used_for_sampling"] is False
    assert evidence["test_split_used"] is False
    assert evidence["train_list_path_base"] == "canonical_dataset_root"
    assert evidence["sample_multiplicity"] == {
        "a-small": 2,
        "b-multi": 2,
        "c-large": 1,
        "n-good": 1,
    }


# ADD 2026-08-28: Portable entries의 runtime order, multiplicity, containment를 검증한다.
def test_runtime_train_view_adapter_preserves_entries_and_dataset_containment(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    dataset_root, contract, records = _write_sampling_package(repository_root)
    portable = build_component_aware_train_view(
        repository_root=repository_root,
        dataset_root=dataset_root,
        train_records=records,
        contract=contract,
        experiment_id=EXPERIMENT_ID,
    )
    runtime = build_runtime_train_view_adapter(
        repository_root=repository_root,
        dataset_root=dataset_root,
        portable_train_view=portable,
        destination=repository_root / "outputs/runtime/train.runtime.txt",
    )

    portable_lines = portable.train_list_path.read_text(encoding="utf-8").splitlines()
    runtime_lines = runtime.train_list_path.read_text(encoding="utf-8").splitlines()
    assert len(runtime_lines) == len(portable_lines) == 6
    assert runtime.entry_count == portable.evidence.expanded_entry_count
    assert runtime.source_train_list_sha256 == portable.evidence.train_list_sha256
    assert runtime_lines == [str((dataset_root / line).resolve()) for line in portable_lines]
    assert all(Path(line).is_relative_to(dataset_root.resolve()) for line in runtime_lines)

    with pytest.raises(ValueError, match="ignored outputs namespace"):
        build_runtime_train_view_adapter(
            repository_root=repository_root,
            dataset_root=dataset_root,
            portable_train_view=portable,
            destination=repository_root / "artifacts/train.runtime.txt",
        )


# ADD 2026-08-28: Val/test record와 non-train path가 sampling boundary를 통과하지 못하게 한다.
def test_sampling_rejects_non_train_records_and_paths(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    dataset_root, _, records = _write_sampling_package(repository_root)

    with pytest.raises(ValueError, match="TRAIN records only"):
        profile_train_samples(
            [replace(records[0], derived_split="val")],
            dataset_root=dataset_root,
            valid_class_ids={0, 1, 2},
        )
    with pytest.raises(ValueError, match="portable train content"):
        profile_train_samples(
            [replace(records[0], image_path="images/test/sealed.png")],
            dataset_root=dataset_root,
            valid_class_ids={0, 1, 2},
        )


# ADD 2026-08-28: Sealed row는 split 외 malformed value도 materialize하지 않는지 검증한다.
def test_manifest_filter_skips_sealed_row_before_semantic_materialization(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    train = asdict(_record("train", split="train", is_negative=True, component_count=0))
    sealed = asdict(_record("sealed", split="test", is_negative=False, component_count=1))
    sealed["image_width"] = "NOT_MATERIALIZED"
    sealed["is_negative"] = "NOT_MATERIALIZED"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DERIVED_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows([train, sealed])

    records = read_derived_manifest(manifest_path, allowed_splits={"train"})

    assert [record.sample_id for record in records] == ["train"]


# ADD 2026-08-28: C4 validator가 test image/label content 없이 train/val package를 검증한다.
def test_experiment_validation_does_not_open_sealed_test_content(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    train = _record("train-positive", split="train", is_negative=False, component_count=1)
    val = _record("val-good", split="val", is_negative=True, component_count=0)
    sealed = _record("sealed-test", split="test", is_negative=False, component_count=1)
    records = [train, val, sealed]
    for record, label in (
        (train, "0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n"),
        (val, ""),
    ):
        image_path = dataset_root / record.image_path
        label_path = dataset_root / record.label_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color=(0, 0, 0)).save(image_path)
        label_path.write_text(label, encoding="utf-8")
    train = replace(train, image_sha256=sha256_file(dataset_root / train.image_path))
    val = replace(val, image_sha256=sha256_file(dataset_root / val.image_path))
    records = [train, val, sealed]
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "bent", 1: "color", 2: "scratch"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest_path = dataset_root / "manifest.csv"
    write_derived_manifest(records, manifest_path)
    contract = _contract(
        manifest_path,
        _counts(
            train_positive=1,
            train_negative=0,
            val_positive=0,
            val_negative=1,
            test_positive=1,
            test_negative=0,
        ),
    )
    (dataset_root / "metadata.json").write_text(
        json.dumps(
            {
                "derived_manifest_sha256": contract.manifest_sha256,
                "semantic_fingerprint_sha256": contract.semantic_fingerprint_sha256,
            }
        ),
        encoding="utf-8",
    )

    validated = validate_experiment_dataset(dataset_root, contract)

    assert [record.derived_split for record in validated] == ["train", "val"]
    assert not (dataset_root / sealed.image_path).exists()
    assert not (dataset_root / sealed.label_path).exists()

    # Future non-sampling controlled replay도 missing sealed test content로 preflight에 성공한다.
    experiment = load_yolo_experiment_config(
        Path("configs/experiments/yolo_segmentation/c4_2a_yolo11n_seg_imgsz1024_seed42.yaml")
    )
    baseline = replace(
        load_yolo_segmentation_config(BASELINE_CONFIG_PATH),
        dataset_contract=contract,
    )
    prepared = prepare_experiment_training_dataset(
        experiment_config=experiment,
        baseline_config=baseline,
        dataset_root=dataset_root,
        repository_root=tmp_path,
        experiment_dir=tmp_path / "outputs/experiments" / experiment.experiment_id,
    )
    assert [record.derived_split for record in prepared.validated_records] == ["train", "val"]
    assert prepared.runtime_dataset_yaml is not None
    assert "test" not in yaml.safe_load(prepared.runtime_dataset_yaml.read_text(encoding="utf-8"))


# ADD 2026-08-28: Experiment runtime YAML이 test key를 선택적으로 생략하는지 검증한다.
def test_runtime_dataset_yaml_can_omit_sealed_test_path(tmp_path: Path) -> None:
    runtime_train_list = tmp_path / "runtime" / "train.runtime.txt"
    runtime_train_list.parent.mkdir(parents=True)
    runtime_train_list.write_text("/tmp/fixture.png\n", encoding="utf-8")
    path = write_runtime_dataset_yaml(
        dataset_root=tmp_path / "dataset",
        destination=tmp_path / "runtime" / "dataset.yaml",
        classes={0: "bent", 1: "color", 2: "scratch"},
        include_test=False,
        train_source=runtime_train_list,
    )

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["train"] == str(runtime_train_list.resolve())
    assert payload["val"] == "images/val"
    assert "test" not in payload

    default_path = write_runtime_dataset_yaml(
        dataset_root=tmp_path / "dataset",
        destination=tmp_path / "runtime" / "dataset.default.yaml",
        classes={0: "bent", 1: "color", 2: "scratch"},
    )
    default_payload = yaml.safe_load(default_path.read_text(encoding="utf-8"))
    assert default_payload["train"] == "images/train"
    assert default_payload["val"] == "images/val"
    assert default_payload["test"] == "images/test"


# ADD 2026-08-28: Pinned Ultralytics의 duplicate path와 cache multiplicity를 검증한다.
def test_ultralytics_dataset_preserves_duplicate_path_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "ultralytics-config"))
    from ultralytics import __version__ as ultralytics_version
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.utils import DEFAULT_CFG

    assert ultralytics_version == "8.4.128"
    image_path = tmp_path / "dataset" / "images" / "train" / "sample.png"
    label_path = tmp_path / "dataset" / "labels" / "train" / "sample.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(image_path)
    label_path.write_text(
        "0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n",
        encoding="utf-8",
    )
    train_list = tmp_path / "train.txt"
    train_list.write_text(f"{image_path.resolve()}\n{image_path.resolve()}\n", encoding="utf-8")
    config = cast(
        Any,
        get_cfg(
            DEFAULT_CFG,
            overrides={"task": "segment", "mode": "train", "imgsz": 32, "batch": 2},
        ),
    )
    data = {"names": {0: "defect"}, "nc": 1, "channels": 3}

    duplicated = cast(
        Any,
        build_yolo_dataset(
            config,
            str(train_list),
            2,
            data,
            mode="train",
            rect=False,
            stride=32,
        ),
    )
    assert len(duplicated) == 2
    assert duplicated.im_files == [str(image_path.resolve()), str(image_path.resolve())]
    assert len(duplicated.labels) == 2

    # 기존 label cache가 있어도 list multiplicity 변경은 hash miss 후 새 count로 반영된다.
    train_list.write_text(f"{image_path.resolve()}\n", encoding="utf-8")
    canonical = cast(
        Any,
        build_yolo_dataset(
            config,
            str(train_list),
            2,
            data,
            mode="train",
            rect=False,
            stride=32,
        ),
    )
    assert len(canonical) == 1
    assert canonical.im_files == [str(image_path.resolve())]
