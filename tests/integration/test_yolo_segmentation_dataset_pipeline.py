"""Integration tests for the supervised-derived YOLO segmentation exporter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image

from ml.datasets.manifest import ManifestRecord, write_manifest_csv
from pipelines.export_yolo_segmentation_dataset import (
    export_yolo_segmentation_dataset,
    load_export_config,
    read_derived_manifest,
    validate_exported_dataset,
)
from shared.hashing import sha256_file


# ADD 2026-08-25: Integration fixture용 fixed-size RGB image를 생성한다.
def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((20, 20, 3), value, dtype=np.uint8)).save(path)


# ADD 2026-08-25: Single/multi-component synthetic binary mask를 생성한다.
def _write_mask(path: Path, *, disconnected: bool) -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:10, 3:12] = 255
    if disconnected:
        mask[14:18, 15:19] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


# ADD 2026-08-25: Real dataset을 대체하는 small Manifest/image/mask fixture를 구성한다.
def _build_source_fixture(root: Path) -> tuple[Path, Path]:
    source_root = root / "source"
    records: list[ManifestRecord] = []
    for class_index, defect_type in enumerate(("bent", "color", "scratch", "flip")):
        for index in range(3):
            image_relative = f"metal_nut/test/{defect_type}/{index:03d}.png"
            mask_relative = f"metal_nut/ground_truth/{defect_type}/{index:03d}_mask.png"
            _write_rgb(source_root / image_relative, 30 + class_index * 30 + index)
            _write_mask(
                source_root / mask_relative,
                disconnected=defect_type == "scratch" and index == 0,
            )
            records.append(
                ManifestRecord(
                    sample_id=f"metal_nut_test_{defect_type}_{index:03d}",
                    category="metal_nut",
                    source_split="test",
                    split="test",
                    defect_type=defect_type,
                    label=1,
                    image_path=image_relative,
                    mask_path=mask_relative,
                    width=20,
                    height=20,
                )
            )
    for index in range(12):
        source_split = "test" if index < 3 else "train"
        split = "test" if index < 3 else "train"
        image_relative = f"metal_nut/{source_split}/good/{index:03d}.png"
        _write_rgb(source_root / image_relative, 180 + index)
        records.append(
            ManifestRecord(
                sample_id=f"metal_nut_{source_split}_good_{index:03d}",
                category="metal_nut",
                source_split=source_split,
                split=split,
                defect_type="good",
                label=0,
                image_path=image_relative,
                mask_path="",
                width=20,
                height=20,
            )
        )
    manifest_path = root / "source_manifest.csv"
    write_manifest_csv(records, manifest_path)
    return source_root, manifest_path


# ADD 2026-08-25: Test별 isolated output을 사용하는 C2-1 config를 반환한다.
def _config(
    *,
    root: Path,
    source_root: Path,
    manifest_path: Path,
    suffix: str,
) -> dict[str, Any]:
    config = load_export_config(Path("configs/data/mvtec_ad_metal_nut_yolo_segmentation.yaml"))
    config["dataset"]["source_root"] = str(source_root)
    config["dataset"]["source_manifest"] = str(manifest_path)
    config["output"] = {
        "dataset_root": str(root / f"dataset-{suffix}"),
        "visualization_root": str(root / f"visuals-{suffix}"),
        "package_path": str(root / f"package-{suffix}.zip"),
    }
    return config


# ADD 2026-08-25: Export, validation, ZIP, lineage와 corruption failure를 통합 검증한다.
def test_export_is_training_ready_deterministic_and_validated(tmp_path: Path) -> None:
    source_root, manifest_path = _build_source_fixture(tmp_path)
    first_config = _config(
        root=tmp_path,
        source_root=source_root,
        manifest_path=manifest_path,
        suffix="first",
    )
    second_config = _config(
        root=tmp_path,
        source_root=source_root,
        manifest_path=manifest_path,
        suffix="second",
    )

    # 같은 source/config/time은 output 위치와 무관하게 같은 dataset bytes를 생성한다.
    first = export_yolo_segmentation_dataset(
        config=first_config,
        created_at_utc="2026-08-25T00:00:00+00:00",
    )
    second = export_yolo_segmentation_dataset(
        config=second_config,
        created_at_utc="2026-08-25T00:00:00+00:00",
    )
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.package_sha256 == second.package_sha256
    assert first.sample_count == 18

    records = read_derived_manifest(first.manifest_path)
    assert len({record.source_image_path for record in records}) == 18
    assert not any(record.defect_type == "flip" for record in records)
    assert sum(record.is_negative for record in records) == 9
    assert sum(not record.is_negative for record in records) == 9
    assert sum(record.polygon_count for record in records) == 10
    assert all(
        (first.dataset_root / record.label_path).read_bytes() == b""
        for record in records
        if record.is_negative
    )
    assert all(record.source_manifest_sha256 == sha256_file(manifest_path) for record in records)

    dataset_yaml = yaml.safe_load(first.dataset_yaml_path.read_text(encoding="utf-8"))
    assert dataset_yaml["path"] == "."
    assert dataset_yaml["names"] == {0: "bent", 1: "color", 2: "scratch"}
    assert not any(
        Path(value).is_absolute() for value in dataset_yaml.values() if isinstance(value, str)
    )
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    assert metadata["sample_counts"] == {
        "test:bent": 1,
        "test:color": 1,
        "test:negative": 3,
        "test:scratch": 1,
        "train:bent": 1,
        "train:color": 1,
        "train:negative": 3,
        "train:scratch": 1,
        "val:bent": 1,
        "val:color": 1,
        "val:negative": 3,
        "val:scratch": 1,
    }
    assert len(first.visualization_paths) == 4

    # Corrupt label이 coordinate validator와 source-lineage validation을 통과하지 못한다.
    positive = next(record for record in records if not record.is_negative)
    (first.dataset_root / positive.label_path).write_text(
        "0 0.0 0.0 1.1 0.0 0.5 0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        validate_exported_dataset(
            output_root=first.dataset_root,
            source_root=source_root,
            source_manifest_path=manifest_path,
            config=first_config,
        )
