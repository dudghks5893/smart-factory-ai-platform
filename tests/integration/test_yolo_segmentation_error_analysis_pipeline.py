"""Portable orchestration test for YOLO validation error-analysis outputs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image

from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord, write_derived_manifest
from ml.training.yolo_segmentation import (
    YoloSegmentationBaselineConfig,
    load_yolo_segmentation_config,
)
from pipelines.analyze_yolo_segmentation_errors import analyze_yolo_segmentation_errors
from services.inference.yolo_segmentation_runtime import (
    YoloSegmentationAdapter,
    YoloSegmentationInstance,
    YoloSegmentationResult,
    YoloSegmentationRuntimeConfig,
)
from shared.hashing import sha256_file

CONFIG_PATH = Path("configs/model/yolo_segmentation_baseline.yaml")
CLASSES = {0: "bent", 1: "color", 2: "scratch"}


# ADD 2026-08-26: Full train/val/test schema의 tiny portable dataset을 생성한다.
def _build_dataset(root: Path) -> tuple[Path, str, dict[str, int]]:
    dataset_root = root / "dataset"
    records: list[DerivedManifestRecord] = []
    counts: dict[str, int] = {}
    for split_index, split in enumerate(("train", "val", "test")):
        counts[split] = 2
        counts[f"{split}_positive"] = 1
        counts[f"{split}_negative"] = 1
        for is_negative in (False, True):
            sample_id = f"{split}-{'good' if is_negative else 'bent'}"
            image_path = dataset_root / "images" / split / f"{sample_id}.png"
            label_path = dataset_root / "labels" / split / f"{sample_id}.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            value = split_index * 20 + (2 if is_negative else 1)
            Image.fromarray(np.full((10, 10, 3), value, dtype=np.uint8)).save(image_path)
            label_path.write_text(
                "" if is_negative else "0 0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6\n",
                encoding="utf-8",
            )
            records.append(
                DerivedManifestRecord(
                    dataset_name="synthetic",
                    dataset_version="v1",
                    derived_task="yolo_segmentation",
                    source_manifest_sha256="a" * 64,
                    source_split="test",
                    source_manifest_split="test",
                    source_image_path=f"source/{sample_id}.png",
                    source_mask_path="" if is_negative else f"source/{sample_id}_mask.png",
                    category="metal_nut",
                    sample_id=sample_id,
                    defect_type="good" if is_negative else "bent",
                    target_class="" if is_negative else "bent",
                    target_class_id="" if is_negative else "0",
                    derived_split=split,
                    is_negative=is_negative,
                    image_width=10,
                    image_height=10,
                    image_path=image_path.relative_to(dataset_root).as_posix(),
                    label_path=label_path.relative_to(dataset_root).as_posix(),
                    image_sha256=sha256_file(image_path),
                    mask_sha256="" if is_negative else "b" * 64,
                    polygon_count=0 if is_negative else 1,
                    component_count=0 if is_negative else 1,
                    hole_count=0,
                    polygon_vertex_count=0 if is_negative else 4,
                    round_trip_iou="" if is_negative else "1.0",
                    pixel_precision="" if is_negative else "1.0",
                    pixel_recall="" if is_negative else "1.0",
                )
            )
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_root / "manifest.csv"
    write_derived_manifest(records, manifest_path)
    manifest_sha = sha256_file(manifest_path)
    semantic_fingerprint = "c" * 64
    (dataset_root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": CLASSES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (dataset_root / "metadata.json").write_text(
        json.dumps(
            {
                "derived_manifest_sha256": manifest_sha,
                "semantic_fingerprint_sha256": semantic_fingerprint,
            }
        ),
        encoding="utf-8",
    )
    return dataset_root, semantic_fingerprint, counts


# ADD 2026-08-26: Synthetic lineage/count를 baseline config에 안전하게 주입한다.
def _config(
    dataset_root: Path, semantic_fingerprint: str, counts: dict[str, int]
) -> YoloSegmentationBaselineConfig:
    base = load_yolo_segmentation_config(CONFIG_PATH)
    return replace(
        base,
        dataset_contract=replace(
            base.dataset_contract,
            manifest_sha256=sha256_file(dataset_root / "manifest.csv"),
            semantic_fingerprint_sha256=semantic_fingerprint,
            sample_counts=counts,
        ),
    )


class FakeRuntime:
    """MPS/GPU-independent normalized result source for the val images only."""

    device = "cpu"

    # ADD 2026-08-26: Dataset lineage를 runtime loader 결과에 노출한다.
    def __init__(self, manifest_sha: str) -> None:
        self.provenance = SimpleNamespace(dataset_manifest_sha256=manifest_sha)

    # ADD 2026-08-26: Positive image에는 one exact mask, negative에는 empty result를 반환한다.
    def predict(
        self, image_rgb: NDArray[np.uint8], *, diagnostic_confidence: float
    ) -> YoloSegmentationResult:
        assert diagnostic_confidence == 0.10
        is_positive = int(image_rgb[0, 0, 0]) == 21
        instances: tuple[YoloSegmentationInstance, ...] = ()
        if is_positive:
            mask = np.zeros((10, 10), dtype=np.bool_)
            mask[2:7, 2:7] = True
            mask.setflags(write=False)
            instances = (YoloSegmentationInstance(0, "bent", 0.9, (2.0, 2.0, 7.0, 7.0), mask),)
        return YoloSegmentationResult(10, 10, "cpu", 1.0, instances)


# ADD 2026-08-26: Val-only analysis가 stable outputs를 만들고 test rows를 실행하지 않는지 검증한다.
def test_error_analysis_pipeline_writes_machine_readable_outputs(tmp_path: Path) -> None:
    dataset_root, semantic_fingerprint, counts = _build_dataset(tmp_path)
    config = _config(dataset_root, semantic_fingerprint, counts)
    artifact_dir = tmp_path / "artifact"
    model_dir = artifact_dir / "model"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "model.pt"
    metadata_path = model_dir / "metadata.json"
    model_path.write_bytes(b"immutable-model")
    metadata_path.write_text('{"fixture": true}\n', encoding="utf-8")
    model_sha = sha256_file(model_path)
    metadata_sha = sha256_file(metadata_path)
    observed_configs: list[YoloSegmentationRuntimeConfig] = []

    def fake_loader(runtime_config: YoloSegmentationRuntimeConfig) -> YoloSegmentationAdapter:
        observed_configs.append(runtime_config)
        return cast(
            YoloSegmentationAdapter,
            FakeRuntime(sha256_file(dataset_root / "manifest.csv")),
        )

    artifacts = analyze_yolo_segmentation_errors(
        config=config,
        dataset_root=dataset_root,
        artifact_dir=artifact_dir,
        output_dir=tmp_path / "analysis",
        requested_device="cpu",
        runtime_loader=fake_loader,
        created_at="2026-08-26T00:00:00+00:00",
    )
    assert observed_configs == [YoloSegmentationRuntimeConfig(artifact_dir, "cpu")]
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in artifacts.sample_analysis_path.read_text().splitlines()]
    assert summary["split"] == "val"
    assert summary["test_split_used"] is False
    assert summary["aggregate"]["sample_count"] == 2
    assert {row["sample_id"] for row in rows} == {"val-bent", "val-good"}
    assert artifacts.per_class_path.is_file()
    assert artifacts.confidence_sweep_path.is_file()
    assert artifacts.error_taxonomy_path.is_file()
    assert artifacts.hypotheses_path.is_file()
    assert len(artifacts.visualization_paths) == 1
    assert sha256_file(model_path) == model_sha
    assert sha256_file(metadata_path) == metadata_sha
