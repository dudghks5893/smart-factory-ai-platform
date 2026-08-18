"""Integration test for MVTec AD preparation."""

import csv
import json
from pathlib import Path

from PIL import Image

from pipelines.prepare_mvtec_ad import PreparationConfig, prepare_mvtec_dataset


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10)).save(path)


def _build_valid_dataset(root: Path) -> Path:
    dataset_root = root / "mvtec_ad"

    for index in range(10):
        _write_png(dataset_root / f"metal_nut/train/good/{index:03d}.png")

    _write_png(dataset_root / "metal_nut/test/good/000.png")
    _write_png(dataset_root / "metal_nut/test/bent/000.png")
    _write_png(dataset_root / "metal_nut/ground_truth/bent/000_mask.png")

    return dataset_root


def test_prepare_pipeline_generates_manifest_and_summary(tmp_path: Path) -> None:
    dataset_root = _build_valid_dataset(tmp_path)
    manifest_path = tmp_path / "output/manifest.csv"
    summary_path = tmp_path / "output/summary.json"

    config = PreparationConfig(
        dataset_root=dataset_root,
        category="metal_nut",
        validation_ratio=0.2,
        random_seed=42,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )

    summary = prepare_mvtec_dataset(config)

    assert summary.train_count == 8
    assert summary.validation_count == 2
    assert summary.test_good_count == 1
    assert summary.test_anomaly_count == 1
    assert summary.manifest_count == 12

    with manifest_path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 12

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["random_seed"] == 42
    assert saved_summary["manifest_count"] == 12
