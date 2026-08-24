"""Integration contract for known-defect feasibility analysis artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ml.datasets.manifest import ManifestRecord, write_manifest_csv
from pipelines.analyze_known_defect_feasibility import (
    METAL_NUT_DEFECTS,
    analyze_known_defect_feasibility,
)


# ADD 2026-08-25: Small four-defect manifest와 binary masks를 integration fixture로 생성한다.
def _write_analysis_fixture(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "mvtec_ad"
    records: list[ManifestRecord] = []
    for index, defect_type in enumerate(METAL_NUT_DEFECTS):
        image_path = dataset_root / f"metal_nut/test/{defect_type}/000.png"
        mask_path = dataset_root / f"metal_nut/ground_truth/{defect_type}/000_mask.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(80 + index * 20, 80, 80)).save(image_path)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1 + index : 3 + index, 2:5] = 255
        Image.fromarray(mask).save(mask_path)
        records.append(
            ManifestRecord(
                sample_id=f"metal_nut_test_{defect_type}_000",
                category="metal_nut",
                source_split="test",
                split="test",
                defect_type=defect_type,
                label=1,
                image_path=f"metal_nut/test/{defect_type}/000.png",
                mask_path=f"metal_nut/ground_truth/{defect_type}/000_mask.png",
                width=8,
                height=8,
            )
        )
    manifest_path = tmp_path / "manifest.csv"
    write_manifest_csv(records, manifest_path)
    return dataset_root, manifest_path


# ADD 2026-08-25: Synthetic manifest로 stable CSV/JSON/montage artifact round-trip을 검증한다.
def test_known_defect_feasibility_pipeline_artifacts(tmp_path: Path) -> None:
    dataset_root, manifest_path = _write_analysis_fixture(tmp_path)
    output_dir = tmp_path / "analysis"

    artifacts = analyze_known_defect_feasibility(
        dataset_name="mvtec_ad",
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        category="metal_nut",
        output_dir=output_dir,
        expected_defects=METAL_NUT_DEFECTS,
        examples_per_defect=1,
    )

    assert artifacts.sample_count == 4
    assert artifacts.component_count == 4
    assert artifacts.defect_counts == {defect_type: 1 for defect_type in METAL_NUT_DEFECTS}
    assert artifacts.sample_metrics_path.is_file()
    assert artifacts.component_metrics_path.is_file()
    assert artifacts.summary_path.is_file()
    assert all(path.is_file() for path in artifacts.visualization_paths)

    with artifacts.sample_metrics_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))

    assert [row["defect_type"] for row in rows] == list(METAL_NUT_DEFECTS)
    assert summary["source_protocol"] == "official_mvtec_ad_test_anomalies_analysis_only"
    assert summary["sample_count"] == 4
    assert summary["mask_count"] == 4
    assert summary["component_count"] == 4
    assert list(summary["defects"]) == list(METAL_NUT_DEFECTS)
