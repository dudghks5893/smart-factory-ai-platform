"""Static safety contract for the tracked Kaggle workbench notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

NOTEBOOK_PATH = Path("notebooks/vision/yolo_segmentation_experiment_workbench.ipynb")


# ADD 2026-08-27: Expected ordered headings, cleared outputs와 explicit execution locks를 검증한다.
def test_workbench_notebook_structure_and_safety() -> None:
    notebook = cast(dict[str, Any], json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8")))
    cells = cast(list[dict[str, Any]], notebook["cells"])
    headings = [
        line.strip()
        for cell in cells
        if cell["cell_type"] == "markdown"
        for line in cast(list[str], cell["source"])
        if line.lstrip().startswith("#")
    ]
    expected = [
        "# 1. Workbench Introduction",
        "## 2. Mode / Experiment Controls",
        "## 3. Environment Verification",
        "## 4. Repository / Git Verification",
        "## 5. Dataset / Manifest Verification",
        "## 6. Experiment Configuration Summary",
        "## 7. Dataset EDA",
        "## 8. Ground Truth Visualization",
        "## 9. Actual Training Augmentation Preview",
        "## 10. 640 vs 1024 Representation Preview",
        "## 11. Training Preflight",
        "## 12. Training Execution",
        "## 13. Epoch Progress / Timing",
        "## 14. GPU Telemetry Summary",
        "## 15. Validation Metrics",
        "## 16. Validation Failure Analysis",
        "## 17. Baseline vs Candidate Comparison",
        "## 18. Artifact / Evidence Review",
        "## 19. Final Test Review — LOCKED FOR C4-2A",
        "## 20. Export Summary",
    ]
    assert headings == expected
    assert all(
        cell.get("execution_count") is None and cell.get("outputs") == []
        for cell in cells
        if cell["cell_type"] == "code"
    )
    serialized = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert "RUN_OFFICIAL_TRAINING = False" in serialized
    assert "OFFICIAL_OVERRIDES = {}" in serialized
    assert "FINAL TEST REVIEW: LOCKED" in serialized
    assert "base64" not in serialized.lower()
    assert len(serialized.encode()) < 100_000


# ADD 2026-08-27: Notebook이 duplicate logic 대신 project APIs를 호출하는지 확인한다.
def test_workbench_notebook_is_thin_and_validation_safe() -> None:
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert "run_yolo_segmentation_experiment" in source
    assert "preview_actual_training_augmentations" in source
    assert "load_workbench_records" in source
    assert "YOLO(" not in source
    assert "derived_split == 'test'" not in source
    assert "split='test'" not in source
