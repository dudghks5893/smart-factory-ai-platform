"""Portable safety contracts for the dedicated C4-4 final-test notebook."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/vision/yolo_final_test_evaluation.ipynb")


# ADD 2026-09-01: Dedicated notebook의 cleared state와 explicit one-time unlock을 검증한다.
def test_yolo_final_test_notebook_is_small_cleared_and_fail_closed() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert len(cells) == 10
    assert len(code_cells) == 4
    assert all(
        cell.get("execution_count") is None and cell.get("outputs") == [] for cell in code_cells
    )
    assert "RUN_FINAL_TEST = False" in source
    assert "C4_4_EXPECTED_COMMIT" in source
    assert "actual_commit != EXPECTED_C4_4_COMMIT" in source
    assert '"--confirm-final-test"' in source
    assert "if RUN_FINAL_TEST is not True" in source
    command_tokens = ('"uv"', '"run"', '"--locked"', '"python"', '"-m"')
    command_positions = [
        source.index(token, source.index("preflight_command")) for token in command_tokens
    ]
    assert command_positions == sorted(command_positions)
    assert '["uv", "sync", "--locked"]' in source
    assert "shell=True" not in source
    assert "read_derived_manifest" not in source
    assert "test image" not in source.lower()
    assert "base64" not in json.dumps(notebook).lower()


# ADD 2026-09-01: Notebook preflight와 execution이 동일 repository CLI를 사용함을 검증한다.
def test_yolo_final_test_notebook_separates_preflight_from_unlock() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_sources = [
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    preflight_cell = next(source for source in code_sources if "preflight_command =" in source)
    execution_cell = next(source for source in code_sources if "execution_command =" in source)
    assert "--confirm-final-test" not in preflight_cell
    assert "pipelines.evaluate_yolo_final_test" in preflight_cell
    assert "--confirm-final-test" in execution_cell
    assert "RUN_FINAL_TEST is not True" in execution_cell
