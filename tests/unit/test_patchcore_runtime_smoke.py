"""Unit contracts for the pure PatchCore runtime smoke CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from pipelines.smoke_patchcore_runtime import (
    RuntimeSmokeSummary,
    _parse_args,
    format_runtime_smoke_summary,
    smoke_patchcore_runtime,
)
from services.inference.runtime import (
    InferenceResult,
    PatchCoreRuntimeConfig,
    ServingProvenance,
)


class _Runtime:
    """Small runtime fake preserving the production smoke interface."""

    # ADD 2026-08-24: Test score와 threshold를 strict runtime result로 구성한다.
    def __init__(self, *, score: float, threshold: float) -> None:
        self.model_name = "patchcore"
        self.category = "metal_nut"
        self.device = "cpu"
        self.provenance = ServingProvenance(
            manifest_sha256="a" * 64,
            artifact_metadata_sha256="b" * 64,
            model_sha256="c" * 64,
            threshold_artifact_sha256="d" * 64,
        )
        self._score = score
        self._threshold = threshold

    # ADD 2026-08-24: Decoded RGB batch를 확인하고 configured inference result를 반환한다.
    def predict(self, image: torch.Tensor) -> InferenceResult:
        assert image.shape == (1, 3, 8, 8)
        return InferenceResult(
            model_name=self.model_name,
            category=self.category,
            is_anomaly=self._score > self._threshold,
            anomaly_score=self._score,
            threshold=self._threshold,
            comparison_operator=">",
        )


# ADD 2026-08-24: Runtime smoke test용 valid RGB PNG를 생성한다.
def _image_path(tmp_path: Path) -> Path:
    path = tmp_path / "metal-nut.png"
    Image.new("RGB", (8, 8), color=(80, 120, 160)).save(path)
    return path


# ADD 2026-08-24: Image decode, runtime config 전달과 strict equal-score normal 판정을 검증한다.
def test_runtime_smoke_reuses_loader_and_strict_threshold(tmp_path: Path) -> None:
    load_calls: list[PatchCoreRuntimeConfig] = []

    # Injected loader로 실제 artifact 크기 없이 smoke orchestration을 검증한다.
    def loader(config: PatchCoreRuntimeConfig) -> _Runtime:
        load_calls.append(config)
        return _Runtime(score=41.2, threshold=41.2)

    summary = smoke_patchcore_runtime(
        artifact_dir=tmp_path / "artifact",
        thresholds_path=tmp_path / "thresholds.json",
        image_path=_image_path(tmp_path),
        requested_device="cpu",
        runtime_loader=loader,
    )

    assert len(load_calls) == 1
    assert load_calls[0].device == "cpu"
    assert summary.result == "NORMAL"
    assert summary.is_anomaly is False
    assert summary.comparison_operator == ">"


# ADD 2026-08-24: Missing image가 expensive runtime loader 호출 전에 거부되는지 검증한다.
def test_runtime_smoke_rejects_missing_image_before_restore(tmp_path: Path) -> None:
    loader_called = False

    def loader(_config: PatchCoreRuntimeConfig) -> _Runtime:
        nonlocal loader_called
        loader_called = True
        return _Runtime(score=50.0, threshold=40.0)

    with pytest.raises(FileNotFoundError, match="Image file not found"):
        smoke_patchcore_runtime(
            artifact_dir=tmp_path / "artifact",
            thresholds_path=tmp_path / "thresholds.json",
            image_path=tmp_path / "missing.png",
            requested_device="cpu",
            runtime_loader=loader,
        )

    assert loader_called is False


# ADD 2026-08-24: Artifact/threshold lineage loader rejection을 숨기지 않고 전달하는지 검증한다.
def test_runtime_smoke_propagates_lineage_mismatch(tmp_path: Path) -> None:
    def rejecting_loader(_config: PatchCoreRuntimeConfig) -> _Runtime:
        raise ValueError("Threshold and model.pt SHA-256 do not match.")

    with pytest.raises(ValueError, match="model.pt SHA-256"):
        smoke_patchcore_runtime(
            artifact_dir=tmp_path / "artifact",
            thresholds_path=tmp_path / "thresholds.json",
            image_path=_image_path(tmp_path),
            requested_device="cpu",
            runtime_loader=rejecting_loader,
        )


# ADD 2026-08-24: Result schema의 모든 identity/decision field가 CLI text에 포함되는지 검증한다.
def test_runtime_smoke_output_contains_complete_contract() -> None:
    output = format_runtime_smoke_summary(
        RuntimeSmokeSummary(
            model_name="patchcore",
            category="metal_nut",
            device="mps",
            image_path=Path("good/000.png"),
            anomaly_score=34.75,
            image_threshold=41.2,
            comparison_operator=">",
            is_anomaly=False,
            result="NORMAL",
        )
    )

    assert "PatchCore runtime smoke: PASS" in output
    assert "Model: patchcore" in output
    assert "Category: metal_nut" in output
    assert "Device: mps" in output
    assert "Image: good/000.png" in output
    assert "Score: 34.75" in output
    assert "Threshold: 41.2" in output
    assert "Comparison: score > threshold" in output
    assert "Is anomaly: false" in output
    assert "Result: NORMAL" in output


# ADD 2026-08-24: CLI parser가 unsupported explicit device를 거부하는지 검증한다.
def test_runtime_smoke_cli_rejects_unsupported_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke_patchcore_runtime",
            "--artifact-dir",
            "artifact",
            "--thresholds",
            "thresholds.json",
            "--image",
            "image.png",
            "--device",
            "metal",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        _parse_args()

    assert exc_info.value.code == 2
