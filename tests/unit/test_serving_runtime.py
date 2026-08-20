"""Unit tests for production PatchCore serving runtime behavior."""

from pathlib import Path
from typing import cast

import pytest
import torch

from ml.training.patchcore import PatchCoreAdapter, PatchCorePrediction
from ml.training.preprocessing import PatchCorePreprocessingConfig, PatchCorePreprocessor
from services.inference.runtime import (
    PatchCoreRuntimeConfig,
    PatchCoreServingRuntime,
    ServingProvenance,
    load_patchcore_runtime,
)


class _ScoreAdapter:
    """Adapter fake that returns one configured score without model files."""

    # ADD 2026-08-19: Runtime unit test에 사용할 image score를 보관한다.
    def __init__(self, score: float) -> None:
        self.score = score

    # ADD 2026-08-19: 기존 PatchCorePrediction contract로 configured score를 반환한다.
    def predict(
        self,
        images: torch.Tensor,
        preprocessor: PatchCorePreprocessor,
    ) -> PatchCorePrediction:
        return PatchCorePrediction(
            scores=torch.tensor([self.score]),
            anomaly_maps=torch.zeros(1, 1, 2, 2),
        )


# ADD 2026-08-19: Small serving runtime에 필요한 preprocessing contract를 생성한다.
def _preprocessor() -> PatchCorePreprocessor:
    return PatchCorePreprocessor(
        PatchCorePreprocessingConfig(
            resize_size=(2, 2),
            center_crop_size=(2, 2),
            image_mean=(0.485, 0.456, 0.406),
            image_std=(0.229, 0.224, 0.225),
        )
    )


# ADD 2026-08-19: Production runtime이 score > threshold strict contract를 적용하는지 검증한다.
@pytest.mark.parametrize(
    ("score", "expected"),
    [(40.1, True), (40.0, False), (39.9, False)],
)
def test_serving_runtime_applies_strict_threshold(score: float, expected: bool) -> None:
    runtime = PatchCoreServingRuntime(
        adapter=cast(PatchCoreAdapter, _ScoreAdapter(score)),
        preprocessor=_preprocessor(),
        model_name="patchcore",
        category="metal_nut",
        device="cpu",
        image_threshold=40.0,
        comparison_operator=">",
        provenance=ServingProvenance(
            manifest_sha256="a" * 64,
            artifact_metadata_sha256="b" * 64,
            model_sha256="c" * 64,
            threshold_artifact_sha256="d" * 64,
        ),
    )

    prediction = runtime.predict(torch.zeros(1, 3, 2, 2))

    assert prediction.is_anomaly is expected
    assert prediction.anomaly_score == pytest.approx(score)
    assert prediction.threshold == 40.0


# ADD 2026-08-19: Production loader가 missing artifact를 model 생성 전에 거부한다.
def test_runtime_loader_rejects_missing_artifact(tmp_path: Path) -> None:
    config = PatchCoreRuntimeConfig(
        artifact_dir=tmp_path / "missing-artifact",
        thresholds_path=tmp_path / "missing-thresholds.json",
        device="cpu",
    )

    with pytest.raises(FileNotFoundError, match="artifact directory not found"):
        load_patchcore_runtime(config)
