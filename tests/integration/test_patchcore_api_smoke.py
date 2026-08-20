"""Integration tests for the PatchCore FastAPI smoke runner."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from torch import Tensor

from pipelines.smoke_patchcore_api import smoke_patchcore_api
from services.api.tooling import validate_prediction_payload
from services.inference.runtime import InferenceResult, ModelRuntime, PatchCoreRuntimeConfig


class _BrightnessRuntime:
    """Fake runtime that separates black and white smoke images deterministically."""

    model_name = "patchcore"
    category = "metal_nut"
    device = "cpu"
    threshold = 0.5

    # ADD 2026-08-20: Smoke fake runtime의 inference 호출 횟수를 초기화한다.
    def __init__(self) -> None:
        self.predict_calls = 0

    # ADD 2026-08-20: Upload image 평균을 strict threshold score로 반환한다.
    def predict(self, image: Tensor) -> InferenceResult:
        self.predict_calls += 1
        score = float(image.mean().item())
        return InferenceResult(
            model_name=self.model_name,
            category=self.category,
            is_anomaly=score > self.threshold,
            anomaly_score=score,
            threshold=self.threshold,
            comparison_operator=">",
        )


# ADD 2026-08-20: Small RGB smoke image를 PNG로 저장한다.
def _write_png(path: Path, value: int) -> None:
    Image.new("RGB", (8, 8), color=(value, value, value)).save(path, format="PNG")


# ADD 2026-08-20: FastAPI lifespan과 HTTP route를 통한 normal/anomaly smoke 성공을 검증한다.
def test_smoke_runner_uses_one_runtime_and_validates_both_labels(tmp_path: Path) -> None:
    normal_path = tmp_path / "normal.png"
    anomaly_path = tmp_path / "anomaly.png"
    _write_png(normal_path, 0)
    _write_png(anomaly_path, 255)
    runtime = _BrightnessRuntime()
    load_calls: list[PatchCoreRuntimeConfig] = []

    # ADD 2026-08-20: Smoke lifespan의 runtime config와 load 횟수를 기록한다.
    def load(config: PatchCoreRuntimeConfig) -> ModelRuntime:
        load_calls.append(config)
        return runtime

    summary = smoke_patchcore_api(
        artifact_dir=tmp_path / "artifact",
        thresholds_path=tmp_path / "thresholds.json",
        normal_image_path=normal_path,
        anomaly_image_path=anomaly_path,
        requested_device="cpu",
        runtime_loader=load,
    )

    assert summary.normal_score == pytest.approx(0.0)
    assert summary.anomaly_score == pytest.approx(1.0)
    assert summary.threshold == pytest.approx(0.5)
    assert runtime.predict_calls == 2
    assert len(load_calls) == 1


# ADD 2026-08-20: Malformed HTTP prediction payload의 smoke schema 검증 실패를 확인한다.
def test_prediction_payload_rejects_malformed_response() -> None:
    with pytest.raises(ValueError, match="does not match"):
        validate_prediction_payload({"is_anomaly": False})


# ADD 2026-08-20: Smoke ground-truth와 반대인 normal/anomaly response를 거부한다.
@pytest.mark.parametrize("expected_is_anomaly", [False, True])
def test_prediction_payload_rejects_wrong_expected_label(expected_is_anomaly: bool) -> None:
    actual_is_anomaly = not expected_is_anomaly
    payload = {
        "model_name": "patchcore",
        "category": "metal_nut",
        "is_anomaly": actual_is_anomaly,
        "anomaly_score": 1.0 if actual_is_anomaly else 0.0,
        "threshold": 0.5,
        "comparison_operator": ">",
    }

    with pytest.raises(ValueError, match="Expected"):
        validate_prediction_payload(payload, expected_is_anomaly=expected_is_anomaly)
