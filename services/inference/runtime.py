"""PatchCore serving runtime isolated from HTTP transport concerns."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

from ml.evaluation.metrics import apply_strict_threshold
from ml.evaluation.thresholds import (
    read_threshold_artifact,
    validate_threshold_provenance,
)
from ml.training.device import SUPPORTED_DEVICES, resolve_device
from ml.training.patchcore import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    PatchCoreAdapter,
    read_artifact_metadata,
)
from ml.training.preprocessing import PatchCorePreprocessor
from shared.hashing import sha256_file


@dataclass(frozen=True)
class PatchCoreRuntimeConfig:
    """File and device settings needed to restore one serving runtime."""

    artifact_dir: Path
    thresholds_path: Path
    device: str

    # ADD 2026-08-19: Serving runtime path와 device configuration을 검증한다.
    def validate(self) -> None:
        """Validate values that do not require loading model artifacts."""
        if not str(self.artifact_dir):
            raise ValueError("artifact_dir must not be empty.")
        if not str(self.thresholds_path):
            raise ValueError("thresholds_path must not be empty.")
        if self.device not in SUPPORTED_DEVICES:
            raise ValueError(f"device must be one of {SUPPORTED_DEVICES}.")


@dataclass(frozen=True)
class InferenceResult:
    """Transport-independent image-level anomaly prediction."""

    model_name: str
    category: str
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    comparison_operator: str


class ModelRuntime(Protocol):
    """Minimal runtime contract consumed by the HTTP API and test fakes."""

    model_name: str
    category: str
    device: str

    # ADD 2026-08-19: 한 image batch의 image-level anomaly 결과를 반환한다.
    def predict(self, image: Tensor) -> InferenceResult:
        """Predict one already-decoded RGB image batch."""
        ...


@dataclass
class PatchCoreServingRuntime:
    """One process-local PatchCore model, preprocessing contract, and threshold."""

    adapter: PatchCoreAdapter
    preprocessor: PatchCorePreprocessor
    model_name: str
    category: str
    device: str
    image_threshold: float
    comparison_operator: str
    _inference_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ADD 2026-08-19: 공유 PatchCore instance에서 strict threshold image inference를 수행한다.
    def predict(self, image: Tensor) -> InferenceResult:
        """Serialize access to the shared model and apply its validation threshold."""
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError("Serving input must have shape (1, 3, height, width).")
        if not image.is_floating_point():
            raise TypeError("Serving input must be a floating-point tensor.")

        # 공유 model state와 accelerator queue를 보호하기 위해 instance별로 직렬화한다.
        with self._inference_lock, torch.inference_mode():
            prediction = self.adapter.predict(image, self.preprocessor)

        scores = prediction.scores.reshape(-1)
        if scores.numel() != 1:
            raise RuntimeError("PatchCore serving prediction must contain exactly one score.")
        is_anomaly = bool(apply_strict_threshold(scores, self.image_threshold)[0].item())
        return InferenceResult(
            model_name=self.model_name,
            category=self.category,
            is_anomaly=is_anomaly,
            anomaly_score=float(scores[0].item()),
            threshold=self.image_threshold,
            comparison_operator=self.comparison_operator,
        )


# ADD 2026-08-19: Artifact와 threshold provenance를 검증한 뒤 PatchCore를 한 번 복원한다.
def load_patchcore_runtime(config: PatchCoreRuntimeConfig) -> ModelRuntime:
    """Restore one ready PatchCore runtime without downloading pretrained weights."""
    config.validate()
    device = resolve_device(config.device)

    # Model load 전에 metadata, threshold와 실제 artifact hash를 모두 검증한다.
    artifact_metadata = read_artifact_metadata(config.artifact_dir)
    thresholds = read_threshold_artifact(config.thresholds_path)
    validate_threshold_provenance(
        thresholds,
        artifact_metadata=artifact_metadata,
        manifest_sha256=artifact_metadata.manifest_sha256,
        artifact_metadata_sha256=sha256_file(config.artifact_dir / METADATA_FILENAME),
        model_sha256=sha256_file(config.artifact_dir / MODEL_FILENAME),
    )

    # 검증된 metadata로 pretrained download 없이 model을 한 번만 복원한다.
    adapter, _ = PatchCoreAdapter.load_artifact(
        config.artifact_dir,
        device,
        metadata=artifact_metadata,
    )
    return PatchCoreServingRuntime(
        adapter=adapter,
        preprocessor=PatchCorePreprocessor(artifact_metadata.preprocessing),
        model_name=artifact_metadata.model_name,
        category=artifact_metadata.category,
        device=str(device),
        image_threshold=thresholds.image_threshold,
        comparison_operator=thresholds.comparison_operator,
    )
