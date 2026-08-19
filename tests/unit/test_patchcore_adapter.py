"""Unit tests for PatchCore embedding collection behavior."""

from pathlib import Path

import pytest
import torch
from torch import Tensor

import ml.training.patchcore as patchcore_module
from ml.training.config import (
    OutputConfig,
    PatchCoreBaselineConfig,
    PatchCoreModelConfig,
    TrainingConfig,
)
from ml.training.patchcore import PatchCoreAdapter
from ml.training.preprocessing import PatchCorePreprocessingConfig, PatchCorePreprocessor
from pipelines.train_patchcore import train_patchcore


class _FakePatchcoreModel:
    # ADD 2026-08-19: Adapter unit test용 PatchCore fake 상태를 초기화한다.
    def __init__(self, **kwargs: object) -> None:
        self.pre_trained = kwargs["pre_trained"]
        self.memory_bank = torch.empty(0)
        self.training = True
        self.sampling_ratio: float | None = None

    # ADD 2026-08-19: Fake model의 device 이동 interface를 제공한다.
    def to(self, device: torch.device) -> "_FakePatchcoreModel":
        return self

    # ADD 2026-08-19: Fake model을 training 상태로 전환한다.
    def train(self) -> "_FakePatchcoreModel":
        self.training = True
        return self

    # ADD 2026-08-19: Fake model을 inference 상태로 전환한다.
    def eval(self) -> "_FakePatchcoreModel":
        self.training = False
        return self

    # ADD 2026-08-19: Fake feature embedding을 생성한다.
    def __call__(self, images: Tensor) -> Tensor:
        return images.mean(dim=(2, 3))

    # ADD 2026-08-19: Fake coreset memory bank를 생성한다.
    def subsample_embedding(self, sampling_ratio: float) -> None:
        self.sampling_ratio = sampling_ratio
        self.memory_bank = torch.ones(2, 3)


# ADD 2026-08-19: 테스트용 PatchCore configuration을 생성한다.
def _config() -> PatchCoreBaselineConfig:
    return PatchCoreBaselineConfig(
        model=PatchCoreModelConfig(
            name="patchcore",
            implementation="anomalib",
            backbone="resnet18",
            layers=("layer1",),
            pretrained=False,
            coreset_sampling_ratio=0.25,
            num_neighbors=1,
        ),
        preprocessing=PatchCorePreprocessingConfig(
            resize_size=(32, 32),
            center_crop_size=(32, 32),
            image_mean=(0.485, 0.456, 0.406),
            image_std=(0.229, 0.224, 0.225),
        ),
        training=TrainingConfig(
            random_seed=42,
            device="cpu",
            batch_size=2,
            num_workers=0,
        ),
        output=OutputConfig(
            artifact_root=Path("artifacts/test"),
            prediction_root=Path("outputs/test"),
        ),
    )


# ADD 2026-08-19: adapter collects normal samples and builds memory bank 테스트 시나리오를 검증한다.
def test_adapter_collects_normal_samples_and_builds_memory_bank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(patchcore_module, "PatchcoreModel", _FakePatchcoreModel)
    config = _config()
    adapter = PatchCoreAdapter.for_training(config, torch.device("cpu"))
    loader = [
        {
            "image": torch.rand(2, 3, 32, 32),
            "label": torch.zeros(2, dtype=torch.int64),
        }
    ]

    sample_count = adapter.fit(loader, PatchCorePreprocessor(config.preprocessing))

    assert sample_count == 2
    assert adapter.model.memory_bank.shape == (2, 3)
    assert adapter.model.sampling_ratio == 0.25
    assert not adapter.model.training


# ADD 2026-08-19: adapter rejects anomalous training label 테스트 시나리오를 검증한다.
def test_adapter_rejects_anomalous_training_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(patchcore_module, "PatchcoreModel", _FakePatchcoreModel)
    config = _config()
    adapter = PatchCoreAdapter.for_training(config, torch.device("cpu"))
    loader = [
        {
            "image": torch.rand(2, 3, 32, 32),
            "label": torch.tensor([0, 1]),
        }
    ]

    with pytest.raises(ValueError, match="only normal labels"):
        adapter.fit(loader, PatchCorePreprocessor(config.preprocessing))


# ADD 2026-08-19: Feature extraction 전에 missing training manifest를 거부하는지 검증한다.
def test_train_pipeline_rejects_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Training manifest not found"):
        train_patchcore(
            config=_config(),
            dataset_root=tmp_path / "mvtec_ad",
            manifest_path=tmp_path / "missing.csv",
            category="metal_nut",
            artifact_dir=tmp_path / "artifact",
            requested_device="cpu",
        )
