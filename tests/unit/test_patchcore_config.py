"""Tests for the PatchCore baseline configuration contract."""

from pathlib import Path

import pytest

from ml.training.config import load_patchcore_config


# ADD 2026-08-19: load patchcore baseline config 테스트 시나리오를 검증한다.
def test_load_patchcore_baseline_config() -> None:
    config = load_patchcore_config(Path("configs/model/patchcore_baseline.yaml"))

    assert config.model.backbone == "wide_resnet50_2"
    assert config.model.layers == ("layer2", "layer3")
    assert config.model.coreset_sampling_ratio == 0.1
    assert config.preprocessing.resize_size == (256, 256)
    assert config.preprocessing.center_crop_size == (224, 224)
    assert config.training.random_seed == 42
    assert config.training.device == "auto"


# ADD 2026-08-19: config rejects crop larger than resize 테스트 시나리오를 검증한다.
def test_config_rejects_crop_larger_than_resize(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
model:
  name: patchcore
  implementation: anomalib
  backbone: resnet18
  layers: [layer1]
  pretrained: false
  coreset_sampling_ratio: 0.1
  num_neighbors: 1
preprocessing:
  resize_size: [32, 32]
  center_crop_size: [64, 64]
  image_mean: [0.485, 0.456, 0.406]
  image_std: [0.229, 0.224, 0.225]
training:
  random_seed: 42
  device: cpu
  batch_size: 1
  num_workers: 0
output:
  artifact_root: artifacts/test
  prediction_root: outputs/test
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="center_crop_size"):
        load_patchcore_config(config_path)
