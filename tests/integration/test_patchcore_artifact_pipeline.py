"""Integration coverage for PatchCore artifact round-trip and raw prediction output."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from PIL import Image

import ml.training.patchcore as patchcore_module
from ml.datasets.manifest import ManifestRecord, write_manifest_csv
from ml.training.config import (
    OutputConfig,
    PatchCoreBaselineConfig,
    PatchCoreModelConfig,
    TrainingConfig,
)
from ml.training.patchcore import MODEL_FILENAME, PatchCoreAdapter, validate_artifact_layout
from ml.training.preprocessing import PatchCorePreprocessingConfig, PatchCorePreprocessor
from ml.training.reproducibility import seed_training
from pipelines.predict_patchcore import predict_patchcore


# ADD 2026-08-19: 테스트용 PatchCore configuration을 생성한다.
def _config(tmp_path: Path) -> PatchCoreBaselineConfig:
    return PatchCoreBaselineConfig(
        model=PatchCoreModelConfig(
            name="patchcore",
            implementation="anomalib",
            backbone="resnet18",
            layers=("layer1",),
            pretrained=True,
            coreset_sampling_ratio=0.1,
            num_neighbors=3,
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
            batch_size=1,
            num_workers=0,
        ),
        output=OutputConfig(
            artifact_root=tmp_path / "artifacts",
            prediction_root=tmp_path / "outputs",
        ),
    )


# ADD 2026-08-19: 테스트에 필요한 dataset fixture를 구성한다.
def _build_manifest(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "mvtec_ad"
    train_path = dataset_root / "metal_nut/train/good/000.png"
    test_path = dataset_root / "metal_nut/test/good/000.png"
    train_path.parent.mkdir(parents=True)
    test_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(64, 128, 192)).save(train_path)
    Image.new("RGB", (32, 32), color=(96, 128, 160)).save(test_path)

    records = [
        ManifestRecord(
            sample_id="metal_nut_train_good_000",
            category="metal_nut",
            source_split="train",
            split="train",
            defect_type="good",
            label=0,
            image_path="metal_nut/train/good/000.png",
            mask_path="",
            width=32,
            height=32,
        ),
        ManifestRecord(
            sample_id="metal_nut_test_good_000",
            category="metal_nut",
            source_split="test",
            split="test",
            defect_type="good",
            label=0,
            image_path="metal_nut/test/good/000.png",
            mask_path="",
            width=32,
            height=32,
        ),
    ]
    manifest_path = tmp_path / "manifest.csv"
    write_manifest_csv(records, manifest_path)
    return dataset_root, manifest_path


# ADD 2026-08-19: 외부 다운로드 없는 artifact round-trip과 prediction 동일성을 검증한다.
def test_artifact_round_trip_preserves_predictions_without_pretrained_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 작은 CPU PatchCore fixture와 동일 manifest 기반 입력을 준비한다.
    config = _config(tmp_path)
    dataset_root, manifest_path = _build_manifest(tmp_path)
    preprocessor = PatchCorePreprocessor(config.preprocessing)
    seed_training(config.training.random_seed)

    # Pretrained download 없이 embedding을 수집하고 prediction A를 생성한다.
    adapter = PatchCoreAdapter(
        model_config=config.model,
        preprocessing_config=config.preprocessing,
        device=torch.device("cpu"),
        pre_trained=False,
    )
    train_images = torch.rand(2, 3, 32, 32)
    train_loader = [{"image": train_images, "label": torch.zeros(2, dtype=torch.int64)}]
    train_sample_count = adapter.fit(train_loader, preprocessor)

    prediction_input = torch.rand(1, 3, 32, 32)
    prediction_a = adapter.predict(prediction_input, preprocessor)
    artifact_dir = tmp_path / "artifact"
    metadata = adapter.save_artifact(
        artifact_dir=artifact_dir,
        category="metal_nut",
        train_sample_count=train_sample_count,
        manifest_path=manifest_path,
        random_seed=config.training.random_seed,
    )

    # Artifact가 tensor state_dict와 required metadata만 포함하는지 확인한다.
    saved_state = torch.load(
        artifact_dir / MODEL_FILENAME,
        map_location="cpu",
        weights_only=True,
    )
    assert isinstance(saved_state, dict)
    assert saved_state["memory_bank"].numel() > 0
    assert metadata.pretrained_used_during_training

    real_patchcore_model = patchcore_module.PatchcoreModel
    constructor_pretrained_values: list[bool] = []

    # ADD 2026-08-19: Artifact 복원 시 pretrained 인자 값을 기록한다.
    def recording_constructor(
        *,
        backbone: str,
        layers: Sequence[str],
        pre_trained: bool,
        num_neighbors: int,
    ) -> object:
        constructor_pretrained_values.append(pre_trained)
        return real_patchcore_model(
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
            num_neighbors=num_neighbors,
        )

    monkeypatch.setattr(patchcore_module, "PatchcoreModel", recording_constructor)
    # 복원 생성자가 pre_trained=False를 사용하도록 기록하면서 prediction B를 생성한다.
    loaded_adapter, loaded_metadata = PatchCoreAdapter.load_artifact(
        artifact_dir,
        torch.device("cpu"),
    )
    prediction_b = loaded_adapter.predict(prediction_input, preprocessor)

    assert constructor_pretrained_values == [False]
    assert loaded_metadata == metadata
    assert torch.allclose(prediction_a.scores, prediction_b.scores, rtol=1e-6, atol=1e-7)
    assert torch.allclose(
        prediction_a.anomaly_maps,
        prediction_b.anomaly_maps,
        rtol=1e-6,
        atol=1e-7,
    )

    # Raw prediction pipeline output을 저장하고 STEP 3에서 lossless하게 재로드한다.
    output_dir = tmp_path / "prediction-output"
    summary = predict_patchcore(
        config=config,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        artifact_dir=artifact_dir,
        output_dir=output_dir,
        split="test",
        requested_device="cpu",
    )

    assert summary.sample_count == 1
    record = json.loads(summary.predictions_path.read_text(encoding="utf-8"))
    assert record["sample_id"] == "metal_nut_test_good_000"
    assert record["split"] == "test"
    assert "raw_anomaly_score" in record
    anomaly_maps = torch.load(summary.anomaly_maps_path, weights_only=True)
    assert anomaly_maps[record["anomaly_map_key"]].shape == (1, 32, 32)
    assert constructor_pretrained_values == [False, False]


# ADD 2026-08-19: 필수 파일이 없는 artifact layout을 model 생성 전에 거부하는지 검증한다.
def test_artifact_layout_requires_model_and_metadata(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="artifact file not found"):
        validate_artifact_layout(artifact_dir)
