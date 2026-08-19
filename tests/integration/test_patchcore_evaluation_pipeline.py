"""Integration tests for PatchCore calibration and fixed-threshold evaluation."""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from ml.datasets.manifest import ManifestRecord, write_manifest_csv
from ml.evaluation.predictions import RawPredictionRecord
from ml.training.patchcore import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    PatchCoreArtifactMetadata,
)
from ml.training.preprocessing import PatchCorePreprocessingConfig
from pipelines.calibrate_patchcore_thresholds import calibrate_patchcore_thresholds
from pipelines.evaluate_patchcore import evaluate_patchcore
from shared.hashing import sha256_file


# ADD 2026-08-19: Evaluation integration fixture의 RGB image를 생성한다.
def _write_image(path: Path, color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(color, color, color)).save(path)


# ADD 2026-08-19: Evaluation integration fixture의 binary anomaly mask를 생성한다.
def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = torch.zeros(8, 8, dtype=torch.uint8)
    mask[:, :4] = 255
    Image.fromarray(mask.numpy(), mode="L").save(path)


# ADD 2026-08-19: Train validation test record를 포함한 synthetic manifest를 구성한다.
def _build_dataset_and_manifest(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "mvtec_ad"
    paths = {
        "train": "metal_nut/train/good/000.png",
        "validation-1": "metal_nut/train/good/001.png",
        "validation-2": "metal_nut/train/good/002.png",
        "test-good": "metal_nut/test/good/000.png",
        "test-bent": "metal_nut/test/bent/000.png",
    }
    for index, path in enumerate(paths.values(), start=1):
        _write_image(dataset_root / path, color=index * 20)
    mask_path = "metal_nut/ground_truth/bent/000_mask.png"
    _write_mask(dataset_root / mask_path)

    records = [
        ManifestRecord(
            sample_id="train",
            category="metal_nut",
            source_split="train",
            split="train",
            defect_type="good",
            label=0,
            image_path=paths["train"],
            mask_path="",
            width=8,
            height=8,
        ),
        ManifestRecord(
            sample_id="validation-1",
            category="metal_nut",
            source_split="train",
            split="validation",
            defect_type="good",
            label=0,
            image_path=paths["validation-1"],
            mask_path="",
            width=8,
            height=8,
        ),
        ManifestRecord(
            sample_id="validation-2",
            category="metal_nut",
            source_split="train",
            split="validation",
            defect_type="good",
            label=0,
            image_path=paths["validation-2"],
            mask_path="",
            width=8,
            height=8,
        ),
        ManifestRecord(
            sample_id="test-good",
            category="metal_nut",
            source_split="test",
            split="test",
            defect_type="good",
            label=0,
            image_path=paths["test-good"],
            mask_path="",
            width=8,
            height=8,
        ),
        ManifestRecord(
            sample_id="test-bent",
            category="metal_nut",
            source_split="test",
            split="test",
            defect_type="bent",
            label=1,
            image_path=paths["test-bent"],
            mask_path=mask_path,
            width=8,
            height=8,
        ),
    ]
    manifest_path = tmp_path / "manifest.csv"
    write_manifest_csv(records, manifest_path)
    return dataset_root, manifest_path


# ADD 2026-08-19: Synthetic artifact metadata와 small tensor model file을 저장한다.
def _write_artifact(tmp_path: Path, manifest_path: Path) -> Path:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    metadata = PatchCoreArtifactMetadata(
        schema_version=1,
        model_name="patchcore",
        implementation="anomalib",
        backbone="resnet18",
        layers=("layer1",),
        num_neighbors=1,
        coreset_sampling_ratio=0.1,
        pretrained_used_during_training=True,
        preprocessing=PatchCorePreprocessingConfig(
            resize_size=(8, 8),
            center_crop_size=(8, 8),
            image_mean=(0.485, 0.456, 0.406),
            image_std=(0.229, 0.224, 0.225),
        ),
        random_seed=42,
        category="metal_nut",
        train_sample_count=1,
        manifest_sha256=sha256_file(manifest_path),
        anomalib_version="2.5.1",
        torch_version="2.13.0",
        torchvision_version="0.28.0",
        python_version="3.12.14",
        created_at="2026-08-19T00:00:00+00:00",
    )
    (artifact_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata.to_json_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save({"memory_bank": torch.ones(1, 1)}, artifact_dir / MODEL_FILENAME)
    return artifact_dir


# ADD 2026-08-19: Raw prediction JSONL과 tensor map integration fixture를 저장한다.
def _write_predictions(
    output_dir: Path,
    records: list[RawPredictionRecord],
    maps: dict[str, torch.Tensor],
) -> tuple[Path, Path]:
    output_dir.mkdir()
    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(record.to_json_dict()) + "\n" for record in records),
        encoding="utf-8",
    )
    maps_path = output_dir / "anomaly_maps.pt"
    torch.save(maps, maps_path)
    return predictions_path, maps_path


# ADD 2026-08-19: Calibration에서 fixed-threshold test metric 저장까지 전체 계약을 검증한다.
def test_calibration_then_evaluation_produces_inspectable_metrics(tmp_path: Path) -> None:
    # Synthetic manifest, artifact와 normal-only validation prediction을 준비한다.
    dataset_root, manifest_path = _build_dataset_and_manifest(tmp_path)
    artifact_dir = _write_artifact(tmp_path, manifest_path)
    validation_records = [
        RawPredictionRecord(
            sample_id="validation-1",
            category="metal_nut",
            defect_type="good",
            label=0,
            split="validation",
            raw_anomaly_score=0.2,
            anomaly_map_key="validation-1",
        ),
        RawPredictionRecord(
            sample_id="validation-2",
            category="metal_nut",
            defect_type="good",
            label=0,
            split="validation",
            raw_anomaly_score=0.4,
            anomaly_map_key="validation-2",
        ),
    ]
    validation_paths = _write_predictions(
        tmp_path / "validation-predictions",
        validation_records,
        {
            "validation-1": torch.full((1, 8, 8), 0.2),
            "validation-2": torch.full((1, 8, 8), 0.3),
        },
    )

    # Validation maxima를 threshold artifact로 저장한다.
    calibration = calibrate_patchcore_thresholds(
        validation_predictions_path=validation_paths[0],
        validation_anomaly_maps_path=validation_paths[1],
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        artifact_dir=artifact_dir,
        output_dir=tmp_path / "threshold-output",
    )
    assert calibration.thresholds.image_threshold == pytest.approx(0.4)
    assert calibration.thresholds.pixel_threshold == pytest.approx(0.3)

    anomaly_map = torch.full((1, 8, 8), 0.2)
    anomaly_map[:, :, :4] = 0.9
    test_paths = _write_predictions(
        tmp_path / "test-predictions",
        [
            RawPredictionRecord(
                sample_id="test-good",
                category="metal_nut",
                defect_type="good",
                label=0,
                split="test",
                raw_anomaly_score=0.3,
                anomaly_map_key="test-good",
            ),
            RawPredictionRecord(
                sample_id="test-bent",
                category="metal_nut",
                defect_type="bent",
                label=1,
                split="test",
                raw_anomaly_score=0.9,
                anomaly_map_key="test-bent",
            ),
        ],
        {
            "test-good": torch.full((1, 8, 8), 0.1),
            "test-bent": anomaly_map,
        },
    )

    # Stored threshold를 이동하지 않고 image/pixel/per-defect metric을 계산한다.
    evaluation = evaluate_patchcore(
        test_predictions_path=test_paths[0],
        test_anomaly_maps_path=test_paths[1],
        thresholds_path=calibration.thresholds_path,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        artifact_dir=artifact_dir,
        output_dir=tmp_path / "evaluation-output",
        batch_size=2,
    )
    metrics = json.loads(evaluation.metrics_path.read_text(encoding="utf-8"))
    assert metrics["threshold_artifact"]["strategy"] == "max_normal_validation"
    assert metrics["threshold_artifact"]["comparison_operator"] == ">"
    assert metrics["image_level"]["auroc"] == pytest.approx(1.0)
    assert metrics["image_level"]["f1"] == pytest.approx(1.0)
    assert metrics["pixel_level"]["auroc"] == pytest.approx(1.0)
    assert metrics["pixel_level"]["f1"] == pytest.approx(1.0)
    assert metrics["per_defect"]["bent"]["recall"] == pytest.approx(1.0)
    assert metrics["per_defect"]["good"]["false_positive_rate"] == pytest.approx(0.0)


# ADD 2026-08-19: Calibration pipeline이 test split prediction 혼입을 거부하는지 검증한다.
def test_calibration_rejects_test_prediction_leakage(tmp_path: Path) -> None:
    dataset_root, manifest_path = _build_dataset_and_manifest(tmp_path)
    artifact_dir = _write_artifact(tmp_path, manifest_path)
    paths = _write_predictions(
        tmp_path / "leaked-predictions",
        [
            RawPredictionRecord(
                sample_id="validation-1",
                category="metal_nut",
                defect_type="good",
                label=0,
                split="test",
                raw_anomaly_score=0.2,
                anomaly_map_key="validation-1",
            )
        ],
        {"validation-1": torch.zeros(1, 8, 8)},
    )

    with pytest.raises(ValueError, match="Prediction split mismatch"):
        calibrate_patchcore_thresholds(
            validation_predictions_path=paths[0],
            validation_anomaly_maps_path=paths[1],
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            artifact_dir=artifact_dir,
            output_dir=tmp_path / "threshold-output",
        )


# ADD 2026-08-19: Evaluation pipeline이 threshold model provenance mismatch를 거부하는지 검증한다.
def test_evaluation_rejects_model_provenance_mismatch(tmp_path: Path) -> None:
    dataset_root, manifest_path = _build_dataset_and_manifest(tmp_path)
    artifact_dir = _write_artifact(tmp_path, manifest_path)
    validation_paths = _write_predictions(
        tmp_path / "validation-predictions",
        [
            RawPredictionRecord(
                sample_id="validation-1",
                category="metal_nut",
                defect_type="good",
                label=0,
                split="validation",
                raw_anomaly_score=0.2,
                anomaly_map_key="validation-1",
            ),
            RawPredictionRecord(
                sample_id="validation-2",
                category="metal_nut",
                defect_type="good",
                label=0,
                split="validation",
                raw_anomaly_score=0.3,
                anomaly_map_key="validation-2",
            ),
        ],
        {
            "validation-1": torch.zeros(1, 8, 8),
            "validation-2": torch.zeros(1, 8, 8),
        },
    )
    calibration = calibrate_patchcore_thresholds(
        validation_predictions_path=validation_paths[0],
        validation_anomaly_maps_path=validation_paths[1],
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        artifact_dir=artifact_dir,
        output_dir=tmp_path / "threshold-output",
    )
    torch.save({"memory_bank": torch.zeros(2, 1)}, artifact_dir / MODEL_FILENAME)

    with pytest.raises(ValueError, match="model.pt SHA-256"):
        evaluate_patchcore(
            test_predictions_path=tmp_path / "missing-predictions.jsonl",
            test_anomaly_maps_path=tmp_path / "missing-maps.pt",
            thresholds_path=calibration.thresholds_path,
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            artifact_dir=artifact_dir,
            output_dir=tmp_path / "evaluation-output",
        )


# ADD 2026-08-19: Evaluation pipeline이 기존 output directory를 overwrite하지 않는지 검증한다.
def test_evaluation_rejects_existing_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation-output"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        evaluate_patchcore(
            test_predictions_path=tmp_path / "missing-predictions.jsonl",
            test_anomaly_maps_path=tmp_path / "missing-maps.pt",
            thresholds_path=tmp_path / "missing-thresholds.json",
            dataset_root=tmp_path / "missing-dataset",
            manifest_path=tmp_path / "missing-manifest.csv",
            artifact_dir=tmp_path / "missing-artifact",
            output_dir=output_dir,
        )
