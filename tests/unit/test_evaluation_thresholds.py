"""Unit tests for max-normal PatchCore threshold calibration."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ml.evaluation.predictions import PredictionBundle, RawPredictionRecord
from ml.evaluation.thresholds import calibrate_max_normal_validation, write_threshold_artifact
from ml.training.patchcore import PatchCoreArtifactMetadata
from ml.training.preprocessing import PatchCorePreprocessingConfig


# ADD 2026-08-19: Threshold test용 PatchCore artifact metadata를 생성한다.
def _artifact_metadata() -> PatchCoreArtifactMetadata:
    return PatchCoreArtifactMetadata(
        schema_version=1,
        model_name="patchcore",
        implementation="anomalib",
        backbone="resnet18",
        layers=("layer1",),
        num_neighbors=1,
        coreset_sampling_ratio=0.1,
        pretrained_used_during_training=True,
        preprocessing=PatchCorePreprocessingConfig(
            resize_size=(2, 2),
            center_crop_size=(2, 2),
            image_mean=(0.485, 0.456, 0.406),
            image_std=(0.229, 0.224, 0.225),
        ),
        random_seed=42,
        category="metal_nut",
        train_sample_count=2,
        manifest_sha256="a" * 64,
        anomalib_version="2.5.1",
        torch_version="2.13.0",
        torchvision_version="0.28.0",
        python_version="3.12.14",
        created_at="2026-08-19T00:00:00+00:00",
    )


# ADD 2026-08-19: Threshold test용 normal validation prediction bundle을 생성한다.
def _validation_bundle() -> PredictionBundle:
    records = tuple(
        RawPredictionRecord(
            sample_id=f"validation-{index}",
            category="metal_nut",
            defect_type="good",
            label=0,
            split="validation",
            raw_anomaly_score=score,
            anomaly_map_key=f"validation-{index}",
        )
        for index, score in enumerate((0.2, 0.7))
    )
    return PredictionBundle(
        records=records,
        scores=torch.tensor([0.2, 0.7], dtype=torch.float64),
        anomaly_maps=torch.tensor(
            [
                [[[0.1, 0.2], [0.3, 0.4]]],
                [[[0.5, 0.6], [0.8, 0.7]]],
            ]
        ),
    )


# ADD 2026-08-19: Image와 pixel threshold가 validation maxima인지 검증한다.
def test_max_normal_validation_uses_image_and_pixel_maxima() -> None:
    thresholds = calibrate_max_normal_validation(
        predictions=_validation_bundle(),
        artifact_metadata=_artifact_metadata(),
        artifact_metadata_sha256="b" * 64,
        model_sha256="c" * 64,
        validation_predictions_sha256="d" * 64,
        validation_anomaly_maps_sha256="e" * 64,
        created_at="2026-08-19T00:00:00+00:00",
    )

    assert thresholds.image_threshold == pytest.approx(0.7)
    assert thresholds.pixel_threshold == pytest.approx(0.8)
    assert thresholds.validation_sample_count == 2
    assert thresholds.validation_pixel_count == 8
    assert thresholds.comparison_operator == ">"


# ADD 2026-08-19: Anomaly validation label이 max-normal contract를 위반하는지 검증한다.
def test_max_normal_validation_rejects_anomaly_label() -> None:
    bundle = _validation_bundle()
    invalid_record = replace(bundle.records[0], label=1)
    invalid_bundle = replace(bundle, records=(invalid_record, bundle.records[1]))

    with pytest.raises(ValueError, match="label to be 0"):
        calibrate_max_normal_validation(
            predictions=invalid_bundle,
            artifact_metadata=_artifact_metadata(),
            artifact_metadata_sha256="b" * 64,
            model_sha256="c" * 64,
            validation_predictions_sha256="d" * 64,
            validation_anomaly_maps_sha256="e" * 64,
            created_at="2026-08-19T00:00:00+00:00",
        )


# ADD 2026-08-19: Threshold root와 nested artifact manifest provenance mismatch를 거부한다.
def test_threshold_artifact_rejects_manifest_provenance_mismatch() -> None:
    thresholds = calibrate_max_normal_validation(
        predictions=_validation_bundle(),
        artifact_metadata=_artifact_metadata(),
        artifact_metadata_sha256="b" * 64,
        model_sha256="c" * 64,
        validation_predictions_sha256="d" * 64,
        validation_anomaly_maps_sha256="e" * 64,
        created_at="2026-08-19T00:00:00+00:00",
    )

    with pytest.raises(ValueError, match="manifest SHA-256"):
        replace(thresholds, manifest_sha256="f" * 64).validate()


# ADD 2026-08-19: Existing threshold JSON file을 overwrite하지 않는지 검증한다.
def test_threshold_artifact_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    thresholds = calibrate_max_normal_validation(
        predictions=_validation_bundle(),
        artifact_metadata=_artifact_metadata(),
        artifact_metadata_sha256="b" * 64,
        model_sha256="c" * 64,
        validation_predictions_sha256="d" * 64,
        validation_anomaly_maps_sha256="e" * 64,
        created_at="2026-08-19T00:00:00+00:00",
    )
    output_path = tmp_path / "thresholds.json"
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_threshold_artifact(thresholds, output_path)
