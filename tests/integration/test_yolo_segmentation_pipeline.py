"""Integration tests for YOLO training, artifact, and test-evaluation orchestration."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from ml.datasets.yolo_segmentation_manifest import (
    DerivedManifestRecord,
    read_derived_manifest,
    write_derived_manifest,
)
from ml.evaluation.yolo_segmentation import PredictionObservation
from ml.training.yolo_segmentation import (
    YoloSegmentationBaselineConfig,
    load_yolo_segmentation_config,
    validate_training_dataset,
    validate_yolo_artifact,
)
from pipelines.evaluate_yolo_segmentation import (
    BackendEvaluationResult,
    evaluate_yolo_segmentation,
)
from pipelines.train_yolo_segmentation import (
    BackendTrainingResult,
    train_yolo_segmentation,
)
from shared.hashing import sha256_file

CONFIG_PATH = Path("configs/model/yolo_segmentation_baseline.yaml")
CLASSES = {0: "bent", 1: "color", 2: "scratch"}


class FakeMetric:
    """Documented metric method surface returned by a fake Ultralytics validation."""

    # ADD 2026-08-25: Integration output용 deterministic overall metric을 반환한다.
    def mean_results(self) -> tuple[float, float, float, float]:
        return 0.8, 0.7, 0.75, 0.55

    # ADD 2026-08-25: Integration output용 deterministic class metric을 반환한다.
    def class_result(self, class_id: int) -> tuple[float, float, float, float]:
        offset = class_id * 0.1
        return 0.8 - offset, 0.7 - offset, 0.75 - offset, 0.55 - offset


class FakeMetrics:
    """Box and mask metric components returned by the injected evaluator."""

    # ADD 2026-08-25: Box와 mask 모두 documented metric surface를 제공한다.
    def __init__(self) -> None:
        self.box = FakeMetric()
        self.seg = FakeMetric()


# ADD 2026-08-25: Minimal image/label/manifest를 가진 portable derived package를 생성한다.
def _build_dataset(root: Path) -> tuple[Path, str, dict[str, int]]:
    dataset_root = root / "dataset"
    records: list[DerivedManifestRecord] = []
    counts = {
        "train": 0,
        "val": 0,
        "test": 0,
        "train_positive": 0,
        "train_negative": 0,
        "val_positive": 0,
        "val_negative": 0,
        "test_positive": 0,
        "test_negative": 0,
    }
    value = 10
    for split_name in ("train", "val", "test"):
        for class_id, defect_type in CLASSES.items():
            sample_id = f"{split_name}-{defect_type}"
            image_path = dataset_root / "images" / split_name / f"{sample_id}.png"
            label_path = dataset_root / "labels" / split_name / f"{sample_id}.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(image_path)
            label_path.write_text(
                f"{class_id} 0.2 0.2 0.8 0.2 0.5 0.8\n",
                encoding="utf-8",
            )
            records.append(
                _record(
                    sample_id=sample_id,
                    split_name=split_name,
                    defect_type=defect_type,
                    class_id=str(class_id),
                    image_path=image_path.relative_to(dataset_root).as_posix(),
                    label_path=label_path.relative_to(dataset_root).as_posix(),
                    image_sha256=sha256_file(image_path),
                    is_negative=False,
                )
            )
            counts[split_name] += 1
            counts[f"{split_name}_positive"] += 1
            value += 1

        sample_id = f"{split_name}-good"
        image_path = dataset_root / "images" / split_name / f"{sample_id}.png"
        label_path = dataset_root / "labels" / split_name / f"{sample_id}.txt"
        Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(image_path)
        label_path.write_text("", encoding="utf-8")
        records.append(
            _record(
                sample_id=sample_id,
                split_name=split_name,
                defect_type="good",
                class_id="",
                image_path=image_path.relative_to(dataset_root).as_posix(),
                label_path=label_path.relative_to(dataset_root).as_posix(),
                image_sha256=sha256_file(image_path),
                is_negative=True,
            )
        )
        counts[split_name] += 1
        counts[f"{split_name}_negative"] += 1
        value += 1

    manifest_path = dataset_root / "manifest.csv"
    write_derived_manifest(records, manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    semantic_fingerprint = "d" * 64
    (dataset_root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": CLASSES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (dataset_root / "metadata.json").write_text(
        json.dumps(
            {
                "derived_manifest_sha256": manifest_sha256,
                "semantic_fingerprint_sha256": semantic_fingerprint,
            }
        ),
        encoding="utf-8",
    )
    return dataset_root, semantic_fingerprint, counts


# ADD 2026-08-25: Synthetic package의 positive/negative manifest row를 생성한다.
def _record(
    *,
    sample_id: str,
    split_name: str,
    defect_type: str,
    class_id: str,
    image_path: str,
    label_path: str,
    image_sha256: str,
    is_negative: bool,
) -> DerivedManifestRecord:
    return DerivedManifestRecord(
        dataset_name="synthetic-derived",
        dataset_version="v1",
        derived_task="yolo_segmentation",
        source_manifest_sha256="a" * 64,
        source_split="test",
        source_manifest_split="test",
        source_image_path=f"source/{sample_id}.png",
        source_mask_path="" if is_negative else f"source/{sample_id}_mask.png",
        category="metal_nut",
        sample_id=sample_id,
        defect_type=defect_type,
        target_class="" if is_negative else defect_type,
        target_class_id=class_id,
        derived_split=split_name,
        is_negative=is_negative,
        image_width=16,
        image_height=16,
        image_path=image_path,
        label_path=label_path,
        image_sha256=image_sha256,
        mask_sha256="" if is_negative else "b" * 64,
        polygon_count=0 if is_negative else 1,
        component_count=0 if is_negative else 1,
        hole_count=0,
        polygon_vertex_count=0 if is_negative else 3,
        round_trip_iou="1.0",
        pixel_precision="1.0",
        pixel_recall="1.0",
    )


# ADD 2026-08-25: Synthetic lineage와 isolated output roots를 baseline config에 주입한다.
def _config(
    root: Path,
    dataset_root: Path,
    semantic_fingerprint: str,
    counts: dict[str, int],
) -> YoloSegmentationBaselineConfig:
    base = load_yolo_segmentation_config(CONFIG_PATH)
    return replace(
        base,
        dataset_contract=replace(
            base.dataset_contract,
            manifest_sha256=sha256_file(dataset_root / "manifest.csv"),
            semantic_fingerprint_sha256=semantic_fingerprint,
            sample_counts=counts,
        ),
        evaluation=replace(base.evaluation, save_visualizations=False),
        output=replace(
            base.output,
            artifact_root=root / "artifacts",
            training_runtime_root=root / "training",
            evaluation_root=root / "evaluation",
        ),
    )


# ADD 2026-08-25: Fake backend로 train/val/test isolation과 project artifacts를 통합 검증한다.
def test_training_and_evaluation_pipeline_contract(tmp_path: Path) -> None:
    dataset_root, semantic_fingerprint, counts = _build_dataset(tmp_path)
    config = _config(tmp_path, dataset_root, semantic_fingerprint, counts)

    # Dataset lineage mismatch는 framework backend 호출 전에 중단된다.
    wrong_contract = replace(config.dataset_contract, manifest_sha256="e" * 64)
    with pytest.raises(ValueError, match="manifest SHA"):
        validate_training_dataset(dataset_root, wrong_contract)

    # Fake trainer는 best checkpoint contract만 반환하고 GPU optimization은 수행하지 않는다.
    def fake_training_runner(
        runner_config: YoloSegmentationBaselineConfig,
        dataset_yaml: Path,
        runtime_root: Path,
        artifact_id: str,
        requested_device: str,
    ) -> BackendTrainingResult:
        runtime_dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
        assert Path(runtime_dataset["path"]).is_absolute()
        assert runtime_dataset["train"] == "images/train"
        assert runtime_dataset["val"] == "images/val"
        assert runtime_dataset["test"] == "images/test"
        assert runner_config.evaluation.split == "test"
        assert requested_device == "cuda"
        checkpoint = runtime_root / artifact_id / "weights" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"fake-yolo-best-checkpoint")
        return BackendTrainingResult(
            best_checkpoint=checkpoint,
            best_epoch=7,
            actual_device="cuda:0",
            framework_version="8.4.128",
            source_checkpoint="weights/best.pt",
        )

    training_result = train_yolo_segmentation(
        config=config,
        dataset_root=dataset_root,
        artifact_id="integration-v1",
        requested_device="cuda",
        training_runner=fake_training_runner,
        created_at="2026-08-25T00:00:00+00:00",
    )
    metadata = validate_yolo_artifact(
        training_result.artifact_dir,
        expected_contract=config.dataset_contract,
    )
    assert metadata.best_epoch == 7
    assert metadata.device == "cuda:0"

    # Independent evaluator는 고정 artifact와 test manifest record만 받는다.
    def fake_evaluation_runner(
        runner_config: YoloSegmentationBaselineConfig,
        checkpoint_path: Path,
        dataset_yaml: Path,
        test_records: list[DerivedManifestRecord],
        output_dir: Path,
        requested_device: str,
    ) -> BackendEvaluationResult:
        assert checkpoint_path.read_bytes() == b"fake-yolo-best-checkpoint"
        assert all(record.derived_split == "test" for record in test_records)
        assert len(test_records) == counts["test"]
        assert dataset_yaml.is_file()
        assert output_dir.is_dir()
        assert requested_device == "cpu"
        observations = tuple(
            PredictionObservation(
                sample_id=record.sample_id,
                defect_type=record.defect_type,
                is_negative=record.is_negative,
                predicted_class_ids=() if record.is_negative else (int(record.target_class_id),),
                confidences=() if record.is_negative else (0.9,),
                segmentation_instance_count=0 if record.is_negative else 1,
            )
            for record in test_records
        )
        return BackendEvaluationResult(
            metrics=FakeMetrics(),
            observations=observations,
            visualization_paths=(),
            actual_device="cpu",
            framework_version="8.4.128",
        )

    evaluation_result = evaluate_yolo_segmentation(
        config=config,
        dataset_root=dataset_root,
        artifact_id="integration-v1",
        requested_device="cpu",
        evaluation_runner=fake_evaluation_runner,
        created_at="2026-08-25T01:00:00+00:00",
    )
    metrics = json.loads(evaluation_result.metrics_path.read_text(encoding="utf-8"))
    negative = json.loads(evaluation_result.negative_analysis_path.read_text(encoding="utf-8"))
    per_class = json.loads(evaluation_result.per_class_metrics_path.read_text(encoding="utf-8"))
    assert metrics["split"] == "test"
    assert metrics["threshold_calibrated_on_test"] is False
    assert metrics["metrics"]["mask"]["map50_95"] == 0.55
    assert negative["false_positive_image_rate"] == 0.0
    assert set(per_class) == {"bent", "color", "scratch"}


# ADD 2026-08-25: Derived manifest의 flip row가 valid lineage로 재서명되어도 training에서 거부된다.
def test_training_dataset_rejects_flip_taxonomy(tmp_path: Path) -> None:
    dataset_root, semantic_fingerprint, counts = _build_dataset(tmp_path)
    manifest_path = dataset_root / "manifest.csv"
    records = read_derived_manifest(manifest_path)
    records[0] = replace(records[0], defect_type="flip", target_class="flip")
    write_derived_manifest(records, manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    (dataset_root / "metadata.json").write_text(
        json.dumps(
            {
                "derived_manifest_sha256": manifest_sha256,
                "semantic_fingerprint_sha256": semantic_fingerprint,
            }
        ),
        encoding="utf-8",
    )
    config = _config(tmp_path, dataset_root, semantic_fingerprint, counts)
    with pytest.raises(ValueError, match="Flip"):
        validate_training_dataset(dataset_root, config.dataset_contract)
