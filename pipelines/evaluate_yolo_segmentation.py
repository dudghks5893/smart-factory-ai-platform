"""Evaluate a fixed YOLO segmentation artifact on the supervised-derived test split."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sized
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
import yaml
from PIL import Image

from ml.datasets.segmentation_annotations import parse_yolo_segmentation_label, rasterize_polygons
from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord, read_derived_manifest
from ml.evaluation.yolo_segmentation import (
    PredictionObservation,
    aggregate_prediction_diagnostics,
    serialize_ultralytics_metrics,
)
from ml.training.device import SUPPORTED_DEVICES, resolve_device
from ml.training.yolo_segmentation import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    YoloSegmentationBaselineConfig,
    load_yolo_segmentation_config,
    validate_artifact_id,
    validate_training_dataset,
    validate_yolo_artifact,
)
from pipelines.train_yolo_segmentation import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATASET_ROOT,
    write_runtime_dataset_yaml,
)
from shared.hashing import sha256_file

METRICS_FILENAME = "metrics.json"
PER_CLASS_METRICS_FILENAME = "per_class_metrics.json"
NEGATIVE_ANALYSIS_FILENAME = "negative_analysis.json"
POSITIVE_ANALYSIS_FILENAME = "positive_analysis.json"
PREDICTIONS_FILENAME = "prediction_summary.jsonl"


@dataclass(frozen=True)
class BackendEvaluationResult:
    """Framework metrics plus test-image diagnostics from one fixed checkpoint."""

    metrics: object
    observations: tuple[PredictionObservation, ...]
    visualization_paths: tuple[Path, ...]
    actual_device: str
    framework_version: str


@dataclass(frozen=True)
class YoloEvaluationResult:
    """Paths produced by one independent derived-test evaluation."""

    output_dir: Path
    metrics_path: Path
    per_class_metrics_path: Path
    negative_analysis_path: Path
    positive_analysis_path: Path
    predictions_path: Path
    visualization_paths: tuple[Path, ...]


type EvaluationRunner = Callable[
    [YoloSegmentationBaselineConfig, Path, Path, list[DerivedManifestRecord], Path, str],
    BackendEvaluationResult,
]


# ADD 2026-08-25: Tensor-like Ultralytics prediction field를 typed Python list로 변환한다.
def _tensor_values(value: object, *, integer: bool) -> list[int] | list[float]:
    if value is None:
        return []
    detached = getattr(value, "detach", None)
    cpu = getattr(detached() if callable(detached) else value, "cpu", None)
    materialized = cpu() if callable(cpu) else value
    tolist = getattr(materialized, "tolist", None)
    if not callable(tolist):
        raise ValueError("Ultralytics prediction field cannot be converted to a list.")
    raw = tolist()
    if not isinstance(raw, list):
        raise ValueError("Ultralytics prediction field must be one-dimensional.")
    return [int(item) for item in raw] if integer else [float(item) for item in raw]


# ADD 2026-08-25: One Ultralytics Results object를 framework-neutral diagnostic으로 변환한다.
def parse_prediction_result(
    result: object,
    *,
    record: DerivedManifestRecord,
) -> PredictionObservation:
    """Reject missing box/mask alignment rather than hiding malformed predictions."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        raise ValueError("Ultralytics result is missing boxes.")
    class_ids = cast(list[int], _tensor_values(getattr(boxes, "cls", None), integer=True))
    confidences = cast(
        list[float],
        _tensor_values(getattr(boxes, "conf", None), integer=False),
    )
    if len(class_ids) != len(confidences):
        raise ValueError("Ultralytics box class/confidence lengths do not match.")
    masks = getattr(result, "masks", None)
    if masks is None:
        mask_count = 0
    else:
        mask_data = getattr(masks, "data", None)
        if mask_data is None:
            raise ValueError("Ultralytics result masks have no data.")
        try:
            mask_count = len(cast(Sized, mask_data))
        except TypeError as exc:
            raise ValueError("Ultralytics result masks are malformed.") from exc
    observation = PredictionObservation(
        sample_id=record.sample_id,
        defect_type=record.defect_type,
        is_negative=record.is_negative,
        predicted_class_ids=tuple(class_ids),
        confidences=tuple(confidences),
        segmentation_instance_count=mask_count,
    )
    return observation


# ADD 2026-08-25: GT overlay와 rendered prediction을 representative image로 저장한다.
def save_prediction_visualization(
    *,
    result: object,
    record: DerivedManifestRecord,
    dataset_root: Path,
    output_path: Path,
    classes: dict[int, str],
) -> None:
    """Persist a two-panel GT/prediction comparison below ignored evaluation outputs."""
    image_path = dataset_root / record.image_path
    label_path = dataset_root / record.label_path
    with Image.open(image_path) as image:
        original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    label_text = label_path.read_text(encoding="utf-8")
    polygons = parse_yolo_segmentation_label(label_text, valid_class_ids=set(classes))
    ground_truth = rasterize_polygons(
        polygons,
        image_width=record.image_width,
        image_height=record.image_height,
    )
    gt_overlay = original.copy()
    gt_overlay[ground_truth] = (
        0.45 * gt_overlay[ground_truth] + 0.55 * np.array([255, 55, 35])
    ).astype(np.uint8)
    plot = getattr(result, "plot", None)
    if not callable(plot):
        raise ValueError("Ultralytics result cannot render prediction visualization.")
    predicted_bgr = np.asarray(plot(), dtype=np.uint8)
    predicted_rgb = predicted_bgr[..., ::-1]
    if predicted_rgb.shape != gt_overlay.shape:
        predicted_rgb = np.asarray(
            Image.fromarray(predicted_rgb).resize((record.image_width, record.image_height))
        )
    combined = np.concatenate((gt_overlay, predicted_rgb), axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(output_path)


# ADD 2026-08-25: Pinned checkpoint를 test val과 fixed-confidence diagnostics에 사용한다.
def run_ultralytics_evaluation(
    config: YoloSegmentationBaselineConfig,
    checkpoint_path: Path,
    dataset_yaml: Path,
    test_records: list[DerivedManifestRecord],
    output_dir: Path,
    requested_device: str,
) -> BackendEvaluationResult:
    """Delegate framework metrics/prediction to Ultralytics without threshold tuning."""
    os.environ.setdefault("YOLO_CONFIG_DIR", str((output_dir / ".ultralytics-config").resolve()))
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version

    resolved_device = resolve_device(requested_device)
    framework_device: str | int | None = None
    if resolved_device.type == "cuda":
        framework_device = 0
    elif requested_device != "auto":
        framework_device = resolved_device.type
    model = YOLO(str(checkpoint_path), task=config.model.task)

    # Framework default metric confidence로 fixed checkpoint의 derived test split을 평가한다.
    metrics = model.val(
        data=str(dataset_yaml),
        split=config.evaluation.split,
        imgsz=config.training.imgsz,
        batch=config.evaluation.batch,
        workers=config.evaluation.workers,
        device=framework_device,
        project=str(output_dir),
        name="ultralytics-validation",
        exist_ok=True,
        plots=False,
        save_json=False,
        verbose=True,
    )

    # 별도 fixed diagnostic confidence로 good FP와 positive behavior를 image별 수집한다.
    runtime_dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(runtime_dataset, dict) or not isinstance(runtime_dataset.get("path"), str):
        raise ValueError("Runtime dataset YAML does not contain an absolute dataset root.")
    dataset_root = Path(runtime_dataset["path"])
    sources = [str((dataset_root / record.image_path).resolve()) for record in test_records]
    raw_results = list(
        model.predict(
            source=sources,
            conf=config.evaluation.diagnostic_confidence,
            imgsz=config.training.imgsz,
            batch=config.evaluation.batch,
            device=framework_device,
            save=False,
            stream=False,
            verbose=False,
        )
    )
    if len(raw_results) != len(test_records):
        raise ValueError("Ultralytics prediction count does not match derived test manifest.")
    observations: list[PredictionObservation] = []
    visualization_paths: list[Path] = []
    visualized_classes: set[str] = set()
    negative_visualized = False
    for result, record in zip(raw_results, test_records, strict=True):
        observation = parse_prediction_result(result, record=record)
        observations.append(observation)
        should_visualize_positive = (
            not record.is_negative and record.defect_type not in visualized_classes
        )
        should_visualize_negative = (
            record.is_negative and bool(observation.predicted_class_ids) and not negative_visualized
        )
        if config.evaluation.save_visualizations and (
            should_visualize_positive or should_visualize_negative
        ):
            visualization_path = (
                output_dir / "visualizations" / f"{record.sample_id}_gt_vs_prediction.png"
            )
            save_prediction_visualization(
                result=result,
                record=record,
                dataset_root=dataset_root,
                output_path=visualization_path,
                classes=config.dataset_contract.classes,
            )
            visualization_paths.append(visualization_path)
            if record.is_negative:
                negative_visualized = True
            else:
                visualized_classes.add(record.defect_type)
    return BackendEvaluationResult(
        metrics=metrics,
        observations=tuple(observations),
        visualization_paths=tuple(visualization_paths),
        actual_device=str(resolved_device),
        framework_version=ultralytics_version,
    )


# ADD 2026-08-25: Stable JSON artifact를 parent 생성 후 overwrite 없이 저장한다.
def _write_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"Evaluation output already exists: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ADD 2026-08-25: Test observation을 audit-friendly JSONL 순서로 기록한다.
def _write_prediction_jsonl(path: Path, observations: tuple[PredictionObservation, ...]) -> None:
    if path.exists():
        raise FileExistsError(f"Evaluation prediction output already exists: {path}")
    with path.open("w", encoding="utf-8") as file:
        for observation in observations:
            file.write(json.dumps(asdict(observation), sort_keys=True) + "\n")


# ADD 2026-08-25: Artifact reload, derived test metric, FP diagnostic와 outputs 저장을 조율한다.
def evaluate_yolo_segmentation(
    *,
    config: YoloSegmentationBaselineConfig,
    dataset_root: Path,
    artifact_id: str,
    requested_device: str | None = None,
    evaluation_runner: EvaluationRunner = run_ultralytics_evaluation,
    created_at: str | None = None,
) -> YoloEvaluationResult:
    """Evaluate only the fixed best artifact; never select epochs or thresholds on test."""
    validate_artifact_id(artifact_id)
    artifact_dir = config.output.artifact_root / artifact_id
    output_dir = config.output.evaluation_root / artifact_id
    if output_dir.exists():
        raise FileExistsError(f"YOLO evaluation output already exists: {output_dir}")

    # Model load 전에 dataset package와 artifact checkpoint lineage를 모두 검증한다.
    validate_training_dataset(dataset_root, config.dataset_contract)
    artifact_metadata = validate_yolo_artifact(
        artifact_dir,
        expected_contract=config.dataset_contract,
    )
    records = read_derived_manifest(dataset_root / "manifest.csv")
    test_records = [record for record in records if record.derived_split == "test"]
    if len(test_records) != config.dataset_contract.sample_counts["test"]:
        raise ValueError("Derived test record count does not match evaluation contract.")
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_yaml = write_runtime_dataset_yaml(
        dataset_root=dataset_root,
        destination=output_dir / "dataset.runtime.yaml",
        classes=config.dataset_contract.classes,
    )
    backend = evaluation_runner(
        config,
        artifact_dir / MODEL_FILENAME,
        dataset_yaml,
        test_records,
        output_dir,
        requested_device or config.training.device,
    )
    if len(backend.observations) != len(test_records):
        raise ValueError("Evaluation diagnostic count does not match derived test split.")

    # Framework box/mask metric과 class result를 documented API에서 분리한다.
    overall_metrics, per_class_metrics = serialize_ultralytics_metrics(
        backend.metrics,
        classes=config.dataset_contract.classes,
    )
    negative_analysis, positive_analysis = aggregate_prediction_diagnostics(
        list(backend.observations),
        classes=config.dataset_contract.classes,
    )
    evaluated_at = created_at or datetime.now(UTC).isoformat()
    metrics_payload = {
        "schema_version": 1,
        "protocol_name": config.dataset_contract.protocol_name,
        "task": config.model.task,
        "category": config.dataset_contract.category,
        "split": "test",
        "sample_count": len(test_records),
        "metrics": overall_metrics,
        "evaluation_confidence_policy": "ultralytics_framework_default",
        "diagnostic_confidence": config.evaluation.diagnostic_confidence,
        "threshold_calibrated_on_test": False,
        "created_at": evaluated_at,
        "environment": {
            "framework": "ultralytics",
            "framework_version": backend.framework_version,
            "torch_version": str(torch.__version__),
            "device": backend.actual_device,
        },
        "provenance": {
            "dataset_manifest_sha256": config.dataset_contract.manifest_sha256,
            "dataset_semantic_fingerprint_sha256": (
                config.dataset_contract.semantic_fingerprint_sha256
            ),
            "checkpoint_sha256": artifact_metadata.checkpoint_sha256,
            "artifact_metadata_sha256": sha256_file(artifact_dir / METADATA_FILENAME),
        },
    }

    # Aggregate/per-class/negative/positive/prediction outputs를 독립 JSON artifact로 저장한다.
    metrics_path = output_dir / METRICS_FILENAME
    per_class_path = output_dir / PER_CLASS_METRICS_FILENAME
    negative_path = output_dir / NEGATIVE_ANALYSIS_FILENAME
    positive_path = output_dir / POSITIVE_ANALYSIS_FILENAME
    predictions_path = output_dir / PREDICTIONS_FILENAME
    _write_json(metrics_path, metrics_payload)
    _write_json(per_class_path, per_class_metrics)
    _write_json(negative_path, negative_analysis)
    _write_json(positive_path, positive_analysis)
    _write_prediction_jsonl(predictions_path, backend.observations)
    return YoloEvaluationResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        per_class_metrics_path=per_class_path,
        negative_analysis_path=negative_path,
        positive_analysis_path=positive_path,
        predictions_path=predictions_path,
        visualization_paths=backend.visualization_paths,
    )


# ADD 2026-08-25: Fixed artifact independent evaluation CLI arguments를 정의한다.
def parse_args() -> argparse.Namespace:
    """Parse config, dataset, artifact identity, and optional evaluation device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--device", choices=SUPPORTED_DEVICES)
    return parser.parse_args()


# ADD 2026-08-25: CLI에서 artifact를 reload해 test evaluation output 위치를 출력한다.
def main() -> int:
    """Run independent test evaluation after Kaggle training has produced an artifact."""
    args = parse_args()
    config = load_yolo_segmentation_config(args.config)
    result = evaluate_yolo_segmentation(
        config=config,
        dataset_root=args.dataset,
        artifact_id=args.artifact_id,
        requested_device=args.device,
    )
    print("YOLO segmentation evaluation: PASS")
    print(f"Metrics: {result.metrics_path}")
    print(f"Per-class metrics: {result.per_class_metrics_path}")
    print(f"Good-negative analysis: {result.negative_analysis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
