"""Train YOLO11n segmentation and export a project-owned best-checkpoint artifact."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from ml.experiments.yolo_epoch_logging import EpochMetricsLogger
from ml.training.device import SUPPORTED_DEVICES, resolve_device
from ml.training.reproducibility import seed_training
from ml.training.yolo_segmentation import (
    ARTIFACT_SCHEMA_VERSION,
    YoloArtifactMetadata,
    YoloSegmentationBaselineConfig,
    build_ultralytics_training_overrides,
    load_yolo_segmentation_config,
    validate_artifact_id,
    validate_training_dataset,
    write_yolo_artifact,
)

DEFAULT_CONFIG_PATH = Path("configs/model/yolo_segmentation_baseline.yaml")
DEFAULT_DATASET_ROOT = Path(
    "data/processed/supervised_derived/mvtec_ad/metal_nut/yolo_segmentation/v1"
)


@dataclass(frozen=True)
class BackendTrainingResult:
    """Minimal information extracted from one completed Ultralytics training run."""

    best_checkpoint: Path
    best_epoch: int
    actual_device: str
    framework_version: str
    source_checkpoint: str


@dataclass(frozen=True)
class YoloTrainingResult:
    """Completed project-owned artifact and its ignored Ultralytics runtime directory."""

    artifact_dir: Path
    runtime_dir: Path
    metadata: YoloArtifactMetadata


type TrainingRunner = Callable[
    [YoloSegmentationBaselineConfig, Path, Path, str, str],
    BackendTrainingResult,
]


# ADD 2026-08-25: Portable package YAML을 Ultralytics가 정확히 해석할 runtime YAML로 변환한다.
def write_runtime_dataset_yaml(
    *,
    dataset_root: Path,
    destination: Path,
    classes: dict[int, str],
) -> Path:
    """Write an ignored absolute-root adapter without mutating the portable dataset artifact."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": classes,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return destination


# ADD 2026-08-25: Ultralytics를 lazy import하고 best epoch/checkpoint만 project contract로 반환한다.
# MODIFY 2026-08-27: Shared trainer overrides와 per-epoch evidence callback을 연결한다.
def run_ultralytics_training(
    config: YoloSegmentationBaselineConfig,
    dataset_yaml: Path,
    runtime_root: Path,
    artifact_id: str,
    requested_device: str,
) -> BackendTrainingResult:
    """Delegate model optimization to the pinned framework without copying its trainer logic."""
    os.environ.setdefault("YOLO_CONFIG_DIR", str((runtime_root / ".ultralytics-config").resolve()))
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version

    resolved_device = resolve_device(requested_device)
    framework_device: str | int | None = None
    if resolved_device.type == "cuda":
        framework_device = 0
    elif requested_device != "auto":
        framework_device = resolved_device.type

    model = YOLO(config.model.weights, task=config.model.task)
    best_epoch = 0
    epoch_logger = EpochMetricsLogger(
        output_path=runtime_root / artifact_id / "epoch_metrics.jsonl",
        total_epochs=config.training.epochs,
    )

    # Validation fitness가 갱신된 epoch를 callback에서 1-based best epoch로 기록한다.
    def record_best_epoch(trainer: Any) -> None:
        nonlocal best_epoch
        if trainer.best_fitness == trainer.fitness:
            best_epoch = int(trainer.epoch) + 1

    model.add_callback("on_fit_epoch_end", record_best_epoch)
    model.add_callback("on_train_epoch_start", epoch_logger.on_train_epoch_start)
    model.add_callback("on_fit_epoch_end", epoch_logger.on_fit_epoch_end)
    training_kwargs: dict[str, Any] = {
        **build_ultralytics_training_overrides(config),
        "data": str(dataset_yaml),
        "device": framework_device,
        "project": str(runtime_root),
        "name": artifact_id,
        "exist_ok": True,
    }
    try:
        model.train(**training_kwargs)
    finally:
        epoch_logger.close()
    trainer = getattr(model, "trainer", None)
    best_checkpoint = Path(getattr(trainer, "best", ""))
    if not best_checkpoint.is_file() or best_epoch <= 0:
        raise RuntimeError(
            "Ultralytics training did not produce a traceable best checkpoint/epoch."
        )
    actual_device = str(getattr(trainer, "device", resolved_device))
    return BackendTrainingResult(
        best_checkpoint=best_checkpoint,
        best_epoch=best_epoch,
        actual_device=actual_device,
        framework_version=ultralytics_version,
        source_checkpoint="weights/best.pt",
    )


# ADD 2026-08-25: Dataset validation부터 best checkpoint artifact 저장까지 training flow를 조율한다.
def train_yolo_segmentation(
    *,
    config: YoloSegmentationBaselineConfig,
    dataset_root: Path,
    artifact_id: str,
    requested_device: str | None = None,
    training_runner: TrainingRunner = run_ultralytics_training,
    created_at: str | None = None,
) -> YoloTrainingResult:
    """Train from train/val only and persist the selected checkpoint with exact lineage."""
    validate_artifact_id(artifact_id)
    artifact_dir = config.output.artifact_root / artifact_id
    runtime_dir = config.output.training_runtime_root / artifact_id
    if artifact_dir.exists() or runtime_dir.exists():
        raise FileExistsError("YOLO artifact or training runtime directory already exists.")

    # 비용이 큰 pretrained download/training 전에 self-contained dataset contract를 검증한다.
    validate_training_dataset(dataset_root, config.dataset_contract)
    runtime_dir.mkdir(parents=True, exist_ok=False)
    dataset_yaml = write_runtime_dataset_yaml(
        dataset_root=dataset_root,
        destination=runtime_dir / "dataset.runtime.yaml",
        classes=config.dataset_contract.classes,
    )
    device = requested_device or config.training.device
    seed_training(config.training.seed)

    # Train/validation만 사용하는 Ultralytics backend로 best checkpoint를 선택한다.
    backend_result = training_runner(
        config, dataset_yaml, config.output.training_runtime_root, artifact_id, device
    )
    metadata = YoloArtifactMetadata(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        model_name=config.model.weights,
        task=config.model.task,
        architecture=config.model.architecture,
        category=config.dataset_contract.category,
        classes=config.dataset_contract.classes,
        seed=config.training.seed,
        dataset_manifest_sha256=config.dataset_contract.manifest_sha256,
        dataset_semantic_fingerprint_sha256=(config.dataset_contract.semantic_fingerprint_sha256),
        training_config={
            "model": asdict(config.model),
            "training": asdict(config.training),
            "dataset_protocol_name": config.dataset_contract.protocol_name,
        },
        created_at=created_at or datetime.now(UTC).isoformat(),
        framework="ultralytics",
        framework_version=backend_result.framework_version,
        torch_version=str(torch.__version__),
        device=backend_result.actual_device,
        best_epoch=backend_result.best_epoch,
        source_checkpoint=backend_result.source_checkpoint,
        checkpoint_sha256="0" * 64,
    )

    # Library run directory와 분리된 checkpoint/metadata만 project artifact로 저장한다.
    saved_metadata = write_yolo_artifact(
        source_checkpoint=backend_result.best_checkpoint,
        artifact_dir=artifact_dir,
        metadata=metadata,
    )
    return YoloTrainingResult(
        artifact_dir=artifact_dir,
        runtime_dir=runtime_dir,
        metadata=saved_metadata,
    )


# ADD 2026-08-25: C2-2 training CLI arguments를 repository convention에 맞게 정의한다.
def parse_args() -> argparse.Namespace:
    """Parse config, dataset, artifact identity, and optional explicit device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--device", choices=SUPPORTED_DEVICES)
    return parser.parse_args()


# ADD 2026-08-25: CLI config/dataset을 검증하고 training artifact summary를 출력한다.
def main() -> int:
    """Run one baseline training lifecycle; actual full run is intended for Kaggle T4."""
    args = parse_args()
    config = load_yolo_segmentation_config(args.config)
    result = train_yolo_segmentation(
        config=config,
        dataset_root=args.dataset,
        artifact_id=args.artifact_id,
        requested_device=args.device,
    )
    print("YOLO segmentation training: PASS")
    print(f"Device: {result.metadata.device}")
    print(f"Best epoch: {result.metadata.best_epoch}")
    print(f"Artifact: {result.artifact_dir}")
    print(f"Runtime outputs: {result.runtime_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
