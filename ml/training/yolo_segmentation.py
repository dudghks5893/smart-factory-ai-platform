"""Configuration and artifact contracts for YOLO segmentation feasibility runs."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from PIL import Image

from ml.datasets.segmentation_annotations import parse_yolo_segmentation_label
from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord, read_derived_manifest
from shared.hashing import is_sha256_digest, sha256_file

MODEL_FILENAME = "model.pt"
METADATA_FILENAME = "metadata.json"
ARTIFACT_SCHEMA_VERSION = 1
EXPECTED_PROTOCOL_NAME = "MVTec AD-derived supervised segmentation feasibility split"
ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
EXPERIMENT_CONTENT_SPLITS = frozenset({"train", "val"})


@dataclass(frozen=True)
class YoloModelConfig:
    """Ultralytics instance-segmentation architecture selection."""

    architecture: str
    weights: str
    task: str
    pretrained: bool

    # ADD 2026-08-25: Baseline architecture와 official checkpoint naming 계약을 검증한다.
    def validate(self) -> None:
        if self.architecture != "yolo11n-seg" or self.weights != "yolo11n-seg.pt":
            raise ValueError("C2-2 baseline must use the pinned yolo11n-seg.pt architecture.")
        if self.task != "segment" or not self.pretrained:
            raise ValueError("C2-2 baseline requires pretrained instance segmentation.")


@dataclass(frozen=True)
class YoloTrainingConfig:
    """Explicit small-data training and early-stopping settings."""

    seed: int
    epochs: int
    imgsz: int
    batch: int
    workers: int
    optimizer: str
    lr0: float | None
    lrf: float | None
    patience: int
    deterministic: bool
    amp: bool
    device: str

    # ADD 2026-08-25: Training resource와 reproducibility parameter bounds를 검증한다.
    def validate(self) -> None:
        if self.seed < 0 or self.epochs <= 0 or self.imgsz <= 0 or self.batch <= 0:
            raise ValueError("YOLO seed/count/size training values are invalid.")
        if self.imgsz % 32 != 0:
            raise ValueError("training.imgsz must be divisible by 32.")
        if self.workers < 0 or self.patience < 0:
            raise ValueError("training.workers and patience must be non-negative.")
        if self.optimizer != "auto" or self.lr0 is not None or self.lrf is not None:
            raise ValueError(
                "Initial baseline must use the documented Ultralytics optimizer/LR defaults."
            )
        if not isinstance(self.deterministic, bool) or not isinstance(self.amp, bool):
            raise ValueError("training deterministic and amp fields must be boolean.")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("training.device must be auto, cpu, mps, or cuda.")


@dataclass(frozen=True)
class YoloEvaluationConfig:
    """Independent fixed-artifact test evaluation settings."""

    split: str
    batch: int
    workers: int
    diagnostic_confidence: float
    save_visualizations: bool

    # ADD 2026-08-25: Test-only evaluation과 diagnostic confidence contract를 검증한다.
    def validate(self) -> None:
        if self.split != "test":
            raise ValueError("C2-2 final evaluation must use only the derived test split.")
        if self.batch <= 0 or self.workers < 0:
            raise ValueError("Evaluation batch/workers values are invalid.")
        if not 0.0 < self.diagnostic_confidence < 1.0:
            raise ValueError("Diagnostic confidence must be in (0, 1).")
        if not isinstance(self.save_visualizations, bool):
            raise ValueError("evaluation.save_visualizations must be boolean.")


@dataclass(frozen=True)
class YoloDatasetContract:
    """Immutable identity and expected counts for one derived dataset package."""

    protocol_name: str
    task: str
    category: str
    manifest_sha256: str
    semantic_fingerprint_sha256: str
    classes: dict[int, str]
    sample_counts: dict[str, int]

    # ADD 2026-08-25: Benchmark naming, class mapping, hashes와 split counts를 검증한다.
    def validate(self) -> None:
        if self.protocol_name != EXPECTED_PROTOCOL_NAME:
            raise ValueError(
                "Dataset protocol name must preserve supervised-derived benchmark isolation."
            )
        if self.task != "yolo_segmentation" or not self.category:
            raise ValueError("Dataset task/category contract is invalid.")
        if self.classes != {0: "bent", 1: "color", 2: "scratch"}:
            raise ValueError("Dataset classes must be exactly bent/color/scratch.")
        if not is_sha256_digest(self.manifest_sha256) or not is_sha256_digest(
            self.semantic_fingerprint_sha256
        ):
            raise ValueError("Dataset contract hashes must be SHA-256 digests.")
        required_counts = {
            "train",
            "val",
            "test",
            "train_positive",
            "train_negative",
            "val_positive",
            "val_negative",
            "test_positive",
            "test_negative",
        }
        if set(self.sample_counts) != required_counts or any(
            count < 0 for count in self.sample_counts.values()
        ):
            raise ValueError("Dataset contract sample counts are incomplete or invalid.")
        for split_name in ("train", "val", "test"):
            if self.sample_counts[split_name] != (
                self.sample_counts[f"{split_name}_positive"]
                + self.sample_counts[f"{split_name}_negative"]
            ):
                raise ValueError(f"Dataset count arithmetic mismatch: {split_name}")


@dataclass(frozen=True)
class YoloOutputConfig:
    """Project-owned artifact and ignored runtime/evaluation roots."""

    artifact_root: Path
    training_runtime_root: Path
    evaluation_root: Path


@dataclass(frozen=True)
class YoloSegmentationBaselineConfig:
    """Complete validated C2-2 baseline configuration."""

    model: YoloModelConfig
    training: YoloTrainingConfig
    evaluation: YoloEvaluationConfig
    dataset_contract: YoloDatasetContract
    output: YoloOutputConfig

    # ADD 2026-08-25: 모든 C2-2 config section의 domain invariant를 검증한다.
    def validate(self) -> None:
        self.model.validate()
        self.training.validate()
        self.evaluation.validate()
        self.dataset_contract.validate()


@dataclass(frozen=True)
class YoloTrainerOverrides:
    """Optional experiment-owned Ultralytics arguments with no global defaults."""

    mosaic: float
    mask_ratio: int
    overlap_mask: bool
    scale: float

    # ADD 2026-08-31: Explicit experiment arguments를 safe Ultralytics scalar로 검증한다.
    def validate(self) -> None:
        if (
            not 0.0 <= self.mosaic <= 1.0
            or self.mask_ratio <= 0
            or type(self.overlap_mask) is not bool
            or self.scale < 0.0
        ):
            raise ValueError("YOLO experiment trainer overrides are invalid.")


# ADD 2026-08-27: Shared trainer args를 만든다. → MODIFY 2026-08-31: C4-2C args를 opt-in 병합한다.
def build_ultralytics_training_overrides(
    config: YoloSegmentationBaselineConfig,
    experiment_overrides: YoloTrainerOverrides | None = None,
) -> dict[str, Any]:
    config.validate()
    result: dict[str, Any] = {
        "task": config.model.task,
        "mode": "train",
        "epochs": config.training.epochs,
        "imgsz": config.training.imgsz,
        "batch": config.training.batch,
        "workers": config.training.workers,
        "optimizer": config.training.optimizer,
        "patience": config.training.patience,
        "seed": config.training.seed,
        "deterministic": config.training.deterministic,
        "amp": config.training.amp,
        "pretrained": config.model.pretrained,
        "plots": True,
        "save": True,
        "verbose": True,
    }
    if experiment_overrides is not None:
        experiment_overrides.validate()
        result.update(asdict(experiment_overrides))
    return result


@dataclass(frozen=True)
class YoloArtifactMetadata:
    """Project-owned checkpoint lineage independent of Ultralytics run directories."""

    schema_version: int
    model_name: str
    task: str
    architecture: str
    category: str
    classes: dict[int, str]
    seed: int
    dataset_manifest_sha256: str
    dataset_semantic_fingerprint_sha256: str
    training_config: dict[str, Any]
    created_at: str
    framework: str
    framework_version: str
    torch_version: str
    device: str
    best_epoch: int
    source_checkpoint: str
    checkpoint_sha256: str

    # ADD 2026-08-25: Artifact metadata를 JSON key contract로 변환한다.
    def to_json_dict(self) -> dict[str, Any]:
        return {**asdict(self), "classes": {str(key): value for key, value in self.classes.items()}}

    # ADD 2026-08-25: Untrusted metadata JSON을 typed artifact contract로 복원한다.
    @classmethod
    def from_json_dict(cls, raw: object) -> YoloArtifactMetadata:
        if not isinstance(raw, dict):
            raise ValueError("YOLO artifact metadata root must be a mapping.")
        values = cast(dict[str, Any], raw)
        try:
            raw_classes = values["classes"]
            if not isinstance(raw_classes, dict):
                raise TypeError
            raw_training_config = values["training_config"]
            if not isinstance(raw_training_config, dict):
                raise TypeError
            metadata = cls(
                schema_version=int(values["schema_version"]),
                model_name=str(values["model_name"]),
                task=str(values["task"]),
                architecture=str(values["architecture"]),
                category=str(values["category"]),
                classes={int(key): str(value) for key, value in raw_classes.items()},
                seed=int(values["seed"]),
                dataset_manifest_sha256=str(values["dataset_manifest_sha256"]),
                dataset_semantic_fingerprint_sha256=str(
                    values["dataset_semantic_fingerprint_sha256"]
                ),
                training_config=cast(dict[str, Any], raw_training_config),
                created_at=str(values["created_at"]),
                framework=str(values["framework"]),
                framework_version=str(values["framework_version"]),
                torch_version=str(values["torch_version"]),
                device=str(values["device"]),
                best_epoch=int(values["best_epoch"]),
                source_checkpoint=str(values["source_checkpoint"]),
                checkpoint_sha256=str(values["checkpoint_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("YOLO artifact metadata is missing or malformed.") from exc
        metadata.validate()
        return metadata

    # ADD 2026-08-25: Artifact task, lineage, version과 checkpoint field를 검증한다.
    def validate(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Unsupported YOLO artifact schema version.")
        if self.model_name != "yolo11n-seg.pt" or self.architecture != "yolo11n-seg":
            raise ValueError("Artifact model architecture does not match the C2-2 baseline.")
        if self.task != "segment" or self.classes != {0: "bent", 1: "color", 2: "scratch"}:
            raise ValueError("Artifact task/classes are invalid.")
        if not self.category or self.seed < 0 or self.best_epoch <= 0:
            raise ValueError("Artifact category, seed, or best epoch is invalid.")
        if not self.framework or not self.framework_version or not self.torch_version:
            raise ValueError("Artifact framework version fields must not be blank.")
        if (
            not self.created_at
            or not self.device
            or not self.source_checkpoint
            or not self.training_config
        ):
            raise ValueError("Artifact runtime/training metadata must not be blank.")
        for digest in (
            self.dataset_manifest_sha256,
            self.dataset_semantic_fingerprint_sha256,
            self.checkpoint_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("Artifact lineage/checkpoint hashes must be SHA-256 digests.")


# ADD 2026-08-25: YAML mapping section을 typed loader가 사용할 mapping으로 좁힌다.
def _mapping(value: object, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section}' must be a mapping.")
    return cast(dict[str, Any], value)


# ADD 2026-08-25: Required config boolean이 integer/string으로 coercion되지 않게 검증한다.
def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean.")
    return value


# ADD 2026-08-25: C2-2 YAML을 typed baseline config로 로드하고 fail-fast 검증한다.
def load_yolo_segmentation_config(path: Path) -> YoloSegmentationBaselineConfig:
    """Load model, runtime, dataset, and output policy without initializing Ultralytics."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        root = _mapping(raw, "root")
        model = _mapping(root["model"], "model")
        training = _mapping(root["training"], "training")
        evaluation = _mapping(root["evaluation"], "evaluation")
        dataset = _mapping(root["dataset_contract"], "dataset_contract")
        output = _mapping(root["output"], "output")
        raw_classes = _mapping(dataset["classes"], "dataset_contract.classes")
        raw_counts = _mapping(dataset["sample_counts"], "dataset_contract.sample_counts")
        config = YoloSegmentationBaselineConfig(
            model=YoloModelConfig(
                architecture=str(model["architecture"]),
                weights=str(model["weights"]),
                task=str(model["task"]),
                pretrained=_boolean(model["pretrained"], "model.pretrained"),
            ),
            training=YoloTrainingConfig(
                seed=int(training["seed"]),
                epochs=int(training["epochs"]),
                imgsz=int(training["imgsz"]),
                batch=int(training["batch"]),
                workers=int(training["workers"]),
                optimizer=str(training["optimizer"]),
                lr0=None if training["lr0"] is None else float(training["lr0"]),
                lrf=None if training["lrf"] is None else float(training["lrf"]),
                patience=int(training["patience"]),
                deterministic=_boolean(training["deterministic"], "training.deterministic"),
                amp=_boolean(training["amp"], "training.amp"),
                device=str(training["device"]),
            ),
            evaluation=YoloEvaluationConfig(
                split=str(evaluation["split"]),
                batch=int(evaluation["batch"]),
                workers=int(evaluation["workers"]),
                diagnostic_confidence=float(evaluation["diagnostic_confidence"]),
                save_visualizations=_boolean(
                    evaluation["save_visualizations"],
                    "evaluation.save_visualizations",
                ),
            ),
            dataset_contract=YoloDatasetContract(
                protocol_name=str(dataset["protocol_name"]),
                task=str(dataset["task"]),
                category=str(dataset["category"]),
                manifest_sha256=str(dataset["manifest_sha256"]),
                semantic_fingerprint_sha256=str(dataset["semantic_fingerprint_sha256"]),
                classes={int(key): str(value) for key, value in raw_classes.items()},
                sample_counts={str(key): int(value) for key, value in raw_counts.items()},
            ),
            output=YoloOutputConfig(
                artifact_root=Path(output["artifact_root"]),
                training_runtime_root=Path(output["training_runtime_root"]),
                evaluation_root=Path(output["evaluation_root"]),
            ),
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid YOLO segmentation config: {path}") from exc
    config.validate()
    return config


# ADD 2026-08-25: CLI artifact ID가 output root 밖의 경로를 만들지 않게 검증한다.
def validate_artifact_id(artifact_id: str) -> None:
    """Reject traversal, separators, blank, and unbounded artifact identifiers."""
    if not ARTIFACT_ID_PATTERN.fullmatch(artifact_id):
        raise ValueError(
            "Artifact ID must use 1-128 ASCII letters, digits, dot, underscore, or dash."
        )


# ADD 2026-08-28: Dataset-level provenance와 portable descriptor를 content 접근 전에 검증한다.
def _validate_dataset_package_metadata(
    dataset_root: Path,
    contract: YoloDatasetContract,
) -> Path:
    dataset_yaml_path = dataset_root / "dataset.yaml"
    manifest_path = dataset_root / "manifest.csv"
    metadata_path = dataset_root / "metadata.json"
    for path in (dataset_yaml_path, manifest_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Training dataset artifact missing: {path}")
    if sha256_file(manifest_path) != contract.manifest_sha256:
        raise ValueError("Training dataset manifest SHA does not match the baseline contract.")

    dataset_yaml = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8"))
    if not isinstance(dataset_yaml, dict) or dataset_yaml.get("names") != contract.classes:
        raise ValueError("Training dataset class mapping does not match the baseline contract.")
    if dataset_yaml.get("path") != "." or any(
        dataset_yaml.get(split_name) != f"images/{split_name}"
        for split_name in ("train", "val", "test")
    ):
        raise ValueError("Training dataset YAML paths are not portable or complete.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("semantic_fingerprint_sha256") != contract.semantic_fingerprint_sha256:
        raise ValueError("Training dataset semantic fingerprint mismatch.")
    if metadata.get("derived_manifest_sha256") != contract.manifest_sha256:
        raise ValueError("Training dataset metadata manifest lineage mismatch.")
    return manifest_path


# ADD 2026-08-28: 선택된 split record만 image/label content와 count contract로 검증한다.
def _validate_dataset_record_content(
    dataset_root: Path,
    contract: YoloDatasetContract,
    records: list[DerivedManifestRecord],
    *,
    expected_splits: frozenset[str],
    count_context: str,
) -> dict[str, int]:
    expected_counts: dict[str, int] = {}
    for split in sorted(expected_splits):
        expected_counts[split] = contract.sample_counts[split]
        for kind in ("positive", "negative"):
            key = f"{split}_{kind}"
            expected_counts[key] = contract.sample_counts[key]
    observed = {key: 0 for key in expected_counts}

    source_images: set[str] = set()
    valid_class_ids = set(contract.classes)
    for record in records:
        if record.source_image_path in source_images:
            raise ValueError("Duplicate source image or derived split leakage detected.")
        source_images.add(record.source_image_path)
        if record.defect_type == "flip":
            raise ValueError("Flip must not enter C2-2 training or evaluation.")
        if (
            record.derived_task != contract.task
            or record.category != contract.category
            or record.derived_split not in expected_splits
        ):
            raise ValueError(f"Derived record task/category/split mismatch: {record.sample_id}")
        image_path = dataset_root / record.image_path
        label_path = dataset_root / record.label_path
        try:
            image_path.resolve().relative_to(dataset_root.resolve())
            label_path.resolve().relative_to(dataset_root.resolve())
        except ValueError as exc:
            raise ValueError("Derived dataset path escapes the package root.") from exc
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Derived training image/label missing: {record.sample_id}")
        with Image.open(image_path) as image:
            if image.size != (record.image_width, record.image_height):
                raise ValueError(f"Derived image dimensions changed: {record.sample_id}")
            image.verify()
        if sha256_file(image_path) != record.image_sha256:
            raise ValueError(f"Derived image SHA mismatch: {record.sample_id}")
        label_text = label_path.read_text(encoding="utf-8")
        if record.is_negative:
            if label_text or record.target_class or record.polygon_count:
                raise ValueError(f"Invalid good-negative label: {record.sample_id}")
        else:
            expected_class_id = next(
                (
                    class_id
                    for class_id, class_name in contract.classes.items()
                    if class_name == record.target_class
                ),
                None,
            )
            if expected_class_id is None or record.target_class_id != str(expected_class_id):
                raise ValueError(f"Positive manifest class mismatch: {record.sample_id}")
            polygons = parse_yolo_segmentation_label(
                label_text,
                valid_class_ids=valid_class_ids,
            )
            if len(polygons) != record.polygon_count or any(
                polygon.class_id != expected_class_id for polygon in polygons
            ):
                raise ValueError(f"Positive polygon count mismatch: {record.sample_id}")
        observed[record.derived_split] += 1
        observed[f"{record.derived_split}_{'negative' if record.is_negative else 'positive'}"] += 1
    if observed != expected_counts:
        raise ValueError(
            f"{count_context} counts mismatch: expected={expected_counts}, actual={observed}"
        )
    return observed


# ADD 2026-08-25: Dataset을 검증한다. → MODIFY 2026-08-28: 검증을 분리해 seal과 공유한다.
def validate_training_dataset(
    dataset_root: Path,
    contract: YoloDatasetContract,
) -> dict[str, int]:
    """Validate package integrity without requiring unavailable raw source masks on Kaggle."""
    manifest_path = _validate_dataset_package_metadata(dataset_root, contract)
    records = read_derived_manifest(manifest_path)
    return _validate_dataset_record_content(
        dataset_root,
        contract,
        records,
        expected_splits=frozenset({"train", "val", "test"}),
        count_context="Training dataset",
    )


# ADD 2026-08-28: C4 experiment가 sealed test content를 열지 않고 train/val만 검증한다.
def validate_experiment_dataset(
    dataset_root: Path,
    contract: YoloDatasetContract,
    *,
    content_splits: frozenset[str] = EXPERIMENT_CONTENT_SPLITS,
) -> tuple[DerivedManifestRecord, ...]:
    """Validate selected content; excluded CSV rows stop after lexical split gating."""
    if not content_splits or not content_splits.issubset(EXPERIMENT_CONTENT_SPLITS):
        raise ValueError("C4 experiment content splits must be a non-empty train/val subset.")
    manifest_path = _validate_dataset_package_metadata(dataset_root, contract)
    records = read_derived_manifest(manifest_path, allowed_splits=set(content_splits))
    _validate_dataset_record_content(
        dataset_root,
        contract,
        records,
        expected_splits=content_splits,
        count_context="C4 experiment dataset",
    )
    return tuple(records)


# ADD 2026-09-01: Explicit C4-4 unlock 뒤 derived test content만 strict contract로 검증한다.
def validate_final_test_dataset(
    dataset_root: Path,
    contract: YoloDatasetContract,
) -> tuple[DerivedManifestRecord, ...]:
    """Validate only test rows after the guarded final-test boundary has opened."""
    manifest_path = _validate_dataset_package_metadata(dataset_root, contract)
    records = read_derived_manifest(manifest_path, allowed_splits={"test"})
    _validate_dataset_record_content(
        dataset_root,
        contract,
        records,
        expected_splits=frozenset({"test"}),
        count_context="C4-4 final-test dataset",
    )
    return tuple(records)


# ADD 2026-08-25: Project-owned checkpoint와 metadata를 overwrite 없이 저장한다.
def write_yolo_artifact(
    *,
    source_checkpoint: Path,
    artifact_dir: Path,
    metadata: YoloArtifactMetadata,
) -> YoloArtifactMetadata:
    """Copy the portable best checkpoint and bind its digest to validated metadata."""
    if artifact_dir.exists():
        raise FileExistsError(f"YOLO artifact directory already exists: {artifact_dir}")
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"Ultralytics best checkpoint not found: {source_checkpoint}")
    metadata.validate()
    artifact_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = artifact_dir / MODEL_FILENAME
    shutil.copyfile(source_checkpoint, checkpoint_path)
    actual_metadata = YoloArtifactMetadata(
        **{**asdict(metadata), "checkpoint_sha256": sha256_file(checkpoint_path)}
    )
    actual_metadata.validate()
    (artifact_dir / METADATA_FILENAME).write_text(
        json.dumps(actual_metadata.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return actual_metadata


# ADD 2026-08-25: Artifact layout, metadata schema와 checkpoint digest를 검증한다.
def validate_yolo_artifact(
    artifact_dir: Path,
    *,
    expected_contract: YoloDatasetContract | None = None,
) -> YoloArtifactMetadata:
    """Validate a fixed model artifact before evaluation or future serving."""
    checkpoint_path = artifact_dir / MODEL_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("YOLO artifact requires model.pt and metadata.json.")
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read YOLO artifact metadata JSON.") from exc
    metadata = YoloArtifactMetadata.from_json_dict(raw)
    if sha256_file(checkpoint_path) != metadata.checkpoint_sha256:
        raise ValueError("YOLO checkpoint SHA does not match artifact metadata.")
    if expected_contract is not None and (
        metadata.dataset_manifest_sha256 != expected_contract.manifest_sha256
        or metadata.dataset_semantic_fingerprint_sha256
        != expected_contract.semantic_fingerprint_sha256
        or metadata.classes != expected_contract.classes
        or metadata.category != expected_contract.category
    ):
        raise ValueError("YOLO artifact dataset lineage does not match evaluation config.")
    return metadata
