"""Validation-only PyTorch and ONNX Runtime parity evidence for frozen YOLO."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.deployment.yolo_onnx import (
    EXPECTED_CLASSES,
    FrozenYoloSource,
    YoloOnnxExportConfig,
    YoloOnnxExportMetadata,
    load_yolo_onnx_artifact,
)
from ml.evaluation.final_benchmark import RepositoryProvenance, resolve_repository_provenance
from ml.evaluation.yolo_confirmation_prediction import predict_c4_2c_instances
from ml.evaluation.yolo_segmentation_error_analysis import (
    PredictedInstance,
    box_iou,
    filter_predictions,
    mask_overlap,
)
from ml.experiments.yolo_final_candidate import materialize_official_candidate_artifact
from ml.training.yolo_segmentation import (
    YoloDatasetContract,
    validate_artifact_id,
    validate_experiment_dataset,
)
from shared.hashing import is_sha256_digest, sha256_file

PARITY_SCHEMA_VERSION = 1
PARITY_STATE = "METRICS_COLLECTED_ACCEPTANCE_PENDING"
PARITY_FILENAME = "parity.json"
PARITY_OUTPUT_ROOT = Path("outputs/deployment/yolo_segmentation/onnx_parity")


@dataclass(frozen=True)
class RuntimeTensorObservation:
    """One finite post-backend tensor exposed by the shared Ultralytics result."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    finite: bool

    # ADD 2026-09-02: Backend output tensor의 name/shape/finite contract를 검증한다.
    def validate(self) -> None:
        if not self.name or not self.dtype or any(value < 0 for value in self.shape):
            raise ValueError("Parity tensor observation contains invalid metadata.")
        if self.finite is not True:
            raise ValueError("Parity backend output contains NaN or Inf.")


@dataclass(frozen=True)
class BackendPrediction:
    """Normalized instances plus observed backend result tensors for one image."""

    instances: tuple[PredictedInstance, ...]
    tensors: tuple[RuntimeTensorObservation, ...]


@dataclass(frozen=True)
class NormalizedPredictionObservation:
    """Compact class/confidence/box/mask identity for one normalized prediction."""

    prediction_index: int
    class_id: int
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    mask_shape: tuple[int, int]
    mask_foreground_pixels: int
    mask_sha256: str

    # ADD 2026-09-02: Per-instance parity observation의 numeric/mask identity를 검증한다.
    def validate(self) -> None:
        if self.prediction_index < 0 or self.class_id not in EXPECTED_CLASSES:
            raise ValueError("Parity prediction observation has an invalid index or class.")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Parity prediction observation has invalid confidence.")
        if len(self.box_xyxy) != 4 or not all(math.isfinite(value) for value in self.box_xyxy):
            raise ValueError("Parity prediction observation has invalid box geometry.")
        if (
            len(self.mask_shape) != 2
            or any(value <= 0 for value in self.mask_shape)
            or self.mask_foreground_pixels <= 0
            or not is_sha256_digest(self.mask_sha256)
        ):
            raise ValueError("Parity prediction observation has invalid mask identity.")


@dataclass(frozen=True)
class ParityInstanceMatch:
    """One deterministic class-neutral spatial match used to measure class agreement."""

    pytorch_index: int
    onnx_index: int
    class_agreement: bool
    confidence_abs_error: float
    box_iou: float
    mask_iou: float

    # ADD 2026-09-02: Matched-instance metrics가 finite probability 범위인지 검증한다.
    def validate(self) -> None:
        values = (self.confidence_abs_error, self.box_iou, self.mask_iou)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Parity match metrics must be finite.")
        if self.confidence_abs_error < 0.0 or not 0.0 <= self.box_iou <= 1.0:
            raise ValueError("Parity confidence/box metrics are out of range.")
        if not 0.0 < self.mask_iou <= 1.0:
            raise ValueError("Parity matching requires positive mask overlap.")


@dataclass(frozen=True)
class SampleParityEvidence:
    """Per-validation-image backend integrity and postprocessed parity evidence."""

    sample_id: str
    split: str
    pytorch_prediction_count: int
    onnx_prediction_count: int
    matched_instance_count: int
    unmatched_pytorch_count: int
    unmatched_onnx_count: int
    pytorch_tensors: tuple[RuntimeTensorObservation, ...]
    onnx_tensors: tuple[RuntimeTensorObservation, ...]
    pytorch_predictions: tuple[NormalizedPredictionObservation, ...]
    onnx_predictions: tuple[NormalizedPredictionObservation, ...]
    matches: tuple[ParityInstanceMatch, ...]

    # ADD 2026-09-02: Sample evidence의 validation split과 count conservation을 검증한다.
    def validate(self) -> None:
        if not self.sample_id or self.split != "val":
            raise ValueError("C5 parity sample must be a named validation row.")
        if (
            min(
                self.pytorch_prediction_count,
                self.onnx_prediction_count,
                self.matched_instance_count,
                self.unmatched_pytorch_count,
                self.unmatched_onnx_count,
            )
            < 0
        ):
            raise ValueError("C5 parity sample counts must be non-negative.")
        if self.matched_instance_count != len(self.matches):
            raise ValueError("C5 parity matched count does not match evidence.")
        if self.pytorch_prediction_count != len(
            self.pytorch_predictions
        ) or self.onnx_prediction_count != len(self.onnx_predictions):
            raise ValueError("C5 parity prediction observations do not match counts.")
        if (
            self.matched_instance_count + self.unmatched_pytorch_count
            != self.pytorch_prediction_count
            or self.matched_instance_count + self.unmatched_onnx_count != self.onnx_prediction_count
        ):
            raise ValueError("C5 parity prediction counts are not conserved.")
        for tensor in (*self.pytorch_tensors, *self.onnx_tensors):
            tensor.validate()
        for prediction in (*self.pytorch_predictions, *self.onnx_predictions):
            prediction.validate()
        for match in self.matches:
            match.validate()


@dataclass(frozen=True)
class YoloOnnxParityEvidence:
    """Deterministic validation-only equivalence metrics without an invented numeric gate."""

    schema_version: int
    parity_id: str
    state: str
    created_at: str
    source_experiment_id: str
    frozen_manifest_sha256: str
    source_model_sha256: str
    onnx_sha256: str
    export_config_sha256: str
    split: str
    test_used: bool
    test_split_used: bool
    sample_count: int
    pytorch_prediction_count: int
    onnx_prediction_count: int
    matched_instance_count: int
    unmatched_pytorch_count: int
    unmatched_onnx_count: int
    class_agreement_count: int
    class_agreement_rate: float | None
    confidence_abs_error: Mapping[str, float | int | None]
    box_iou: Mapping[str, float | int | None]
    mask_iou: Mapping[str, float | int | None]
    structural_gates_passed: bool
    numeric_acceptance: str
    samples: tuple[SampleParityEvidence, ...]
    environment: Mapping[str, str]
    repository: Mapping[str, str | bool]

    # ADD 2026-09-02: Parity evidence가 validation-only metrics-first policy를 보존하는지 검증한다.
    def validate(self) -> None:
        validate_artifact_id(self.parity_id)
        if (
            self.schema_version != PARITY_SCHEMA_VERSION
            or self.state != PARITY_STATE
            or self.split != "val"
            or self.test_used is not False
            or self.test_split_used is not False
        ):
            raise ValueError("C5-2 parity lifecycle or test seal is invalid.")
        if self.numeric_acceptance != "PENDING_APPROVED_TOLERANCES":
            raise ValueError("C5-2 must not claim numeric acceptance before tolerance approval.")
        if self.structural_gates_passed is not True:
            raise ValueError("C5-2 structural gates must pass before evidence publication.")
        _validate_timestamp(self.created_at)
        if not self.source_experiment_id:
            raise ValueError("C5-2 source experiment identity is missing.")
        for digest in (
            self.frozen_manifest_sha256,
            self.source_model_sha256,
            self.onnx_sha256,
            self.export_config_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C5-2 parity evidence contains an invalid SHA-256.")
        if self.sample_count != len(self.samples) or self.sample_count <= 0:
            raise ValueError("C5-2 parity evidence requires non-empty validation samples.")
        if set(self.repository) != {"git_commit", "working_tree_dirty"}:
            raise ValueError("C5-2 repository provenance fields are invalid.")
        if type(self.repository["working_tree_dirty"]) is not bool:
            raise ValueError("C5-2 working_tree_dirty must be boolean.")
        repository = RepositoryProvenance(
            git_commit=str(self.repository["git_commit"]),
            working_tree_dirty=bool(self.repository["working_tree_dirty"]),
        )
        repository.validate()
        if repository.working_tree_dirty:
            raise ValueError("Official C5-2 parity requires a clean repository state.")
        required_environment = {
            "python_version",
            "platform",
            "python_implementation",
            "torch_version",
            "ultralytics_version",
            "onnxruntime_version",
            "pytorch_device",
            "onnxruntime_provider",
        }
        if set(self.environment) != required_environment or any(
            not isinstance(value, str) or not value for value in self.environment.values()
        ):
            raise ValueError("C5-2 parity environment fields are invalid.")
        totals = {
            "pytorch_prediction_count": sum(item.pytorch_prediction_count for item in self.samples),
            "onnx_prediction_count": sum(item.onnx_prediction_count for item in self.samples),
            "matched_instance_count": sum(item.matched_instance_count for item in self.samples),
            "unmatched_pytorch_count": sum(item.unmatched_pytorch_count for item in self.samples),
            "unmatched_onnx_count": sum(item.unmatched_onnx_count for item in self.samples),
        }
        if any(getattr(self, name) != value for name, value in totals.items()):
            raise ValueError("C5-2 aggregate counts do not match sample evidence.")
        expected_class_agreements = sum(
            match.class_agreement for item in self.samples for match in item.matches
        )
        if self.class_agreement_count != expected_class_agreements:
            raise ValueError("C5-2 class agreement count does not match sample evidence.")
        expected_rate = (
            expected_class_agreements / self.matched_instance_count
            if self.matched_instance_count
            else None
        )
        if self.class_agreement_rate != expected_rate:
            raise ValueError("C5-2 class agreement rate is inconsistent.")
        for sample in self.samples:
            sample.validate()
        try:
            json.dumps(asdict(self), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("C5-2 parity evidence must be strict JSON data.") from exc

    # ADD 2026-09-02: Parity evidence를 deterministic strict JSON bytes로 직렬화한다.
    def to_json_bytes(self) -> bytes:
        self.validate()
        return (json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


class PredictionModel(Protocol):
    """Minimal Ultralytics prediction and initialized predictor surface."""

    predictor: object
    names: Mapping[int, str]

    def predict(self, **kwargs: object) -> Sequence[object]: ...


class _CapturingModel:
    """Transparent predictor wrapper retaining the one shared result for inspection."""

    def __init__(self, model: PredictionModel) -> None:
        self.model = model
        self.result: object | None = None

    def predict(self, **kwargs: object) -> Sequence[object]:
        results = list(self.model.predict(**kwargs))
        self.result = results[0] if len(results) == 1 else None
        return results


type ProvenanceResolver = Callable[[Path], RepositoryProvenance]


# ADD 2026-09-02: Parity evidence timestamp가 timezone-aware ISO-8601인지 검증한다.
def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("C5-2 timestamp must be ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("C5-2 timestamp must include a timezone offset.")


# ADD 2026-09-02: Numeric-like backend value를 ndarray로 복원하고 finite 여부를 기록한다.
def observe_runtime_tensor(name: str, value: object) -> RuntimeTensorObservation:
    detached = getattr(value, "detach", None)
    materialized = detached() if callable(detached) else value
    cpu = getattr(materialized, "cpu", None)
    materialized = cpu() if callable(cpu) else materialized
    numpy_method = getattr(materialized, "numpy", None)
    array = np.asarray(numpy_method() if callable(numpy_method) else materialized)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Parity backend tensor is not numeric: {name}")
    observation = RuntimeTensorObservation(
        name=name,
        dtype=str(array.dtype),
        shape=tuple(int(value) for value in array.shape),
        finite=bool(np.isfinite(array).all()),
    )
    observation.validate()
    return observation


# ADD 2026-09-02: Shared Ultralytics result에서 boxes/classes/confidences/masks shape를 수집한다.
def _observe_result_tensors(result: object) -> tuple[RuntimeTensorObservation, ...]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        raise ValueError("Parity backend result is missing boxes.")
    values = (
        ("boxes.xyxy", getattr(boxes, "xyxy", None)),
        ("boxes.cls", getattr(boxes, "cls", None)),
        ("boxes.conf", getattr(boxes, "conf", None)),
    )
    observations = [
        observe_runtime_tensor(name, value) for name, value in values if value is not None
    ]
    masks = getattr(result, "masks", None)
    if masks is not None:
        observations.append(observe_runtime_tensor("masks.data", getattr(masks, "data", None)))
    if len(observations) != 3 and len(observations) != 4:
        raise ValueError("Parity backend result tensor schema is incomplete.")
    return tuple(observations)


# ADD 2026-09-02: C4-2C normalization을 그대로 사용해 한 backend prediction을 관측한다.
def predict_backend(
    *,
    model: PredictionModel,
    record: DerivedManifestRecord,
    dataset_root: Path,
    imgsz: int,
    diagnostic_confidence: float = 0.25,
) -> BackendPrediction:
    if record.derived_split != "val":
        raise ValueError("C5 parity prediction accepts validation rows only.")
    source_path = (dataset_root / record.image_path).resolve()
    capturing = _CapturingModel(model)
    raw_instances = predict_c4_2c_instances(
        model=capturing,
        source_image_path=source_path,
        image_width=record.image_width,
        image_height=record.image_height,
        imgsz=imgsz,
        device="cpu",
        valid_class_ids=set(EXPECTED_CLASSES),
    )
    if capturing.result is None:
        raise ValueError("C5 parity backend did not return exactly one result.")
    instances = filter_predictions({record.sample_id: raw_instances}, diagnostic_confidence)[
        record.sample_id
    ]
    _validate_instances(instances, record=record)
    return BackendPrediction(
        instances=instances,
        tensors=_observe_result_tensors(capturing.result),
    )


# ADD 2026-09-02: Normalized instance의 class/confidence/box/mask integrity를 검증한다.
def _validate_instances(
    instances: tuple[PredictedInstance, ...],
    *,
    record: DerivedManifestRecord,
) -> None:
    for instance in instances:
        if instance.class_id not in EXPECTED_CLASSES or not math.isfinite(instance.confidence):
            raise ValueError("Parity prediction contains invalid class or confidence.")
        if not 0.0 <= instance.confidence <= 1.0:
            raise ValueError("Parity prediction confidence is outside [0, 1].")
        if instance.mask.dtype != np.bool_ or instance.mask.shape != (
            record.image_height,
            record.image_width,
        ):
            raise ValueError("Parity prediction mask is not source-size boolean data.")
        if not instance.mask.any() or not all(math.isfinite(value) for value in instance.box_xyxy):
            raise ValueError("Parity prediction mask/box geometry is invalid.")


# ADD 2026-09-02: Normalized prediction을 compact numeric/geometry/mask-hash evidence로 만든다.
def _prediction_observations(
    instances: tuple[PredictedInstance, ...],
) -> tuple[NormalizedPredictionObservation, ...]:
    observations = tuple(
        NormalizedPredictionObservation(
            prediction_index=index,
            class_id=instance.class_id,
            confidence=instance.confidence,
            box_xyxy=instance.box_xyxy,
            mask_shape=(int(instance.mask.shape[0]), int(instance.mask.shape[1])),
            mask_foreground_pixels=int(np.count_nonzero(instance.mask)),
            mask_sha256=hashlib.sha256(instance.mask.tobytes(order="C")).hexdigest(),
        )
        for index, instance in enumerate(instances)
    )
    for observation in observations:
        observation.validate()
    return observations


# ADD 2026-09-02: Positive mask overlap을 maximum-first로 소비해 backend instances를 대응한다.
def match_backend_predictions(
    pytorch: tuple[PredictedInstance, ...],
    onnx: tuple[PredictedInstance, ...],
) -> tuple[ParityInstanceMatch, ...]:
    candidates: list[ParityInstanceMatch] = []
    for pytorch_index, left in enumerate(pytorch):
        for onnx_index, right in enumerate(onnx):
            overlap, _, _ = mask_overlap(left.mask, right.mask)
            if overlap <= 0.0:
                continue
            candidates.append(
                ParityInstanceMatch(
                    pytorch_index=pytorch_index,
                    onnx_index=onnx_index,
                    class_agreement=left.class_id == right.class_id,
                    confidence_abs_error=abs(left.confidence - right.confidence),
                    box_iou=box_iou(left.box_xyxy, right.box_xyxy),
                    mask_iou=overlap,
                )
            )
    selected: list[ParityInstanceMatch] = []
    used_pytorch: set[int] = set()
    used_onnx: set[int] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (-item.mask_iou, item.pytorch_index, item.onnx_index),
    ):
        if candidate.pytorch_index in used_pytorch or candidate.onnx_index in used_onnx:
            continue
        candidate.validate()
        selected.append(candidate)
        used_pytorch.add(candidate.pytorch_index)
        used_onnx.add(candidate.onnx_index)
    return tuple(sorted(selected, key=lambda item: (item.pytorch_index, item.onnx_index)))


# ADD 2026-09-02: One validation sample의 count-preserving parity evidence를 만든다.
def build_sample_parity(
    *,
    record: DerivedManifestRecord,
    pytorch: BackendPrediction,
    onnx: BackendPrediction,
) -> SampleParityEvidence:
    if record.derived_split != "val":
        raise ValueError("C5 parity evidence rejects non-validation rows.")
    matches = match_backend_predictions(pytorch.instances, onnx.instances)
    evidence = SampleParityEvidence(
        sample_id=record.sample_id,
        split=record.derived_split,
        pytorch_prediction_count=len(pytorch.instances),
        onnx_prediction_count=len(onnx.instances),
        matched_instance_count=len(matches),
        unmatched_pytorch_count=len(pytorch.instances) - len(matches),
        unmatched_onnx_count=len(onnx.instances) - len(matches),
        pytorch_tensors=pytorch.tensors,
        onnx_tensors=onnx.tensors,
        pytorch_predictions=_prediction_observations(pytorch.instances),
        onnx_predictions=_prediction_observations(onnx.instances),
        matches=matches,
    )
    evidence.validate()
    return evidence


# ADD 2026-09-02: Empty-safe parity metric distribution을 deterministic mapping으로 만든다.
def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Parity metric distribution contains NaN or Inf.")
    return {
        "count": len(values),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


# ADD 2026-09-02: Sample evidence를 metrics-only aggregate contract로 결합한다.
def build_parity_evidence(
    *,
    parity_id: str,
    created_at: str,
    source: FrozenYoloSource,
    export_metadata: YoloOnnxExportMetadata,
    samples: tuple[SampleParityEvidence, ...],
    provenance: RepositoryProvenance,
) -> YoloOnnxParityEvidence:
    matches = [match for sample in samples for match in sample.matches]
    matched_count = len(matches)
    class_agreement_count = sum(match.class_agreement for match in matches)
    evidence = YoloOnnxParityEvidence(
        schema_version=PARITY_SCHEMA_VERSION,
        parity_id=parity_id,
        state=PARITY_STATE,
        created_at=created_at,
        source_experiment_id=source.candidate.selected_experiment_id,
        frozen_manifest_sha256=source.manifest_sha256,
        source_model_sha256=source.candidate.model_sha256,
        onnx_sha256=export_metadata.onnx_sha256,
        export_config_sha256=export_metadata.export_config_sha256,
        split="val",
        test_used=False,
        test_split_used=False,
        sample_count=len(samples),
        pytorch_prediction_count=sum(item.pytorch_prediction_count for item in samples),
        onnx_prediction_count=sum(item.onnx_prediction_count for item in samples),
        matched_instance_count=matched_count,
        unmatched_pytorch_count=sum(item.unmatched_pytorch_count for item in samples),
        unmatched_onnx_count=sum(item.unmatched_onnx_count for item in samples),
        class_agreement_count=class_agreement_count,
        class_agreement_rate=(class_agreement_count / matched_count if matched_count else None),
        confidence_abs_error=_distribution([item.confidence_abs_error for item in matches]),
        box_iou=_distribution([item.box_iou for item in matches]),
        mask_iou=_distribution([item.mask_iou for item in matches]),
        structural_gates_passed=True,
        numeric_acceptance="PENDING_APPROVED_TOLERANCES",
        samples=samples,
        environment=_parity_environment(),
        repository=provenance.to_json_dict(),
    )
    evidence.validate()
    return evidence


# ADD 2026-09-02: Frozen source와 ONNX metadata identity를 parity 실행 전에 교차 검증한다.
def verify_parity_artifact_identity(
    *,
    source: FrozenYoloSource,
    config: YoloOnnxExportConfig,
    metadata: YoloOnnxExportMetadata,
) -> None:
    expected = {
        "export_id": config.export_id,
        "source_experiment_id": source.candidate.selected_experiment_id,
        "frozen_manifest_sha256": source.manifest_sha256,
        "official_package_sha256": source.candidate.official_package_sha256,
        "source_model_sha256": source.candidate.model_sha256,
        "source_metadata_sha256": source.candidate.metadata_sha256,
        "dataset_manifest_sha256": source.candidate.dataset_manifest_sha256,
        "export_config_sha256": sha256_file(config.config_path),
        "test_used": False,
        "test_split_used": False,
    }
    mismatches = [name for name, value in expected.items() if getattr(metadata, name) != value]
    if mismatches:
        raise ValueError("ONNX artifact does not match frozen source: " + ", ".join(mismatches))


# ADD 2026-09-02: Test rows를 materialize/open하지 않고 validation content만 검증한다.
def load_parity_validation_records(
    dataset_root: Path,
    contract: YoloDatasetContract,
) -> tuple[DerivedManifestRecord, ...]:
    records = validate_experiment_dataset(
        dataset_root,
        contract,
        content_splits=frozenset({"val"}),
    )
    if not records or any(record.derived_split != "val" for record in records):
        raise ValueError("C5-2 parity requires non-empty validation-only records.")
    return records


# ADD 2026-09-02: Ultralytics backend가 requested PyTorch/ONNX Runtime engine인지 확인한다.
def _load_backend(path: Path, *, expected_backend: str) -> PredictionModel:
    from ultralytics import YOLO

    model = YOLO(str(path), task="segment")
    if model.task != "segment":
        raise ValueError("C5-2 backend model is not YOLO segmentation.")
    if expected_backend not in {"pytorch", "onnxruntime"}:
        raise ValueError("C5-2 backend selector is invalid.")
    return cast(PredictionModel, model)


# ADD 2026-09-02: First prediction 후 Ultralytics AutoBackend engine identity를 검증한다.
def _verify_loaded_backend(model: PredictionModel, *, expected_backend: str) -> None:
    names = {int(key): str(value) for key, value in model.names.items()}
    if names != EXPECTED_CLASSES:
        raise RuntimeError("C5-2 backend classes changed from bent/color/scratch.")
    backend = getattr(getattr(model, "predictor", None), "model", None)
    if backend is None:
        raise RuntimeError("C5-2 Ultralytics predictor backend was not initialized.")
    if expected_backend == "pytorch":
        if getattr(backend, "format", None) != "pt":
            raise RuntimeError("C5-2 source execution did not use the PyTorch backend.")
        return
    session = getattr(backend, "session", None)
    if getattr(backend, "format", None) != "onnx" or session is None:
        raise RuntimeError("C5-2 exported execution did not use ONNX Runtime.")
    if type(session).__module__.split(".")[0] != "onnxruntime":
        raise RuntimeError("C5-2 ONNX session is not provided by ONNX Runtime.")
    get_providers = getattr(session, "get_providers", None)
    if not callable(get_providers) or get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("C5-2 ONNX Runtime parity must use CPUExecutionProvider only.")


# ADD 2026-09-02: Validation rows only로 exact PyTorch와 ONNX Runtime parity evidence를 생성한다.
def evaluate_frozen_yolo_onnx_parity(
    *,
    source: FrozenYoloSource,
    config: YoloOnnxExportConfig,
    onnx_artifact_dir: Path,
    dataset_root: Path,
    parity_id: str,
    created_at: str,
    provenance_resolver: ProvenanceResolver = resolve_repository_provenance,
) -> Path:
    config.validate()
    validate_artifact_id(parity_id)
    try:
        config.config_path.resolve().relative_to(source.repository_root.resolve())
    except ValueError as exc:
        raise ValueError("C5-2 export config must remain inside repository_root.") from exc
    expected_onnx_dir = (source.repository_root / config.output_root / config.export_id).resolve()
    if onnx_artifact_dir.resolve() != expected_onnx_dir:
        raise ValueError("C5-2 ONNX artifact must use the repository ignored export namespace.")
    metadata = load_yolo_onnx_artifact(onnx_artifact_dir)
    verify_parity_artifact_identity(source=source, config=config, metadata=metadata)
    provenance = provenance_resolver(source.repository_root)
    provenance.validate()
    if provenance.working_tree_dirty:
        raise ValueError("Official C5-2 parity requires a clean committed repository state.")

    # Test rows are lexically gated before typed materialization; only validation content is opened.
    records = load_parity_validation_records(dataset_root, source.baseline.dataset_contract)

    output_dir = source.repository_root / PARITY_OUTPUT_ROOT / parity_id
    staging_dir = output_dir.parent / f".{parity_id}.staging"
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError("C5-2 parity output namespace already exists.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="smartfactory-c5-parity-") as temporary:
        artifact_dir = Path(temporary) / "source-artifact"
        materialize_official_candidate_artifact(
            package_path=source.package_path,
            candidate=source.candidate,
            evidence=source.evidence,
            artifact_dir=artifact_dir,
        )
        source_model = artifact_dir / "model" / "model.pt"
        if sha256_file(source_model) != source.candidate.model_sha256:
            raise RuntimeError("C5-2 materialized source model changed frozen identity.")
        pytorch_model = _load_backend(source_model, expected_backend="pytorch")
        onnx_model = _load_backend(onnx_artifact_dir / "model.onnx", expected_backend="onnxruntime")
        sample_evidence: list[SampleParityEvidence] = []
        for index, record in enumerate(records):
            pytorch_prediction = predict_backend(
                model=pytorch_model,
                record=record,
                dataset_root=dataset_root,
                imgsz=config.imgsz,
                diagnostic_confidence=config.parity.diagnostic_confidence,
            )
            onnx_prediction = predict_backend(
                model=onnx_model,
                record=record,
                dataset_root=dataset_root,
                imgsz=config.imgsz,
                diagnostic_confidence=config.parity.diagnostic_confidence,
            )
            if index == 0:
                _verify_loaded_backend(pytorch_model, expected_backend="pytorch")
                _verify_loaded_backend(onnx_model, expected_backend="onnxruntime")
            sample_evidence.append(
                build_sample_parity(
                    record=record,
                    pytorch=pytorch_prediction,
                    onnx=onnx_prediction,
                )
            )
        evidence = build_parity_evidence(
            parity_id=parity_id,
            created_at=created_at,
            source=source,
            export_metadata=metadata,
            samples=tuple(sample_evidence),
            provenance=provenance,
        )
        staging_dir.mkdir(exist_ok=False)
        try:
            (staging_dir / PARITY_FILENAME).write_bytes(evidence.to_json_bytes())
            if sha256_file(source_model) != source.candidate.model_sha256:
                raise RuntimeError("C5-2 parity execution changed frozen source bytes.")
            if sha256_file(onnx_artifact_dir / "model.onnx") != metadata.onnx_sha256:
                raise RuntimeError("C5-2 parity execution changed ONNX bytes.")
            staging_dir.rename(output_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
    return output_dir / PARITY_FILENAME


# ADD 2026-09-02: Parity runtime의 dependency/backend version을 evidence에 기록한다.
def _parity_environment() -> dict[str, str]:
    import onnxruntime  # type: ignore[import-untyped]
    import torch
    from ultralytics import __version__ as ultralytics_version

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "python_implementation": sys.implementation.name,
        "torch_version": str(torch.__version__),
        "ultralytics_version": ultralytics_version,
        "onnxruntime_version": str(onnxruntime.__version__),
        "pytorch_device": "cpu",
        "onnxruntime_provider": "CPUExecutionProvider",
    }
