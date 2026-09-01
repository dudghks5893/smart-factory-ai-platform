"""Validation-only PyTorch FP32 versus TensorRT FP16 characterization evidence."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import platform
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch

from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.deployment.yolo_onnx import (
    EXPECTED_CLASSES,
    FrozenYoloSource,
)
from ml.deployment.yolo_onnx_parity import (
    BackendPrediction,
    NormalizedPredictionObservation,
    ParityInstanceMatch,
    RuntimeTensorObservation,
    load_parity_validation_records,
    match_backend_predictions,
    observe_runtime_tensor,
)
from ml.deployment.yolo_tensorrt import (
    EXPECTED_ONNX_ARTIFACT_ROOT,
    EXPECTED_ONNX_EXPORT_ID,
    TENSORRT_ENGINE_FILENAME,
    YoloTensorRtExportConfig,
    YoloTensorRtExportMetadata,
    load_yolo_tensorrt_artifact,
    verify_tensorrt_source_onnx_identity,
)
from ml.evaluation.final_benchmark import RepositoryProvenance, resolve_repository_provenance
from ml.evaluation.yolo_confirmation_prediction import predict_c4_2c_instances
from ml.evaluation.yolo_segmentation_error_analysis import (
    PredictedInstance,
    filter_predictions,
)
from ml.experiments.yolo_final_candidate import materialize_official_candidate_artifact
from ml.training.yolo_segmentation import validate_artifact_id
from shared.hashing import is_sha256_digest, sha256_file

TENSORRT_PARITY_SCHEMA_VERSION = 1
TENSORRT_PARITY_STATE = "TENSORRT_FP16_METRICS_COLLECTED_ACCEPTANCE_PENDING"
TENSORRT_PARITY_FILENAME = "parity.json"
TENSORRT_PARITY_OUTPUT_ROOT = Path("outputs/deployment/yolo_segmentation/tensorrt_parity")


@dataclass(frozen=True)
class TensorRtSampleParityEvidence:
    """Per-validation-image PyTorch versus TensorRT normalized prediction evidence."""

    sample_id: str
    split: str
    pytorch_prediction_count: int
    tensorrt_prediction_count: int
    matched_instance_count: int
    unmatched_pytorch_count: int
    unmatched_tensorrt_count: int
    pytorch_tensors: tuple[RuntimeTensorObservation, ...]
    tensorrt_tensors: tuple[RuntimeTensorObservation, ...]
    pytorch_predictions: tuple[NormalizedPredictionObservation, ...]
    tensorrt_predictions: tuple[NormalizedPredictionObservation, ...]
    matches: tuple[ParityInstanceMatch, ...]

    # ADD 2026-09-02: FP16 sample evidence의 validation split과 count conservation을 검증한다.
    def validate(self) -> None:
        if not self.sample_id or self.split != "val":
            raise ValueError("C5-3 parity sample must be a named validation row.")
        if (
            min(
                self.pytorch_prediction_count,
                self.tensorrt_prediction_count,
                self.matched_instance_count,
                self.unmatched_pytorch_count,
                self.unmatched_tensorrt_count,
            )
            < 0
        ):
            raise ValueError("C5-3 parity sample counts must be non-negative.")
        if self.matched_instance_count != len(self.matches):
            raise ValueError("C5-3 matched count does not match sample evidence.")
        if self.pytorch_prediction_count != len(
            self.pytorch_predictions
        ) or self.tensorrt_prediction_count != len(self.tensorrt_predictions):
            raise ValueError("C5-3 prediction observations do not match counts.")
        if (
            self.matched_instance_count + self.unmatched_pytorch_count
            != self.pytorch_prediction_count
            or self.matched_instance_count + self.unmatched_tensorrt_count
            != self.tensorrt_prediction_count
        ):
            raise ValueError("C5-3 sample prediction counts are not conserved.")
        for tensor in (*self.pytorch_tensors, *self.tensorrt_tensors):
            tensor.validate()
        for prediction in (*self.pytorch_predictions, *self.tensorrt_predictions):
            prediction.validate()
        for match in self.matches:
            match.validate()


@dataclass(frozen=True)
class TensorRtLatencyBenchmark:
    """Characterization-only end-to-end single-image runtime comparison."""

    scope: str
    sample_id: str
    warmup_iterations: int
    measured_iterations: int
    pytorch_latency_ms: Mapping[str, float | int]
    tensorrt_latency_ms: Mapping[str, float | int]
    speedup_ratio: float

    # ADD 2026-09-02: Benchmark metrics가 finite positive observation인지 검증한다.
    def validate(self) -> None:
        if (
            self.scope != "ultralytics_end_to_end_single_image"
            or not self.sample_id
            or self.warmup_iterations != 10
            or self.measured_iterations != 50
        ):
            raise ValueError("C5-3 latency benchmark metadata changed from the repository policy.")
        _validate_latency_distribution(self.pytorch_latency_ms, expected_count=50)
        _validate_latency_distribution(self.tensorrt_latency_ms, expected_count=50)
        if not math.isfinite(self.speedup_ratio) or self.speedup_ratio <= 0.0:
            raise ValueError("C5-3 TensorRT speedup observation must be finite and positive.")


@dataclass(frozen=True)
class YoloTensorRtParityEvidence:
    """Metrics-first TensorRT FP16 characterization without predeclared acceptance thresholds."""

    schema_version: int
    parity_id: str
    state: str
    created_at: str
    source_experiment_id: str
    frozen_manifest_sha256: str
    source_model_sha256: str
    source_onnx_sha256: str
    engine_sha256: str
    tensorrt_config_sha256: str
    split: str
    test_used: bool
    test_split_used: bool
    sample_count: int
    pytorch_prediction_count: int
    tensorrt_prediction_count: int
    matched_instance_count: int
    unmatched_pytorch_count: int
    unmatched_tensorrt_count: int
    class_agreement_count: int
    class_agreement_rate: float | None
    confidence_abs_error: Mapping[str, float | int | None]
    box_iou: Mapping[str, float | int | None]
    mask_iou: Mapping[str, float | int | None]
    latency: TensorRtLatencyBenchmark
    structural_gates_passed: bool
    numeric_acceptance: str
    samples: tuple[TensorRtSampleParityEvidence, ...]
    environment: Mapping[str, str]
    repository: Mapping[str, str | bool]

    # ADD 2026-09-02: C5-3 evidence가 validation-only characterization을 보존하는지 검증한다.
    def validate(self) -> None:
        validate_artifact_id(self.parity_id)
        if (
            self.schema_version != TENSORRT_PARITY_SCHEMA_VERSION
            or self.state != TENSORRT_PARITY_STATE
            or self.split != "val"
            or self.test_used is not False
            or self.test_split_used is not False
        ):
            raise ValueError("C5-3 TensorRT parity lifecycle or test seal is invalid.")
        if self.numeric_acceptance != "PENDING_TENSORRT_FP16_TOLERANCE_APPROVAL":
            raise ValueError("C5-3 must not claim FP16 numeric acceptance before policy approval.")
        if self.structural_gates_passed is not True:
            raise ValueError("C5-3 structural gates must pass before evidence publication.")
        _validate_timestamp(self.created_at)
        for digest in (
            self.frozen_manifest_sha256,
            self.source_model_sha256,
            self.source_onnx_sha256,
            self.engine_sha256,
            self.tensorrt_config_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C5-3 TensorRT parity evidence contains invalid SHA-256.")
        if (
            not self.source_experiment_id
            or self.sample_count != len(self.samples)
            or self.sample_count <= 0
        ):
            raise ValueError("C5-3 TensorRT parity requires non-empty validation samples.")
        repository = _repository_provenance(self.repository)
        if repository.working_tree_dirty:
            raise ValueError("Official C5-3 TensorRT parity requires a clean repository.")
        required_environment = {
            "python_version",
            "platform",
            "python_implementation",
            "torch_version",
            "ultralytics_version",
            "tensorrt_version",
            "cuda_runtime_version",
            "gpu_name",
            "gpu_compute_capability",
            "pytorch_device",
            "tensorrt_device",
        }
        if set(self.environment) != required_environment or any(
            not isinstance(value, str) or not value for value in self.environment.values()
        ):
            raise ValueError("C5-3 TensorRT parity environment fields are invalid.")

        totals = {
            "pytorch_prediction_count": sum(item.pytorch_prediction_count for item in self.samples),
            "tensorrt_prediction_count": sum(
                item.tensorrt_prediction_count for item in self.samples
            ),
            "matched_instance_count": sum(item.matched_instance_count for item in self.samples),
            "unmatched_pytorch_count": sum(item.unmatched_pytorch_count for item in self.samples),
            "unmatched_tensorrt_count": sum(item.unmatched_tensorrt_count for item in self.samples),
        }
        if any(getattr(self, name) != value for name, value in totals.items()):
            raise ValueError("C5-3 aggregate counts do not match sample evidence.")
        expected_agreements = sum(
            match.class_agreement for sample in self.samples for match in sample.matches
        )
        if self.class_agreement_count != expected_agreements:
            raise ValueError("C5-3 class agreement count does not match sample evidence.")
        expected_rate = (
            expected_agreements / self.matched_instance_count
            if self.matched_instance_count
            else None
        )
        if self.class_agreement_rate != expected_rate:
            raise ValueError("C5-3 class agreement rate is inconsistent.")
        for sample in self.samples:
            sample.validate()
        self.latency.validate()
        try:
            json.dumps(asdict(self), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("C5-3 TensorRT parity evidence must be strict JSON data.") from exc

    # ADD 2026-09-02: Characterization evidence를 deterministic strict JSON bytes로 직렬화한다.
    def to_json_bytes(self) -> bytes:
        self.validate()
        return (json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


class PredictionModel(Protocol):
    """Minimal Ultralytics prediction surface used by PyTorch and TensorRT backends."""

    predictor: object
    names: Mapping[int, str]

    def predict(self, **kwargs: object) -> Sequence[object]: ...


class _CapturingModel:
    """Transparent prediction wrapper retaining the one Ultralytics result for tensor inspection."""

    def __init__(self, model: PredictionModel) -> None:
        self.model = model
        self.result: object | None = None

    def predict(self, **kwargs: object) -> Sequence[object]:
        results = list(self.model.predict(**kwargs))
        self.result = results[0] if len(results) == 1 else None
        return results


# ADD 2026-09-02: Parity evidence timestamp가 timezone-aware ISO-8601인지 검증한다.
def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("C5-3 parity timestamp must be ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("C5-3 parity timestamp must include a timezone offset.")


# ADD 2026-09-02: Repository provenance mapping을 strict typed object로 검증한다.
def _repository_provenance(raw: Mapping[str, str | bool]) -> RepositoryProvenance:
    if set(raw) != {"git_commit", "working_tree_dirty"}:
        raise ValueError("C5-3 parity repository provenance fields are invalid.")
    if type(raw["working_tree_dirty"]) is not bool:
        raise ValueError("C5-3 parity working_tree_dirty must be boolean.")
    provenance = RepositoryProvenance(
        git_commit=str(raw["git_commit"]),
        working_tree_dirty=cast(bool, raw["working_tree_dirty"]),
    )
    provenance.validate()
    return provenance


# ADD 2026-09-02: Ultralytics result의 finite boxes/classes/confidences/masks tensors를 기록한다.
def _observe_result_tensors(result: object) -> tuple[RuntimeTensorObservation, ...]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        raise ValueError("C5-3 backend result is missing boxes.")
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
    if len(observations) not in {3, 4}:
        raise ValueError("C5-3 backend result tensor schema is incomplete.")
    return tuple(observations)


# ADD 2026-09-02: C4-2C normalization을 GPU backend에 그대로 적용해 one-image prediction을 관측한다.
def predict_cuda_backend(
    *,
    model: PredictionModel,
    record: DerivedManifestRecord,
    dataset_root: Path,
    imgsz: int,
    device: int,
    diagnostic_confidence: float = 0.25,
) -> BackendPrediction:
    if record.derived_split != "val":
        raise ValueError("C5-3 TensorRT parity prediction accepts validation rows only.")
    capturing = _CapturingModel(model)
    raw_instances = predict_c4_2c_instances(
        model=capturing,
        source_image_path=(dataset_root / record.image_path).resolve(),
        image_width=record.image_width,
        image_height=record.image_height,
        imgsz=imgsz,
        device=str(device),
        valid_class_ids=set(EXPECTED_CLASSES),
    )
    if capturing.result is None:
        raise ValueError("C5-3 backend did not return exactly one result.")
    instances = filter_predictions({record.sample_id: raw_instances}, diagnostic_confidence)[
        record.sample_id
    ]
    _validate_instances(instances, record=record)
    return BackendPrediction(instances=instances, tensors=_observe_result_tensors(capturing.result))


# ADD 2026-09-02: Normalized GPU instance의 class/confidence/box/mask integrity를 검증한다.
def _validate_instances(
    instances: tuple[PredictedInstance, ...],
    *,
    record: DerivedManifestRecord,
) -> None:
    for instance in instances:
        if instance.class_id not in EXPECTED_CLASSES or not math.isfinite(instance.confidence):
            raise ValueError("C5-3 prediction contains invalid class or confidence.")
        if not 0.0 <= instance.confidence <= 1.0:
            raise ValueError("C5-3 prediction confidence is outside [0, 1].")
        if instance.mask.dtype != np.bool_ or instance.mask.shape != (
            record.image_height,
            record.image_width,
        ):
            raise ValueError("C5-3 prediction mask is not source-size boolean data.")
        if not instance.mask.any() or not all(math.isfinite(value) for value in instance.box_xyxy):
            raise ValueError("C5-3 prediction mask/box geometry is invalid.")


# ADD 2026-09-02: Prediction을 compact class/confidence/geometry/mask hash evidence로 만든다.
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


# ADD 2026-09-02: Validation sample의 PyTorch/TensorRT count parity evidence를 만든다.
def build_tensorrt_sample_parity(
    *,
    record: DerivedManifestRecord,
    pytorch: BackendPrediction,
    tensorrt: BackendPrediction,
) -> TensorRtSampleParityEvidence:
    if record.derived_split != "val":
        raise ValueError("C5-3 parity evidence rejects non-validation rows.")
    matches = match_backend_predictions(pytorch.instances, tensorrt.instances)
    evidence = TensorRtSampleParityEvidence(
        sample_id=record.sample_id,
        split=record.derived_split,
        pytorch_prediction_count=len(pytorch.instances),
        tensorrt_prediction_count=len(tensorrt.instances),
        matched_instance_count=len(matches),
        unmatched_pytorch_count=len(pytorch.instances) - len(matches),
        unmatched_tensorrt_count=len(tensorrt.instances) - len(matches),
        pytorch_tensors=pytorch.tensors,
        tensorrt_tensors=tensorrt.tensors,
        pytorch_predictions=_prediction_observations(pytorch.instances),
        tensorrt_predictions=_prediction_observations(tensorrt.instances),
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
        raise ValueError("C5-3 parity metric distribution contains NaN or Inf.")
    return {
        "count": len(values),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


# ADD 2026-09-02: Positive latency samples를 p50/p95 포함 deterministic summary로 만든다.
def summarize_latency_ms(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("C5-3 latency distribution requires at least one measurement.")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or bool((array <= 0.0).any()):
        raise ValueError("C5-3 latency measurements must be finite and positive.")
    return {
        "count": len(values),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


# ADD 2026-09-02: Saved latency distribution field set과 finite positive values를 검증한다.
def _validate_latency_distribution(
    mapping: Mapping[str, float | int],
    *,
    expected_count: int,
) -> None:
    if set(mapping) != {"count", "min", "mean", "p50", "p95", "max"}:
        raise ValueError("C5-3 latency distribution fields are invalid.")
    count = mapping["count"]
    if type(count) is not int or count != expected_count:
        raise ValueError("C5-3 latency measurement count is invalid.")
    for name in ("min", "mean", "p50", "p95", "max"):
        value = mapping[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("C5-3 latency distribution contains invalid values.")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("C5-3 latency distribution must contain finite positive values.")


# ADD 2026-09-02: CUDA synchronization을 포함한 Ultralytics end-to-end one-image latency를 측정한다.
def benchmark_prediction_model(
    *,
    model: PredictionModel,
    source_path: Path,
    imgsz: int,
    device: int,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, float | int]:
    kwargs: dict[str, object] = {
        "source": str(source_path),
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "retina_masks": False,
        "imgsz": imgsz,
        "device": str(device),
        "save": False,
        "stream": False,
        "verbose": False,
    }
    for _ in range(warmup_iterations):
        results = list(model.predict(**kwargs))
        if len(results) != 1:
            raise RuntimeError("C5-3 benchmark requires one result per prediction.")
    torch.cuda.synchronize(device)

    elapsed_ms: list[float] = []
    for _ in range(measured_iterations):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        results = list(model.predict(**kwargs))
        torch.cuda.synchronize(device)
        if len(results) != 1:
            raise RuntimeError("C5-3 benchmark requires one result per prediction.")
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
    return summarize_latency_ms(elapsed_ms)


# ADD 2026-09-02: Pinned Ultralytics model path를 requested PyTorch/TensorRT backend로 로드한다.
def _load_backend(path: Path) -> PredictionModel:
    from ultralytics import YOLO

    model = YOLO(str(path), task="segment")
    if model.task != "segment":
        raise ValueError("C5-3 backend model is not YOLO segmentation.")
    return cast(PredictionModel, model)


# ADD 2026-09-02: First prediction 후 PyTorch FP32 또는 TensorRT engine backend identity를 검증한다.
def _verify_loaded_backend(model: PredictionModel, *, expected_backend: str) -> None:
    names = {int(key): str(value) for key, value in model.names.items()}
    if names != EXPECTED_CLASSES:
        raise RuntimeError("C5-3 backend classes changed from bent/color/scratch.")
    backend = getattr(getattr(model, "predictor", None), "model", None)
    if backend is None:
        raise RuntimeError("C5-3 Ultralytics predictor backend was not initialized.")
    if expected_backend == "pytorch":
        if getattr(backend, "format", None) != "pt":
            raise RuntimeError("C5-3 reference execution did not use PyTorch backend.")
        if bool(getattr(backend, "fp16", False)):
            raise RuntimeError("C5-3 reference PyTorch backend must remain FP32.")
        return
    if expected_backend != "tensorrt":
        raise ValueError("C5-3 backend selector is invalid.")
    if getattr(backend, "format", None) != "engine":
        raise RuntimeError("C5-3 candidate execution did not use TensorRT engine backend.")
    if getattr(backend, "context", None) is None:
        raise RuntimeError("C5-3 TensorRT execution context was not initialized.")


# ADD 2026-09-02: TensorRT import/version을 GPU runtime에서 lazy resolve한다.
def _tensorrt_version() -> str:
    try:
        module = importlib.import_module("tensorrt")
    except ModuleNotFoundError as exc:
        raise RuntimeError("C5-3 TensorRT parity requires the TensorRT Python runtime.") from exc
    return str(module.__version__)


# ADD 2026-09-02: Current GPU runtime identity를 characterization evidence로 기록한다.
def _parity_environment(device: int) -> dict[str, str]:
    if not torch.cuda.is_available() or device < 0 or device >= torch.cuda.device_count():
        raise RuntimeError("C5-3 TensorRT parity requires an available CUDA device.")
    props = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    cuda_runtime = torch.version.cuda
    if not cuda_runtime:
        raise RuntimeError("C5-3 parity runtime does not expose a CUDA version.")
    from ultralytics import __version__ as ultralytics_version

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "python_implementation": sys.implementation.name,
        "torch_version": str(torch.__version__),
        "ultralytics_version": ultralytics_version,
        "tensorrt_version": _tensorrt_version(),
        "cuda_runtime_version": str(cuda_runtime),
        "gpu_name": str(props.name),
        "gpu_compute_capability": f"{capability[0]}.{capability[1]}",
        "pytorch_device": f"cuda:{device}",
        "tensorrt_device": f"cuda:{device}",
    }


# ADD 2026-09-02: Engine build와 parity runtime의 GPU/runtime identity를 검증한다.
def verify_tensorrt_runtime_environment(
    *,
    engine_metadata: YoloTensorRtExportMetadata,
    environment: Mapping[str, str],
) -> None:
    expected = {
        "tensorrt_version": engine_metadata.environment["tensorrt_version"],
        "cuda_runtime_version": engine_metadata.environment["cuda_runtime_version"],
        "gpu_name": engine_metadata.environment["gpu_name"],
        "gpu_compute_capability": engine_metadata.environment["gpu_compute_capability"],
    }
    mismatches = [name for name, value in expected.items() if environment.get(name) != value]
    if mismatches:
        raise RuntimeError(
            "C5-3 TensorRT parity runtime differs from engine build environment: "
            + ", ".join(sorted(mismatches))
        )


# ADD 2026-09-02: TensorRT artifact를 frozen source와 exact accepted ONNX identity에 교차 검증한다.
def verify_tensorrt_parity_artifact_identity(
    *,
    source: FrozenYoloSource,
    config: YoloTensorRtExportConfig,
    engine_metadata: YoloTensorRtExportMetadata,
) -> None:
    expected = {
        "export_id": config.export_id,
        "source_experiment_id": source.candidate.selected_experiment_id,
        "frozen_manifest_sha256": source.manifest_sha256,
        "source_model_sha256": source.candidate.model_sha256,
        "source_onnx_sha256": config.source_onnx_sha256,
        "source_onnx_metadata_sha256": config.source_onnx_metadata_sha256,
        "source_onnx_export_config_sha256": config.source_onnx_export_config_sha256,
        "tensorrt_config_sha256": sha256_file(config.config_path),
        "test_used": False,
        "test_split_used": False,
    }
    mismatches = [
        name
        for name, expected_value in expected.items()
        if getattr(engine_metadata, name) != expected_value
    ]
    if mismatches:
        raise ValueError(
            "C5-3 TensorRT artifact does not match frozen source: " + ", ".join(sorted(mismatches))
        )


# ADD 2026-09-02: Sample evidence와 benchmark를 TensorRT characterization contract로 결합한다.
def build_tensorrt_parity_evidence(
    *,
    parity_id: str,
    created_at: str,
    source: FrozenYoloSource,
    engine_metadata: YoloTensorRtExportMetadata,
    samples: tuple[TensorRtSampleParityEvidence, ...],
    latency: TensorRtLatencyBenchmark,
    provenance: RepositoryProvenance,
    environment: Mapping[str, str],
) -> YoloTensorRtParityEvidence:
    matches = [match for sample in samples for match in sample.matches]
    matched_count = len(matches)
    agreement_count = sum(match.class_agreement for match in matches)
    evidence = YoloTensorRtParityEvidence(
        schema_version=TENSORRT_PARITY_SCHEMA_VERSION,
        parity_id=parity_id,
        state=TENSORRT_PARITY_STATE,
        created_at=created_at,
        source_experiment_id=source.candidate.selected_experiment_id,
        frozen_manifest_sha256=source.manifest_sha256,
        source_model_sha256=source.candidate.model_sha256,
        source_onnx_sha256=engine_metadata.source_onnx_sha256,
        engine_sha256=engine_metadata.engine_sha256,
        tensorrt_config_sha256=engine_metadata.tensorrt_config_sha256,
        split="val",
        test_used=False,
        test_split_used=False,
        sample_count=len(samples),
        pytorch_prediction_count=sum(item.pytorch_prediction_count for item in samples),
        tensorrt_prediction_count=sum(item.tensorrt_prediction_count for item in samples),
        matched_instance_count=matched_count,
        unmatched_pytorch_count=sum(item.unmatched_pytorch_count for item in samples),
        unmatched_tensorrt_count=sum(item.unmatched_tensorrt_count for item in samples),
        class_agreement_count=agreement_count,
        class_agreement_rate=(agreement_count / matched_count if matched_count else None),
        confidence_abs_error=_distribution([item.confidence_abs_error for item in matches]),
        box_iou=_distribution([item.box_iou for item in matches]),
        mask_iou=_distribution([item.mask_iou for item in matches]),
        latency=latency,
        structural_gates_passed=True,
        numeric_acceptance="PENDING_TENSORRT_FP16_TOLERANCE_APPROVAL",
        samples=samples,
        environment=dict(environment),
        repository=provenance.to_json_dict(),
    )
    evidence.validate()
    return evidence


# ADD 2026-09-02: Validation-only rows에서 PyTorch FP32 GPU와 TensorRT FP16 metrics를 수집한다.
def evaluate_frozen_yolo_tensorrt_parity(
    *,
    source: FrozenYoloSource,
    config: YoloTensorRtExportConfig,
    onnx_artifact_dir: Path,
    tensorrt_artifact_dir: Path,
    dataset_root: Path,
    parity_id: str,
    created_at: str,
) -> Path:
    config.validate()
    validate_artifact_id(parity_id)
    _validate_timestamp(created_at)
    root = source.repository_root.resolve()
    try:
        config.config_path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("C5-3 TensorRT config must remain inside repository_root.") from exc

    expected_engine_dir = (root / config.output_root / config.export_id).resolve()
    if tensorrt_artifact_dir.resolve() != expected_engine_dir:
        raise ValueError("C5-3 TensorRT artifact must use the ignored repository namespace.")
    expected_onnx_dir = (root / EXPECTED_ONNX_ARTIFACT_ROOT / EXPECTED_ONNX_EXPORT_ID).resolve()
    if onnx_artifact_dir.resolve() != expected_onnx_dir:
        raise ValueError("C5-3 source ONNX must use the ignored C5-1 namespace.")

    verify_tensorrt_source_onnx_identity(
        repository_root=root,
        artifact_dir=onnx_artifact_dir,
        config=config,
    )
    engine_metadata = load_yolo_tensorrt_artifact(tensorrt_artifact_dir)
    verify_tensorrt_parity_artifact_identity(
        source=source,
        config=config,
        engine_metadata=engine_metadata,
    )

    provenance = resolve_repository_provenance(root)
    provenance.validate()
    if provenance.working_tree_dirty:
        raise ValueError("Official C5-3 TensorRT parity requires a clean committed repository.")

    environment = _parity_environment(config.device)
    verify_tensorrt_runtime_environment(
        engine_metadata=engine_metadata,
        environment=environment,
    )
    records = load_parity_validation_records(dataset_root, source.baseline.dataset_contract)

    output_dir = root / TENSORRT_PARITY_OUTPUT_ROOT / parity_id
    staging_dir = output_dir.parent / f".{parity_id}.staging"
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError("C5-3 TensorRT parity output namespace already exists.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="smartfactory-c5-trt-parity-") as temporary:
        artifact_dir = Path(temporary) / "source-artifact"
        materialize_official_candidate_artifact(
            package_path=source.package_path,
            candidate=source.candidate,
            evidence=source.evidence,
            artifact_dir=artifact_dir,
        )
        source_model = artifact_dir / "model" / "model.pt"
        if sha256_file(source_model) != source.candidate.model_sha256:
            raise RuntimeError("C5-3 materialized source model changed frozen identity.")

        pytorch_model = _load_backend(source_model)
        tensorrt_model = _load_backend(tensorrt_artifact_dir / TENSORRT_ENGINE_FILENAME)
        sample_evidence: list[TensorRtSampleParityEvidence] = []
        for index, record in enumerate(records):
            pytorch_prediction = predict_cuda_backend(
                model=pytorch_model,
                record=record,
                dataset_root=dataset_root,
                imgsz=config.imgsz,
                device=config.device,
                diagnostic_confidence=config.parity.diagnostic_confidence,
            )
            tensorrt_prediction = predict_cuda_backend(
                model=tensorrt_model,
                record=record,
                dataset_root=dataset_root,
                imgsz=config.imgsz,
                device=config.device,
                diagnostic_confidence=config.parity.diagnostic_confidence,
            )
            if index == 0:
                _verify_loaded_backend(pytorch_model, expected_backend="pytorch")
                _verify_loaded_backend(tensorrt_model, expected_backend="tensorrt")
            sample_evidence.append(
                build_tensorrt_sample_parity(
                    record=record,
                    pytorch=pytorch_prediction,
                    tensorrt=tensorrt_prediction,
                )
            )

        benchmark_record = records[0]
        benchmark_source = (dataset_root / benchmark_record.image_path).resolve()
        pytorch_latency = benchmark_prediction_model(
            model=pytorch_model,
            source_path=benchmark_source,
            imgsz=config.imgsz,
            device=config.device,
            warmup_iterations=config.parity.benchmark.warmup_iterations,
            measured_iterations=config.parity.benchmark.measured_iterations,
        )
        tensorrt_latency = benchmark_prediction_model(
            model=tensorrt_model,
            source_path=benchmark_source,
            imgsz=config.imgsz,
            device=config.device,
            warmup_iterations=config.parity.benchmark.warmup_iterations,
            measured_iterations=config.parity.benchmark.measured_iterations,
        )
        pytorch_mean = float(pytorch_latency["mean"])
        tensorrt_mean = float(tensorrt_latency["mean"])
        latency = TensorRtLatencyBenchmark(
            scope=config.parity.benchmark.scope,
            sample_id=benchmark_record.sample_id,
            warmup_iterations=config.parity.benchmark.warmup_iterations,
            measured_iterations=config.parity.benchmark.measured_iterations,
            pytorch_latency_ms=pytorch_latency,
            tensorrt_latency_ms=tensorrt_latency,
            speedup_ratio=pytorch_mean / tensorrt_mean,
        )
        latency.validate()

        evidence = build_tensorrt_parity_evidence(
            parity_id=parity_id,
            created_at=created_at,
            source=source,
            engine_metadata=engine_metadata,
            samples=tuple(sample_evidence),
            latency=latency,
            provenance=provenance,
            environment=environment,
        )
        staging_dir.mkdir(exist_ok=False)
        try:
            (staging_dir / TENSORRT_PARITY_FILENAME).write_bytes(evidence.to_json_bytes())
            if sha256_file(source_model) != source.candidate.model_sha256:
                raise RuntimeError("C5-3 parity changed frozen source model bytes.")
            if sha256_file(onnx_artifact_dir / "model.onnx") != engine_metadata.source_onnx_sha256:
                raise RuntimeError("C5-3 parity changed accepted ONNX bytes.")
            if (
                sha256_file(tensorrt_artifact_dir / TENSORRT_ENGINE_FILENAME)
                != engine_metadata.engine_sha256
            ):
                raise RuntimeError("C5-3 parity changed TensorRT engine bytes.")
            staging_dir.rename(output_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
    return output_dir / TENSORRT_PARITY_FILENAME
