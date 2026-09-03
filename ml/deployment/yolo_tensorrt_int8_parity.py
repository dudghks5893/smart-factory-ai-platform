"""C5-4C validation-only PyTorch FP32 versus exact TensorRT INT8 characterization."""

from __future__ import annotations

import importlib
import json
import platform
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import yaml

from ml.deployment.yolo_onnx import EXPECTED_CLASSES, FrozenYoloSource
from ml.deployment.yolo_onnx_parity import load_parity_validation_records
from ml.deployment.yolo_tensorrt_int8 import YoloTensorRtInt8Config
from ml.deployment.yolo_tensorrt_int8_engine import (
    INT8_ENGINE_FILENAME,
    INT8_ENGINE_METADATA_FILENAME,
    Int8EngineMetadata,
    YoloTensorRtInt8EngineConfig,
)
from ml.deployment.yolo_tensorrt_parity import (
    TensorRtLatencyBenchmark,
    TensorRtSampleParityEvidence,
    benchmark_prediction_model,
    build_tensorrt_sample_parity,
    predict_cuda_backend,
)
from ml.evaluation.final_benchmark import RepositoryProvenance, resolve_repository_provenance
from ml.experiments.yolo_final_candidate import materialize_official_candidate_artifact
from ml.training.yolo_segmentation import validate_artifact_id
from shared.hashing import is_sha256_digest, sha256_file

INT8_CHARACTERIZATION_SCHEMA_VERSION = 1
INT8_CHARACTERIZATION_STATE = "TENSORRT_INT8_METRICS_COLLECTED_ACCEPTANCE_PENDING"
INT8_CHARACTERIZATION_FILENAME = "characterization.json"
DEFAULT_TENSORRT_INT8_CHARACTERIZATION_CONFIG = Path(
    "configs/deployment/yolo_tensorrt_int8_characterization.yaml"
)

EXPECTED_INT8_QUANTIZATION_CONFIG_SHA256 = (
    "18309302e45855e506628bb5e262886fc2cb366f8758fc100c55aaf6dbf3c37a"
)
EXPECTED_ENGINE_EXPORT_ID = "c5_4b2_yolo11n_seg_tensorrt_int8_qdq"
EXPECTED_ENGINE_SHA256 = "4f397d59741f4efb7832087030b890a0fe059a657d074a3b07cdeb54493e8971"
EXPECTED_ENGINE_METADATA_SHA256 = "d44de78cc89fea67d6b351c2ba92f76dda0242386f4b6f14e216740ca682461e"
EXPECTED_ENGINE_RUN_SUMMARY_SHA256 = (
    "a5de13fc8e616b6071eebc0d76f0f88cdff32181e8d49db5cdd143113aef113f"
)
EXPECTED_ENGINE_CONFIG_SHA256 = "63eebcac04d11c9247bf7543fe18d0798758ab20cc734d2b18bfbece4eaf6b41"
EXPECTED_ENGINE_EVIDENCE_ZIP_SHA256 = (
    "0cba556981b12a95b25feb324d0ff02b9cadeda6bde056b46e27eb7698f66b00"
)
EXPECTED_ENGINE_BUILD_COMMIT = "7835291c8fb123eba6acfa839977f94093c2f3ac"
EXPECTED_ENGINE_TENSORRT_VERSION = "10.13.3.9.post1"
EXPECTED_ENGINE_CUDA_RUNTIME_VERSION = "12.8"
EXPECTED_ENGINE_GPU_NAME = "Tesla T4"
EXPECTED_ENGINE_GPU_COMPUTE_CAPABILITY = "7.5"
EXPECTED_ENGINE_TORCH_VERSION = "2.10.0+cu128"
EXPECTED_ENGINE_ULTRALYTICS_VERSION = "8.4.128"

EXPECTED_FP16_ENGINE_SHA256 = "9bbbe5297e6cc55bcea877a79f45485ee7e1e5e6a831ad5276aedc8e3d904037"
EXPECTED_FP16_POLICY_SHA256 = "4f8f81a70417e380062358a9f3888d4fe0fa236fdfbc7b04da2616356833bfd9"
EXPECTED_OUTPUT_ROOT = Path("outputs/deployment/yolo_segmentation/tensorrt_int8_characterization")


@dataclass(frozen=True)
class Int8CharacterizationSource:
    """Exact C5-4B2 engine and runtime identity."""

    int8_quantization_config_sha256: str
    engine_export_id: str
    engine_sha256: str
    engine_metadata_sha256: str
    engine_run_summary_sha256: str
    engine_config_sha256: str
    engine_evidence_zip_sha256: str
    engine_build_repository_commit: str
    engine_tensorrt_version: str
    engine_cuda_runtime_version: str
    engine_gpu_name: str
    engine_gpu_compute_capability: str
    engine_torch_version: str
    engine_ultralytics_version: str

    # ADD 2026-09-04: C5-4C가 successful B2 exact engine과 build runtime에서 이탈하지 않게 한다.
    def validate(self) -> None:
        expected = {
            "int8_quantization_config_sha256": EXPECTED_INT8_QUANTIZATION_CONFIG_SHA256,
            "engine_export_id": EXPECTED_ENGINE_EXPORT_ID,
            "engine_sha256": EXPECTED_ENGINE_SHA256,
            "engine_metadata_sha256": EXPECTED_ENGINE_METADATA_SHA256,
            "engine_run_summary_sha256": EXPECTED_ENGINE_RUN_SUMMARY_SHA256,
            "engine_config_sha256": EXPECTED_ENGINE_CONFIG_SHA256,
            "engine_evidence_zip_sha256": EXPECTED_ENGINE_EVIDENCE_ZIP_SHA256,
            "engine_build_repository_commit": EXPECTED_ENGINE_BUILD_COMMIT,
            "engine_tensorrt_version": EXPECTED_ENGINE_TENSORRT_VERSION,
            "engine_cuda_runtime_version": EXPECTED_ENGINE_CUDA_RUNTIME_VERSION,
            "engine_gpu_name": EXPECTED_ENGINE_GPU_NAME,
            "engine_gpu_compute_capability": EXPECTED_ENGINE_GPU_COMPUTE_CAPABILITY,
            "engine_torch_version": EXPECTED_ENGINE_TORCH_VERSION,
            "engine_ultralytics_version": EXPECTED_ENGINE_ULTRALYTICS_VERSION,
        }
        for name, expected_value in expected.items():
            if getattr(self, name) != expected_value:
                raise ValueError(f"C5-4C source field {name} changed from frozen B2 evidence.")
        for digest in (
            self.int8_quantization_config_sha256,
            self.engine_sha256,
            self.engine_metadata_sha256,
            self.engine_run_summary_sha256,
            self.engine_config_sha256,
            self.engine_evidence_zip_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C5-4C source contains invalid SHA-256.")


@dataclass(frozen=True)
class Int8CharacterizationBenchmark:
    """Same benchmark boundary used in the C5-4A contract."""

    warmup_iterations: int
    measured_iterations: int
    sample_selector: str
    scope: str

    def validate(self) -> None:
        if (
            type(self.warmup_iterations) is not int
            or type(self.measured_iterations) is not int
            or self.warmup_iterations != 10
            or self.measured_iterations != 50
            or self.sample_selector != "first_validation_sample"
            or self.scope != "ultralytics_end_to_end_single_image"
        ):
            raise ValueError("C5-4C benchmark boundary changed from C5-4A.")


@dataclass(frozen=True)
class Int8CharacterizationPolicy:
    """Metrics-first validation-only policy without numeric acceptance thresholds."""

    split: str
    sample_count: int
    test_used: bool
    test_split_used: bool
    reference_backend: str
    candidate_backend: str
    comparison_baseline: str
    diagnostic_confidence: float
    numeric_acceptance: str
    numeric_thresholds: None
    benchmark: Int8CharacterizationBenchmark

    # ADD 2026-09-04: C5-4C가 val 28장 metrics-only characterization으로만 동작하게 한다.
    def validate(self) -> None:
        if type(self.sample_count) is not int:
            raise TypeError("C5-4C sample_count must be int.")
        if type(self.test_used) is not bool or type(self.test_split_used) is not bool:
            raise TypeError("C5-4C leakage flags must be booleans.")
        if (
            self.split != "val"
            or self.sample_count != 28
            or self.test_used is not False
            or self.test_split_used is not False
            or self.reference_backend != "pytorch_fp32_gpu"
            or self.candidate_backend != "tensorrt_int8"
            or self.comparison_baseline != "accepted_tensorrt_fp16"
            or self.diagnostic_confidence != 0.25
            or self.numeric_acceptance != "PENDING_TENSORRT_INT8_TOLERANCE_APPROVAL"
            or self.numeric_thresholds is not None
        ):
            raise ValueError("C5-4C characterization policy changed without review.")
        self.benchmark.validate()


@dataclass(frozen=True)
class HistoricalFp16Baseline:
    """Accepted FP16 result retained only as historical context."""

    comparison_mode: str
    acceptance_state: str
    engine_sha256: str
    policy_sha256: str
    tensorrt_version: str
    cuda_runtime_version: str
    gpu_name: str
    gpu_compute_capability: str
    torch_version: str
    ultralytics_version: str
    pytorch_mean_latency_ms: float
    tensorrt_fp16_mean_latency_ms: float
    speedup_ratio: float

    # ADD 2026-09-04: FP16 결과를 runtime-mismatched historical context로 명시한다.
    def validate(self) -> None:
        if (
            self.comparison_mode != "historical_context_only_runtime_mismatch"
            or self.acceptance_state != "TENSORRT_FP16_PARITY_ACCEPTED"
            or self.engine_sha256 != EXPECTED_FP16_ENGINE_SHA256
            or self.policy_sha256 != EXPECTED_FP16_POLICY_SHA256
            or self.tensorrt_version != "10.13.3.9.post1"
            or self.cuda_runtime_version != "13.0"
            or self.gpu_name != "Tesla T4"
            or self.gpu_compute_capability != "7.5"
            or self.torch_version != "2.13.0+cu130"
            or self.ultralytics_version != "8.4.128"
        ):
            raise ValueError("C5-4C historical FP16 baseline identity changed.")
        if not is_sha256_digest(self.engine_sha256) or not is_sha256_digest(self.policy_sha256):
            raise ValueError("C5-4C historical FP16 baseline contains invalid SHA-256.")
        expected_metrics = (
            (self.pytorch_mean_latency_ms, 31.130911420002576),
            (self.tensorrt_fp16_mean_latency_ms, 25.844023020001714),
            (self.speedup_ratio, 1.2045690949860681),
        )
        if any(abs(value - expected) > 1e-12 for value, expected in expected_metrics):
            raise ValueError("C5-4C historical FP16 latency context changed.")


@dataclass(frozen=True)
class YoloTensorRtInt8CharacterizationConfig:
    """Top-level C5-4C characterization contract."""

    schema_version: int
    characterization_id: str
    output_root: Path
    source: Int8CharacterizationSource
    characterization: Int8CharacterizationPolicy
    historical_fp16: HistoricalFp16Baseline
    config_path: Path

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported C5-4C config schema.")
        validate_artifact_id(self.characterization_id)
        if self.output_root != EXPECTED_OUTPUT_ROOT:
            raise ValueError("C5-4C output_root changed.")
        self.source.validate()
        self.characterization.validate()
        self.historical_fp16.validate()


@dataclass(frozen=True)
class Int8CharacterizationEvidence:
    """Validation-only INT8 characterization awaiting a separately frozen policy."""

    schema_version: int
    characterization_id: str
    state: str
    created_at: str
    source_experiment_id: str
    frozen_manifest_sha256: str
    source_model_sha256: str
    int8_quantization_config_sha256: str
    engine_export_id: str
    engine_sha256: str
    engine_metadata_sha256: str
    engine_config_sha256: str
    engine_evidence_zip_sha256: str
    engine_build_repository_commit: str
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
    historical_fp16: HistoricalFp16Baseline
    structural_gates_passed: bool
    numeric_acceptance: str
    samples: tuple[TensorRtSampleParityEvidence, ...]
    environment: Mapping[str, str]
    repository: Mapping[str, str | bool]

    # ADD 2026-09-04: C5-4C evidence의 val-only seal, metrics-first state, aggregate를 검증한다.
    def validate(self) -> None:
        validate_artifact_id(self.characterization_id)
        if (
            self.schema_version != INT8_CHARACTERIZATION_SCHEMA_VERSION
            or self.state != INT8_CHARACTERIZATION_STATE
            or self.split != "val"
            or self.test_used is not False
            or self.test_split_used is not False
            or self.sample_count != 28
            or self.structural_gates_passed is not True
            or self.numeric_acceptance != "PENDING_TENSORRT_INT8_TOLERANCE_APPROVAL"
        ):
            raise ValueError("C5-4C evidence lifecycle or data seal is invalid.")
        _validate_timestamp(self.created_at)
        for digest in (
            self.frozen_manifest_sha256,
            self.source_model_sha256,
            self.int8_quantization_config_sha256,
            self.engine_sha256,
            self.engine_metadata_sha256,
            self.engine_config_sha256,
            self.engine_evidence_zip_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C5-4C evidence contains invalid SHA-256.")
        if (
            self.engine_export_id != EXPECTED_ENGINE_EXPORT_ID
            or self.engine_sha256 != EXPECTED_ENGINE_SHA256
            or self.engine_metadata_sha256 != EXPECTED_ENGINE_METADATA_SHA256
            or self.engine_config_sha256 != EXPECTED_ENGINE_CONFIG_SHA256
            or self.engine_evidence_zip_sha256 != EXPECTED_ENGINE_EVIDENCE_ZIP_SHA256
            or self.engine_build_repository_commit != EXPECTED_ENGINE_BUILD_COMMIT
        ):
            raise ValueError("C5-4C evidence engine identity changed.")

        repository = _repository_provenance(self.repository)
        if repository.working_tree_dirty:
            raise ValueError("Official C5-4C run requires a clean repository.")
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
            raise ValueError("C5-4C environment fields are invalid.")

        totals = {
            "pytorch_prediction_count": sum(item.pytorch_prediction_count for item in self.samples),
            "tensorrt_prediction_count": sum(
                item.tensorrt_prediction_count for item in self.samples
            ),
            "matched_instance_count": sum(item.matched_instance_count for item in self.samples),
            "unmatched_pytorch_count": sum(item.unmatched_pytorch_count for item in self.samples),
            "unmatched_tensorrt_count": sum(item.unmatched_tensorrt_count for item in self.samples),
        }
        if self.sample_count != len(self.samples) or any(
            getattr(self, name) != value for name, value in totals.items()
        ):
            raise ValueError("C5-4C aggregate counts do not match sample evidence.")
        agreement_count = sum(
            match.class_agreement for sample in self.samples for match in sample.matches
        )
        expected_rate = (
            agreement_count / self.matched_instance_count if self.matched_instance_count else None
        )
        if (
            self.class_agreement_count != agreement_count
            or self.class_agreement_rate != expected_rate
        ):
            raise ValueError("C5-4C class agreement aggregate is inconsistent.")
        for sample in self.samples:
            sample.validate()
        self.latency.validate()
        self.historical_fp16.validate()
        try:
            json.dumps(asdict(self), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("C5-4C evidence must be strict JSON data.") from exc

    def to_json_bytes(self) -> bytes:
        self.validate()
        return (json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


class PredictionModel(Protocol):
    """Minimal Ultralytics prediction surface."""

    predictor: object
    names: Mapping[int, str]

    def predict(self, **kwargs: object) -> Sequence[object]: ...


def _mapping(raw: object, *, field: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"C5-4C {field} must be a mapping.")
    return cast(dict[str, Any], raw)


def load_yolo_tensorrt_int8_characterization_config(
    path: Path,
) -> YoloTensorRtInt8CharacterizationConfig:
    try:
        raw_obj: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("Cannot read C5-4C characterization config.") from exc

    raw = _mapping(raw_obj, field="config root")
    try:
        output_root = Path(str(raw.pop("output_root")))
        source = Int8CharacterizationSource(**_mapping(raw.pop("source"), field="source"))
        characterization_raw = _mapping(raw.pop("characterization"), field="characterization")
        benchmark = Int8CharacterizationBenchmark(
            **_mapping(characterization_raw.pop("benchmark"), field="characterization.benchmark")
        )
        characterization = Int8CharacterizationPolicy(
            **characterization_raw,
            benchmark=benchmark,
        )
        historical = HistoricalFp16Baseline(
            **_mapping(raw.pop("historical_fp16"), field="historical_fp16")
        )
        config = YoloTensorRtInt8CharacterizationConfig(
            **raw,
            output_root=output_root,
            source=source,
            characterization=characterization,
            historical_fp16=historical,
            config_path=path.resolve(),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("C5-4C config fields do not match schema.") from exc
    config.validate()
    return config


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("C5-4C timestamp must be ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("C5-4C timestamp must include timezone offset.")


def _repository_provenance(raw: Mapping[str, str | bool]) -> RepositoryProvenance:
    if set(raw) != {"git_commit", "working_tree_dirty"}:
        raise ValueError("C5-4C repository provenance fields are invalid.")
    if type(raw["working_tree_dirty"]) is not bool:
        raise TypeError("C5-4C working_tree_dirty must be boolean.")
    provenance = RepositoryProvenance(
        git_commit=str(raw["git_commit"]),
        working_tree_dirty=cast(bool, raw["working_tree_dirty"]),
    )
    provenance.validate()
    return provenance


def _metric_distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("C5-4C metric distribution contains NaN or Inf.")
    return {
        "count": len(values),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


# ADD 2026-09-04: Existing C5-4A characterization boundary와 C5-4C runtime contract를 교차 검증한다.
def verify_c5_4a_characterization_contract(
    int8_config: YoloTensorRtInt8Config,
    characterization_config: YoloTensorRtInt8CharacterizationConfig,
) -> None:
    int8_config.validate()
    characterization_config.validate()
    if (
        sha256_file(int8_config.config_path)
        != characterization_config.source.int8_quantization_config_sha256
    ):
        raise ValueError("C5-4C C5-4A INT8 config SHA mismatch.")
    observed = int8_config.characterization
    expected = characterization_config.characterization
    if (
        observed.split != expected.split
        or observed.sample_count != expected.sample_count
        or observed.test_used is not False
        or observed.test_split_used is not False
        or observed.reference_backend != expected.reference_backend
        or observed.candidate_backend != expected.candidate_backend
        or observed.comparison_baseline != expected.comparison_baseline
        or observed.diagnostic_confidence != expected.diagnostic_confidence
        or observed.acceptance_mode != "metrics_only_pending_tensorrt_int8_tolerance_approval"
        or observed.numeric_thresholds is not None
        or observed.benchmark.warmup_iterations != expected.benchmark.warmup_iterations
        or observed.benchmark.measured_iterations != expected.benchmark.measured_iterations
        or observed.benchmark.sample_selector != expected.benchmark.sample_selector
        or observed.benchmark.scope != expected.benchmark.scope
    ):
        raise ValueError("C5-4C changed the frozen C5-4A characterization boundary.")


# ADD 2026-09-04: Exact B2 model.engine/metadata bytes와 semantic provenance를 검증한다.
def load_exact_int8_engine_metadata(
    *,
    artifact_dir: Path,
    engine_config: YoloTensorRtInt8EngineConfig,
    characterization_config: YoloTensorRtInt8CharacterizationConfig,
) -> Int8EngineMetadata:
    engine_path = artifact_dir / INT8_ENGINE_FILENAME
    metadata_path = artifact_dir / INT8_ENGINE_METADATA_FILENAME
    if not engine_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("C5-4C exact B2 engine artifact is incomplete.")
    if sha256_file(engine_path) != characterization_config.source.engine_sha256:
        raise ValueError("C5-4C engine SHA mismatch.")
    if sha256_file(metadata_path) != characterization_config.source.engine_metadata_sha256:
        raise ValueError("C5-4C engine metadata SHA mismatch.")
    if (
        sha256_file(engine_config.config_path)
        != characterization_config.source.engine_config_sha256
    ):
        raise ValueError("C5-4C B2 engine config SHA mismatch.")

    raw_obj: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw = _mapping(raw_obj, field="B2 engine metadata")
    try:
        metadata = Int8EngineMetadata(**raw)
    except TypeError as exc:
        raise ValueError("C5-4C B2 engine metadata fields are invalid.") from exc
    metadata.validate(config=engine_config)

    if (
        metadata.export_id != characterization_config.source.engine_export_id
        or metadata.engine_sha256 != characterization_config.source.engine_sha256
        or metadata.tensorrt_int8_engine_config_sha256
        != characterization_config.source.engine_config_sha256
        or metadata.repository.get("git_commit")
        != characterization_config.source.engine_build_repository_commit
        or metadata.repository.get("working_tree_dirty") is not False
        or metadata.validation_used is not False
        or metadata.test_used is not False
        or metadata.test_split_used is not False
    ):
        raise ValueError("C5-4C B2 engine semantic identity changed.")
    expected_environment = {
        "tensorrt_version": characterization_config.source.engine_tensorrt_version,
        "cuda_runtime_version": characterization_config.source.engine_cuda_runtime_version,
        "gpu_name": characterization_config.source.engine_gpu_name,
        "gpu_compute_capability": characterization_config.source.engine_gpu_compute_capability,
        "torch_version": characterization_config.source.engine_torch_version,
        "ultralytics_version": characterization_config.source.engine_ultralytics_version,
    }
    mismatches = [
        name
        for name, expected_value in expected_environment.items()
        if metadata.environment.get(name) != expected_value
    ]
    if mismatches:
        raise ValueError("C5-4C B2 engine environment metadata mismatch: " + ", ".join(mismatches))
    return metadata


def _runtime_environment(device: int) -> dict[str, str]:
    if not torch.cuda.is_available() or device < 0 or device >= torch.cuda.device_count():
        raise RuntimeError("C5-4C requires an available CUDA device.")
    try:
        tensorrt = importlib.import_module("tensorrt")
    except ModuleNotFoundError as exc:
        raise RuntimeError("C5-4C requires TensorRT runtime.") from exc
    from ultralytics import __version__ as ultralytics_version

    props = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    cuda_runtime = torch.version.cuda
    if not cuda_runtime:
        raise RuntimeError("C5-4C PyTorch runtime does not expose CUDA version.")
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "python_implementation": sys.implementation.name,
        "torch_version": str(torch.__version__),
        "ultralytics_version": ultralytics_version,
        "tensorrt_version": str(tensorrt.__version__),
        "cuda_runtime_version": str(cuda_runtime),
        "gpu_name": str(props.name),
        "gpu_compute_capability": f"{capability[0]}.{capability[1]}",
        "pytorch_device": f"cuda:{device}",
        "tensorrt_device": f"cuda:{device}",
    }


# ADD 2026-09-04: Characterization runtime을 exact B2 build runtime identity와 맞춘다.
def verify_int8_characterization_runtime(
    *,
    config: YoloTensorRtInt8CharacterizationConfig,
    environment: Mapping[str, str],
) -> None:
    expected = {
        "torch_version": config.source.engine_torch_version,
        "ultralytics_version": config.source.engine_ultralytics_version,
        "tensorrt_version": config.source.engine_tensorrt_version,
        "cuda_runtime_version": config.source.engine_cuda_runtime_version,
        "gpu_name": config.source.engine_gpu_name,
        "gpu_compute_capability": config.source.engine_gpu_compute_capability,
    }
    mismatches = [name for name, value in expected.items() if environment.get(name) != value]
    if mismatches:
        raise RuntimeError(
            "C5-4C runtime differs from exact B2 build environment: "
            + ", ".join(sorted(mismatches))
        )


def _load_backend(path: Path) -> PredictionModel:
    from ultralytics import YOLO

    model = YOLO(str(path), task="segment")
    if model.task != "segment":
        raise ValueError("C5-4C backend model is not YOLO segmentation.")
    return cast(PredictionModel, model)


def _verify_loaded_backend(model: PredictionModel, *, expected_backend: str) -> None:
    names = {int(key): str(value) for key, value in model.names.items()}
    if names != EXPECTED_CLASSES:
        raise RuntimeError("C5-4C backend classes changed from bent/color/scratch.")
    backend = getattr(getattr(model, "predictor", None), "model", None)
    if backend is None:
        raise RuntimeError("C5-4C Ultralytics predictor backend was not initialized.")
    if expected_backend == "pytorch":
        if getattr(backend, "format", None) != "pt":
            raise RuntimeError("C5-4C reference execution did not use PyTorch backend.")
        if bool(getattr(backend, "fp16", False)):
            raise RuntimeError("C5-4C reference PyTorch backend must remain FP32.")
        return
    if expected_backend != "tensorrt":
        raise ValueError("C5-4C backend selector is invalid.")
    if getattr(backend, "format", None) != "engine":
        raise RuntimeError("C5-4C candidate execution did not use TensorRT engine backend.")
    if getattr(backend, "context", None) is None:
        raise RuntimeError("C5-4C TensorRT execution context was not initialized.")


# ADD 2026-09-04: Samples/latency를 threshold-free INT8 characterization evidence로 집계한다.
def build_int8_characterization_evidence(
    *,
    config: YoloTensorRtInt8CharacterizationConfig,
    created_at: str,
    source: FrozenYoloSource,
    samples: tuple[TensorRtSampleParityEvidence, ...],
    latency: TensorRtLatencyBenchmark,
    provenance: RepositoryProvenance,
    environment: Mapping[str, str],
) -> Int8CharacterizationEvidence:
    matches = [match for sample in samples for match in sample.matches]
    matched_count = len(matches)
    agreement_count = sum(match.class_agreement for match in matches)
    evidence = Int8CharacterizationEvidence(
        schema_version=INT8_CHARACTERIZATION_SCHEMA_VERSION,
        characterization_id=config.characterization_id,
        state=INT8_CHARACTERIZATION_STATE,
        created_at=created_at,
        source_experiment_id=source.candidate.selected_experiment_id,
        frozen_manifest_sha256=source.manifest_sha256,
        source_model_sha256=source.candidate.model_sha256,
        int8_quantization_config_sha256=config.source.int8_quantization_config_sha256,
        engine_export_id=config.source.engine_export_id,
        engine_sha256=config.source.engine_sha256,
        engine_metadata_sha256=config.source.engine_metadata_sha256,
        engine_config_sha256=config.source.engine_config_sha256,
        engine_evidence_zip_sha256=config.source.engine_evidence_zip_sha256,
        engine_build_repository_commit=config.source.engine_build_repository_commit,
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
        class_agreement_rate=agreement_count / matched_count if matched_count else None,
        confidence_abs_error=_metric_distribution([item.confidence_abs_error for item in matches]),
        box_iou=_metric_distribution([item.box_iou for item in matches]),
        mask_iou=_metric_distribution([item.mask_iou for item in matches]),
        latency=latency,
        historical_fp16=config.historical_fp16,
        structural_gates_passed=True,
        numeric_acceptance=config.characterization.numeric_acceptance,
        samples=samples,
        environment=dict(environment),
        repository=provenance.to_json_dict(),
    )
    evidence.validate()
    return evidence


# ADD 2026-09-04: Exact B2 engine을 rebuild 없이 복원해 val 28장 metrics를 수집한다.
def evaluate_frozen_yolo_tensorrt_int8_characterization(
    *,
    source: FrozenYoloSource,
    int8_config: YoloTensorRtInt8Config,
    engine_config: YoloTensorRtInt8EngineConfig,
    characterization_config: YoloTensorRtInt8CharacterizationConfig,
    int8_engine_artifact_dir: Path,
    dataset_root: Path,
    created_at: str,
) -> Path:
    verify_c5_4a_characterization_contract(int8_config, characterization_config)
    engine_config.validate()
    _validate_timestamp(created_at)
    root = source.repository_root.resolve()

    expected_engine_dir = (root / engine_config.output_root / engine_config.export_id).resolve()
    if int8_engine_artifact_dir.resolve() != expected_engine_dir:
        raise ValueError("C5-4C engine must use the ignored B2 repository artifact namespace.")
    load_exact_int8_engine_metadata(
        artifact_dir=int8_engine_artifact_dir,
        engine_config=engine_config,
        characterization_config=characterization_config,
    )

    if source.manifest_sha256 != int8_config.source.frozen_manifest_sha256:
        raise ValueError("C5-4C frozen manifest changed from C5-4A.")
    if source.candidate.model_sha256 != int8_config.source.model_sha256:
        raise ValueError("C5-4C frozen model changed from C5-4A.")

    provenance = resolve_repository_provenance(root)
    provenance.validate()
    if provenance.working_tree_dirty:
        raise ValueError("Official C5-4C run requires a clean committed repository.")

    environment = _runtime_environment(engine_config.device)
    verify_int8_characterization_runtime(config=characterization_config, environment=environment)
    records = load_parity_validation_records(dataset_root, source.baseline.dataset_contract)
    if len(records) != characterization_config.characterization.sample_count:
        raise ValueError("C5-4C validation record count changed from 28.")

    output_dir = (
        root / characterization_config.output_root / characterization_config.characterization_id
    )
    staging_dir = output_dir.parent / f".{characterization_config.characterization_id}.staging"
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError("C5-4C output namespace already exists.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    engine_path = int8_engine_artifact_dir / INT8_ENGINE_FILENAME
    engine_sha_before = sha256_file(engine_path)
    with tempfile.TemporaryDirectory(prefix="smartfactory-c5-int8-characterization-") as temporary:
        artifact_dir = Path(temporary) / "source-artifact"
        materialize_official_candidate_artifact(
            package_path=source.package_path,
            candidate=source.candidate,
            evidence=source.evidence,
            artifact_dir=artifact_dir,
        )
        source_model = artifact_dir / "model" / "model.pt"
        if sha256_file(source_model) != source.candidate.model_sha256:
            raise RuntimeError("C5-4C materialized source model changed frozen identity.")

        pytorch_model = _load_backend(source_model)
        int8_model = _load_backend(engine_path)
        sample_evidence: list[TensorRtSampleParityEvidence] = []
        for index, record in enumerate(records):
            pytorch_prediction = predict_cuda_backend(
                model=pytorch_model,
                record=record,
                dataset_root=dataset_root,
                imgsz=int8_config.imgsz,
                device=int8_config.device,
                diagnostic_confidence=int8_config.characterization.diagnostic_confidence,
            )
            int8_prediction = predict_cuda_backend(
                model=int8_model,
                record=record,
                dataset_root=dataset_root,
                imgsz=int8_config.imgsz,
                device=int8_config.device,
                diagnostic_confidence=int8_config.characterization.diagnostic_confidence,
            )
            if index == 0:
                _verify_loaded_backend(pytorch_model, expected_backend="pytorch")
                _verify_loaded_backend(int8_model, expected_backend="tensorrt")
            sample_evidence.append(
                build_tensorrt_sample_parity(
                    record=record,
                    pytorch=pytorch_prediction,
                    tensorrt=int8_prediction,
                )
            )

        benchmark_record = records[0]
        benchmark_source = (dataset_root / benchmark_record.image_path).resolve()
        benchmark = characterization_config.characterization.benchmark
        pytorch_latency = benchmark_prediction_model(
            model=pytorch_model,
            source_path=benchmark_source,
            imgsz=int8_config.imgsz,
            device=int8_config.device,
            warmup_iterations=benchmark.warmup_iterations,
            measured_iterations=benchmark.measured_iterations,
        )
        int8_latency = benchmark_prediction_model(
            model=int8_model,
            source_path=benchmark_source,
            imgsz=int8_config.imgsz,
            device=int8_config.device,
            warmup_iterations=benchmark.warmup_iterations,
            measured_iterations=benchmark.measured_iterations,
        )
        latency = TensorRtLatencyBenchmark(
            scope=benchmark.scope,
            sample_id=benchmark_record.sample_id,
            warmup_iterations=benchmark.warmup_iterations,
            measured_iterations=benchmark.measured_iterations,
            pytorch_latency_ms=pytorch_latency,
            tensorrt_latency_ms=int8_latency,
            speedup_ratio=float(pytorch_latency["mean"]) / float(int8_latency["mean"]),
        )
        latency.validate()

        evidence = build_int8_characterization_evidence(
            config=characterization_config,
            created_at=created_at,
            source=source,
            samples=tuple(sample_evidence),
            latency=latency,
            provenance=provenance,
            environment=environment,
        )
        staging_dir.mkdir(exist_ok=False)
        try:
            (staging_dir / INT8_CHARACTERIZATION_FILENAME).write_bytes(evidence.to_json_bytes())
            if sha256_file(source_model) != source.candidate.model_sha256:
                raise RuntimeError("C5-4C changed frozen source model bytes.")
            if sha256_file(engine_path) != engine_sha_before:
                raise RuntimeError("C5-4C changed exact B2 engine bytes.")
            staging_dir.rename(output_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    return output_dir / INT8_CHARACTERIZATION_FILENAME
