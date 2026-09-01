"""Exact-ONNX TensorRT FP16 engine build contracts for frozen YOLO segmentation."""

from __future__ import annotations

import importlib
import json
import platform
import shutil
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from ml.deployment.yolo_onnx import (
    ONNX_METADATA_FILENAME,
    ONNX_MODEL_FILENAME,
    YoloOnnxExportMetadata,
    load_yolo_onnx_artifact,
)
from ml.evaluation.final_benchmark import RepositoryProvenance, resolve_repository_provenance
from ml.training.yolo_segmentation import validate_artifact_id
from shared.hashing import is_sha256_digest, sha256_file

TENSORRT_EXPORT_SCHEMA_VERSION = 1
DEFAULT_TENSORRT_EXPORT_CONFIG = Path("configs/export/yolo_segmentation_tensorrt_fp16.yaml")
TENSORRT_ENGINE_FILENAME = "model.engine"
TENSORRT_METADATA_FILENAME = "metadata.json"
EXPECTED_ONNX_EXPORT_ID = "c5_1_yolo11n_seg_fp32_static_opset18"
EXPECTED_ONNX_EXPORT_COMMIT = "643ed9386a61bd2bf0c041f92a10b809b6d52c3e"
EXPECTED_ONNX_SHA256 = "f916325bb126d174de9c1fdfc24802eec11c46014f723fbf3ba3b3c1755c1490"
EXPECTED_ONNX_METADATA_SHA256 = "3286861db66cb4c4f886d2fd71f8f13b749b019bd0d57249f54a025d43b11fcd"
EXPECTED_ONNX_EXPORT_CONFIG_SHA256 = (
    "f1c2ef5045fdd89d964b2dc79c501580c9f55c2a1d38f38f13cf4794bafd0e85"
)
EXPECTED_ONNX_ARTIFACT_ROOT = Path("artifacts/deployment/yolo_segmentation/onnx")
EXPECTED_TENSORRT_OUTPUT_ROOT = Path("artifacts/deployment/yolo_segmentation/tensorrt")


@dataclass(frozen=True)
class TensorRtBenchmarkPolicy:
    """Characterization-only end-to-end latency sampling contract."""

    warmup_iterations: int
    measured_iterations: int
    sample_selector: str
    scope: str

    # ADD 2026-09-02: C5-3 characterization benchmark boundary를 고정한다.
    def validate(self) -> None:
        if type(self.warmup_iterations) is not int or type(self.measured_iterations) is not int:
            raise TypeError("C5-3 benchmark iteration counts must be integers.")
        if self.warmup_iterations != 10 or self.measured_iterations != 50:
            raise ValueError("C5-3 benchmark iteration policy changed without review.")
        if (
            self.sample_selector != "first_validation_sample"
            or self.scope != "ultralytics_end_to_end_single_image"
        ):
            raise ValueError("C5-3 benchmark scope changed without review.")


@dataclass(frozen=True)
class YoloTensorRtParityPolicy:
    """Validation-only TensorRT FP16 characterization policy without numeric acceptance."""

    split: str
    test_used: bool
    test_split_used: bool
    prediction_initial_confidence: float
    diagnostic_confidence: float
    prediction_iou: float
    max_detections: int
    retina_masks: bool
    mask_threshold: float
    mask_resize: str
    association: str
    acceptance_mode: str
    require_finite_outputs: bool
    require_valid_output_shapes: bool
    require_valid_class_ids: bool
    numeric_thresholds: None
    benchmark: TensorRtBenchmarkPolicy

    # ADD 2026-09-02: FP16 characterization이 C4 normalization과 no-test seal을 유지하는지 검증한다.
    def validate(self) -> None:
        boolean_values = (
            self.test_used,
            self.test_split_used,
            self.retina_masks,
            self.require_finite_outputs,
            self.require_valid_output_shapes,
            self.require_valid_class_ids,
        )
        if any(type(value) is not bool for value in boolean_values):
            raise TypeError("C5-3 parity boolean fields must be strict booleans.")
        if type(self.max_detections) is not int or any(
            type(value) not in {int, float}
            for value in (
                self.prediction_initial_confidence,
                self.diagnostic_confidence,
                self.prediction_iou,
                self.mask_threshold,
            )
        ):
            raise TypeError("C5-3 parity numeric fields have invalid types.")
        if self.split != "val" or self.test_used is not False or self.test_split_used is not False:
            raise ValueError("C5-3 TensorRT characterization must remain validation-only.")
        if (
            self.prediction_initial_confidence != 0.001
            or self.diagnostic_confidence != 0.25
            or self.prediction_iou != 0.7
            or self.max_detections != 300
            or self.retina_masks is not False
            or self.mask_threshold != 0.5
            or self.mask_resize != "opencv_inter_nearest"
        ):
            raise ValueError("C5-3 prediction normalization changed from the C4-2C contract.")
        if self.association != "greedy_max_mask_iou_positive_overlap":
            raise ValueError("C5-3 parity association policy is invalid.")
        if self.acceptance_mode != "metrics_only_pending_tensorrt_fp16_tolerance_approval":
            raise ValueError("C5-3 must characterize before defining FP16 numeric acceptance.")
        if (
            self.require_finite_outputs is not True
            or self.require_valid_output_shapes is not True
            or self.require_valid_class_ids is not True
            or self.numeric_thresholds is not None
        ):
            raise ValueError("C5-3 structural gates are incomplete or numeric gates leaked in.")
        self.benchmark.validate()


@dataclass(frozen=True)
class YoloTensorRtExportConfig:
    """Static batch-1 TensorRT FP16 engine build from the accepted exact ONNX."""

    schema_version: int
    export_id: str
    format: str
    task: str
    precision: str
    batch: int
    imgsz: int
    dynamic: bool
    workspace_gib: int
    device: int
    source_onnx_sha256: str
    source_onnx_metadata_sha256: str
    source_onnx_export_config_sha256: str
    output_root: Path
    parity: YoloTensorRtParityPolicy
    config_path: Path

    # ADD 2026-09-02: C5-3A engine build를 exact ONNX 기반 static FP16 contract로 제한한다.
    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or type(self.batch) is not int
            or type(self.imgsz) is not int
            or type(self.dynamic) is not bool
            or type(self.workspace_gib) is not int
            or type(self.device) is not int
        ):
            raise TypeError("C5-3 TensorRT config contains invalid scalar types.")
        if self.schema_version != TENSORRT_EXPORT_SCHEMA_VERSION:
            raise ValueError("Unsupported C5-3 TensorRT config schema version.")
        validate_artifact_id(self.export_id)
        if self.format != "engine" or self.task != "segment" or self.precision != "fp16":
            raise ValueError("C5-3A supports only YOLO segmentation TensorRT FP16 engines.")
        if (
            self.batch != 1
            or self.imgsz != 640
            or self.dynamic is not False
            or self.workspace_gib != 4
            or self.device != 0
        ):
            raise ValueError("C5-3A static FP16 build parameters changed without review.")
        if self.output_root != EXPECTED_TENSORRT_OUTPUT_ROOT:
            raise ValueError("C5-3 TensorRT output_root must remain in ignored artifacts/.")
        expected_hashes = (
            (self.source_onnx_sha256, EXPECTED_ONNX_SHA256),
            (self.source_onnx_metadata_sha256, EXPECTED_ONNX_METADATA_SHA256),
            (self.source_onnx_export_config_sha256, EXPECTED_ONNX_EXPORT_CONFIG_SHA256),
        )
        for observed, expected in expected_hashes:
            if (
                not isinstance(observed, str)
                or not is_sha256_digest(observed)
                or observed != expected
            ):
                raise ValueError("C5-3A source ONNX identity changed from accepted C5-2 bytes.")
        self.parity.validate()


@dataclass(frozen=True)
class TensorRtTensorContract:
    """One TensorRT engine I/O tensor."""

    name: str
    mode: str
    dtype: str
    shape: tuple[int, ...]

    # ADD 2026-09-02: TensorRT engine I/O가 static positive tensor인지 검증한다.
    def validate(self) -> None:
        if (
            not self.name
            or self.mode not in {"INPUT", "OUTPUT"}
            or not self.dtype
            or not self.shape
            or any(value <= 0 for value in self.shape)
        ):
            raise ValueError("C5-3 TensorRT I/O tensor contract is invalid.")


@dataclass(frozen=True)
class TensorRtEngineContract:
    """Static TensorRT segmentation engine interface and device-memory observation."""

    io_tensors: tuple[TensorRtTensorContract, ...]
    device_memory_size_bytes: int

    # ADD 2026-09-02: Engine interface가 frozen YOLO11n-seg 640 graph와 일치하는지 검증한다.
    def validate(self, *, config: YoloTensorRtExportConfig) -> None:
        if type(self.device_memory_size_bytes) is not int or self.device_memory_size_bytes < 0:
            raise ValueError("C5-3 TensorRT device memory size is invalid.")
        if len(self.io_tensors) != 3:
            raise ValueError("C5-3 TensorRT engine requires one input and two outputs.")
        for tensor in self.io_tensors:
            tensor.validate()
        by_name = {tensor.name: tensor for tensor in self.io_tensors}
        if set(by_name) != {"images", "output0", "output1"}:
            raise ValueError("C5-3 TensorRT engine tensor names changed from accepted ONNX.")
        if by_name["images"].mode != "INPUT" or by_name["images"].shape != (
            config.batch,
            3,
            config.imgsz,
            config.imgsz,
        ):
            raise ValueError("C5-3 TensorRT input contract is invalid.")
        if by_name["output0"].mode != "OUTPUT" or by_name["output0"].shape != (1, 39, 8400):
            raise ValueError("C5-3 TensorRT output0 shape changed from accepted ONNX.")
        if by_name["output1"].mode != "OUTPUT" or by_name["output1"].shape != (
            1,
            32,
            160,
            160,
        ):
            raise ValueError("C5-3 TensorRT output1 shape changed from accepted ONNX.")


@dataclass(frozen=True)
class YoloTensorRtExportMetadata:
    """Source, engine, environment, and repository provenance for one FP16 build."""

    schema_version: int
    artifact_type: str
    export_state: str
    export_id: str
    created_at: str
    source_experiment_id: str
    frozen_manifest_sha256: str
    source_model_sha256: str
    source_model_family: str
    source_task: str
    dataset_manifest_sha256: str
    source_onnx_sha256: str
    source_onnx_metadata_sha256: str
    source_onnx_export_config_sha256: str
    source_onnx_export_commit: str
    tensorrt_config_sha256: str
    tensorrt_config: Mapping[str, Any]
    engine_sha256: str
    engine_size_bytes: int
    engine: Mapping[str, Any]
    environment: Mapping[str, str]
    repository: Mapping[str, str | bool]
    test_used: bool
    test_split_used: bool

    # ADD 2026-09-02: TensorRT metadata의 exact ONNX/FP16/GPU/no-test provenance를 검증한다.
    def validate(self) -> None:
        if (
            self.schema_version != TENSORRT_EXPORT_SCHEMA_VERSION
            or self.artifact_type != "yolo_segmentation_tensorrt"
            or self.export_state != "TENSORRT_FP16_ENGINE_BUILT"
        ):
            raise ValueError("C5-3A TensorRT metadata lifecycle is invalid.")
        validate_artifact_id(self.export_id)
        if (
            self.source_model_family != "yolo11n-seg"
            or self.source_task != "segment"
            or self.test_used is not False
            or self.test_split_used is not False
        ):
            raise ValueError("C5-3A TensorRT metadata changed model/task or test seal.")
        for digest in (
            self.frozen_manifest_sha256,
            self.source_model_sha256,
            self.dataset_manifest_sha256,
            self.source_onnx_sha256,
            self.source_onnx_metadata_sha256,
            self.source_onnx_export_config_sha256,
            self.tensorrt_config_sha256,
            self.engine_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C5-3A TensorRT metadata contains an invalid SHA-256.")
        if (
            self.source_onnx_sha256 != EXPECTED_ONNX_SHA256
            or self.source_onnx_metadata_sha256 != EXPECTED_ONNX_METADATA_SHA256
            or self.source_onnx_export_config_sha256 != EXPECTED_ONNX_EXPORT_CONFIG_SHA256
            or self.source_onnx_export_commit != EXPECTED_ONNX_EXPORT_COMMIT
        ):
            raise ValueError("C5-3A TensorRT metadata is not bound to the accepted ONNX artifact.")
        if self.engine_size_bytes <= 0:
            raise ValueError("C5-3A TensorRT engine size must be positive.")
        _validate_timestamp(self.created_at)
        repository = _repository_provenance(self.repository)
        if repository.working_tree_dirty:
            raise ValueError("Official C5-3A TensorRT build requires a clean repository.")
        required_environment = {
            "python_version",
            "platform",
            "python_implementation",
            "torch_version",
            "ultralytics_version",
            "tensorrt_version",
            "cuda_runtime_version",
            "cuda_available",
            "gpu_name",
            "gpu_compute_capability",
            "gpu_total_memory_bytes",
        }
        if set(self.environment) != required_environment or any(
            not isinstance(value, str) or not value for value in self.environment.values()
        ):
            raise ValueError("C5-3A TensorRT environment fields are invalid.")
        if self.environment["cuda_available"] != "true":
            raise ValueError("C5-3A TensorRT metadata requires a CUDA GPU environment.")
        config = _config_contract_from_mapping(self.tensorrt_config)
        if config.export_id != self.export_id:
            raise ValueError("C5-3A embedded config export_id does not match metadata.")
        engine = _engine_contract_from_mapping(self.engine)
        engine.validate(config=config)
        try:
            json.dumps(asdict(self), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("C5-3A TensorRT metadata must be strict JSON data.") from exc

    # ADD 2026-09-02: TensorRT metadata를 deterministic strict JSON bytes로 직렬화한다.
    def to_json_bytes(self) -> bytes:
        self.validate()
        return (json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()

    # ADD 2026-09-02: Untrusted TensorRT metadata JSON을 typed strict contract로 복원한다.
    @classmethod
    def from_json_dict(cls, raw: object) -> YoloTensorRtExportMetadata:
        if not isinstance(raw, dict):
            raise ValueError("C5-3A TensorRT metadata root must be an object.")
        try:
            metadata = cls(**cast(dict[str, Any], raw))
        except TypeError as exc:
            raise ValueError("C5-3A TensorRT metadata fields do not match schema.") from exc
        metadata.validate()
        return metadata


@dataclass(frozen=True)
class YoloTensorRtExportArtifacts:
    """Published TensorRT engine and metadata paths."""

    output_dir: Path
    engine_path: Path
    metadata_path: Path
    metadata: YoloTensorRtExportMetadata


type ProvenanceResolver = Callable[[Path], RepositoryProvenance]
type EngineBuilder = Callable[[Path, Path, YoloTensorRtExportConfig], None]
type EngineInspector = Callable[[Path], TensorRtEngineContract]
type EnvironmentResolver = Callable[[int], Mapping[str, str]]


# ADD 2026-09-02: Timestamp가 timezone-aware ISO-8601인지 검증한다.
def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("C5-3 timestamp must be ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("C5-3 timestamp must include a timezone offset.")


# ADD 2026-09-02: Repository provenance mapping을 strict typed object로 검증한다.
def _repository_provenance(raw: Mapping[str, str | bool]) -> RepositoryProvenance:
    if set(raw) != {"git_commit", "working_tree_dirty"}:
        raise ValueError("C5-3 repository provenance fields are invalid.")
    if type(raw["working_tree_dirty"]) is not bool:
        raise ValueError("C5-3 working_tree_dirty must be boolean.")
    provenance = RepositoryProvenance(
        git_commit=str(raw["git_commit"]),
        working_tree_dirty=cast(bool, raw["working_tree_dirty"]),
    )
    provenance.validate()
    return provenance


# ADD 2026-09-02: Repository-owned path가 repository_root 밖으로 이탈하지 않게 한다.
def _repository_path(repository_root: Path, path: Path, *, field: str) -> Path:
    root = repository_root.resolve()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"C5-3 {field} must remain inside repository_root.") from exc
    return resolved


# ADD 2026-09-02: TensorRT module을 local CI import path에 강제하지 않고 GPU runtime에서만 로드한다.
def _import_tensorrt() -> Any:
    try:
        return importlib.import_module("tensorrt")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "C5-3 TensorRT runtime is unavailable. "
            "Run this stage in the pinned NVIDIA GPU environment."
        ) from exc


# ADD 2026-09-02: Repository TensorRT FP16 YAML을 strict typed contract로 로드한다.
def load_yolo_tensorrt_export_config(path: Path) -> YoloTensorRtExportConfig:
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("Cannot read C5-3 TensorRT export config.") from exc
    if not isinstance(raw, dict):
        raise ValueError("C5-3 TensorRT export config root must be a mapping.")
    values = cast(dict[str, Any], raw)
    expected = {
        "schema_version",
        "export_id",
        "format",
        "task",
        "precision",
        "batch",
        "imgsz",
        "dynamic",
        "workspace_gib",
        "device",
        "source_onnx_sha256",
        "source_onnx_metadata_sha256",
        "source_onnx_export_config_sha256",
        "output_root",
        "parity",
    }
    if set(values) != expected or not isinstance(values.get("parity"), dict):
        raise ValueError("C5-3 TensorRT export config fields do not match schema.")
    parity_values = cast(dict[str, Any], values["parity"])
    benchmark_raw = parity_values.get("benchmark")
    if not isinstance(benchmark_raw, dict):
        raise ValueError("C5-3 TensorRT benchmark config must be a mapping.")
    parity_values = dict(parity_values)
    parity_values["benchmark"] = TensorRtBenchmarkPolicy(**cast(dict[str, Any], benchmark_raw))
    try:
        config = YoloTensorRtExportConfig(
            **{key: value for key, value in values.items() if key not in {"output_root", "parity"}},
            output_root=Path(str(values["output_root"])),
            parity=YoloTensorRtParityPolicy(**parity_values),
            config_path=path.resolve(),
        )
    except TypeError as exc:
        raise ValueError("C5-3 TensorRT export config typed values are invalid.") from exc
    config.validate()
    return config


# ADD 2026-09-02: Embedded config mapping을 metadata validation용 typed contract로 복원한다.
def _config_contract_from_mapping(raw: Mapping[str, Any]) -> YoloTensorRtExportConfig:
    values = dict(raw)
    parity_raw = values.pop("parity", None)
    output_root = values.pop("output_root", None)
    if not isinstance(parity_raw, dict) or output_root is None:
        raise ValueError("C5-3 embedded TensorRT config is incomplete.")
    parity_values = dict(parity_raw)
    benchmark_raw = parity_values.get("benchmark")
    if not isinstance(benchmark_raw, dict):
        raise ValueError("C5-3 embedded benchmark config is invalid.")
    parity_values["benchmark"] = TensorRtBenchmarkPolicy(**cast(dict[str, Any], benchmark_raw))
    try:
        config = YoloTensorRtExportConfig(
            **values,
            output_root=Path(str(output_root)),
            parity=YoloTensorRtParityPolicy(**parity_values),
            config_path=Path("<embedded>"),
        )
    except TypeError as exc:
        raise ValueError("C5-3 embedded TensorRT config fields are invalid.") from exc
    config.validate()
    return config


# ADD 2026-09-02: TensorRT engine mapping을 strict typed I/O contract로 복원한다.
def _engine_contract_from_mapping(raw: Mapping[str, Any]) -> TensorRtEngineContract:
    if set(raw) != {"io_tensors", "device_memory_size_bytes"}:
        raise ValueError("C5-3 embedded engine fields are invalid.")
    tensors_raw = raw["io_tensors"]
    if not isinstance(tensors_raw, list | tuple):
        raise ValueError("C5-3 embedded engine io_tensors must be an array.")
    tensors: list[TensorRtTensorContract] = []
    for item in tensors_raw:
        if not isinstance(item, dict) or set(item) != {"name", "mode", "dtype", "shape"}:
            raise ValueError("C5-3 embedded engine tensor fields are invalid.")
        shape = item["shape"]
        if not isinstance(shape, list | tuple):
            raise ValueError("C5-3 embedded engine tensor shape must be an array.")
        tensors.append(
            TensorRtTensorContract(
                name=str(item["name"]),
                mode=str(item["mode"]),
                dtype=str(item["dtype"]),
                shape=tuple(int(value) for value in shape),
            )
        )
    memory = raw["device_memory_size_bytes"]
    if type(memory) is not int:
        raise ValueError("C5-3 embedded engine memory size must be an integer.")
    return TensorRtEngineContract(io_tensors=tuple(tensors), device_memory_size_bytes=memory)


# ADD 2026-09-02: Exact preserved C5-1 ONNX binary와 metadata identity만 TensorRT source로 허용한다.
def verify_tensorrt_source_onnx_identity(
    *,
    repository_root: Path,
    artifact_dir: Path,
    config: YoloTensorRtExportConfig,
) -> YoloOnnxExportMetadata:
    expected_dir = (
        repository_root.resolve() / EXPECTED_ONNX_ARTIFACT_ROOT / EXPECTED_ONNX_EXPORT_ID
    ).resolve()
    if artifact_dir.resolve() != expected_dir:
        raise ValueError("C5-3 source ONNX must use the repository ignored C5-1 namespace.")
    metadata = load_yolo_onnx_artifact(artifact_dir)
    metadata_sha = sha256_file(artifact_dir / ONNX_METADATA_FILENAME)
    if (
        metadata.export_id != EXPECTED_ONNX_EXPORT_ID
        or metadata.onnx_sha256 != config.source_onnx_sha256
        or metadata_sha != config.source_onnx_metadata_sha256
        or metadata.export_config_sha256 != config.source_onnx_export_config_sha256
        or str(metadata.repository["git_commit"]) != EXPECTED_ONNX_EXPORT_COMMIT
        or metadata.test_used is not False
        or metadata.test_split_used is not False
    ):
        raise ValueError("C5-3 source ONNX artifact does not match accepted C5-2 identity.")
    return metadata


# ADD 2026-09-02: TensorRT builder가 실행될 GPU/CUDA/runtime identity를 기록한다.
def resolve_tensorrt_environment(device: int) -> Mapping[str, str]:
    trt = _import_tensorrt()
    if not torch.cuda.is_available() or device < 0 or device >= torch.cuda.device_count():
        raise RuntimeError("C5-3A TensorRT build requires an available CUDA device.")
    props = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    cuda_runtime = torch.version.cuda
    if not cuda_runtime:
        raise RuntimeError("C5-3A PyTorch runtime does not expose a CUDA runtime version.")
    from ultralytics import __version__ as ultralytics_version

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "python_implementation": sys.implementation.name,
        "torch_version": str(torch.__version__),
        "ultralytics_version": ultralytics_version,
        "tensorrt_version": str(trt.__version__),
        "cuda_runtime_version": str(cuda_runtime),
        "cuda_available": "true",
        "gpu_name": str(props.name),
        "gpu_compute_capability": f"{capability[0]}.{capability[1]}",
        "gpu_total_memory_bytes": str(int(props.total_memory)),
    }


# ADD 2026-09-02: TensorRT Python API로 exact ONNX를 static FP16 serialized engine으로 build한다.
def build_tensorrt_fp16_engine(
    onnx_path: Path,
    engine_path: Path,
    config: YoloTensorRtExportConfig,
) -> None:
    trt = _import_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)
    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        errors = [str(parser.get_error(index)) for index in range(int(parser.num_errors))]
        detail = " | ".join(errors) if errors else "unknown parser error"
        raise RuntimeError(f"C5-3A TensorRT ONNX parse failed: {detail}")
    if not bool(builder.platform_has_fast_fp16):
        raise RuntimeError("C5-3A GPU does not report fast FP16 support.")

    builder_config = builder.create_builder_config()
    workspace_bytes = config.workspace_gib * 1024**3
    set_pool = getattr(builder_config, "set_memory_pool_limit", None)
    if callable(set_pool):
        set_pool(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        builder_config.max_workspace_size = workspace_bytes
    builder_config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, builder_config)
    if serialized is None:
        raise RuntimeError("C5-3A TensorRT builder did not produce serialized engine bytes.")
    engine_path.write_bytes(bytes(serialized))


# ADD 2026-09-02: Serialized engine의 static I/O shape와 device memory를 관측한다.
def inspect_tensorrt_engine(engine_path: Path) -> TensorRtEngineContract:
    trt = _import_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("C5-3A TensorRT engine deserialization failed.")

    tensors: list[TensorRtTensorContract] = []
    if hasattr(engine, "num_io_tensors"):
        for index in range(int(engine.num_io_tensors)):
            name = str(engine.get_tensor_name(index))
            mode = str(engine.get_tensor_mode(name)).split(".")[-1].upper()
            dtype = str(engine.get_tensor_dtype(name))
            shape = tuple(int(value) for value in engine.get_tensor_shape(name))
            tensors.append(TensorRtTensorContract(name=name, mode=mode, dtype=dtype, shape=shape))
    else:
        for index in range(int(engine.num_bindings)):
            name = str(engine.get_binding_name(index))
            mode = "INPUT" if bool(engine.binding_is_input(index)) else "OUTPUT"
            dtype = str(engine.get_binding_dtype(index))
            shape = tuple(int(value) for value in engine.get_binding_shape(index))
            tensors.append(TensorRtTensorContract(name=name, mode=mode, dtype=dtype, shape=shape))

    memory = getattr(engine, "device_memory_size_v2", None)
    if memory is None:
        memory = getattr(engine, "device_memory_size", None)
    memory_size_bytes = 0 if memory is None else int(memory)
    return TensorRtEngineContract(
        io_tensors=tuple(tensors),
        device_memory_size_bytes=memory_size_bytes,
    )


# ADD 2026-09-02: Exact ONNX에서 TensorRT FP16 engine과 provenance metadata를 atomic publish한다.
def export_frozen_yolo_tensorrt(
    *,
    repository_root: Path,
    onnx_artifact_dir: Path,
    config: YoloTensorRtExportConfig,
    created_at: str,
    provenance_resolver: ProvenanceResolver = resolve_repository_provenance,
    engine_builder: EngineBuilder = build_tensorrt_fp16_engine,
    engine_inspector: EngineInspector = inspect_tensorrt_engine,
    environment_resolver: EnvironmentResolver = resolve_tensorrt_environment,
) -> YoloTensorRtExportArtifacts:
    root = repository_root.resolve()
    config.validate()
    _validate_timestamp(created_at)
    _repository_path(root, config.config_path, field="TensorRT export config")

    provenance = provenance_resolver(root)
    provenance.validate()
    if provenance.working_tree_dirty:
        raise ValueError("Official C5-3A TensorRT build requires a clean committed repository.")

    onnx_metadata = verify_tensorrt_source_onnx_identity(
        repository_root=root,
        artifact_dir=onnx_artifact_dir,
        config=config,
    )
    environment = dict(environment_resolver(config.device))
    if environment.get("cuda_available") != "true":
        raise RuntimeError("C5-3A TensorRT build requires CUDA.")

    output_root = _repository_path(
        root,
        config.output_root,
        field="TensorRT artifact output root",
    )
    output_dir = output_root / config.export_id
    staging_dir = output_root / f".{config.export_id}.staging"
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError("C5-3A TensorRT export namespace already exists.")
    output_root.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(exist_ok=False)

    try:
        engine_path = staging_dir / TENSORRT_ENGINE_FILENAME
        source_onnx_path = onnx_artifact_dir / ONNX_MODEL_FILENAME
        before_onnx_sha = sha256_file(source_onnx_path)
        engine_builder(source_onnx_path, engine_path, config)
        if not engine_path.is_file() or engine_path.stat().st_size <= 0:
            raise RuntimeError("C5-3A TensorRT builder did not publish a non-empty engine.")
        if sha256_file(source_onnx_path) != before_onnx_sha:
            raise RuntimeError("C5-3A TensorRT build changed the source ONNX bytes.")

        engine_contract = engine_inspector(engine_path)
        engine_contract.validate(config=config)
        metadata = YoloTensorRtExportMetadata(
            schema_version=TENSORRT_EXPORT_SCHEMA_VERSION,
            artifact_type="yolo_segmentation_tensorrt",
            export_state="TENSORRT_FP16_ENGINE_BUILT",
            export_id=config.export_id,
            created_at=created_at,
            source_experiment_id=onnx_metadata.source_experiment_id,
            frozen_manifest_sha256=onnx_metadata.frozen_manifest_sha256,
            source_model_sha256=onnx_metadata.source_model_sha256,
            source_model_family=onnx_metadata.source_model_family,
            source_task=onnx_metadata.source_task,
            dataset_manifest_sha256=onnx_metadata.dataset_manifest_sha256,
            source_onnx_sha256=onnx_metadata.onnx_sha256,
            source_onnx_metadata_sha256=sha256_file(onnx_artifact_dir / ONNX_METADATA_FILENAME),
            source_onnx_export_config_sha256=onnx_metadata.export_config_sha256,
            source_onnx_export_commit=str(onnx_metadata.repository["git_commit"]),
            tensorrt_config_sha256=sha256_file(config.config_path),
            tensorrt_config=_export_config_mapping(config),
            engine_sha256=sha256_file(engine_path),
            engine_size_bytes=engine_path.stat().st_size,
            engine=asdict(engine_contract),
            environment=environment,
            repository=provenance.to_json_dict(),
            test_used=False,
            test_split_used=False,
        )
        metadata_path = staging_dir / TENSORRT_METADATA_FILENAME
        metadata_path.write_bytes(metadata.to_json_bytes())
        staging_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return YoloTensorRtExportArtifacts(
        output_dir=output_dir,
        engine_path=output_dir / TENSORRT_ENGINE_FILENAME,
        metadata_path=output_dir / TENSORRT_METADATA_FILENAME,
        metadata=metadata,
    )


# ADD 2026-09-02: Config를 path-free stable evidence mapping으로 변환한다.
def _export_config_mapping(config: YoloTensorRtExportConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("config_path")
    payload["output_root"] = str(config.output_root)
    return payload


# ADD 2026-09-02: Ignored TensorRT artifact metadata와 engine bytes identity를 함께 검증한다.
def load_yolo_tensorrt_artifact(artifact_dir: Path) -> YoloTensorRtExportMetadata:
    engine_path = artifact_dir / TENSORRT_ENGINE_FILENAME
    metadata_path = artifact_dir / TENSORRT_METADATA_FILENAME
    if not engine_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("C5-3A TensorRT artifact requires model.engine and metadata.json.")
    try:
        raw: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read C5-3A TensorRT artifact metadata.") from exc
    metadata = YoloTensorRtExportMetadata.from_json_dict(raw)
    if sha256_file(engine_path) != metadata.engine_sha256:
        raise ValueError("C5-3A TensorRT engine SHA does not match metadata.")
    return metadata
