"""C5-4B2 TensorRT INT8 engine build from exact explicit-Q/DQ ONNX."""

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

import onnx
import torch
import yaml

from ml.deployment.yolo_onnx import EXPECTED_CLASSES
from ml.deployment.yolo_tensorrt_int8_quantization import inspect_qdq_graph
from ml.evaluation.final_benchmark import RepositoryProvenance, resolve_repository_provenance
from ml.training.yolo_segmentation import validate_artifact_id
from shared.hashing import is_sha256_digest, sha256_file

INT8_ENGINE_SCHEMA_VERSION = 1
DEFAULT_TENSORRT_INT8_ENGINE_CONFIG = Path(
    "configs/export/yolo_segmentation_tensorrt_int8_engine.yaml"
)
INT8_ENGINE_FILENAME = "model.engine"
INT8_ENGINE_METADATA_FILENAME = "metadata.json"
EXPECTED_QDQ_ONNX_FILENAME = "model.int8.qdq.onnx"
EXPECTED_QDQ_METADATA_FILENAME = "metadata.json"
EXPECTED_QDQ_ONNX_SHA256 = "d7c9af3ab3c2f71e88de26be71abe80f113f2e1c359d2a532a24079fa9b4dd00"
EXPECTED_QDQ_METADATA_SHA256 = "8c3b215082ba111d4f932f4e021a9bc11866c49ecec788a52f20b2f9fe244fa7"
EXPECTED_QDQ_EVIDENCE_ZIP_SHA256 = (
    "00f925d0ce5f6106d441822e419a039a736831c0d2c13835cfd01b62fad50990"
)
EXPECTED_QDQ_RUN_SUMMARY_SHA256 = "c6b4dd790ae9a2ff312b9336d46c87f0efc03f3a2364ddda0b014a3f4405a60c"
EXPECTED_QDQ_RUN_COMMIT = "8e489c80ef9527a044b100cc96172d179947e051"
EXPECTED_QDQ_INT8_CONTRACT_SHA256 = (
    "18309302e45855e506628bb5e262886fc2cb366f8758fc100c55aaf6dbf3c37a"
)
EXPECTED_QDQ_QUANTIZATION_ID = "c5_4a_yolo11n_seg_tensorrt_int8_qdq_ptq"
EXPECTED_QDQ_OPSET = 19
EXPECTED_Q_COUNT = 211
EXPECTED_DQ_COUNT = 211
EXPECTED_CALIBRATION_COUNT = 84
EXPECTED_OUTPUT_ROOT = Path("artifacts/deployment/yolo_segmentation/tensorrt_int8/engine")


@dataclass(frozen=True)
class Int8EngineSourcePolicy:
    """Exact successful C5-4B1 Q/DQ artifact identity."""

    quantization_id: str
    qdq_onnx_sha256: str
    qdq_metadata_sha256: str
    qdq_evidence_zip_sha256: str
    qdq_run_summary_sha256: str
    qdq_run_commit: str
    qdq_int8_contract_sha256: str
    qdq_opset: int
    quantize_linear_count: int
    dequantize_linear_count: int
    calibration_sample_count: int
    validation_used: bool
    test_used: bool
    test_split_used: bool

    # ADD 2026-09-03: B2 source를 successful B1 exact Q/DQ evidence로 고정한다.
    def validate(self) -> None:
        expected_pairs = (
            (self.quantization_id, EXPECTED_QDQ_QUANTIZATION_ID),
            (self.qdq_onnx_sha256, EXPECTED_QDQ_ONNX_SHA256),
            (self.qdq_metadata_sha256, EXPECTED_QDQ_METADATA_SHA256),
            (self.qdq_evidence_zip_sha256, EXPECTED_QDQ_EVIDENCE_ZIP_SHA256),
            (self.qdq_run_summary_sha256, EXPECTED_QDQ_RUN_SUMMARY_SHA256),
            (self.qdq_run_commit, EXPECTED_QDQ_RUN_COMMIT),
            (self.qdq_int8_contract_sha256, EXPECTED_QDQ_INT8_CONTRACT_SHA256),
        )
        if any(observed != expected for observed, expected in expected_pairs):
            raise ValueError("C5-4B2 source identity changed from successful C5-4B1 evidence.")
        for digest in (
            self.qdq_onnx_sha256,
            self.qdq_metadata_sha256,
            self.qdq_evidence_zip_sha256,
            self.qdq_run_summary_sha256,
            self.qdq_int8_contract_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C5-4B2 source contains an invalid SHA-256.")
        if (
            self.qdq_opset != EXPECTED_QDQ_OPSET
            or self.quantize_linear_count != EXPECTED_Q_COUNT
            or self.dequantize_linear_count != EXPECTED_DQ_COUNT
            or self.calibration_sample_count != EXPECTED_CALIBRATION_COUNT
            or self.validation_used is not False
            or self.test_used is not False
            or self.test_split_used is not False
        ):
            raise ValueError("C5-4B2 source graph or data seal changed from C5-4B1.")


@dataclass(frozen=True)
class Int8EngineBuildPolicy:
    """TensorRT explicit quantization build behavior."""

    explicit_quantization: bool
    strongly_typed_network: bool
    builder_int8_flag: bool
    builder_fp16_flag: bool
    legacy_calibrator: bool

    # ADD 2026-09-03: Q/DQ build에서 precision flag와 calibrator 재사용을 금지한다.
    def validate(self) -> None:
        values = (
            self.explicit_quantization,
            self.strongly_typed_network,
            self.builder_int8_flag,
            self.builder_fp16_flag,
            self.legacy_calibrator,
        )
        if any(type(value) is not bool for value in values):
            raise TypeError("C5-4B2 build policy requires strict booleans.")
        if (
            self.explicit_quantization is not True
            or self.strongly_typed_network is not True
            or self.builder_int8_flag is not False
            or self.builder_fp16_flag is not False
            or self.legacy_calibrator is not False
        ):
            raise ValueError(
                "C5-4B2 must use strongly typed explicit Q/DQ without precision flags."
            )


@dataclass(frozen=True)
class YoloTensorRtInt8EngineConfig:
    """Static C5-4B2 TensorRT engine contract."""

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
    output_root: Path
    source: Int8EngineSourcePolicy
    build: Int8EngineBuildPolicy
    config_path: Path

    # ADD 2026-09-03: C5-4B2 static TensorRT INT8 engine boundary를 검증한다.
    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or type(self.batch) is not int
            or type(self.imgsz) is not int
            or type(self.dynamic) is not bool
            or type(self.workspace_gib) is not int
            or type(self.device) is not int
        ):
            raise TypeError("C5-4B2 config scalar types are invalid.")
        if self.schema_version != INT8_ENGINE_SCHEMA_VERSION:
            raise ValueError("Unsupported C5-4B2 schema version.")
        validate_artifact_id(self.export_id)
        if self.format != "engine" or self.task != "segment" or self.precision != "int8_qdq":
            raise ValueError("C5-4B2 supports only explicit-Q/DQ TensorRT segmentation engines.")
        if (
            self.batch != 1
            or self.imgsz != 640
            or self.dynamic is not False
            or self.workspace_gib != 4
            or self.device != 0
            or self.output_root != EXPECTED_OUTPUT_ROOT
        ):
            raise ValueError("C5-4B2 static engine parameters changed without review.")
        self.source.validate()
        self.build.validate()


@dataclass(frozen=True)
class Int8EngineTensor:
    """One TensorRT engine I/O tensor."""

    name: str
    mode: str
    dtype: str
    shape: tuple[int, ...]

    def validate(self) -> None:
        if (
            not self.name
            or self.mode not in {"INPUT", "OUTPUT"}
            or not self.dtype
            or not self.shape
            or any(value <= 0 for value in self.shape)
        ):
            raise ValueError("C5-4B2 engine tensor contract is invalid.")


@dataclass(frozen=True)
class Int8EngineContract:
    """TensorRT INT8 engine external interface."""

    io_tensors: tuple[Int8EngineTensor, ...]
    device_memory_size_bytes: int

    # ADD 2026-09-03: Engine I/O names/shapes를 frozen segmentation graph와 대조한다.
    def validate(self, *, config: YoloTensorRtInt8EngineConfig) -> None:
        if type(self.device_memory_size_bytes) is not int or self.device_memory_size_bytes < 0:
            raise ValueError("C5-4B2 engine memory observation is invalid.")
        if len(self.io_tensors) != 3:
            raise ValueError("C5-4B2 engine requires one input and two outputs.")
        for tensor in self.io_tensors:
            tensor.validate()
        by_name = {tensor.name: tensor for tensor in self.io_tensors}
        if set(by_name) != {"images", "output0", "output1"}:
            raise ValueError("C5-4B2 engine tensor names changed.")
        if by_name["images"].mode != "INPUT" or by_name["images"].shape != (
            config.batch,
            3,
            config.imgsz,
            config.imgsz,
        ):
            raise ValueError("C5-4B2 engine input shape is invalid.")
        if by_name["output0"].mode != "OUTPUT" or by_name["output0"].shape != (1, 39, 8400):
            raise ValueError("C5-4B2 output0 shape is invalid.")
        if by_name["output1"].mode != "OUTPUT" or by_name["output1"].shape != (
            1,
            32,
            160,
            160,
        ):
            raise ValueError("C5-4B2 output1 shape is invalid.")


@dataclass(frozen=True)
class Int8EngineMetadata:
    """C5-4B2 engine provenance."""

    schema_version: int
    artifact_type: str
    state: str
    export_id: str
    created_at: str
    source_qdq_onnx_sha256: str
    source_qdq_metadata_sha256: str
    source_qdq_evidence_zip_sha256: str
    source_qdq_run_commit: str
    tensorrt_int8_engine_config_sha256: str
    engine_sha256: str
    engine_size_bytes: int
    engine: Mapping[str, Any]
    environment: Mapping[str, str]
    repository: Mapping[str, str | bool]
    validation_used: bool
    test_used: bool
    test_split_used: bool

    # ADD 2026-09-03: B2 metadata가 exact B1 source와 no-test seal을 유지하는지 검증한다.
    def validate(self, *, config: YoloTensorRtInt8EngineConfig) -> None:
        if (
            self.schema_version != INT8_ENGINE_SCHEMA_VERSION
            or self.artifact_type != "yolo_segmentation_tensorrt_int8"
            or self.state != "TENSORRT_INT8_ENGINE_BUILT"
            or self.export_id != config.export_id
        ):
            raise ValueError("C5-4B2 metadata lifecycle is invalid.")
        for digest in (
            self.source_qdq_onnx_sha256,
            self.source_qdq_metadata_sha256,
            self.source_qdq_evidence_zip_sha256,
            self.tensorrt_int8_engine_config_sha256,
            self.engine_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C5-4B2 metadata contains invalid SHA-256.")
        if (
            self.source_qdq_onnx_sha256 != config.source.qdq_onnx_sha256
            or self.source_qdq_metadata_sha256 != config.source.qdq_metadata_sha256
            or self.source_qdq_evidence_zip_sha256 != config.source.qdq_evidence_zip_sha256
            or self.source_qdq_run_commit != config.source.qdq_run_commit
            or self.engine_size_bytes <= 0
            or self.validation_used is not False
            or self.test_used is not False
            or self.test_split_used is not False
        ):
            raise ValueError("C5-4B2 metadata source or data seal is invalid.")
        _validate_timestamp(self.created_at)
        provenance = _repository_provenance(self.repository)
        if provenance.working_tree_dirty:
            raise ValueError("Official C5-4B2 build requires a clean repository.")
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
            raise ValueError("C5-4B2 environment fields are incomplete.")
        if self.environment["cuda_available"] != "true":
            raise ValueError("C5-4B2 requires CUDA.")
        contract = _engine_contract_from_mapping(self.engine)
        contract.validate(config=config)
        try:
            json.dumps(asdict(self), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("C5-4B2 metadata must be strict JSON data.") from exc

    def to_json_bytes(self, *, config: YoloTensorRtInt8EngineConfig) -> bytes:
        self.validate(config=config)
        return (json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


@dataclass(frozen=True)
class Int8EngineArtifacts:
    """Published C5-4B2 engine files."""

    output_dir: Path
    engine_path: Path
    metadata_path: Path
    metadata: Int8EngineMetadata


type ProvenanceResolver = Callable[[Path], RepositoryProvenance]
type EnvironmentResolver = Callable[[int], Mapping[str, str]]
type EngineBuilder = Callable[[Path, Path, YoloTensorRtInt8EngineConfig], None]
type EngineInspector = Callable[[Path, YoloTensorRtInt8EngineConfig], Int8EngineContract]


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("C5-4B2 timestamp must be ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("C5-4B2 timestamp requires timezone offset.")


def _repository_provenance(raw: Mapping[str, str | bool]) -> RepositoryProvenance:
    if set(raw) != {"git_commit", "working_tree_dirty"}:
        raise ValueError("C5-4B2 repository provenance fields are invalid.")
    if type(raw["working_tree_dirty"]) is not bool:
        raise TypeError("C5-4B2 working_tree_dirty must be boolean.")
    provenance = RepositoryProvenance(
        git_commit=str(raw["git_commit"]),
        working_tree_dirty=cast(bool, raw["working_tree_dirty"]),
    )
    provenance.validate()
    return provenance


def _import_tensorrt() -> Any:
    try:
        return importlib.import_module("tensorrt")
    except ModuleNotFoundError as exc:
        raise RuntimeError("C5-4B2 requires TensorRT in the NVIDIA GPU runtime.") from exc


def load_yolo_tensorrt_int8_engine_config(path: Path) -> YoloTensorRtInt8EngineConfig:
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("Cannot read C5-4B2 TensorRT INT8 config.") from exc
    if not isinstance(raw, dict):
        raise ValueError("C5-4B2 config root must be a mapping.")
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
        "output_root",
        "source",
        "build",
    }
    if (
        set(values) != expected
        or not isinstance(values["source"], dict)
        or not isinstance(values["build"], dict)
    ):
        raise ValueError("C5-4B2 config fields do not match schema.")
    try:
        config = YoloTensorRtInt8EngineConfig(
            **{
                key: value
                for key, value in values.items()
                if key not in {"output_root", "source", "build"}
            },
            output_root=Path(str(values["output_root"])),
            source=Int8EngineSourcePolicy(**cast(dict[str, Any], values["source"])),
            build=Int8EngineBuildPolicy(**cast(dict[str, Any], values["build"])),
            config_path=path.resolve(),
        )
    except TypeError as exc:
        raise ValueError("C5-4B2 config typed values are invalid.") from exc
    config.validate()
    return config


# ADD 2026-09-03: Exact C5-4B1 Q/DQ ONNX와 metadata만 B2 source로 허용한다.
def verify_qdq_source(
    *,
    artifact_dir: Path,
    config: YoloTensorRtInt8EngineConfig,
) -> Mapping[str, Any]:
    model_path = artifact_dir / EXPECTED_QDQ_ONNX_FILENAME
    metadata_path = artifact_dir / EXPECTED_QDQ_METADATA_FILENAME
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("C5-4B2 source Q/DQ artifact is incomplete.")
    if sha256_file(model_path) != config.source.qdq_onnx_sha256:
        raise ValueError("C5-4B2 source Q/DQ ONNX SHA mismatch.")
    if sha256_file(metadata_path) != config.source.qdq_metadata_sha256:
        raise ValueError("C5-4B2 source Q/DQ metadata SHA mismatch.")

    raw: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("C5-4B2 source metadata must be an object.")
    values = cast(dict[str, Any], raw)
    repository = values.get("repository")
    if (
        values.get("state") != "INT8_QDQ_ONNX_QUANTIZED"
        or values.get("quantization_id") != config.source.quantization_id
        or values.get("quantized_onnx_sha256") != config.source.qdq_onnx_sha256
        or values.get("int8_contract_sha256") != config.source.qdq_int8_contract_sha256
        or values.get("quantize_linear_count") != config.source.quantize_linear_count
        or values.get("dequantize_linear_count") != config.source.dequantize_linear_count
        or values.get("calibration_sample_count") != config.source.calibration_sample_count
        or values.get("validation_used") is not False
        or values.get("test_used") is not False
        or values.get("test_split_used") is not False
        or not isinstance(repository, dict)
        or repository.get("git_commit") != config.source.qdq_run_commit
        or repository.get("working_tree_dirty") is not False
    ):
        raise ValueError("C5-4B2 source metadata changed from successful B1 evidence.")

    graph = inspect_qdq_graph(model_path)
    if (
        graph.quantize_linear_count != config.source.quantize_linear_count
        or graph.dequantize_linear_count != config.source.dequantize_linear_count
    ):
        raise ValueError("C5-4B2 source Q/DQ graph counts changed.")

    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    opset = next(item.version for item in model.opset_import if item.domain in {"", "ai.onnx"})
    if opset != config.source.qdq_opset:
        raise ValueError("C5-4B2 source Q/DQ opset changed.")
    return values


def _encode_engine_container(
    serialized_engine: bytes,
    config: YoloTensorRtInt8EngineConfig,
) -> bytes:
    if not serialized_engine:
        raise ValueError("C5-4B2 serialized engine must be non-empty.")
    header = {
        "args": {"dynamic": config.dynamic, "nms": False},
        "batch": config.batch,
        "channels": 3,
        "imgsz": [config.imgsz, config.imgsz],
        "names": dict(EXPECTED_CLASSES),
        "stride": 32,
        "task": config.task,
    }
    header_bytes = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return len(header_bytes).to_bytes(4, "little", signed=True) + header_bytes + serialized_engine


def _read_engine_container(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    if len(data) < 5:
        raise ValueError("C5-4B2 engine container is too small.")
    length = int.from_bytes(data[:4], "little", signed=True)
    if length <= 0 or 4 + length >= len(data):
        raise ValueError("C5-4B2 engine header length is invalid.")
    raw: object = json.loads(data[4 : 4 + length].decode())
    if not isinstance(raw, dict):
        raise ValueError("C5-4B2 engine header must be an object.")
    return cast(dict[str, Any], raw), data[4 + length :]


def _validate_engine_header(
    header: Mapping[str, Any],
    config: YoloTensorRtInt8EngineConfig,
) -> None:
    names_raw = header.get("names")
    if not isinstance(names_raw, dict):
        raise ValueError("C5-4B2 engine header is missing class names.")
    names = {int(key): str(value) for key, value in names_raw.items()}
    if (
        names != EXPECTED_CLASSES
        or header.get("task") != config.task
        or header.get("batch") != config.batch
        or header.get("channels") != 3
        or header.get("imgsz") != [config.imgsz, config.imgsz]
        or header.get("stride") != 32
        or header.get("args") != {"dynamic": False, "nms": False}
    ):
        raise ValueError("C5-4B2 Ultralytics engine header changed frozen identity.")


def resolve_tensorrt_int8_environment(device: int) -> Mapping[str, str]:
    trt = _import_tensorrt()
    if not torch.cuda.is_available() or device < 0 or device >= torch.cuda.device_count():
        raise RuntimeError("C5-4B2 requires an available CUDA device.")
    props = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    if not torch.version.cuda:
        raise RuntimeError("C5-4B2 PyTorch runtime does not expose CUDA version.")
    from ultralytics import __version__ as ultralytics_version

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "python_implementation": sys.implementation.name,
        "torch_version": str(torch.__version__),
        "ultralytics_version": ultralytics_version,
        "tensorrt_version": str(trt.__version__),
        "cuda_runtime_version": str(torch.version.cuda),
        "cuda_available": "true",
        "gpu_name": str(props.name),
        "gpu_compute_capability": f"{capability[0]}.{capability[1]}",
        "gpu_total_memory_bytes": str(int(props.total_memory)),
    }


# ADD 2026-09-03: Strongly typed network가 Q/DQ precision을 그대로 해석하게 build한다.
def build_tensorrt_int8_engine(
    qdq_onnx_path: Path,
    engine_path: Path,
    config: YoloTensorRtInt8EngineConfig,
) -> None:
    config.build.validate()
    trt = _import_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(qdq_onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(int(parser.num_errors))]
        detail = " | ".join(errors) if errors else "unknown parser error"
        raise RuntimeError(f"C5-4B2 TensorRT Q/DQ parse failed: {detail}")

    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        config.workspace_gib * 1024**3,
    )
    serialized = builder.build_serialized_network(network, builder_config)
    if serialized is None:
        raise RuntimeError("C5-4B2 TensorRT builder returned no engine.")
    engine_path.write_bytes(_encode_engine_container(bytes(serialized), config))


def inspect_tensorrt_int8_engine(
    engine_path: Path,
    config: YoloTensorRtInt8EngineConfig,
) -> Int8EngineContract:
    header, serialized = _read_engine_container(engine_path)
    _validate_engine_header(header, config)
    trt = _import_tensorrt()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("C5-4B2 TensorRT engine deserialization failed.")

    tensors: list[Int8EngineTensor] = []
    for index in range(int(engine.num_io_tensors)):
        name = str(engine.get_tensor_name(index))
        tensors.append(
            Int8EngineTensor(
                name=name,
                mode=str(engine.get_tensor_mode(name)).split(".")[-1].upper(),
                dtype=str(engine.get_tensor_dtype(name)),
                shape=tuple(int(value) for value in engine.get_tensor_shape(name)),
            )
        )
    memory_raw = getattr(engine, "device_memory_size_v2", None)
    if memory_raw is None:
        memory_raw = getattr(engine, "device_memory_size", None)
    if memory_raw is None:
        memory_size_bytes = 0
    elif isinstance(memory_raw, int):
        memory_size_bytes = memory_raw
    else:
        memory_size_bytes = int(cast(Any, memory_raw))
    return Int8EngineContract(
        io_tensors=tuple(tensors),
        device_memory_size_bytes=memory_size_bytes,
    )


def _engine_contract_from_mapping(raw: Mapping[str, Any]) -> Int8EngineContract:
    if set(raw) != {"io_tensors", "device_memory_size_bytes"}:
        raise ValueError("C5-4B2 embedded engine fields are invalid.")
    items = raw["io_tensors"]
    if not isinstance(items, list | tuple):
        raise TypeError("C5-4B2 io_tensors must be an array.")
    tensors: list[Int8EngineTensor] = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {"name", "mode", "dtype", "shape"}:
            raise ValueError("C5-4B2 embedded engine tensor fields are invalid.")
        shape = item["shape"]
        if not isinstance(shape, list | tuple):
            raise ValueError("C5-4B2 embedded engine tensor shape must be an array.")
        tensors.append(
            Int8EngineTensor(
                name=str(item["name"]),
                mode=str(item["mode"]),
                dtype=str(item["dtype"]),
                shape=tuple(int(value) for value in shape),
            )
        )
    memory = raw["device_memory_size_bytes"]
    if type(memory) is not int:
        raise TypeError("C5-4B2 device memory must be integer.")
    return Int8EngineContract(tuple(tensors), memory)


# ADD 2026-09-03: Exact B1 Q/DQ ONNX에서 engine과 provenance를 atomic publish한다.
def export_tensorrt_int8_engine(
    *,
    repository_root: Path,
    qdq_artifact_dir: Path,
    config: YoloTensorRtInt8EngineConfig,
    created_at: str,
    provenance_resolver: ProvenanceResolver = resolve_repository_provenance,
    environment_resolver: EnvironmentResolver = resolve_tensorrt_int8_environment,
    engine_builder: EngineBuilder = build_tensorrt_int8_engine,
    engine_inspector: EngineInspector = inspect_tensorrt_int8_engine,
) -> Int8EngineArtifacts:
    root = repository_root.resolve()
    config.validate()
    _validate_timestamp(created_at)
    provenance = provenance_resolver(root)
    provenance.validate()
    if provenance.working_tree_dirty:
        raise ValueError("Official C5-4B2 build requires a clean repository.")

    source_model = qdq_artifact_dir / EXPECTED_QDQ_ONNX_FILENAME
    source_sha_before = sha256_file(source_model)
    verify_qdq_source(artifact_dir=qdq_artifact_dir, config=config)

    output_dir = root / config.output_root / config.export_id
    staging = output_dir.parent / f".{config.export_id}.staging"
    if output_dir.exists() or staging.exists():
        raise FileExistsError("C5-4B2 output namespace already exists.")
    staging.mkdir(parents=True)

    try:
        engine_path = staging / INT8_ENGINE_FILENAME
        engine_builder(source_model, engine_path, config)
        if sha256_file(source_model) != source_sha_before:
            raise RuntimeError("C5-4B2 source Q/DQ ONNX bytes changed during build.")
        contract = engine_inspector(engine_path, config)
        contract.validate(config=config)
        environment = environment_resolver(config.device)

        metadata = Int8EngineMetadata(
            schema_version=INT8_ENGINE_SCHEMA_VERSION,
            artifact_type="yolo_segmentation_tensorrt_int8",
            state="TENSORRT_INT8_ENGINE_BUILT",
            export_id=config.export_id,
            created_at=created_at,
            source_qdq_onnx_sha256=config.source.qdq_onnx_sha256,
            source_qdq_metadata_sha256=config.source.qdq_metadata_sha256,
            source_qdq_evidence_zip_sha256=config.source.qdq_evidence_zip_sha256,
            source_qdq_run_commit=config.source.qdq_run_commit,
            tensorrt_int8_engine_config_sha256=sha256_file(config.config_path),
            engine_sha256=sha256_file(engine_path),
            engine_size_bytes=engine_path.stat().st_size,
            engine=asdict(contract),
            environment=environment,
            repository=asdict(provenance),
            validation_used=False,
            test_used=False,
            test_split_used=False,
        )
        metadata_path = staging / INT8_ENGINE_METADATA_FILENAME
        metadata_path.write_bytes(metadata.to_json_bytes(config=config))
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return Int8EngineArtifacts(
        output_dir=output_dir,
        engine_path=output_dir / INT8_ENGINE_FILENAME,
        metadata_path=output_dir / INT8_ENGINE_METADATA_FILENAME,
        metadata=metadata,
    )
