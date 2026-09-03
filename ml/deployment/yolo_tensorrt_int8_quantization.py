"""C5-4B1 ModelOpt INT8 explicit-Q/DQ ONNX generation and provenance."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import onnx
from numpy.typing import NDArray

from ml.datasets.yolo_segmentation_manifest import (
    DerivedManifestRecord,
    read_derived_manifest,
)
from ml.deployment.yolo_onnx import (
    ONNX_METADATA_FILENAME,
    ONNX_MODEL_FILENAME,
    load_yolo_onnx_artifact,
)
from ml.deployment.yolo_tensorrt_int8 import YoloTensorRtInt8Config
from ml.evaluation.final_benchmark import RepositoryProvenance, resolve_repository_provenance
from shared.hashing import is_sha256_digest, sha256_bytes, sha256_file

INT8_QDQ_SCHEMA_VERSION = 1
INT8_QDQ_ONNX_FILENAME = "model.int8.qdq.onnx"
INT8_QDQ_METADATA_FILENAME = "metadata.json"
INT8_QDQ_OUTPUT_ROOT = Path("artifacts/deployment/yolo_segmentation/tensorrt_int8/qdq")
DATASET_MANIFEST_FILENAME = "manifest.csv"
MODEL_OPT_PACKAGE = "nvidia-modelopt"
MODEL_OPT_MODULE = "modelopt.onnx.quantization"
CALIBRATION_EXECUTION_PROVIDERS = ("cpu",)


@dataclass(frozen=True)
class QdqGraphContract:
    """Observable Q/DQ graph identity needed before TensorRT engine build."""

    quantize_linear_count: int
    dequantize_linear_count: int
    input_names: tuple[str, ...]
    input_shapes: tuple[tuple[int | str, ...], ...]
    output_names: tuple[str, ...]
    output_shapes: tuple[tuple[int | str, ...], ...]

    # ADD 2026-09-02: ModelOpt 결과가 explicit Q/DQ와 source I/O shape를 유지하는지 검증한다.
    def validate_against(self, source: QdqGraphContract) -> None:
        if self.quantize_linear_count <= 0 or self.dequantize_linear_count <= 0:
            raise ValueError("C5-4B1 quantized ONNX must contain explicit Q/DQ nodes.")
        if self.input_names != source.input_names or self.input_shapes != source.input_shapes:
            raise ValueError("C5-4B1 quantized ONNX changed source input identity.")
        if self.output_names != source.output_names or self.output_shapes != source.output_shapes:
            raise ValueError("C5-4B1 quantized ONNX changed source output identity.")


@dataclass(frozen=True)
class Int8QdqMetadata:
    """Immutable evidence describing one quantized Q/DQ ONNX artifact."""

    schema_version: int
    artifact_type: str
    state: str
    quantization_id: str
    created_at: str
    source_onnx_sha256: str
    source_onnx_metadata_sha256: str
    int8_contract_sha256: str
    dataset_manifest_sha256: str
    calibration_split: str
    calibration_sample_count: int
    calibration_sample_ids_sha256: str
    calibration_execution_providers: tuple[str, ...]
    modelopt_version: str
    quantized_onnx_sha256: str
    quantized_onnx_size_bytes: int
    quantize_linear_count: int
    dequantize_linear_count: int
    environment: Mapping[str, str]
    repository: Mapping[str, str | bool]
    validation_used: bool
    test_used: bool
    test_split_used: bool

    # ADD 2026-09-02: C5-4B1 metadata가 train-only calibration과
    # exact source identity를 보존하는지 검증한다.
    def validate(self, *, config: YoloTensorRtInt8Config) -> None:
        if self.schema_version != INT8_QDQ_SCHEMA_VERSION:
            raise ValueError("Unsupported C5-4B1 metadata schema version.")
        if (
            self.artifact_type != "yolo_segmentation_int8_qdq_onnx"
            or self.state != "INT8_QDQ_ONNX_QUANTIZED"
            or self.quantization_id != config.quantization_id
        ):
            raise ValueError("C5-4B1 metadata artifact identity is invalid.")
        _validate_timestamp(self.created_at)
        digests = (
            self.source_onnx_sha256,
            self.source_onnx_metadata_sha256,
            self.int8_contract_sha256,
            self.dataset_manifest_sha256,
            self.calibration_sample_ids_sha256,
            self.quantized_onnx_sha256,
        )
        if any(not is_sha256_digest(value) for value in digests):
            raise ValueError("C5-4B1 metadata contains an invalid SHA-256.")
        if (
            self.source_onnx_sha256 != config.source.onnx_sha256
            or self.source_onnx_metadata_sha256 != config.source.onnx_metadata_sha256
            or self.dataset_manifest_sha256 != config.source.dataset_manifest_sha256
        ):
            raise ValueError("C5-4B1 metadata is not bound to the frozen source contract.")
        if (
            self.calibration_split != config.calibration.split
            or self.calibration_sample_count != config.calibration.sample_count
            or self.calibration_execution_providers != CALIBRATION_EXECUTION_PROVIDERS
            or self.modelopt_version != config.quantizer.version
        ):
            raise ValueError("C5-4B1 metadata changed the frozen calibration/toolchain contract.")
        if (
            self.validation_used is not False
            or self.test_used is not False
            or self.test_split_used is not False
        ):
            raise ValueError("C5-4B1 metadata violates the validation/final-test seal.")
        if (
            self.quantized_onnx_size_bytes <= 0
            or self.quantize_linear_count <= 0
            or self.dequantize_linear_count <= 0
        ):
            raise ValueError("C5-4B1 metadata does not describe a valid Q/DQ ONNX.")
        repository = _repository_provenance(self.repository)
        if repository.working_tree_dirty:
            raise ValueError("Official C5-4B1 quantization requires a clean repository.")
        required_environment = {
            "python_version",
            "platform",
            "python_implementation",
            "modelopt_version",
            "onnx_version",
            "opencv_version",
        }
        if set(self.environment) != required_environment or any(
            not isinstance(value, str) or not value for value in self.environment.values()
        ):
            raise ValueError("C5-4B1 environment fields are invalid.")
        try:
            json.dumps(asdict(self), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("C5-4B1 metadata must be strict JSON data.") from exc

    # ADD 2026-09-02: Q/DQ metadata를 deterministic strict JSON bytes로 직렬화한다.
    def to_json_bytes(self, *, config: YoloTensorRtInt8Config) -> bytes:
        self.validate(config=config)
        return (json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


@dataclass(frozen=True)
class Int8QdqArtifacts:
    """Published C5-4B1 Q/DQ ONNX and metadata paths."""

    output_dir: Path
    onnx_path: Path
    metadata_path: Path
    metadata: Int8QdqMetadata


class Int8CalibrationDataReader:
    """One-image-at-a-time deterministic calibration reader for ModelOpt/ORT."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        records: Sequence[DerivedManifestRecord],
        config: YoloTensorRtInt8Config,
    ) -> None:
        self._dataset_root = dataset_root
        self._records = tuple(records)
        self._config = config
        self._index = 0

    # ADD 2026-09-02: Manifest sample_id 순서대로 train image 한 장씩 calibration batch로 제공한다.
    # MODIFY 2026-09-03: ModelOpt 0.46.0의 사전 inference가 요구하는
    # non-consuming get_first contract를 지원한다.
    def get_first(self) -> dict[str, NDArray[np.float32]]:
        if not self._records:
            raise ValueError("C5-4B1 calibration reader has no train records.")
        record = self._records[0]
        image_path = self._dataset_root / record.image_path
        return {
            self._config.calibration.input_name: preprocess_calibration_image(
                image_path,
                imgsz=self._config.imgsz,
            )
        }

    def get_next(self) -> dict[str, NDArray[np.float32]] | None:
        if self._index >= len(self._records):
            return None
        record = self._records[self._index]
        self._index += 1
        image_path = self._dataset_root / record.image_path
        return {
            self._config.calibration.input_name: preprocess_calibration_image(
                image_path,
                imgsz=self._config.imgsz,
            )
        }

    # ADD 2026-09-02: ModelOpt가 calibration reader를 재사용할 때 deterministic 시작점으로 되감는다.
    def rewind(self) -> None:
        self._index = 0


type Quantizer = Callable[..., Any]
type ProvenanceResolver = Callable[[Path], RepositoryProvenance]


# ADD 2026-09-02: Timestamp가 timezone-aware ISO-8601인지 검증한다.
def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("C5-4B1 timestamp must be ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("C5-4B1 timestamp must include a timezone offset.")


# ADD 2026-09-02: Repository provenance mapping을 strict typed object로 복원한다.
def _repository_provenance(raw: Mapping[str, str | bool]) -> RepositoryProvenance:
    if set(raw) != {"git_commit", "working_tree_dirty"}:
        raise ValueError("C5-4B1 repository provenance fields are invalid.")
    if type(raw["working_tree_dirty"]) is not bool:
        raise ValueError("C5-4B1 working_tree_dirty must be boolean.")
    provenance = RepositoryProvenance(
        git_commit=str(raw["git_commit"]),
        working_tree_dirty=cast(bool, raw["working_tree_dirty"]),
    )
    provenance.validate()
    return provenance


# ADD 2026-09-02: Repository-owned path가 repository root 밖으로 이탈하지 않게 한다.
def _repository_path(repository_root: Path, path: Path, *, field: str) -> Path:
    root = repository_root.resolve()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"C5-4B1 {field} must remain inside repository_root.") from exc
    return resolved


# ADD 2026-09-02: ModelOpt import를 local/macOS CI에 강제하지 않고
# quantization runtime에서만 로드한다.
def resolve_modelopt_quantizer(*, expected_version: str) -> Quantizer:
    try:
        installed_version = importlib.metadata.version(MODEL_OPT_PACKAGE)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("C5-4B1 requires nvidia-modelopt in the quantization runtime.") from exc
    if installed_version != expected_version:
        raise RuntimeError(
            f"C5-4B1 requires nvidia-modelopt=={expected_version}, found {installed_version}."
        )
    try:
        module = importlib.import_module(MODEL_OPT_MODULE)
    except ModuleNotFoundError as exc:
        raise RuntimeError("C5-4B1 ModelOpt ONNX quantization module is unavailable.") from exc
    quantize = getattr(module, "quantize", None)
    if not callable(quantize):
        raise RuntimeError("C5-4B1 ModelOpt quantize API is unavailable.")
    return cast(Quantizer, quantize)


# ADD 2026-09-02: Frozen derived manifest에서 train 84장만 sample_id 오름차순으로 선택한다.
def load_calibration_records(
    *,
    dataset_root: Path,
    config: YoloTensorRtInt8Config,
) -> tuple[DerivedManifestRecord, ...]:
    manifest_path = dataset_root / DATASET_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"C5-4B1 dataset manifest not found: {manifest_path}")
    if sha256_file(manifest_path) != config.source.dataset_manifest_sha256:
        raise ValueError("C5-4B1 dataset manifest SHA does not match the frozen contract.")

    records = read_derived_manifest(manifest_path, allowed_splits={"train"})
    ordered = tuple(sorted(records, key=lambda record: record.sample_id))
    if len(ordered) != config.calibration.sample_count:
        raise ValueError("C5-4B1 calibration sample count is not exactly 84.")
    sample_ids = tuple(record.sample_id for record in ordered)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("C5-4B1 calibration sample IDs are not unique.")

    for record in ordered:
        if record.derived_split != "train":
            raise ValueError("C5-4B1 calibration record escaped the train split.")
        image_path = dataset_root / record.image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"C5-4B1 calibration image missing: {record.sample_id}")
        if sha256_file(image_path) != record.image_sha256:
            raise ValueError(f"C5-4B1 calibration image SHA mismatch: {record.sample_id}")
    return ordered


# ADD 2026-09-02: Ultralytics static 640 letterbox와 동일한
# padding/normalization으로 ONNX input을 만든다.
def preprocess_calibration_image(
    image_path: Path,
    *,
    imgsz: int,
) -> NDArray[np.float32]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"C5-4B1 cannot decode calibration image: {image_path}")

    height, width = image.shape[:2]
    ratio = min(imgsz / height, imgsz / width)
    resized_width = round(width * ratio)
    resized_height = round(height * ratio)
    if (resized_width, resized_height) != (width, height):
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

    pad_width = imgsz - resized_width
    pad_height = imgsz - resized_height
    half_width = pad_width / 2
    half_height = pad_height / 2
    left = round(half_width - 0.1)
    right = round(half_width + 0.1)
    top = round(half_height - 0.1)
    bottom = round(half_height + 0.1)
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if image.shape[:2] != (imgsz, imgsz):
        raise ValueError("C5-4B1 letterbox did not produce the frozen static input size.")

    rgb = image[:, :, ::-1]
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    batch = np.expand_dims(chw, axis=0).astype(np.float32)
    batch *= np.float32(1.0 / 255.0)
    return batch


# ADD 2026-09-02: Calibration sample identity list를 stable SHA-256 evidence로 축약한다.
def calibration_sample_ids_sha256(records: Sequence[DerivedManifestRecord]) -> str:
    payload = "".join(f"{record.sample_id}\n" for record in records).encode()
    return sha256_bytes(payload)


# ADD 2026-09-02: ONNX external I/O shape와 Q/DQ node 수를 framework-independent하게 관측한다.
def inspect_qdq_graph(path: Path) -> QdqGraphContract:
    model = onnx.load(path)
    onnx.checker.check_model(model)

    def shape_of(value_info: Any) -> tuple[int | str, ...]:
        dims: list[int | str] = []
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                dims.append(str(dim.dim_param))
            else:
                dims.append("?")
        return tuple(dims)

    node_types = [node.op_type for node in model.graph.node]
    return QdqGraphContract(
        quantize_linear_count=node_types.count("QuantizeLinear"),
        dequantize_linear_count=node_types.count("DequantizeLinear"),
        input_names=tuple(item.name for item in model.graph.input),
        input_shapes=tuple(shape_of(item) for item in model.graph.input),
        output_names=tuple(item.name for item in model.graph.output),
        output_shapes=tuple(shape_of(item) for item in model.graph.output),
    )


# ADD 2026-09-02: ModelOpt 0.46.0을 frozen PTQ kwargs로 호출해 explicit Q/DQ ONNX를 생성한다.
def run_modelopt_quantization(
    *,
    source_onnx_path: Path,
    output_onnx_path: Path,
    calibration_reader: Int8CalibrationDataReader,
    config: YoloTensorRtInt8Config,
    quantizer: Quantizer | None = None,
) -> None:
    selected_quantizer = quantizer or resolve_modelopt_quantizer(
        expected_version=config.quantizer.version
    )
    selected_quantizer(
        str(source_onnx_path),
        quantize_mode=config.quantizer.quantize_mode,
        calibration_data_reader=calibration_reader,
        calibration_method=config.quantizer.calibration_method,
        calibration_eps=list(CALIBRATION_EXECUTION_PROVIDERS),
        output_path=str(output_onnx_path),
        high_precision_dtype=config.quantizer.high_precision_dtype,
        simplify=config.quantizer.simplify,
    )
    if not output_onnx_path.is_file() or output_onnx_path.stat().st_size <= 0:
        raise RuntimeError("C5-4B1 ModelOpt did not publish a non-empty Q/DQ ONNX.")


# ADD 2026-09-02: ModelOpt/ONNX/OpenCV runtime identity를 quantization evidence에 기록한다.
def resolve_quantization_environment(*, expected_modelopt_version: str) -> Mapping[str, str]:
    installed_modelopt = importlib.metadata.version(MODEL_OPT_PACKAGE)
    if installed_modelopt != expected_modelopt_version:
        raise RuntimeError("C5-4B1 ModelOpt environment version changed from the contract.")
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "python_implementation": sys.implementation.name,
        "modelopt_version": installed_modelopt,
        "onnx_version": str(onnx.__version__),
        "opencv_version": str(cv2.__version__),
    }


# ADD 2026-09-02: Exact accepted ONNX와 train-only calibration으로 Q/DQ ONNX를 atomic publish한다.
def export_int8_qdq_onnx(
    *,
    repository_root: Path,
    onnx_artifact_dir: Path,
    dataset_root: Path,
    config: YoloTensorRtInt8Config,
    created_at: str,
    provenance_resolver: ProvenanceResolver = resolve_repository_provenance,
    quantizer: Quantizer | None = None,
    environment: Mapping[str, str] | None = None,
) -> Int8QdqArtifacts:
    root = repository_root.resolve()
    config.validate()
    _validate_timestamp(created_at)

    provenance = provenance_resolver(root)
    provenance.validate()
    if provenance.working_tree_dirty:
        raise ValueError("Official C5-4B1 quantization requires a clean committed repository.")

    source_metadata = load_yolo_onnx_artifact(onnx_artifact_dir)
    source_onnx_path = onnx_artifact_dir / ONNX_MODEL_FILENAME
    source_metadata_path = onnx_artifact_dir / ONNX_METADATA_FILENAME
    if (
        sha256_file(source_onnx_path) != config.source.onnx_sha256
        or sha256_file(source_metadata_path) != config.source.onnx_metadata_sha256
        or source_metadata.onnx_sha256 != config.source.onnx_sha256
        or source_metadata.export_config_sha256 != config.source.onnx_export_config_sha256
        or str(source_metadata.repository["git_commit"]) != config.source.onnx_export_commit
        or source_metadata.test_used is not False
        or source_metadata.test_split_used is not False
    ):
        raise ValueError("C5-4B1 source ONNX does not match the accepted C5-2 identity.")

    records = load_calibration_records(dataset_root=dataset_root, config=config)
    reader = Int8CalibrationDataReader(
        dataset_root=dataset_root,
        records=records,
        config=config,
    )

    output_root = _repository_path(
        root,
        INT8_QDQ_OUTPUT_ROOT,
        field="Q/DQ artifact output root",
    )
    output_dir = output_root / config.quantization_id
    staging_dir = output_root / f".{config.quantization_id}.staging"
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError("C5-4B1 Q/DQ artifact namespace already exists.")
    output_root.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(exist_ok=False)

    try:
        candidate_path = staging_dir / INT8_QDQ_ONNX_FILENAME
        source_sha_before = sha256_file(source_onnx_path)
        run_modelopt_quantization(
            source_onnx_path=source_onnx_path,
            output_onnx_path=candidate_path,
            calibration_reader=reader,
            config=config,
            quantizer=quantizer,
        )
        if sha256_file(source_onnx_path) != source_sha_before:
            raise RuntimeError("C5-4B1 quantization changed the accepted source ONNX bytes.")

        source_graph = inspect_qdq_graph(source_onnx_path)
        candidate_graph = inspect_qdq_graph(candidate_path)
        candidate_graph.validate_against(source_graph)

        runtime_environment = dict(
            environment
            if environment is not None
            else resolve_quantization_environment(
                expected_modelopt_version=config.quantizer.version
            )
        )
        metadata = Int8QdqMetadata(
            schema_version=INT8_QDQ_SCHEMA_VERSION,
            artifact_type="yolo_segmentation_int8_qdq_onnx",
            state="INT8_QDQ_ONNX_QUANTIZED",
            quantization_id=config.quantization_id,
            created_at=created_at,
            source_onnx_sha256=config.source.onnx_sha256,
            source_onnx_metadata_sha256=config.source.onnx_metadata_sha256,
            int8_contract_sha256=sha256_file(config.config_path),
            dataset_manifest_sha256=config.source.dataset_manifest_sha256,
            calibration_split=config.calibration.split,
            calibration_sample_count=len(records),
            calibration_sample_ids_sha256=calibration_sample_ids_sha256(records),
            calibration_execution_providers=CALIBRATION_EXECUTION_PROVIDERS,
            modelopt_version=config.quantizer.version,
            quantized_onnx_sha256=sha256_file(candidate_path),
            quantized_onnx_size_bytes=candidate_path.stat().st_size,
            quantize_linear_count=candidate_graph.quantize_linear_count,
            dequantize_linear_count=candidate_graph.dequantize_linear_count,
            environment=runtime_environment,
            repository=provenance.to_json_dict(),
            validation_used=False,
            test_used=False,
            test_split_used=False,
        )
        metadata_path = staging_dir / INT8_QDQ_METADATA_FILENAME
        metadata_path.write_bytes(metadata.to_json_bytes(config=config))
        staging_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return Int8QdqArtifacts(
        output_dir=output_dir,
        onnx_path=output_dir / INT8_QDQ_ONNX_FILENAME,
        metadata_path=output_dir / INT8_QDQ_METADATA_FILENAME,
        metadata=metadata,
    )
