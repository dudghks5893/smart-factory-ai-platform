"""Frozen-candidate YOLO segmentation ONNX export contracts."""

from __future__ import annotations

import json
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from ml.evaluation.final_benchmark import RepositoryProvenance, resolve_repository_provenance
from ml.experiments.yolo_final_candidate import (
    FINAL_CANDIDATE_SELECTION_BASIS,
    FINAL_CANDIDATE_STATE,
    FINAL_TEST_STATE,
    FinalCandidateManifest,
    OfficialCandidateEvidence,
    load_final_candidate_manifest,
    load_official_candidate_evidence,
    materialize_official_candidate_artifact,
    verify_official_candidate_identity,
)
from ml.experiments.yolo_segmentation import (
    YoloExperimentConfig,
    load_yolo_experiment_config,
)
from ml.training.yolo_segmentation import (
    YoloSegmentationBaselineConfig,
    load_yolo_segmentation_config,
    validate_artifact_id,
    validate_yolo_artifact,
)
from shared.hashing import is_sha256_digest, sha256_file

EXPECTED_FROZEN_MANIFEST_SHA256 = "2a26b1bc03a1876f828e12a625c69c76af5e8c5713e3f64be699feffe2e8aa09"
DEFAULT_FROZEN_MANIFEST = Path("configs/model/yolo_segmentation_final_candidate.json")
DEFAULT_EXPORT_CONFIG = Path("configs/export/yolo_segmentation_onnx.yaml")
ONNX_MODEL_FILENAME = "model.onnx"
ONNX_METADATA_FILENAME = "metadata.json"
ONNX_EXPORT_SCHEMA_VERSION = 1
EXPECTED_CLASSES = {0: "bent", 1: "color", 2: "scratch"}


@dataclass(frozen=True)
class YoloOnnxParityPolicy:
    """Predeclared validation-only equivalence evidence policy."""

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

    # ADD 2026-09-02: C5 parity가 validation-only metrics-first boundary인지 검증한다.
    def validate(self) -> None:
        typed_values = (
            self.test_used,
            self.test_split_used,
            self.retina_masks,
            self.require_finite_outputs,
            self.require_valid_output_shapes,
            self.require_valid_class_ids,
        )
        if any(type(value) is not bool for value in typed_values):
            raise TypeError("C5 parity boolean fields must be strict booleans.")
        if type(self.max_detections) is not int or any(
            type(value) not in {int, float}
            for value in (
                self.prediction_initial_confidence,
                self.diagnostic_confidence,
                self.prediction_iou,
                self.mask_threshold,
            )
        ):
            raise TypeError("C5 parity numeric fields have invalid types.")
        if self.split != "val" or self.test_used is not False or self.test_split_used is not False:
            raise ValueError("C5 ONNX parity must remain validation-only with test excluded.")
        if (
            self.prediction_initial_confidence != 0.001
            or self.diagnostic_confidence != 0.25
            or self.prediction_iou != 0.7
            or self.max_detections != 300
            or self.retina_masks is not False
            or self.mask_threshold != 0.5
            or self.mask_resize != "opencv_inter_nearest"
        ):
            raise ValueError("C5 parity prediction normalization changed from C4-2C.")
        if self.association != "greedy_max_mask_iou_positive_overlap":
            raise ValueError("C5 parity association policy is invalid.")
        if self.acceptance_mode != "metrics_only_pending_numeric_tolerance_approval":
            raise ValueError("C5 parity must not invent unapproved numeric tolerances.")
        if (
            self.require_finite_outputs is not True
            or self.require_valid_output_shapes is not True
            or self.require_valid_class_ids is not True
            or self.numeric_thresholds is not None
        ):
            raise ValueError(
                "C5 structural parity gates are incomplete or numeric gates leaked in."
            )


@dataclass(frozen=True)
class YoloOnnxExportConfig:
    """Conservative FP32 static-shape ONNX export configuration."""

    schema_version: int
    export_id: str
    format: str
    task: str
    precision: str
    batch: int
    imgsz: int
    dynamic: bool
    simplify: bool
    nms: bool
    opset: int
    device: str
    output_root: Path
    parity: YoloOnnxParityPolicy
    config_path: Path

    # ADD 2026-09-02: First C5 export가 static FP32 segmentation foundation인지 검증한다.
    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or type(self.batch) is not int
            or type(self.imgsz) is not int
            or type(self.opset) is not int
            or type(self.dynamic) is not bool
            or type(self.simplify) is not bool
            or type(self.nms) is not bool
        ):
            raise TypeError("C5-1 export config contains invalid scalar types.")
        if self.schema_version != ONNX_EXPORT_SCHEMA_VERSION:
            raise ValueError("Unsupported C5 ONNX export config schema version.")
        validate_artifact_id(self.export_id)
        if self.format != "onnx" or self.task != "segment" or self.precision != "fp32":
            raise ValueError("C5-1 supports only FP32 YOLO segmentation ONNX export.")
        if (
            self.batch != 1
            or self.imgsz != 640
            or self.dynamic is not False
            or self.simplify is not False
            or self.nms is not False
            or self.opset != 18
            or self.device != "cpu"
        ):
            raise ValueError("C5-1 conservative export parameters changed without review.")
        if self.output_root != Path("artifacts/deployment/yolo_segmentation/onnx"):
            raise ValueError("C5-1 output_root must remain in the ignored artifact namespace.")
        self.parity.validate()


@dataclass(frozen=True)
class OnnxTensorContract:
    """One static ONNX graph input or output tensor."""

    name: str
    dtype: str
    shape: tuple[int, ...]

    # ADD 2026-09-02: ONNX graph tensor identity와 static positive shape를 검증한다.
    def validate(self) -> None:
        if (
            not self.name
            or not self.dtype
            or not self.shape
            or any(value <= 0 for value in self.shape)
        ):
            raise ValueError(
                "ONNX tensor contract requires name, dtype, and static positive shape."
            )


@dataclass(frozen=True)
class OnnxGraphContract:
    """Checked segmentation graph interface emitted by the pinned exporter."""

    opset: int
    inputs: tuple[OnnxTensorContract, ...]
    outputs: tuple[OnnxTensorContract, ...]

    # ADD 2026-09-02: Static batch-1 segmentation graph interface를 검증한다.
    def validate(self, *, config: YoloOnnxExportConfig) -> None:
        if self.opset != config.opset:
            raise ValueError("Exported ONNX opset does not match the repository config.")
        if len(self.inputs) != 1 or self.inputs[0].name != "images":
            raise ValueError("YOLO ONNX graph requires exactly one images input.")
        if self.inputs[0].shape != (config.batch, 3, config.imgsz, config.imgsz):
            raise ValueError("YOLO ONNX graph input shape is not the configured static shape.")
        if len(self.outputs) != 2 or {item.name for item in self.outputs} != {
            "output0",
            "output1",
        }:
            raise ValueError("YOLO segmentation ONNX graph requires output0 and output1.")
        outputs_by_name = {item.name: item for item in self.outputs}
        if outputs_by_name["output0"].shape != (1, 39, 8400) or outputs_by_name[
            "output1"
        ].shape != (1, 32, 160, 160):
            raise ValueError(
                "YOLO11n-seg ONNX raw output shapes changed from the frozen 640 graph."
            )
        for tensor in (*self.inputs, *self.outputs):
            tensor.validate()
            if tensor.dtype != "TensorProto.FLOAT":
                raise ValueError("C5-1 ONNX graph must preserve FP32 input and outputs.")


@dataclass(frozen=True)
class FrozenYoloSource:
    """Verified C4-3 pointer, package evidence, and repository configuration."""

    repository_root: Path
    manifest_path: Path
    manifest_sha256: str
    package_path: Path
    candidate: FinalCandidateManifest
    evidence: OfficialCandidateEvidence
    experiment: YoloExperimentConfig
    baseline: YoloSegmentationBaselineConfig


@dataclass(frozen=True)
class YoloOnnxExportMetadata:
    """Machine-readable source, graph, environment, and Git provenance."""

    schema_version: int
    artifact_type: str
    export_state: str
    export_id: str
    created_at: str
    source_experiment_id: str
    frozen_manifest_sha256: str
    official_package_sha256: str
    source_model_sha256: str
    source_metadata_sha256: str
    source_model_family: str
    source_task: str
    dataset_manifest_sha256: str
    export_config_sha256: str
    export_config: Mapping[str, Any]
    onnx_sha256: str
    onnx_size_bytes: int
    graph: Mapping[str, Any]
    environment: Mapping[str, str]
    repository: Mapping[str, str | bool]
    test_used: bool
    test_split_used: bool

    # ADD 2026-09-02: Export evidence가 frozen source와 no-test FP32 contract를 보존하는지 검증한다.
    def validate(self) -> None:
        if (
            self.schema_version != ONNX_EXPORT_SCHEMA_VERSION
            or self.artifact_type != "yolo_segmentation_onnx"
            or self.export_state != "ONNX_EXPORT_COMPLETED"
        ):
            raise ValueError("C5-1 ONNX metadata lifecycle is invalid.")
        validate_artifact_id(self.export_id)
        if (
            self.source_model_family != "yolo11n-seg"
            or self.source_task != "segment"
            or self.test_used is not False
            or self.test_split_used is not False
        ):
            raise ValueError("C5-1 ONNX metadata changed the frozen model or test seal.")
        for digest in (
            self.frozen_manifest_sha256,
            self.official_package_sha256,
            self.source_model_sha256,
            self.source_metadata_sha256,
            self.dataset_manifest_sha256,
            self.export_config_sha256,
            self.onnx_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C5-1 ONNX metadata contains an invalid SHA-256.")
        if self.onnx_size_bytes <= 0:
            raise ValueError("C5-1 ONNX artifact size must be positive.")
        _validate_timestamp(self.created_at)
        if self.repository.get("working_tree_dirty") is not False:
            raise ValueError("Official C5-1 export requires a clean repository state.")
        if set(self.repository) != {"git_commit", "working_tree_dirty"}:
            raise ValueError("C5-1 repository provenance fields are invalid.")
        RepositoryProvenance(
            git_commit=str(self.repository["git_commit"]),
            working_tree_dirty=cast(bool, self.repository["working_tree_dirty"]),
        ).validate()
        required_environment = {
            "python_version",
            "platform",
            "torch_version",
            "ultralytics_version",
            "onnx_version",
            "python_implementation",
        }
        if set(self.environment) != required_environment or any(
            not isinstance(value, str) or not value for value in self.environment.values()
        ):
            raise ValueError("C5-1 export environment fields are invalid.")
        graph = _graph_contract_from_mapping(self.graph)
        config = _config_contract_from_mapping(self.export_config)
        graph.validate(config=config)
        try:
            json.dumps(asdict(self), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("C5-1 ONNX metadata must be strict JSON data.") from exc

    # ADD 2026-09-02: Export metadata를 deterministic strict JSON bytes로 직렬화한다.
    def to_json_bytes(self) -> bytes:
        self.validate()
        return (json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()

    # ADD 2026-09-02: Untrusted export metadata JSON을 strict field set으로 복원한다.
    @classmethod
    def from_json_dict(cls, raw: object) -> YoloOnnxExportMetadata:
        if not isinstance(raw, dict):
            raise ValueError("C5-1 ONNX metadata root must be an object.")
        try:
            metadata = cls(**cast(dict[str, Any], raw))
        except TypeError as exc:
            raise ValueError("C5-1 ONNX metadata fields do not match the schema.") from exc
        metadata.validate()
        return metadata


@dataclass(frozen=True)
class YoloOnnxExportArtifacts:
    """Completed ignored ONNX artifact and metadata paths."""

    output_dir: Path
    model_path: Path
    metadata_path: Path
    metadata: YoloOnnxExportMetadata


type OnnxExporter = Callable[[Path, YoloOnnxExportConfig], Path]
type GraphInspector = Callable[[Path], OnnxGraphContract]
type ProvenanceResolver = Callable[[Path], RepositoryProvenance]


# ADD 2026-09-02: Embedded export mapping을 path-free typed config contract로 복원한다.
def _config_contract_from_mapping(raw: Mapping[str, Any]) -> YoloOnnxExportConfig:
    values = dict(raw)
    parity_raw = values.pop("parity", None)
    output_root = values.pop("output_root", None)
    if not isinstance(parity_raw, dict) or output_root is None:
        raise ValueError("C5-1 embedded export config is incomplete.")
    try:
        config = YoloOnnxExportConfig(
            **values,
            output_root=Path(str(output_root)),
            parity=YoloOnnxParityPolicy(**cast(dict[str, Any], parity_raw)),
            config_path=Path("<embedded>"),
        )
    except TypeError as exc:
        raise ValueError("C5-1 embedded export config fields are invalid.") from exc
    config.validate()
    return config


# ADD 2026-09-02: Embedded ONNX graph mapping을 strict tensor contract로 복원한다.
def _graph_contract_from_mapping(raw: Mapping[str, Any]) -> OnnxGraphContract:
    if set(raw) != {"opset", "inputs", "outputs"}:
        raise ValueError("C5-1 embedded ONNX graph fields are invalid.")

    def tensors(value: object) -> tuple[OnnxTensorContract, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError("C5-1 embedded ONNX tensors must be an array.")
        result: list[OnnxTensorContract] = []
        for item in value:
            if not isinstance(item, dict) or set(item) != {"name", "dtype", "shape"}:
                raise ValueError("C5-1 embedded ONNX tensor fields are invalid.")
            shape = item["shape"]
            if not isinstance(shape, list | tuple):
                raise ValueError("C5-1 embedded ONNX tensor shape must be an array.")
            result.append(
                OnnxTensorContract(
                    name=str(item["name"]),
                    dtype=str(item["dtype"]),
                    shape=tuple(int(value) for value in shape),
                )
            )
        return tuple(result)

    if type(raw["opset"]) is not int:
        raise ValueError("C5-1 embedded ONNX opset must be an integer.")
    return OnnxGraphContract(
        opset=raw["opset"],
        inputs=tensors(raw["inputs"]),
        outputs=tensors(raw["outputs"]),
    )


# ADD 2026-09-02: Repository-owned path를 root 안에서 cwd-independent resolve한다.
def _repository_path(repository_root: Path, path: Path, *, field: str) -> Path:
    root = repository_root.resolve()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"C5 {field} must remain inside repository_root.") from exc
    return resolved


# ADD 2026-09-02: External package path를 repository cwd와 무관하게 resolve한다.
def _external_path(repository_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repository_root.resolve() / path).resolve()


# ADD 2026-09-02: Machine-readable timestamp가 timezone-aware ISO-8601인지 검증한다.
def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("C5 timestamp must be ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("C5 timestamp must include a timezone offset.")


# ADD 2026-09-02: C5 export/parity config를 strict typed contract로 로드한다.
def load_yolo_onnx_export_config(path: Path) -> YoloOnnxExportConfig:
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("Cannot read C5 ONNX export config.") from exc
    if not isinstance(raw, dict):
        raise ValueError("C5 ONNX export config root must be a mapping.")
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
        "simplify",
        "nms",
        "opset",
        "device",
        "output_root",
        "parity",
    }
    if set(values) != expected or not isinstance(values.get("parity"), dict):
        raise ValueError("C5 ONNX export config fields do not match the schema.")
    try:
        parity = YoloOnnxParityPolicy(**cast(dict[str, Any], values["parity"]))
        config = YoloOnnxExportConfig(
            **{
                key: value
                for key, value in values.items()
                if key != "parity" and key != "output_root"
            },
            output_root=Path(str(values["output_root"])),
            parity=parity,
            config_path=path.resolve(),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("C5 ONNX export config contains invalid typed values.") from exc
    config.validate()
    return config


# ADD 2026-09-02: Frozen manifest/package/config만 검증하고 dataset/test에는 접근하지 않는다.
def prepare_frozen_yolo_source(
    *,
    repository_root: Path,
    manifest_path: Path,
    package_path: Path,
) -> FrozenYoloSource:
    root = repository_root.resolve()
    frozen_path = _repository_path(root, manifest_path, field="frozen manifest")
    manifest_sha256 = sha256_file(frozen_path)
    if manifest_sha256 != EXPECTED_FROZEN_MANIFEST_SHA256:
        raise ValueError("C5 frozen candidate Manifest SHA-256 is not the C4-3 identity.")
    candidate = load_final_candidate_manifest(frozen_path)
    if (
        candidate.selection_state != FINAL_CANDIDATE_STATE
        or candidate.selection_basis != FINAL_CANDIDATE_SELECTION_BASIS
        or candidate.final_test_state != FINAL_TEST_STATE
        or candidate.test_used is not False
        or candidate.test_split_used is not False
        or candidate.model_family != "yolo11n-seg"
        or candidate.task != "segment"
    ):
        raise ValueError("C5 source is not the exact validation-selected frozen candidate.")
    official_package = _external_path(root, package_path)
    evidence = load_official_candidate_evidence(
        official_package,
        expected_package_sha256=candidate.official_package_sha256,
    )
    verify_official_candidate_identity(candidate, evidence)
    experiment_path = _repository_path(
        root,
        Path("configs/experiments/yolo_segmentation") / f"{candidate.selected_experiment_id}.yaml",
        field="experiment config",
    )
    if sha256_file(experiment_path) != candidate.experiment_config_sha256:
        raise ValueError("C5 source experiment config changed from the frozen identity.")
    experiment = load_yolo_experiment_config(experiment_path)
    baseline = experiment.training_config(
        load_yolo_segmentation_config(experiment.baseline_config_path)
    )
    if (
        experiment.experiment_id != candidate.selected_experiment_id
        or baseline.model.architecture != candidate.model_family
        or baseline.model.task != candidate.task
    ):
        raise ValueError("C5 source config model/task identity changed from the frozen candidate.")
    return FrozenYoloSource(
        repository_root=root,
        manifest_path=frozen_path,
        manifest_sha256=manifest_sha256,
        package_path=official_package,
        candidate=candidate,
        evidence=evidence,
        experiment=experiment,
        baseline=baseline,
    )


# ADD 2026-09-02: Pinned Ultralytics exporter로 static FP32 segmentation ONNX를 생성한다.
def _export_with_ultralytics(model_path: Path, config: YoloOnnxExportConfig) -> Path:
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version

    model = YOLO(str(model_path), task=config.task)
    if ultralytics_version != "8.4.128" or model.task != "segment":
        raise ValueError("C5-1 requires pinned Ultralytics 8.4.128 segmentation export.")
    names = {int(key): str(value) for key, value in model.names.items()}
    if names != EXPECTED_CLASSES:
        raise ValueError("C5-1 source model classes changed from bent/color/scratch.")
    exported = Path(
        str(
            model.export(
                format="onnx",
                imgsz=config.imgsz,
                batch=config.batch,
                dynamic=config.dynamic,
                simplify=config.simplify,
                opset=config.opset,
                nms=config.nms,
                device=config.device,
                half=False,
                int8=False,
            )
        )
    )
    if not exported.is_file() or exported.suffix != ".onnx":
        raise RuntimeError("Pinned Ultralytics exporter did not produce one ONNX file.")
    return exported


# ADD 2026-09-02: ONNX checker와 static graph I/O schema를 repository metadata로 복원한다.
def inspect_onnx_graph(path: Path) -> OnnxGraphContract:
    import onnx

    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    opsets = [item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}]
    if len(opsets) != 1:
        raise ValueError("C5-1 ONNX graph must declare exactly one ai.onnx opset.")

    def tensor_contract(value: Any) -> OnnxTensorContract:
        tensor_type = value.type.tensor_type
        shape = tuple(int(dimension.dim_value) for dimension in tensor_type.shape.dim)
        return OnnxTensorContract(
            name=str(value.name),
            dtype=str(onnx.helper.tensor_dtype_to_string(tensor_type.elem_type)),
            shape=shape,
        )

    contract = OnnxGraphContract(
        opset=int(opsets[0]),
        inputs=tuple(tensor_contract(value) for value in model.graph.input),
        outputs=tuple(tensor_contract(value) for value in model.graph.output),
    )
    return contract


# ADD 2026-09-02: Exact frozen model에서 ONNX binary와 provenance metadata를 atomic publish한다.
def export_frozen_yolo_onnx(
    *,
    source: FrozenYoloSource,
    config: YoloOnnxExportConfig,
    created_at: str,
    provenance_resolver: ProvenanceResolver = resolve_repository_provenance,
    exporter: OnnxExporter = _export_with_ultralytics,
    graph_inspector: GraphInspector = inspect_onnx_graph,
) -> YoloOnnxExportArtifacts:
    config.validate()
    _validate_timestamp(created_at)
    _repository_path(
        source.repository_root,
        config.config_path,
        field="ONNX export config",
    )
    provenance = provenance_resolver(source.repository_root)
    provenance.validate()
    if provenance.working_tree_dirty:
        raise ValueError("Official C5-1 export requires a clean committed repository state.")
    if config.imgsz != source.baseline.training.imgsz:
        raise ValueError("C5-1 image size does not match the frozen C4-2C training contract.")
    output_root = _repository_path(
        source.repository_root,
        config.output_root,
        field="ONNX artifact output root",
    )
    output_dir = output_root / config.export_id
    staging_dir = output_root / f".{config.export_id}.staging"
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError("C5-1 ONNX export namespace already exists.")

    with tempfile.TemporaryDirectory(prefix="smartfactory-c5-onnx-") as temporary:
        artifact_dir = Path(temporary) / "source-artifact"
        materialize_official_candidate_artifact(
            package_path=source.package_path,
            candidate=source.candidate,
            evidence=source.evidence,
            artifact_dir=artifact_dir,
        )
        model_dir = artifact_dir / "model"
        artifact_metadata = validate_yolo_artifact(
            model_dir,
            expected_contract=source.baseline.dataset_contract,
        )
        source_model_path = model_dir / "model.pt"
        if (
            artifact_metadata.architecture != source.candidate.model_family
            or artifact_metadata.task != source.candidate.task
            or sha256_file(source_model_path) != source.candidate.model_sha256
        ):
            raise ValueError("C5-1 materialized source model is not the frozen candidate.")

        # Verified checkpoint만 pinned Ultralytics exporter에 전달한다.
        exported_path = exporter(source_model_path, config)
        if sha256_file(source_model_path) != source.candidate.model_sha256:
            raise RuntimeError("C5-1 exporter changed the materialized source checkpoint.")
        graph = graph_inspector(exported_path)
        graph.validate(config=config)
        output_root.mkdir(parents=True, exist_ok=True)
        staging_dir.mkdir(exist_ok=False)
        try:
            onnx_path = staging_dir / ONNX_MODEL_FILENAME
            shutil.copyfile(exported_path, onnx_path)
            metadata = YoloOnnxExportMetadata(
                schema_version=ONNX_EXPORT_SCHEMA_VERSION,
                artifact_type="yolo_segmentation_onnx",
                export_state="ONNX_EXPORT_COMPLETED",
                export_id=config.export_id,
                created_at=created_at,
                source_experiment_id=source.candidate.selected_experiment_id,
                frozen_manifest_sha256=source.manifest_sha256,
                official_package_sha256=source.candidate.official_package_sha256,
                source_model_sha256=source.candidate.model_sha256,
                source_metadata_sha256=source.candidate.metadata_sha256,
                source_model_family=source.candidate.model_family,
                source_task=source.candidate.task,
                dataset_manifest_sha256=source.candidate.dataset_manifest_sha256,
                export_config_sha256=sha256_file(config.config_path),
                export_config=_export_config_mapping(config),
                onnx_sha256=sha256_file(onnx_path),
                onnx_size_bytes=onnx_path.stat().st_size,
                graph=asdict(graph),
                environment=_export_environment(),
                repository=provenance.to_json_dict(),
                test_used=False,
                test_split_used=False,
            )
            metadata_path = staging_dir / ONNX_METADATA_FILENAME
            metadata_path.write_bytes(metadata.to_json_bytes())
            staging_dir.rename(output_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
    return YoloOnnxExportArtifacts(
        output_dir=output_dir,
        model_path=output_dir / ONNX_MODEL_FILENAME,
        metadata_path=output_dir / ONNX_METADATA_FILENAME,
        metadata=metadata,
    )


# ADD 2026-09-02: Config를 path-free stable evidence mapping으로 변환한다.
def _export_config_mapping(config: YoloOnnxExportConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("config_path")
    payload["output_root"] = str(config.output_root)
    return payload


# ADD 2026-09-02: Export runtime dependency/environment version을 기록한다.
def _export_environment() -> dict[str, str]:
    import onnx
    from ultralytics import __version__ as ultralytics_version

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "ultralytics_version": ultralytics_version,
        "onnx_version": str(onnx.__version__),
        "python_implementation": sys.implementation.name,
    }


# ADD 2026-09-02: Ignored ONNX artifact metadata와 binary identity를 함께 검증한다.
def load_yolo_onnx_artifact(artifact_dir: Path) -> YoloOnnxExportMetadata:
    model_path = artifact_dir / ONNX_MODEL_FILENAME
    metadata_path = artifact_dir / ONNX_METADATA_FILENAME
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("C5-1 ONNX artifact requires model.onnx and metadata.json.")
    try:
        raw: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read C5-1 ONNX artifact metadata.") from exc
    metadata = YoloOnnxExportMetadata.from_json_dict(raw)
    if sha256_file(model_path) != metadata.onnx_sha256:
        raise ValueError("C5-1 ONNX binary SHA does not match export metadata.")
    return metadata
