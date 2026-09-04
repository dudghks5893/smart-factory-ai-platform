"""C6-5C DeepStream/L4 TensorRT INT8 deployment contracts."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shared.hashing import is_sha256_digest, sha256_file

DEFAULT_DEEPSTREAM_TENSORRT_CONFIG = Path("configs/streaming/yolo_deepstream_tensorrt_int8.json")

EXPECTED_BUILD_ID = "c6_5c_deepstream_l4_yolo11n_seg_int8_qdq_v1"
EXPECTED_C6_5B_CLOSURE_COMMIT = "3fc791f4ef1919099f61d40e49f8033882beab6c"

EXPECTED_SOURCE_ARCHIVE_SHA256 = "00f925d0ce5f6106d441822e419a039a736831c0d2c13835cfd01b62fad50990"
EXPECTED_SOURCE_ARCHIVE_BYTES = 5_352_519
EXPECTED_QDQ_RUN_COMMIT = "8e489c80ef9527a044b100cc96172d179947e051"
EXPECTED_QDQ_OPSET = 19
EXPECTED_Q_COUNT = 211
EXPECTED_DQ_COUNT = 211
EXPECTED_CALIBRATION_COUNT = 84

EXPECTED_SOURCE_FILES: dict[str, tuple[str, int]] = {
    "SHA256SUMS.txt": (
        "39b0d5b2bc139f74dfedeb052c86fc22f4dd281705c16e6702a905534144bc53",
        352,
    ),
    "metadata.json": (
        "8c3b215082ba111d4f932f4e021a9bc11866c49ecec788a52f20b2f9fe244fa7",
        1_517,
    ),
    "model.int8.qdq.onnx": (
        "d7c9af3ab3c2f71e88de26be71abe80f113f2e1c359d2a532a24079fa9b4dd00",
        6_195_984,
    ),
    "run_summary.json": (
        "c6b4dd790ae9a2ff312b9336d46c87f0efc03f3a2364ddda0b014a3f4405a60c",
        847,
    ),
    "yolo_segmentation_tensorrt_int8.yaml": (
        "18309302e45855e506628bb5e262886fc2cb366f8758fc100c55aaf6dbf3c37a",
        2_098,
    ),
}

EXPECTED_IMAGE_TAG = "nvcr.io/nvidia/deepstream:9.1-samples-multiarch"
EXPECTED_IMAGE_ID = "sha256:4f80b374e4a5086552825fe0f5bdd015c8cfd3dbe430cdde5ce9572e80e01583"
EXPECTED_REPO_DIGEST = (
    "nvcr.io/nvidia/deepstream@"
    "sha256:4f80b374e4a5086552825fe0f5bdd015c8cfd3dbe430cdde5ce9572e80e01583"
)
EXPECTED_GPU_NAME = "NVIDIA L4"
EXPECTED_GPU_COMPUTE_CAPABILITY = "8.9"
EXPECTED_DRIVER_VERSION = "595.84"
EXPECTED_DEEPSTREAM_VERSION = "9.1.0"
EXPECTED_TENSORRT_VERSION = "10.16.1.11"

EXPECTED_WORKSPACE_BYTES = 4_294_967_296
EXPECTED_PLAN_FILENAME = "model.plan"
EXPECTED_OUTPUT_ROOT = Path(
    "artifacts/deployment/yolo_segmentation/deepstream_l4/tensorrt_int8/"
    "c6_5c_deepstream_l4_yolo11n_seg_int8_qdq_v1"
)


@dataclass(frozen=True)
class SourceFileIdentity:
    """One member of the accepted C5-4B1 evidence archive."""

    sha256: str
    size_bytes: int

    # ADD 2026-09-05: Source member의 exact SHA-256과 byte size를 검증한다.
    def validate(self) -> None:
        if not is_sha256_digest(self.sha256):
            raise ValueError("C6-5C source member SHA-256 is invalid.")

        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("C6-5C source member size_bytes must be positive int.")


@dataclass(frozen=True)
class QdqSourceIdentity:
    """Exact accepted C5-4B1 Q/DQ source lineage."""

    evidence_zip_sha256: str
    evidence_zip_size_bytes: int
    qdq_run_commit: str
    qdq_opset: int
    quantize_linear_count: int
    dequantize_linear_count: int
    calibration_sample_count: int
    files: dict[str, SourceFileIdentity]

    # ADD 2026-09-05: C6-5C source를 exact accepted C5-4B1 lineage에 고정한다.
    def validate(self) -> None:
        if not is_sha256_digest(self.evidence_zip_sha256):
            raise ValueError("C6-5C source archive SHA-256 is invalid.")

        if (
            self.evidence_zip_sha256 != EXPECTED_SOURCE_ARCHIVE_SHA256
            or self.evidence_zip_size_bytes != EXPECTED_SOURCE_ARCHIVE_BYTES
            or self.qdq_run_commit != EXPECTED_QDQ_RUN_COMMIT
            or self.qdq_opset != EXPECTED_QDQ_OPSET
            or self.quantize_linear_count != EXPECTED_Q_COUNT
            or self.dequantize_linear_count != EXPECTED_DQ_COUNT
            or self.calibration_sample_count != EXPECTED_CALIBRATION_COUNT
        ):
            raise ValueError("C6-5C Q/DQ source identity changed.")

        if set(self.files) != set(EXPECTED_SOURCE_FILES):
            raise ValueError("C6-5C source archive member set changed.")

        for filename, expected in EXPECTED_SOURCE_FILES.items():
            identity = self.files[filename]
            identity.validate()

            expected_sha, expected_bytes = expected

            if identity.sha256 != expected_sha or identity.size_bytes != expected_bytes:
                raise ValueError(f"C6-5C source member identity changed: {filename}")


@dataclass(frozen=True)
class DeepStreamTensorRtRuntime:
    """Exact DeepStream/L4 TensorRT runtime identity."""

    image_tag: str
    image_id: str
    repo_digest: str
    gpu_name: str
    gpu_compute_capability: str
    driver_version: str
    deepstream_version: str
    tensorrt_python_version: str

    # ADD 2026-09-05: C6-5C runtime을 exact DeepStream 9.1/L4 identity에 고정한다.
    def validate(self) -> None:
        expected = {
            "image_tag": EXPECTED_IMAGE_TAG,
            "image_id": EXPECTED_IMAGE_ID,
            "repo_digest": EXPECTED_REPO_DIGEST,
            "gpu_name": EXPECTED_GPU_NAME,
            "gpu_compute_capability": EXPECTED_GPU_COMPUTE_CAPABILITY,
            "driver_version": EXPECTED_DRIVER_VERSION,
            "deepstream_version": EXPECTED_DEEPSTREAM_VERSION,
            "tensorrt_python_version": EXPECTED_TENSORRT_VERSION,
        }

        mismatches = [name for name, value in expected.items() if getattr(self, name) != value]

        if mismatches:
            raise ValueError("C6-5C runtime identity changed: " + ", ".join(mismatches))


@dataclass(frozen=True)
class TensorRtBuildPolicy:
    """Frozen TensorRT explicit-Q/DQ raw-plan build policy."""

    strongly_typed: bool
    explicit_quantization: bool
    builder_int8_flag: bool
    builder_fp16_flag: bool
    legacy_calibrator: bool
    workspace_bytes: int
    plan_filename: str
    output_root: Path
    persist_raw_plan: bool

    # ADD 2026-09-05: Strongly-typed explicit-Q/DQ L4 raw-plan build contract를 검증한다.
    def validate(self) -> None:
        bool_values = (
            self.strongly_typed,
            self.explicit_quantization,
            self.builder_int8_flag,
            self.builder_fp16_flag,
            self.legacy_calibrator,
            self.persist_raw_plan,
        )

        if any(type(value) is not bool for value in bool_values):
            raise TypeError("C6-5C build policy flags must be strict booleans.")

        if type(self.workspace_bytes) is not int:
            raise TypeError("C6-5C workspace_bytes must be int.")

        if (
            self.strongly_typed is not True
            or self.explicit_quantization is not True
            or self.builder_int8_flag is not False
            or self.builder_fp16_flag is not False
            or self.legacy_calibrator is not False
            or self.workspace_bytes != EXPECTED_WORKSPACE_BYTES
            or self.plan_filename != EXPECTED_PLAN_FILENAME
            or self.output_root != EXPECTED_OUTPUT_ROOT
            or self.persist_raw_plan is not True
        ):
            raise ValueError("C6-5C TensorRT build policy changed.")


@dataclass(frozen=True)
class EngineTensorContract:
    """One frozen external TensorRT engine tensor."""

    name: str
    mode: str
    dtype: str
    shape: tuple[int, ...]

    # ADD 2026-09-05: Engine tensor name/mode/dtype/static shape를 검증한다.
    def validate(self) -> None:
        if not self.name:
            raise ValueError("C6-5C engine tensor name must be non-empty.")

        if self.mode not in {"INPUT", "OUTPUT"}:
            raise ValueError("C6-5C engine tensor mode is invalid.")

        if self.dtype != "FLOAT":
            raise ValueError("C6-5C external engine tensor dtype must be FLOAT.")

        if not self.shape or any(value <= 0 for value in self.shape):
            raise ValueError("C6-5C engine tensor shape must be static and positive.")


@dataclass(frozen=True)
class TensorRtDiagnosticsPolicy:
    """Policy for interpreting TensorRT diagnostic output."""

    record_tensorrt_stderr: bool
    error_line_is_automatic_failure: bool
    fatal_conditions: tuple[str, ...]

    # ADD 2026-09-05: Tactic-skip diagnostics와 structural build failure를 구분한다.
    def validate(self) -> None:
        expected_fatal = (
            "parser_failure",
            "serialized_plan_build_failure",
            "deserialize_failure",
            "engine_io_contract_mismatch",
            "container_nonzero_exit",
        )

        if (
            self.record_tensorrt_stderr is not True
            or self.error_line_is_automatic_failure is not False
            or self.fatal_conditions != expected_fatal
        ):
            raise ValueError("C6-5C TensorRT diagnostics policy changed.")


@dataclass(frozen=True)
class C65CScopePolicy:
    """Frozen C6-5C build-stage scope restrictions."""

    require_clean_repository: bool
    network_allowed: bool
    overwrite_allowed: bool
    c5_engine_rebuild_allowed: bool
    application_inference_during_engine_build: bool
    dataset_used: bool
    validation_used: bool
    test_used: bool
    final_test_used: bool
    segmentation_decode_allowed: bool
    overlay_allowed: bool

    # ADD 2026-09-05: C6-5C build를 no-data/no-test/no-overlay boundary로 제한한다.
    def validate(self) -> None:
        values = (
            self.require_clean_repository,
            self.network_allowed,
            self.overwrite_allowed,
            self.c5_engine_rebuild_allowed,
            self.application_inference_during_engine_build,
            self.dataset_used,
            self.validation_used,
            self.test_used,
            self.final_test_used,
            self.segmentation_decode_allowed,
            self.overlay_allowed,
        )

        if any(type(value) is not bool for value in values):
            raise TypeError("C6-5C scope policy flags must be strict booleans.")

        if (
            self.require_clean_repository is not True
            or self.network_allowed is not False
            or self.overwrite_allowed is not False
            or self.c5_engine_rebuild_allowed is not False
            or self.application_inference_during_engine_build is not False
            or self.dataset_used is not False
            or self.validation_used is not False
            or self.test_used is not False
            or self.final_test_used is not False
            or self.segmentation_decode_allowed is not False
            or self.overlay_allowed is not False
        ):
            raise ValueError("C6-5C scope policy changed.")


@dataclass(frozen=True)
class DeepStreamTensorRtConfig:
    """Top-level C6-5C source/runtime/build/I/O contract."""

    schema_version: int
    build_id: str
    c6_5b_closure_commit: str
    source: QdqSourceIdentity
    runtime: DeepStreamTensorRtRuntime
    build: TensorRtBuildPolicy
    engine_io: tuple[EngineTensorContract, ...]
    diagnostics: TensorRtDiagnosticsPolicy
    policy: C65CScopePolicy
    config_path: Path

    # ADD 2026-09-05: C6-5C top-level contract와 nested identities를 fail-closed 검증한다.
    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported C6-5C config schema.")

        if self.build_id != EXPECTED_BUILD_ID:
            raise ValueError("Unexpected C6-5C build_id.")

        if self.c6_5b_closure_commit != EXPECTED_C6_5B_CLOSURE_COMMIT:
            raise ValueError("C6-5C baseline closure commit changed.")

        self.source.validate()
        self.runtime.validate()
        self.build.validate()
        self.diagnostics.validate()
        self.policy.validate()

        for tensor in self.engine_io:
            tensor.validate()

        observed = {
            tensor.name: {
                "mode": tensor.mode,
                "dtype": tensor.dtype,
                "shape": tensor.shape,
            }
            for tensor in self.engine_io
        }

        if observed != expected_engine_io():
            raise ValueError("C6-5C engine I/O contract changed.")


# ADD 2026-09-05: JSON value를 strict object mapping으로 변환한다.
def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")

    return cast(dict[str, Any], value)


# ADD 2026-09-05: JSON array를 strict list로 변환한다.
def _array(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array.")

    return cast(list[Any], value)


# ADD 2026-09-05: JSON mapping의 field set을 exact schema와 비교한다.
def _require_fields(
    mapping: dict[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(mapping) != expected:
        raise ValueError(f"{label} fields do not match schema.")


# ADD 2026-09-05: JSON string scalar를 strict str로 검증한다.
def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str.")

    return value


# ADD 2026-09-05: JSON integer scalar를 strict int로 검증한다.
def _integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be int.")

    return cast(int, value)


# ADD 2026-09-05: JSON boolean scalar를 strict bool로 검증한다.
def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be bool.")

    return cast(bool, value)


# ADD 2026-09-05: Frozen external TensorRT engine I/O contract를 반환한다.
def expected_engine_io() -> dict[str, dict[str, object]]:
    return {
        "images": {
            "mode": "INPUT",
            "dtype": "FLOAT",
            "shape": (1, 3, 640, 640),
        },
        "output0": {
            "mode": "OUTPUT",
            "dtype": "FLOAT",
            "shape": (1, 39, 8400),
        },
        "output1": {
            "mode": "OUTPUT",
            "dtype": "FLOAT",
            "shape": (1, 32, 160, 160),
        },
    }


# ADD 2026-09-05: Frozen JSON config를 typed C6-5C contract로 로드한다.
def load_deepstream_tensorrt_config(
    path: Path = DEFAULT_DEEPSTREAM_TENSORRT_CONFIG,
) -> DeepStreamTensorRtConfig:
    raw_object: object = json.loads(path.read_text(encoding="utf-8"))
    raw = _mapping(raw_object, label="C6-5C config")

    _require_fields(
        raw,
        {
            "schema_version",
            "build_id",
            "c6_5b_closure_commit",
            "source",
            "runtime",
            "build",
            "engine_io",
            "diagnostics",
            "policy",
        },
        label="C6-5C config",
    )

    source_raw = _mapping(raw["source"], label="source")
    runtime_raw = _mapping(raw["runtime"], label="runtime")
    build_raw = _mapping(raw["build"], label="build")
    diagnostics_raw = _mapping(raw["diagnostics"], label="diagnostics")
    policy_raw = _mapping(raw["policy"], label="policy")

    _require_fields(
        source_raw,
        {
            "evidence_zip_sha256",
            "evidence_zip_size_bytes",
            "qdq_run_commit",
            "qdq_opset",
            "quantize_linear_count",
            "dequantize_linear_count",
            "calibration_sample_count",
            "files",
        },
        label="source",
    )

    _require_fields(
        runtime_raw,
        {
            "image_tag",
            "image_id",
            "repo_digest",
            "gpu_name",
            "gpu_compute_capability",
            "driver_version",
            "deepstream_version",
            "tensorrt_python_version",
        },
        label="runtime",
    )

    _require_fields(
        build_raw,
        {
            "strongly_typed",
            "explicit_quantization",
            "builder_int8_flag",
            "builder_fp16_flag",
            "legacy_calibrator",
            "workspace_bytes",
            "plan_filename",
            "output_root",
            "persist_raw_plan",
        },
        label="build",
    )

    _require_fields(
        diagnostics_raw,
        {
            "record_tensorrt_stderr",
            "error_line_is_automatic_failure",
            "fatal_conditions",
        },
        label="diagnostics",
    )

    _require_fields(
        policy_raw,
        {
            "require_clean_repository",
            "network_allowed",
            "overwrite_allowed",
            "c5_engine_rebuild_allowed",
            "application_inference_during_engine_build",
            "dataset_used",
            "validation_used",
            "test_used",
            "final_test_used",
            "segmentation_decode_allowed",
            "overlay_allowed",
        },
        label="policy",
    )

    files_raw = _mapping(source_raw["files"], label="source.files")
    files: dict[str, SourceFileIdentity] = {}

    for filename, value in files_raw.items():
        member = _mapping(value, label=f"source.files.{filename}")

        _require_fields(
            member,
            {"sha256", "size_bytes"},
            label=f"source.files.{filename}",
        )

        files[filename] = SourceFileIdentity(
            sha256=_string(
                member["sha256"],
                label=f"source.files.{filename}.sha256",
            ),
            size_bytes=_integer(
                member["size_bytes"],
                label=f"source.files.{filename}.size_bytes",
            ),
        )

    engine_io_raw = _array(raw["engine_io"], label="engine_io")
    engine_io: list[EngineTensorContract] = []

    for index, value in enumerate(engine_io_raw):
        tensor = _mapping(value, label=f"engine_io[{index}]")

        _require_fields(
            tensor,
            {"name", "mode", "dtype", "shape"},
            label=f"engine_io[{index}]",
        )

        shape_raw = _array(
            tensor["shape"],
            label=f"engine_io[{index}].shape",
        )

        engine_io.append(
            EngineTensorContract(
                name=_string(
                    tensor["name"],
                    label=f"engine_io[{index}].name",
                ),
                mode=_string(
                    tensor["mode"],
                    label=f"engine_io[{index}].mode",
                ),
                dtype=_string(
                    tensor["dtype"],
                    label=f"engine_io[{index}].dtype",
                ),
                shape=tuple(
                    _integer(
                        dimension,
                        label=f"engine_io[{index}].shape",
                    )
                    for dimension in shape_raw
                ),
            )
        )

    fatal_raw = _array(
        diagnostics_raw["fatal_conditions"],
        label="diagnostics.fatal_conditions",
    )

    config = DeepStreamTensorRtConfig(
        schema_version=_integer(
            raw["schema_version"],
            label="schema_version",
        ),
        build_id=_string(
            raw["build_id"],
            label="build_id",
        ),
        c6_5b_closure_commit=_string(
            raw["c6_5b_closure_commit"],
            label="c6_5b_closure_commit",
        ),
        source=QdqSourceIdentity(
            evidence_zip_sha256=_string(
                source_raw["evidence_zip_sha256"],
                label="source.evidence_zip_sha256",
            ),
            evidence_zip_size_bytes=_integer(
                source_raw["evidence_zip_size_bytes"],
                label="source.evidence_zip_size_bytes",
            ),
            qdq_run_commit=_string(
                source_raw["qdq_run_commit"],
                label="source.qdq_run_commit",
            ),
            qdq_opset=_integer(
                source_raw["qdq_opset"],
                label="source.qdq_opset",
            ),
            quantize_linear_count=_integer(
                source_raw["quantize_linear_count"],
                label="source.quantize_linear_count",
            ),
            dequantize_linear_count=_integer(
                source_raw["dequantize_linear_count"],
                label="source.dequantize_linear_count",
            ),
            calibration_sample_count=_integer(
                source_raw["calibration_sample_count"],
                label="source.calibration_sample_count",
            ),
            files=files,
        ),
        runtime=DeepStreamTensorRtRuntime(
            image_tag=_string(
                runtime_raw["image_tag"],
                label="runtime.image_tag",
            ),
            image_id=_string(
                runtime_raw["image_id"],
                label="runtime.image_id",
            ),
            repo_digest=_string(
                runtime_raw["repo_digest"],
                label="runtime.repo_digest",
            ),
            gpu_name=_string(
                runtime_raw["gpu_name"],
                label="runtime.gpu_name",
            ),
            gpu_compute_capability=_string(
                runtime_raw["gpu_compute_capability"],
                label="runtime.gpu_compute_capability",
            ),
            driver_version=_string(
                runtime_raw["driver_version"],
                label="runtime.driver_version",
            ),
            deepstream_version=_string(
                runtime_raw["deepstream_version"],
                label="runtime.deepstream_version",
            ),
            tensorrt_python_version=_string(
                runtime_raw["tensorrt_python_version"],
                label="runtime.tensorrt_python_version",
            ),
        ),
        build=TensorRtBuildPolicy(
            strongly_typed=_boolean(
                build_raw["strongly_typed"],
                label="build.strongly_typed",
            ),
            explicit_quantization=_boolean(
                build_raw["explicit_quantization"],
                label="build.explicit_quantization",
            ),
            builder_int8_flag=_boolean(
                build_raw["builder_int8_flag"],
                label="build.builder_int8_flag",
            ),
            builder_fp16_flag=_boolean(
                build_raw["builder_fp16_flag"],
                label="build.builder_fp16_flag",
            ),
            legacy_calibrator=_boolean(
                build_raw["legacy_calibrator"],
                label="build.legacy_calibrator",
            ),
            workspace_bytes=_integer(
                build_raw["workspace_bytes"],
                label="build.workspace_bytes",
            ),
            plan_filename=_string(
                build_raw["plan_filename"],
                label="build.plan_filename",
            ),
            output_root=Path(
                _string(
                    build_raw["output_root"],
                    label="build.output_root",
                )
            ),
            persist_raw_plan=_boolean(
                build_raw["persist_raw_plan"],
                label="build.persist_raw_plan",
            ),
        ),
        engine_io=tuple(engine_io),
        diagnostics=TensorRtDiagnosticsPolicy(
            record_tensorrt_stderr=_boolean(
                diagnostics_raw["record_tensorrt_stderr"],
                label="diagnostics.record_tensorrt_stderr",
            ),
            error_line_is_automatic_failure=_boolean(
                diagnostics_raw["error_line_is_automatic_failure"],
                label="diagnostics.error_line_is_automatic_failure",
            ),
            fatal_conditions=tuple(
                _string(
                    value,
                    label="diagnostics.fatal_conditions",
                )
                for value in fatal_raw
            ),
        ),
        policy=C65CScopePolicy(
            require_clean_repository=_boolean(
                policy_raw["require_clean_repository"],
                label="policy.require_clean_repository",
            ),
            network_allowed=_boolean(
                policy_raw["network_allowed"],
                label="policy.network_allowed",
            ),
            overwrite_allowed=_boolean(
                policy_raw["overwrite_allowed"],
                label="policy.overwrite_allowed",
            ),
            c5_engine_rebuild_allowed=_boolean(
                policy_raw["c5_engine_rebuild_allowed"],
                label="policy.c5_engine_rebuild_allowed",
            ),
            application_inference_during_engine_build=_boolean(
                policy_raw["application_inference_during_engine_build"],
                label="policy.application_inference_during_engine_build",
            ),
            dataset_used=_boolean(
                policy_raw["dataset_used"],
                label="policy.dataset_used",
            ),
            validation_used=_boolean(
                policy_raw["validation_used"],
                label="policy.validation_used",
            ),
            test_used=_boolean(
                policy_raw["test_used"],
                label="policy.test_used",
            ),
            final_test_used=_boolean(
                policy_raw["final_test_used"],
                label="policy.final_test_used",
            ),
            segmentation_decode_allowed=_boolean(
                policy_raw["segmentation_decode_allowed"],
                label="policy.segmentation_decode_allowed",
            ),
            overlay_allowed=_boolean(
                policy_raw["overlay_allowed"],
                label="policy.overlay_allowed",
            ),
        ),
        config_path=path.resolve(),
    )

    config.validate()
    return config


# ADD 2026-09-05: C5-4B1 ZIP과 모든 source member를 exact hash/bytes로 검증한다.
def validate_source_archive(
    path: Path,
    config: DeepStreamTensorRtConfig,
) -> dict[str, dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"C6-5C source archive not found: {path}")

    if sha256_file(path) != config.source.evidence_zip_sha256:
        raise ValueError("C6-5C source archive SHA-256 changed.")

    if path.stat().st_size != config.source.evidence_zip_size_bytes:
        raise ValueError("C6-5C source archive byte size changed.")

    result: dict[str, dict[str, object]] = {}

    with zipfile.ZipFile(path, "r") as archive:
        bad = archive.testzip()

        if bad is not None:
            raise ValueError(f"C6-5C source ZIP integrity failed: {bad}")

        names = {info.filename for info in archive.infolist() if not info.is_dir()}

        if names != set(config.source.files):
            raise ValueError("C6-5C source archive member set changed.")

        for filename, identity in config.source.files.items():
            data = archive.read(filename)
            digest = hashlib.sha256(data).hexdigest()

            if digest != identity.sha256 or len(data) != identity.size_bytes:
                raise ValueError(f"C6-5C source member identity changed: {filename}")

            result[filename] = {
                "sha256": digest,
                "size_bytes": len(data),
            }

    return result


# ADD 2026-09-05: Container stdout의 단일 build-result marker를 strict JSON object로 읽는다.
def parse_container_build_payload(
    stdout: str,
    *,
    prefix: str = "C6_5C_BUILD_RESULT=",
) -> dict[str, Any]:
    matches = [line[len(prefix) :] for line in stdout.splitlines() if line.startswith(prefix)]

    if len(matches) != 1:
        raise ValueError("C6-5C container result marker count must be exactly one.")

    raw: object = json.loads(matches[0])

    return _mapping(
        raw,
        label="C6-5C build payload",
    )


# ADD 2026-09-05: Build result를 frozen runtime/source/I/O/scope contract와 대조한다.
def validate_container_build_payload(
    payload: dict[str, Any],
    config: DeepStreamTensorRtConfig,
) -> None:
    required = {
        "status",
        "tensorrt_version",
        "source_qdq_onnx_sha256",
        "source_qdq_onnx_bytes",
        "parser_success",
        "parser_error_count",
        "strongly_typed",
        "workspace_bytes",
        "build_succeeded",
        "deserialize_succeeded",
        "plan_filename",
        "plan_sha256",
        "plan_bytes",
        "io_tensors",
        "application_inference_executed",
        "dataset_used",
        "validation_used",
        "test_used",
        "final_test_used",
    }

    _require_fields(
        payload,
        required,
        label="C6-5C build payload",
    )

    qdq = config.source.files["model.int8.qdq.onnx"]

    if (
        payload["status"] != "passed"
        or payload["tensorrt_version"] != config.runtime.tensorrt_python_version
        or payload["source_qdq_onnx_sha256"] != qdq.sha256
        or payload["source_qdq_onnx_bytes"] != qdq.size_bytes
        or payload["parser_success"] is not True
        or payload["parser_error_count"] != 0
        or payload["strongly_typed"] is not True
        or payload["workspace_bytes"] != config.build.workspace_bytes
        or payload["build_succeeded"] is not True
        or payload["deserialize_succeeded"] is not True
        or payload["plan_filename"] != config.build.plan_filename
        or payload["application_inference_executed"] is not False
        or payload["dataset_used"] is not False
        or payload["validation_used"] is not False
        or payload["test_used"] is not False
        or payload["final_test_used"] is not False
    ):
        raise ValueError("C6-5C build payload structural contract failed.")

    plan_sha = payload["plan_sha256"]

    if not isinstance(plan_sha, str) or not is_sha256_digest(plan_sha):
        raise ValueError("C6-5C generated plan SHA-256 is invalid.")

    plan_bytes = payload["plan_bytes"]

    if type(plan_bytes) is not int or plan_bytes <= 0:
        raise ValueError("C6-5C generated plan size is invalid.")

    raw_tensors = _array(
        payload["io_tensors"],
        label="C6-5C payload io_tensors",
    )

    observed: dict[str, dict[str, object]] = {}

    for index, value in enumerate(raw_tensors):
        tensor = _mapping(
            value,
            label=f"C6-5C payload io_tensors[{index}]",
        )

        _require_fields(
            tensor,
            {"name", "mode", "dtype", "shape"},
            label=f"C6-5C payload io_tensors[{index}]",
        )

        shape_raw = _array(
            tensor["shape"],
            label=f"C6-5C payload io_tensors[{index}].shape",
        )

        name = _string(
            tensor["name"],
            label=f"C6-5C payload io_tensors[{index}].name",
        )

        if name in observed:
            raise ValueError("C6-5C payload contains duplicate tensor name.")

        observed[name] = {
            "mode": _string(
                tensor["mode"],
                label=f"C6-5C payload io_tensors[{index}].mode",
            ),
            "dtype": _string(
                tensor["dtype"],
                label=f"C6-5C payload io_tensors[{index}].dtype",
            ),
            "shape": tuple(
                _integer(
                    dimension,
                    label=f"C6-5C payload io_tensors[{index}].shape",
                )
                for dimension in shape_raw
            ),
        }

    if observed != expected_engine_io():
        raise ValueError("C6-5C payload engine I/O contract changed.")
