"""C6-5A contracts for probing the accepted C5 TensorRT engine in DeepStream."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from shared.hashing import is_sha256_digest, sha256_file

DEFAULT_DEEPSTREAM_COMPATIBILITY_CONFIG = Path(
    "configs/streaming/yolo_deepstream_c5_engine_compatibility.json"
)

EXPECTED_PROBE_ID = "c6_5a_c5_engine_deepstream_compatibility_v1"

EXPECTED_C5_CLOSURE_COMMIT = "88e9b0b2440e99b6dfd2594bdc9a4947eff75187"
EXPECTED_ENGINE_BUILD_COMMIT = "7835291c8fb123eba6acfa839977f94093c2f3ac"
EXPECTED_EVIDENCE_ZIP_SHA256 = "0cba556981b12a95b25feb324d0ff02b9cadeda6bde056b46e27eb7698f66b00"

EXPECTED_ENGINE_SIZE_BYTES = 7_607_387
EXPECTED_ENGINE_HEADER_LENGTH = 155
EXPECTED_SERIALIZED_ENGINE_SIZE_BYTES = 7_607_228

EXPECTED_FILE_HASHES = {
    "metadata.json": ("d44de78cc89fea67d6b351c2ba92f76dda0242386f4b6f14e216740ca682461e"),
    "model.engine": ("4f397d59741f4efb7832087030b890a0fe059a657d074a3b07cdeb54493e8971"),
    "run_summary.json": ("a5de13fc8e616b6071eebc0d76f0f88cdff32181e8d49db5cdd143113aef113f"),
    "yolo_segmentation_tensorrt_int8_engine.yaml": (
        "63eebcac04d11c9247bf7543fe18d0798758ab20cc734d2b18bfbece4eaf6b41"
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


@dataclass(frozen=True)
class C5SourceBundle:
    """Exact accepted C5-4B2 evidence identity used by C6-5A."""

    c5_closure_commit: str
    engine_build_commit: str
    evidence_zip_sha256: str
    engine_size_bytes: int
    header_length: int
    files: dict[str, str]
    rebuild_allowed: bool

    # ADD 2026-09-04: C6-5A source를 accepted C5-4B2 evidence identity에 고정한다.
    def validate(self) -> None:
        if type(self.engine_size_bytes) is not int:
            raise TypeError("C6-5A engine_size_bytes must be int.")
        if type(self.header_length) is not int:
            raise TypeError("C6-5A header_length must be int.")
        if type(self.rebuild_allowed) is not bool:
            raise TypeError("C6-5A rebuild_allowed must be bool.")

        if (
            self.c5_closure_commit != EXPECTED_C5_CLOSURE_COMMIT
            or self.engine_build_commit != EXPECTED_ENGINE_BUILD_COMMIT
            or self.evidence_zip_sha256 != EXPECTED_EVIDENCE_ZIP_SHA256
            or self.engine_size_bytes != EXPECTED_ENGINE_SIZE_BYTES
            or self.header_length != EXPECTED_ENGINE_HEADER_LENGTH
            or self.files != EXPECTED_FILE_HASHES
            or self.rebuild_allowed is not False
        ):
            raise ValueError("C6-5A C5 source bundle identity changed.")

        for digest in (self.evidence_zip_sha256, *self.files.values()):
            if not is_sha256_digest(digest):
                raise ValueError("C6-5A source bundle contains invalid SHA-256.")


@dataclass(frozen=True)
class DeepStreamRuntimeIdentity:
    """Observed DeepStream/L4 runtime identity for the compatibility probe."""

    image_tag: str
    image_id: str
    repo_digest: str
    gpu_name: str
    gpu_compute_capability: str
    driver_version: str
    deepstream_version: str
    tensorrt_python_version: str

    # ADD 2026-09-04: C6-5A observation을 exact L4/DeepStream runtime identity에 묶는다.
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

        mismatches = [
            name
            for name, expected_value in expected.items()
            if getattr(self, name) != expected_value
        ]
        if mismatches:
            raise ValueError("C6-5A DeepStream runtime identity changed: " + ", ".join(mismatches))


@dataclass(frozen=True)
class CompatibilityPolicy:
    """Restrictions for the C6-5A deserialize-only compatibility probe."""

    mode: str
    require_clean_repository: bool
    inference_allowed: bool
    dataset_used: bool
    validation_used: bool
    test_used: bool
    final_test_used: bool

    # ADD 2026-09-04: C6-5A를 deserialize-only/no-data/no-inference probe로 제한한다.
    def validate(self) -> None:
        bool_values = (
            self.require_clean_repository,
            self.inference_allowed,
            self.dataset_used,
            self.validation_used,
            self.test_used,
            self.final_test_used,
        )
        if any(type(value) is not bool for value in bool_values):
            raise TypeError("C6-5A policy flags must be strict booleans.")

        if (
            self.mode != "deserialize_only"
            or self.require_clean_repository is not True
            or self.inference_allowed is not False
            or self.dataset_used is not False
            or self.validation_used is not False
            or self.test_used is not False
            or self.final_test_used is not False
        ):
            raise ValueError("C6-5A compatibility policy changed.")


@dataclass(frozen=True)
class DeepStreamCompatibilityConfig:
    """Top-level C6-5A source/runtime/probe contract."""

    schema_version: int
    probe_id: str
    source_bundle: C5SourceBundle
    runtime: DeepStreamRuntimeIdentity
    policy: CompatibilityPolicy
    config_path: Path

    # ADD 2026-09-04: C6-5A config schema와 nested contracts를 fail-closed 검증한다.
    def validate(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError("C6-5A schema_version must be int.")
        if self.schema_version != 1:
            raise ValueError("Unsupported C6-5A config schema.")
        if self.probe_id != EXPECTED_PROBE_ID:
            raise ValueError("Unexpected C6-5A probe_id.")

        self.source_bundle.validate()
        self.runtime.validate()
        self.policy.validate()


@dataclass(frozen=True)
class EngineContainerObservation:
    """Validated C5 custom container header and serialized TensorRT plan."""

    header_length: int
    header: dict[str, Any]
    serialized_engine: bytes

    # ADD 2026-09-04: C5 custom container framing과 serialized plan size를 검증한다.
    def validate(self) -> None:
        if self.header_length != EXPECTED_ENGINE_HEADER_LENGTH:
            raise ValueError("C6-5A engine header length changed.")
        if self.header != expected_engine_header():
            raise ValueError("C6-5A engine header changed.")
        if len(self.serialized_engine) != EXPECTED_SERIALIZED_ENGINE_SIZE_BYTES:
            raise ValueError("C6-5A serialized TensorRT plan size changed.")


# ADD 2026-09-04: JSON value를 strict object mapping으로 검증한다.
def _as_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return cast(dict[str, Any], value)


# ADD 2026-09-04: Dependency-free JSON config를 typed C6-5A contract로 로드한다.
def load_deepstream_compatibility_config(
    path: Path = DEFAULT_DEEPSTREAM_COMPATIBILITY_CONFIG,
) -> DeepStreamCompatibilityConfig:
    """Load and validate the frozen C6-5A JSON compatibility contract."""

    raw_object: object = json.loads(path.read_text(encoding="utf-8"))
    raw = _as_mapping(raw_object, label="C6-5A config")

    if set(raw) != {
        "schema_version",
        "probe_id",
        "source_bundle",
        "runtime",
        "policy",
    }:
        raise ValueError("C6-5A config fields do not match schema.")

    source_raw = _as_mapping(raw["source_bundle"], label="source_bundle")
    runtime_raw = _as_mapping(raw["runtime"], label="runtime")
    policy_raw = _as_mapping(raw["policy"], label="policy")
    files_raw = _as_mapping(source_raw.get("files"), label="source_bundle.files")

    try:
        source = C5SourceBundle(
            c5_closure_commit=cast(str, source_raw["c5_closure_commit"]),
            engine_build_commit=cast(str, source_raw["engine_build_commit"]),
            evidence_zip_sha256=cast(str, source_raw["evidence_zip_sha256"]),
            engine_size_bytes=cast(int, source_raw["engine_size_bytes"]),
            header_length=cast(int, source_raw["header_length"]),
            files={str(filename): str(digest) for filename, digest in files_raw.items()},
            rebuild_allowed=cast(bool, source_raw["rebuild_allowed"]),
        )

        runtime = DeepStreamRuntimeIdentity(
            image_tag=cast(str, runtime_raw["image_tag"]),
            image_id=cast(str, runtime_raw["image_id"]),
            repo_digest=cast(str, runtime_raw["repo_digest"]),
            gpu_name=cast(str, runtime_raw["gpu_name"]),
            gpu_compute_capability=cast(
                str,
                runtime_raw["gpu_compute_capability"],
            ),
            driver_version=cast(str, runtime_raw["driver_version"]),
            deepstream_version=cast(str, runtime_raw["deepstream_version"]),
            tensorrt_python_version=cast(
                str,
                runtime_raw["tensorrt_python_version"],
            ),
        )

        policy = CompatibilityPolicy(
            mode=cast(str, policy_raw["mode"]),
            require_clean_repository=cast(
                bool,
                policy_raw["require_clean_repository"],
            ),
            inference_allowed=cast(bool, policy_raw["inference_allowed"]),
            dataset_used=cast(bool, policy_raw["dataset_used"]),
            validation_used=cast(bool, policy_raw["validation_used"]),
            test_used=cast(bool, policy_raw["test_used"]),
            final_test_used=cast(bool, policy_raw["final_test_used"]),
        )
    except KeyError as exc:
        raise ValueError("C6-5A config required field is missing.") from exc

    config = DeepStreamCompatibilityConfig(
        schema_version=cast(int, raw["schema_version"]),
        probe_id=cast(str, raw["probe_id"]),
        source_bundle=source,
        runtime=runtime,
        policy=policy,
        config_path=path.resolve(),
    )
    config.validate()
    return config


# ADD 2026-09-04: Accepted C5 engine의 decoded JSON header identity를 반환한다.
def expected_engine_header() -> dict[str, Any]:
    """Return the exact decoded Ultralytics header expected from the C5 engine."""

    return {
        "args": {
            "dynamic": False,
            "nms": False,
        },
        "batch": 1,
        "channels": 3,
        "imgsz": [640, 640],
        "names": {
            "0": "bent",
            "1": "color",
            "2": "scratch",
        },
        "stride": 32,
        "task": "segment",
    }


# ADD 2026-09-04: C5 custom engine container에서 JSON header와 TensorRT plan을 분리한다.
def read_engine_container(path: Path) -> EngineContainerObservation:
    """Decode the accepted C5 engine wrapper without deserializing TensorRT."""

    if not path.is_file():
        raise FileNotFoundError(f"C6-5A engine not found: {path}")

    data = path.read_bytes()
    if len(data) < 5:
        raise ValueError("C6-5A engine container is too small.")

    header_length = int.from_bytes(
        data[:4],
        byteorder="little",
        signed=True,
    )
    if header_length <= 0 or 4 + header_length >= len(data):
        raise ValueError("C6-5A engine header length is invalid.")

    header_object: object = json.loads(data[4 : 4 + header_length].decode("utf-8"))
    header = _as_mapping(
        header_object,
        label="C6-5A engine header",
    )

    observation = EngineContainerObservation(
        header_length=header_length,
        header=header,
        serialized_engine=data[4 + header_length :],
    )
    observation.validate()
    return observation


# ADD 2026-09-04: C5 SHA256SUMS를 normalized filename-to-digest mapping으로 읽는다.
def read_sha256sums(path: Path) -> dict[str, str]:
    """Read the restored C5 evidence SHA256SUMS manifest."""

    if not path.is_file():
        raise FileNotFoundError(f"C6-5A SHA256SUMS not found: {path}")

    result: dict[str, str] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError("C6-5A SHA256SUMS contains malformed line.")

        digest, filename = parts
        filename = filename.lstrip("*")
        if filename.startswith("./"):
            filename = filename[2:]

        if not filename:
            raise ValueError("C6-5A SHA256SUMS contains empty filename.")
        if not is_sha256_digest(digest):
            raise ValueError("C6-5A SHA256SUMS contains invalid SHA-256.")
        if filename in result:
            raise ValueError("C6-5A SHA256SUMS contains duplicate filename.")

        result[filename] = digest

    return result


# ADD 2026-09-04: Restored C5 evidence files와 engine framing을 exact accepted identity로 검증한다.
def validate_c5_source_bundle(
    *,
    input_dir: Path,
    config: DeepStreamCompatibilityConfig,
) -> dict[str, Any]:
    """Validate restored C5-4B2 evidence before any TensorRT operation."""

    if not input_dir.is_dir():
        raise FileNotFoundError(f"C6-5A input directory not found: {input_dir}")

    source = config.source_bundle

    observed_hashes: dict[str, str] = {}
    for filename, expected_digest in source.files.items():
        path = input_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"C6-5A required C5 file missing: {path}")

        observed_digest = sha256_file(path)
        if observed_digest != expected_digest:
            raise ValueError(f"C6-5A C5 file SHA mismatch: {filename}")

        observed_hashes[filename] = observed_digest

    manifest_hashes = read_sha256sums(input_dir / "SHA256SUMS.txt")
    if manifest_hashes != source.files:
        raise ValueError("C6-5A restored SHA256SUMS differs from accepted C5 identities.")

    engine_path = input_dir / "model.engine"
    engine_size = engine_path.stat().st_size
    if engine_size != source.engine_size_bytes:
        raise ValueError("C6-5A C5 engine file size changed.")

    observation = read_engine_container(engine_path)

    return {
        "engine_path": str(engine_path.resolve()),
        "engine_sha256": observed_hashes["model.engine"],
        "engine_size_bytes": engine_size,
        "header_length": observation.header_length,
        "serialized_tensorrt_bytes": len(observation.serialized_engine),
        "task": observation.header["task"],
        "batch": observation.header["batch"],
        "channels": observation.header["channels"],
        "imgsz": observation.header["imgsz"],
        "class_names": observation.header["names"],
        "engine_rebuilt": False,
        "dataset_used": False,
        "validation_used": False,
        "test_used": False,
        "final_test_used": False,
    }


COMPATIBLE_STATE = "C5_ENGINE_DEEPSTREAM_RUNTIME_COMPATIBLE"
INCOMPATIBLE_STATE = "C5_ENGINE_DEEPSTREAM_RUNTIME_INCOMPATIBLE"

COMPATIBLE_REASON = "TENSORRT_DESERIALIZATION_COMPATIBLE"
VERSION_INCOMPATIBLE_REASON = "TENSORRT_VERSION_INCOMPATIBLE"
DESERIALIZATION_INCOMPATIBLE_REASON = "TENSORRT_DESERIALIZATION_INCOMPATIBLE"

CONTAINER_RESULT_PREFIX = "C6_5A_RESULT_JSON="
C6_5A_CONTAINER_LABEL = "c6_5a_probe=1"


# ADD 2026-09-04: Exact DeepStream digest와 local image ID가 일치하는지 검증한다.
def inspect_deepstream_image(
    config: DeepStreamCompatibilityConfig,
) -> dict[str, Any]:
    """Inspect the exact local DeepStream image required by C6-5A."""

    image_id_result = subprocess.run(
        [
            "sudo",
            "docker",
            "image",
            "inspect",
            config.runtime.repo_digest,
            "--format",
            "{{.Id}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    repo_digests_result = subprocess.run(
        [
            "sudo",
            "docker",
            "image",
            "inspect",
            config.runtime.repo_digest,
            "--format",
            "{{json .RepoDigests}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    image_id = image_id_result.stdout.strip()

    digests_object: object = json.loads(repo_digests_result.stdout.strip())
    if not isinstance(digests_object, list):
        raise RuntimeError("C6-5A Docker RepoDigests output must be a JSON array.")

    repo_digests = tuple(str(value) for value in digests_object)

    if image_id != config.runtime.image_id:
        raise RuntimeError("C6-5A DeepStream local image ID changed.")

    if config.runtime.repo_digest not in repo_digests:
        raise RuntimeError("C6-5A required DeepStream repo digest is not present locally.")

    return {
        "image_tag": config.runtime.image_tag,
        "image_id": image_id,
        "repo_digest": config.runtime.repo_digest,
        "repo_digests": list(repo_digests),
    }


# ADD 2026-09-04: DeepStream container에서 실행할 dependency-free deserialize probe source를 만든다.
def build_container_probe_source(
    config: DeepStreamCompatibilityConfig,
) -> str:
    """Build the stdlib-plus-TensorRT probe executed inside DeepStream."""

    expected_header_json = json.dumps(
        expected_engine_header(),
        sort_keys=True,
        separators=(",", ":"),
    )

    template = r"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import tensorrt as trt

ENGINE_PATH = Path("/c6-5-inputs/model.engine")

EXPECTED_SHA256 = __EXPECTED_SHA256__
EXPECTED_ENGINE_SIZE = __EXPECTED_ENGINE_SIZE__
EXPECTED_HEADER_LENGTH = __EXPECTED_HEADER_LENGTH__
EXPECTED_SERIALIZED_SIZE = __EXPECTED_SERIALIZED_SIZE__
EXPECTED_HEADER = json.loads(__EXPECTED_HEADER_JSON__)

data = ENGINE_PATH.read_bytes()

if len(data) != EXPECTED_ENGINE_SIZE:
    raise RuntimeError("Container engine file size changed.")

engine_sha256 = hashlib.sha256(data).hexdigest()
if engine_sha256 != EXPECTED_SHA256:
    raise RuntimeError("Container engine SHA-256 changed.")

header_length = int.from_bytes(
    data[:4],
    byteorder="little",
    signed=True,
)
if header_length != EXPECTED_HEADER_LENGTH:
    raise RuntimeError("Container engine header length changed.")

header = json.loads(
    data[4:4 + header_length].decode("utf-8")
)
if header != EXPECTED_HEADER:
    raise RuntimeError("Container engine header changed.")

serialized = data[4 + header_length:]
if len(serialized) != EXPECTED_SERIALIZED_SIZE:
    raise RuntimeError("Container TensorRT plan size changed.")

gpu_result = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
)

gpu_parts = [
    part.strip()
    for part in gpu_result.stdout.strip().split(",")
]
if len(gpu_parts) != 4:
    raise RuntimeError("Unexpected nvidia-smi output.")

deepstream_result = subprocess.run(
    ["deepstream-app", "--version-all"],
    check=True,
    capture_output=True,
    text=True,
)

deepstream_text = (
    deepstream_result.stdout
    + "\n"
    + deepstream_result.stderr
)

deepstream_match = re.search(
    r"deepstream-app(?:\s+version)?\s+([0-9]+(?:\.[0-9]+)+)",
    deepstream_text,
    flags=re.IGNORECASE,
)
if deepstream_match is None:
    raise RuntimeError(
        "DeepStream version could not be parsed."
    )

logger = trt.Logger(trt.Logger.VERBOSE)
runtime = trt.Runtime(logger)

exception_type = None
exception_message = None

try:
    engine = runtime.deserialize_cuda_engine(serialized)
except Exception as exc:
    engine = None
    exception_type = type(exc).__name__
    exception_message = str(exc)

io_tensors = []

if engine is not None:
    for index in range(int(engine.num_io_tensors)):
        tensor_name = str(engine.get_tensor_name(index))

        io_tensors.append(
            {
                "name": tensor_name,
                "mode": str(
                    engine.get_tensor_mode(tensor_name)
                ),
                "dtype": str(
                    engine.get_tensor_dtype(tensor_name)
                ),
                "shape": [
                    int(value)
                    for value in engine.get_tensor_shape(
                        tensor_name
                    )
                ],
            }
        )

payload = {
    "status": (
        "compatible"
        if engine is not None
        else "incompatible"
    ),
    "python_tensorrt_version": str(trt.__version__),
    "deepstream_version": deepstream_match.group(1),
    "gpu_name": gpu_parts[0],
    "driver_version": gpu_parts[1],
    "gpu_memory_mib": gpu_parts[2],
    "gpu_compute_capability": gpu_parts[3],
    "engine_sha256": engine_sha256,
    "engine_file_bytes": len(data),
    "header_length": header_length,
    "serialized_tensorrt_bytes": len(serialized),
    "task": header.get("task"),
    "num_io_tensors": len(io_tensors),
    "io_tensors": io_tensors,
    "deserialize_exception_type": exception_type,
    "deserialize_exception_message": exception_message,
    "inference_executed": False,
    "engine_rebuilt": False,
    "dataset_used": False,
    "validation_used": False,
    "test_used": False,
    "final_test_used": False,
}

print(
    "C6_5A_RESULT_JSON="
    + json.dumps(
        payload,
        sort_keys=True,
        allow_nan=False,
    )
)
""".strip()

    return (
        template.replace(
            "__EXPECTED_SHA256__",
            repr(config.source_bundle.files["model.engine"]),
        )
        .replace(
            "__EXPECTED_ENGINE_SIZE__",
            str(EXPECTED_ENGINE_SIZE_BYTES),
        )
        .replace(
            "__EXPECTED_HEADER_LENGTH__",
            str(EXPECTED_ENGINE_HEADER_LENGTH),
        )
        .replace(
            "__EXPECTED_SERIALIZED_SIZE__",
            str(EXPECTED_SERIALIZED_ENGINE_SIZE_BYTES),
        )
        .replace(
            "__EXPECTED_HEADER_JSON__",
            repr(expected_header_json),
        )
        + "\n"
    )


# ADD 2026-09-04: Exact digest와 read-only C5 evidence mount Docker command를 만든다.
# MODIFY 2026-09-04: stdin probe 전달을 위해 interactive mode를 추가한다.
def build_docker_probe_command(
    *,
    input_dir: Path,
    config: DeepStreamCompatibilityConfig,
) -> list[str]:
    """Build the isolated DeepStream deserialize-only Docker command."""

    return [
        "sudo",
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--interactive",
        "--label",
        C6_5A_CONTAINER_LABEL,
        "--gpus",
        "all",
        "--entrypoint",
        "python3",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "-v",
        f"{input_dir.resolve()}:/c6-5-inputs:ro",
        config.runtime.repo_digest,
        "-",
    ]


# ADD 2026-09-04: Container stdout에서 marker가 붙은 compatibility JSON만 추출한다.
def parse_container_probe_payload(
    stdout: str,
) -> dict[str, Any]:
    """Parse the marked JSON result emitted by the container probe."""

    for line in reversed(stdout.splitlines()):
        if not line.startswith(CONTAINER_RESULT_PREFIX):
            continue

        raw_object: object = json.loads(line[len(CONTAINER_RESULT_PREFIX) :])
        if not isinstance(raw_object, dict):
            raise RuntimeError("C6-5A container result must be a JSON object.")

        return cast(dict[str, Any], raw_object)

    raise RuntimeError("C6-5A container did not emit its result marker.")


# ADD 2026-09-04: Container observation이 frozen runtime/source identity와 일치하는지 검증한다.
def validate_container_probe_payload(
    *,
    payload: dict[str, Any],
    config: DeepStreamCompatibilityConfig,
) -> None:
    """Fail closed on runtime, source, or no-inference boundary drift."""

    expected = {
        "python_tensorrt_version": (config.runtime.tensorrt_python_version),
        "deepstream_version": config.runtime.deepstream_version,
        "gpu_name": config.runtime.gpu_name,
        "driver_version": config.runtime.driver_version,
        "gpu_compute_capability": (config.runtime.gpu_compute_capability),
        "engine_sha256": (config.source_bundle.files["model.engine"]),
        "engine_file_bytes": (config.source_bundle.engine_size_bytes),
        "header_length": config.source_bundle.header_length,
        "serialized_tensorrt_bytes": (EXPECTED_SERIALIZED_ENGINE_SIZE_BYTES),
        "task": "segment",
        "inference_executed": False,
        "engine_rebuilt": False,
        "dataset_used": False,
        "validation_used": False,
        "test_used": False,
        "final_test_used": False,
    }

    mismatches = [
        field for field, expected_value in expected.items() if payload.get(field) != expected_value
    ]

    if mismatches:
        raise RuntimeError("C6-5A container observation identity changed: " + ", ".join(mismatches))

    if payload.get("status") not in {
        "compatible",
        "incompatible",
    }:
        raise RuntimeError("C6-5A container returned invalid compatibility status.")


# ADD 2026-09-04: TensorRT logger evidence를 stable compatibility reason으로 분류한다.
def classify_compatibility_reason(
    *,
    status: str,
    diagnostic_text: str,
) -> str:
    """Classify deserialize outcome without inferring hardware incompatibility."""

    if status == "compatible":
        return COMPATIBLE_REASON

    if status != "incompatible":
        raise ValueError("C6-5A compatibility status is invalid.")

    normalized_diagnostics = diagnostic_text.lower()

    if (
        "engine plan file is not compatible with this version of tensorrt" in normalized_diagnostics
        or "checkengineversioncompatible" in normalized_diagnostics
    ):
        return VERSION_INCOMPATIBLE_REASON

    return DESERIALIZATION_INCOMPATIBLE_REASON


# ADD 2026-09-04: Exact DeepStream image에서 deserialize-only probe를 실행한다.
def run_deepstream_deserialize_probe(
    *,
    input_dir: Path,
    config: DeepStreamCompatibilityConfig,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Run one no-network, no-inference TensorRT deserialize observation."""

    if type(timeout_seconds) is not int:
        raise TypeError("C6-5A timeout_seconds must be int.")
    if timeout_seconds <= 0:
        raise ValueError("C6-5A timeout_seconds must be positive.")

    source_observation = validate_c5_source_bundle(
        input_dir=input_dir,
        config=config,
    )

    image_observation = inspect_deepstream_image(config)

    command = build_docker_probe_command(
        input_dir=input_dir,
        config=config,
    )

    result = subprocess.run(
        command,
        input=build_container_probe_source(config),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "C6-5A Docker probe failed as an infrastructure error: "
            f"exit={result.returncode}\n"
            f"{result.stderr}"
        )

    payload = parse_container_probe_payload(result.stdout)

    validate_container_probe_payload(
        payload=payload,
        config=config,
    )

    status = str(payload["status"])

    diagnostics = result.stdout + "\n" + result.stderr

    reason = classify_compatibility_reason(
        status=status,
        diagnostic_text=diagnostics,
    )

    state = COMPATIBLE_STATE if status == "compatible" else INCOMPATIBLE_STATE

    return {
        "state": state,
        "compatibility": status,
        "reason": reason,
        "source": source_observation,
        "runtime": payload,
        "container_image": image_observation,
        "probe_exit_code": result.returncode,
        "probe_stdout": result.stdout,
        "probe_stderr": result.stderr,
    }


# ADD 2026-09-04: Git commit과 working-tree 상태를 canonical evidence provenance로 조회한다.
def resolve_repository_identity(
    repo: Path,
) -> dict[str, str | bool]:
    """Return the current Git commit and whether the working tree is dirty."""

    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    commit = commit_result.stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("C6-5A repository commit must be a full Git SHA.")

    return {
        "git_commit": commit,
        "working_tree_dirty": bool(status_result.stdout.strip()),
    }


# ADD 2026-09-04: Canonical C6-5A observation을 Git 외부 JSON evidence로 기록한다.
def write_deepstream_compatibility_evidence(
    *,
    input_dir: Path,
    output_path: Path,
    repo: Path,
    config_path: Path = DEFAULT_DEEPSTREAM_COMPATIBILITY_CONFIG,
) -> Path:
    """Run the clean-commit probe and write canonical evidence outside Git."""

    repo = repo.resolve()
    output_path = output_path.resolve()

    config = load_deepstream_compatibility_config(config_path)

    repository = resolve_repository_identity(repo)

    if config.policy.require_clean_repository and repository["working_tree_dirty"] is not False:
        raise RuntimeError("C6-5A canonical probe requires a clean repository.")

    try:
        output_path.relative_to(repo)
    except ValueError:
        pass
    else:
        raise ValueError("C6-5A canonical evidence must be outside the Git repository.")

    if output_path.exists():
        raise FileExistsError(f"C6-5A evidence already exists: {output_path}")

    observation = run_deepstream_deserialize_probe(
        input_dir=input_dir,
        config=config,
    )

    evidence = {
        "schema_version": 1,
        "probe_id": config.probe_id,
        "observed_at": datetime.now(UTC).isoformat(),
        "state": observation["state"],
        "compatibility": observation["compatibility"],
        "reason": observation["reason"],
        "source": observation["source"],
        "runtime": observation["runtime"],
        "container_image": observation["container_image"],
        "repository": repository,
        "policy": asdict(config.policy),
        "config_sha256": sha256_file(config.config_path),
        "probe_exit_code": observation["probe_exit_code"],
        "probe_stdout": observation["probe_stdout"],
        "probe_stderr": observation["probe_stderr"],
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path
