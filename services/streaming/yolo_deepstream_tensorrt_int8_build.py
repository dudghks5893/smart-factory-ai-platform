"""C6-5C host orchestration for canonical DeepStream/L4 TensorRT plan builds."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from services.streaming.yolo_deepstream_tensorrt_int8 import (
    DEFAULT_DEEPSTREAM_TENSORRT_CONFIG,
    DeepStreamTensorRtConfig,
    expected_engine_io,
    load_deepstream_tensorrt_config,
    parse_container_build_payload,
    validate_container_build_payload,
    validate_source_archive,
)
from shared.hashing import sha256_bytes, sha256_file

BUILD_STATE = "DEEPSTREAM_L4_TENSORRT_INT8_PLAN_BUILT"
BUILD_METADATA_FILENAME = "metadata.json"
BUILD_TIMEOUT_SECONDS = 300

CONTAINER_LABEL = "c6_5c_engine_build=1"
CONTAINER_RESULT_PREFIX = "C6_5C_BUILD_RESULT="


@dataclass(frozen=True)
class RepositoryIdentity:
    """Clean repository identity used for one canonical C6-5C build."""

    git_commit: str
    working_tree_dirty: bool

    # ADD 2026-09-05: Canonical C6-5C build가 clean Git state인지 검증한다.
    def validate(self) -> None:
        if not self.git_commit:
            raise ValueError("C6-5C repository commit must be non-empty.")

        if type(self.working_tree_dirty) is not bool:
            raise TypeError("C6-5C working_tree_dirty must be bool.")

        if self.working_tree_dirty:
            raise ValueError("C6-5C canonical build requires a clean repository.")


@dataclass(frozen=True)
class DockerImageIdentity:
    """Observed exact DeepStream Docker image identity."""

    image_id: str
    repo_digests: tuple[str, ...]

    # ADD 2026-09-05: Docker image ID와 required repo digest를 frozen runtime과 대조한다.
    def validate(self, config: DeepStreamTensorRtConfig) -> None:
        if self.image_id != config.runtime.image_id:
            raise ValueError("C6-5C DeepStream image ID changed.")

        if config.runtime.repo_digest not in self.repo_digests:
            raise ValueError("C6-5C DeepStream repo digest is missing.")


@dataclass(frozen=True)
class HostGpuIdentity:
    """Observed host NVIDIA GPU identity."""

    gpu_name: str
    driver_version: str
    compute_capability: str

    # ADD 2026-09-05: Host L4/driver/compute-capability identity를 exact config와 대조한다.
    def validate(self, config: DeepStreamTensorRtConfig) -> None:
        if (
            self.gpu_name != config.runtime.gpu_name
            or self.driver_version != config.runtime.driver_version
            or self.compute_capability != config.runtime.gpu_compute_capability
        ):
            raise ValueError("C6-5C host GPU runtime identity changed.")


# ADD 2026-09-05: Host subprocess 실행을 text/captured-output contract로 통일한다.
def _run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(command),
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )


# ADD 2026-09-05: Canonical build 직전 현재 Git commit과 dirty state를 확인한다.
def resolve_repository_identity(repo: Path) -> RepositoryIdentity:
    commit_result = _run_command(
        ("git", "rev-parse", "HEAD"),
        cwd=repo,
    )

    if commit_result.returncode != 0:
        raise RuntimeError("C6-5C failed to resolve repository commit.")

    status_result = _run_command(
        ("git", "status", "--porcelain"),
        cwd=repo,
    )

    if status_result.returncode != 0:
        raise RuntimeError("C6-5C failed to inspect repository status.")

    identity = RepositoryIdentity(
        git_commit=commit_result.stdout.strip(),
        working_tree_dirty=bool(status_result.stdout.strip()),
    )
    identity.validate()

    return identity


# ADD 2026-09-05: Exact DeepStream image ID와 repo digest를 local Docker에서 확인한다.
def inspect_deepstream_image(
    config: DeepStreamTensorRtConfig,
) -> DockerImageIdentity:
    image_result = _run_command(
        (
            "sudo",
            "docker",
            "image",
            "inspect",
            config.runtime.image_tag,
            "--format",
            "{{.Id}}",
        )
    )

    if image_result.returncode != 0:
        raise RuntimeError("C6-5C failed to inspect DeepStream image ID.")

    digests_result = _run_command(
        (
            "sudo",
            "docker",
            "image",
            "inspect",
            config.runtime.image_tag,
            "--format",
            "{{json .RepoDigests}}",
        )
    )

    if digests_result.returncode != 0:
        raise RuntimeError("C6-5C failed to inspect DeepStream repo digests.")

    raw_digests: object = json.loads(digests_result.stdout.strip())

    if not isinstance(raw_digests, list) or not all(
        isinstance(value, str) for value in raw_digests
    ):
        raise ValueError("C6-5C Docker RepoDigests must be string array.")

    identity = DockerImageIdentity(
        image_id=image_result.stdout.strip(),
        repo_digests=tuple(cast(str, value) for value in raw_digests),
    )
    identity.validate(config)

    return identity


# ADD 2026-09-05: Host nvidia-smi에서 L4/driver/compute capability를 exact 확인한다.
def inspect_host_gpu(
    config: DeepStreamTensorRtConfig,
) -> HostGpuIdentity:
    result = _run_command(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        )
    )

    if result.returncode != 0:
        raise RuntimeError("C6-5C failed to inspect host GPU.")

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    if len(lines) != 1:
        raise ValueError("C6-5C requires exactly one visible host GPU.")

    parts = [part.strip() for part in lines[0].split(",")]

    if len(parts) != 3:
        raise ValueError("C6-5C nvidia-smi GPU identity format changed.")

    identity = HostGpuIdentity(
        gpu_name=parts[0],
        driver_version=parts[1],
        compute_capability=parts[2],
    )
    identity.validate(config)

    return identity


# ADD 2026-09-05: Canonical build 전에 labeled C6-5C container가 없는지 확인한다.
def assert_no_build_containers() -> None:
    result = _run_command(
        (
            "sudo",
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label={CONTAINER_LABEL}",
        )
    )

    if result.returncode != 0:
        raise RuntimeError("C6-5C failed to inspect labeled containers.")

    if result.stdout.strip():
        raise RuntimeError("C6-5C labeled build container already exists.")


# ADD 2026-09-05: 실패/timeout 이후 labeled C6-5C container를 즉시 제거한다.
def cleanup_build_containers() -> None:
    result = _run_command(
        (
            "sudo",
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label={CONTAINER_LABEL}",
        )
    )

    if result.returncode != 0:
        raise RuntimeError("C6-5C failed to inspect containers during cleanup.")

    container_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    if not container_ids:
        return

    removal = _run_command(
        (
            "sudo",
            "docker",
            "rm",
            "-f",
            *container_ids,
        )
    )

    if removal.returncode != 0:
        raise RuntimeError("C6-5C failed to remove labeled build containers.")


# ADD 2026-09-05: Exact source ZIP에서 Q/DQ ONNX만 temporary input으로 복원한다.
def restore_qdq_source(
    *,
    source_archive: Path,
    input_dir: Path,
    config: DeepStreamTensorRtConfig,
) -> Path:
    validate_source_archive(
        source_archive,
        config,
    )

    input_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    filename = "model.int8.qdq.onnx"
    identity = config.source.files[filename]

    with zipfile.ZipFile(
        source_archive,
        "r",
    ) as archive:
        data = archive.read(filename)

    target = input_dir / filename

    with target.open("xb") as handle:
        handle.write(data)

    if sha256_file(target) != identity.sha256 or target.stat().st_size != identity.size_bytes:
        raise RuntimeError("C6-5C restored Q/DQ ONNX identity changed.")

    return target


# ADD 2026-09-05: Frozen contract를 dependency-free TensorRT container builder source로 변환한다.
def build_container_source(
    config: DeepStreamTensorRtConfig,
) -> str:
    expected_io = {
        name: {
            "mode": contract["mode"],
            "dtype": contract["dtype"],
            "shape": list(
                cast(
                    tuple[int, ...],
                    contract["shape"],
                )
            ),
        }
        for name, contract in expected_engine_io().items()
    }

    qdq = config.source.files["model.int8.qdq.onnx"]

    template = """from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tensorrt as trt

EXPECTED_TRT_VERSION = __TRT_VERSION__
EXPECTED_ONNX_SHA256 = __ONNX_SHA__
EXPECTED_ONNX_BYTES = __ONNX_BYTES__
EXPECTED_WORKSPACE_BYTES = __WORKSPACE_BYTES__
EXPECTED_IO = json.loads(__EXPECTED_IO__)
OUTPUT_FILENAME = __OUTPUT_FILENAME__
RESULT_PREFIX = __RESULT_PREFIX__


# ADD 2026-09-05: TensorRT enum을 stable name string으로 변환한다.
def enum_name(value: object) -> str:
    name = getattr(value, "name", None)

    if isinstance(name, str):
        return name

    text = str(value)

    if "." in text:
        return text.rsplit(".", 1)[-1]

    return text


# ADD 2026-09-05: TensorRT engine I/O tensor를 strict JSON record로 만든다.
def tensor_record(
    engine: trt.ICudaEngine,
    index: int,
) -> dict[str, object]:
    name = engine.get_tensor_name(index)

    return {
        "name": name,
        "mode": enum_name(
            engine.get_tensor_mode(name)
        ),
        "dtype": enum_name(
            engine.get_tensor_dtype(name)
        ),
        "shape": list(
            engine.get_tensor_shape(name)
        ),
    }


# ADD 2026-09-05: Exact Q/DQ ONNX를 strongly-typed raw TensorRT plan으로 build한다.
def main() -> None:
    if trt.__version__ != EXPECTED_TRT_VERSION:
        raise RuntimeError(
            "C6-5C TensorRT version changed."
        )

    onnx_path = Path(
        "/input/model.int8.qdq.onnx"
    )
    output_path = Path(
        "/output"
    ) / OUTPUT_FILENAME

    onnx_bytes = onnx_path.read_bytes()
    onnx_sha = hashlib.sha256(
        onnx_bytes
    ).hexdigest()

    if (
        onnx_sha != EXPECTED_ONNX_SHA256
        or len(onnx_bytes) != EXPECTED_ONNX_BYTES
    ):
        raise RuntimeError(
            "C6-5C Q/DQ ONNX identity changed."
        )

    logger = trt.Logger(
        trt.Logger.WARNING
    )
    builder = trt.Builder(logger)

    flags = (
        1
        << int(
            trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED
        )
    )

    network = builder.create_network(
        flags
    )

    parser = trt.OnnxParser(
        network,
        logger,
    )

    parsed = parser.parse(
        onnx_bytes
    )

    parser_errors = [
        str(
            parser.get_error(index)
        )
        for index in range(
            parser.num_errors
        )
    ]

    if not parsed:
        raise RuntimeError(
            "C6-5C ONNX parser failed: "
            + " | ".join(parser_errors)
        )

    if parser_errors:
        raise RuntimeError(
            "C6-5C ONNX parser reported errors."
        )

    build_config = (
        builder.create_builder_config()
    )

    build_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        EXPECTED_WORKSPACE_BYTES,
    )

    serialized = (
        builder.build_serialized_network(
            network,
            build_config,
        )
    )

    if serialized is None:
        raise RuntimeError(
            "C6-5C TensorRT plan build failed."
        )

    plan_bytes = bytes(serialized)
    plan_sha = hashlib.sha256(
        plan_bytes
    ).hexdigest()

    runtime = trt.Runtime(logger)

    engine = (
        runtime.deserialize_cuda_engine(
            serialized
        )
    )

    if engine is None:
        raise RuntimeError(
            "C6-5C TensorRT plan deserialize failed."
        )

    records = [
        tensor_record(
            engine,
            index,
        )
        for index in range(
            engine.num_io_tensors
        )
    ]

    observed = {
        str(record["name"]): {
            "mode": record["mode"],
            "dtype": record["dtype"],
            "shape": record["shape"],
        }
        for record in records
    }

    if observed != EXPECTED_IO:
        raise RuntimeError(
            "C6-5C TensorRT engine I/O contract changed."
        )

    with output_path.open("xb") as handle:
        handle.write(plan_bytes)

    payload = {
        "status": "passed",
        "tensorrt_version": trt.__version__,
        "source_qdq_onnx_sha256": onnx_sha,
        "source_qdq_onnx_bytes": len(onnx_bytes),
        "parser_success": True,
        "parser_error_count": len(parser_errors),
        "strongly_typed": True,
        "workspace_bytes": EXPECTED_WORKSPACE_BYTES,
        "build_succeeded": True,
        "deserialize_succeeded": True,
        "plan_filename": output_path.name,
        "plan_sha256": plan_sha,
        "plan_bytes": len(plan_bytes),
        "io_tensors": records,
        "application_inference_executed": False,
        "dataset_used": False,
        "validation_used": False,
        "test_used": False,
        "final_test_used": False,
    }

    print(
        RESULT_PREFIX
        + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
"""

    return (
        template.replace(
            "__TRT_VERSION__",
            repr(config.runtime.tensorrt_python_version),
        )
        .replace(
            "__ONNX_SHA__",
            repr(qdq.sha256),
        )
        .replace(
            "__ONNX_BYTES__",
            str(qdq.size_bytes),
        )
        .replace(
            "__WORKSPACE_BYTES__",
            str(config.build.workspace_bytes),
        )
        .replace(
            "__EXPECTED_IO__",
            repr(
                json.dumps(
                    expected_io,
                    sort_keys=True,
                )
            ),
        )
        .replace(
            "__OUTPUT_FILENAME__",
            repr(config.build.plan_filename),
        )
        .replace(
            "__RESULT_PREFIX__",
            repr(CONTAINER_RESULT_PREFIX),
        )
    )


# ADD 2026-09-05: Exact digest/no-network/GPU volume boundary의 Docker command를 만든다.
def build_docker_command(
    config: DeepStreamTensorRtConfig,
    *,
    input_dir: Path,
    output_dir: Path,
) -> tuple[str, ...]:
    return (
        "sudo",
        "docker",
        "run",
        "--rm",
        "--interactive",
        "--runtime=nvidia",
        "--network",
        "none",
        "--gpus",
        "all",
        "--label",
        CONTAINER_LABEL,
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "-v",
        f"{input_dir.resolve()}:/input:ro",
        "-v",
        f"{output_dir.resolve()}:/output:rw",
        "--entrypoint",
        "python3",
        config.runtime.repo_digest,
        "-",
    )


# ADD 2026-09-05: Generated raw plan의 hash/bytes가 container payload와 동일한지 검증한다.
def validate_plan_file(
    plan_path: Path,
    payload: dict[str, Any],
) -> None:
    if not plan_path.is_file():
        raise FileNotFoundError(f"C6-5C generated plan not found: {plan_path}")

    plan_sha = payload["plan_sha256"]
    plan_bytes = payload["plan_bytes"]

    if not isinstance(plan_sha, str):
        raise TypeError("C6-5C payload plan_sha256 must be str.")

    if type(plan_bytes) is not int:
        raise TypeError("C6-5C payload plan_bytes must be int.")

    if sha256_file(plan_path) != plan_sha or plan_path.stat().st_size != plan_bytes:
        raise ValueError("C6-5C generated plan identity does not match payload.")


# ADD 2026-09-05: Strict sorted JSON bytes를 canonical metadata/evidence에 사용한다.
def _json_bytes(
    value: dict[str, Any],
) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


# ADD 2026-09-05: Canonical plan provenance metadata를 runtime/build diagnostics와 결합한다.
def build_metadata(
    *,
    config: DeepStreamTensorRtConfig,
    repository: RepositoryIdentity,
    image: DockerImageIdentity,
    gpu: HostGpuIdentity,
    source_members: dict[str, dict[str, object]],
    payload: dict[str, Any],
    container_stdout: str,
    container_stderr: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "deepstream_l4_tensorrt_int8_plan",
        "stage": "C6-5C",
        "state": BUILD_STATE,
        "created_at": datetime.now(UTC).isoformat(),
        "repository": {
            "git_commit": repository.git_commit,
            "working_tree_dirty": repository.working_tree_dirty,
        },
        "config_sha256": sha256_file(config.config_path),
        "source": {
            "evidence_zip_sha256": config.source.evidence_zip_sha256,
            "evidence_zip_size_bytes": config.source.evidence_zip_size_bytes,
            "qdq_run_commit": config.source.qdq_run_commit,
            "members": source_members,
        },
        "runtime": {
            "image_tag": config.runtime.image_tag,
            "image_id": image.image_id,
            "repo_digests": list(image.repo_digests),
            "gpu_name": gpu.gpu_name,
            "driver_version": gpu.driver_version,
            "gpu_compute_capability": gpu.compute_capability,
            "deepstream_version": config.runtime.deepstream_version,
            "tensorrt_python_version": config.runtime.tensorrt_python_version,
        },
        "build": payload,
        "artifact": {
            "output_root": str(config.build.output_root),
            "plan_filename": config.build.plan_filename,
            "metadata_filename": BUILD_METADATA_FILENAME,
            "plan_sha256": payload["plan_sha256"],
            "plan_bytes": payload["plan_bytes"],
        },
        "diagnostics": {
            "container_return_code": 0,
            "container_stdout_sha256": sha256_bytes(container_stdout.encode("utf-8")),
            "container_stderr_sha256": sha256_bytes(container_stderr.encode("utf-8")),
            "tensorrt_stderr_recorded": True,
            "error_line_is_automatic_failure": (config.diagnostics.error_line_is_automatic_failure),
        },
        "scope": {
            "network_allowed": config.policy.network_allowed,
            "c5_engine_rebuild_allowed": (config.policy.c5_engine_rebuild_allowed),
            "application_inference_executed": False,
            "dataset_used": False,
            "validation_used": False,
            "test_used": False,
            "final_test_used": False,
            "segmentation_decode_executed": False,
            "overlay_executed": False,
        },
    }


# ADD 2026-09-05: Clean foundation에서 plan+metadata+external evidence를 fail-closed 생성한다.
def write_deepstream_tensorrt_build_evidence(
    *,
    source_archive: Path,
    evidence_output: Path,
    repo: Path,
    config_path: Path = DEFAULT_DEEPSTREAM_TENSORRT_CONFIG,
) -> Path:
    repo = repo.resolve()
    source_archive = source_archive.resolve()
    evidence_output = evidence_output.resolve()

    config = load_deepstream_tensorrt_config(config_path)

    if evidence_output.is_relative_to(repo):
        raise ValueError("C6-5C canonical evidence must be outside repository.")

    if evidence_output.exists():
        raise FileExistsError(f"C6-5C evidence already exists: {evidence_output}")

    final_output_dir = (repo / config.build.output_root).resolve()

    if not final_output_dir.is_relative_to(repo):
        raise ValueError("C6-5C artifact output must remain inside repository artifact namespace.")

    if final_output_dir.exists():
        raise FileExistsError(f"C6-5C artifact output already exists: {final_output_dir}")

    repository = resolve_repository_identity(repo)

    source_members = validate_source_archive(
        source_archive,
        config,
    )

    image = inspect_deepstream_image(config)
    gpu = inspect_host_gpu(config)

    assert_no_build_containers()

    published_output = False

    try:
        with tempfile.TemporaryDirectory(prefix="c6-5c-build-") as temporary_root:
            temporary = Path(temporary_root)

            input_dir = temporary / "input"
            output_dir = temporary / "output"

            restore_qdq_source(
                source_archive=source_archive,
                input_dir=input_dir,
                config=config,
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=False,
            )

            source = build_container_source(config)

            command = build_docker_command(
                config,
                input_dir=input_dir,
                output_dir=output_dir,
            )

            completed = _run_command(
                command,
                input_text=source,
                timeout_seconds=BUILD_TIMEOUT_SECONDS,
            )

            if completed.returncode != 0:
                tail = completed.stderr[-4000:]

                raise RuntimeError(
                    "C6-5C TensorRT build container failed "
                    f"with exit {completed.returncode}: {tail}"
                )

            payload = parse_container_build_payload(
                completed.stdout,
                prefix=CONTAINER_RESULT_PREFIX,
            )

            validate_container_build_payload(
                payload,
                config,
            )

            temporary_plan = output_dir / config.build.plan_filename

            validate_plan_file(
                temporary_plan,
                payload,
            )

            metadata = build_metadata(
                config=config,
                repository=repository,
                image=image,
                gpu=gpu,
                source_members=source_members,
                payload=payload,
                container_stdout=completed.stdout,
                container_stderr=completed.stderr,
            )

            metadata_bytes = _json_bytes(metadata)

            metadata_sha = sha256_bytes(metadata_bytes)

            evidence = {
                **metadata,
                "evidence_type": ("c6_5c_deepstream_l4_tensorrt_int8_build"),
                "artifact_metadata_sha256": metadata_sha,
                "container_stdout": completed.stdout,
                "container_stderr": completed.stderr,
            }

            evidence_bytes = _json_bytes(evidence)

            final_output_dir.mkdir(
                parents=True,
                exist_ok=False,
            )
            published_output = True

            final_plan = final_output_dir / config.build.plan_filename

            with (
                temporary_plan.open("rb") as source_handle,
                final_plan.open("xb") as target_handle,
            ):
                shutil.copyfileobj(
                    source_handle,
                    target_handle,
                )

            if sha256_file(final_plan) != payload["plan_sha256"]:
                raise RuntimeError("C6-5C published plan SHA-256 changed.")

            metadata_path = final_output_dir / BUILD_METADATA_FILENAME

            with metadata_path.open("xb") as handle:
                handle.write(metadata_bytes)

            evidence_output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with evidence_output.open("xb") as handle:
                handle.write(evidence_bytes)

            return evidence_output

    except Exception:
        if published_output:
            shutil.rmtree(
                final_output_dir,
                ignore_errors=True,
            )

        if evidence_output.exists():
            evidence_output.unlink()

        raise

    finally:
        cleanup_build_containers()
