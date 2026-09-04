"""C6-5B DeepStream NVDEC/NVMM smoke contracts and canonical runner."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from shared.hashing import is_sha256_digest, sha256_bytes, sha256_file

DEFAULT_DEEPSTREAM_NVMM_CONFIG = Path("configs/streaming/yolo_deepstream_nvmm_smoke.json")

EXPECTED_SMOKE_ID = "c6_5b_deepstream_nvdec_nvmm_v1"
EXPECTED_C6_5A_CLOSURE_COMMIT = "28b5eab7f23d08bd8c50caa09b2a74f4fabc9236"

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
EXPECTED_GSTREAMER_VERSION = "1.24.2"
EXPECTED_CONTAINER_TOOLKIT_VERSION = "1.20.0"

EXPECTED_NVDEC_PACKAGE = "libnvidia-decode-595"
EXPECTED_NVDEC_PACKAGE_VERSION = "595.84-0ubuntu0.24.04.1"
EXPECTED_REQUIRED_LIBRARY = "libnvcuvid.so.1"
EXPECTED_CDI_SPEC = Path("/var/run/cdi/nvidia.yaml")
EXPECTED_CDI_SPEC_SHA256 = "7e803d06c447d2322fa5a17c22a5149173d596f3eb3b68b3238850dc74704739"
EXPECTED_CDI_ENTRY = "libnvcuvid.so.595.84"

EXPECTED_SAMPLE_PATH = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264"
EXPECTED_SAMPLE_SHA256 = "5f29353a6ec4727bd49fb523efc207d643e6638f4e5c56f060e1b61291aa6ea2"
EXPECTED_SAMPLE_BYTES = 14_759_548

EXPECTED_DECODER_VERSION = "1.14.0"
EXPECTED_DECODER_FILENAME = (
    "/usr/lib/x86_64-linux-gnu/gstreamer-1.0/deepstream/libgstnvvideo4linux2.so"
)
EXPECTED_CONVERTER_VERSION = "1.2.3"
EXPECTED_CONVERTER_FILENAME = (
    "/usr/lib/x86_64-linux-gnu/gstreamer-1.0/deepstream/libgstnvvideoconvert.so"
)

CONTAINER_RESULT_PREFIX = "C6_5B_RESULT_JSON="
CONTAINER_LABEL = "c6_5b_smoke=1"
SMOKE_STATE = "DEEPSTREAM_NVDEC_NVMM_SMOKE_COMPLETED"


@dataclass(frozen=True)
class DeepStreamNvmmRuntime:
    """Exact C6-5B DeepStream/L4 runtime identity."""

    image_tag: str
    image_id: str
    repo_digest: str
    gpu_name: str
    gpu_compute_capability: str
    driver_version: str
    deepstream_version: str
    gstreamer_version: str
    container_toolkit_version: str

    # ADD 2026-09-05: C6-5B runtime을 exact L4/DeepStream image identity에 고정한다.
    def validate(self) -> None:
        expected = {
            "image_tag": EXPECTED_IMAGE_TAG,
            "image_id": EXPECTED_IMAGE_ID,
            "repo_digest": EXPECTED_REPO_DIGEST,
            "gpu_name": EXPECTED_GPU_NAME,
            "gpu_compute_capability": EXPECTED_GPU_COMPUTE_CAPABILITY,
            "driver_version": EXPECTED_DRIVER_VERSION,
            "deepstream_version": EXPECTED_DEEPSTREAM_VERSION,
            "gstreamer_version": EXPECTED_GSTREAMER_VERSION,
            "container_toolkit_version": EXPECTED_CONTAINER_TOOLKIT_VERSION,
        }

        mismatches = [name for name, value in expected.items() if getattr(self, name) != value]
        if mismatches:
            raise ValueError("C6-5B runtime identity changed: " + ", ".join(mismatches))


@dataclass(frozen=True)
class NvdecHostDependency:
    """Host dependency required to inject NVDEC into the container."""

    package: str
    package_version: str
    required_library: str
    cdi_spec_path: Path
    cdi_spec_sha256: str
    cdi_required_entry: str
    cdi_refresh_required: bool

    # ADD 2026-09-05: NVDEC host package와 canonical CDI observation을 검증한다.
    def validate(self) -> None:
        if (
            self.package != EXPECTED_NVDEC_PACKAGE
            or self.package_version != EXPECTED_NVDEC_PACKAGE_VERSION
            or self.required_library != EXPECTED_REQUIRED_LIBRARY
            or self.cdi_spec_path != EXPECTED_CDI_SPEC
            or self.cdi_spec_sha256 != EXPECTED_CDI_SPEC_SHA256
            or self.cdi_required_entry != EXPECTED_CDI_ENTRY
            or self.cdi_refresh_required is not True
        ):
            raise ValueError("C6-5B NVDEC host dependency identity changed.")

        if not is_sha256_digest(self.cdi_spec_sha256):
            raise ValueError("C6-5B CDI SHA-256 is invalid.")


@dataclass(frozen=True)
class NvmmSampleIdentity:
    """Exact DeepStream sample used by the C6-5B smoke."""

    path: str
    sha256: str
    size_bytes: int
    codec: str

    # ADD 2026-09-05: C6-5B H264 fixture를 exact sample hash와 bytes에 고정한다.
    def validate(self) -> None:
        if type(self.size_bytes) is not int:
            raise TypeError("C6-5B sample size_bytes must be int.")

        if (
            self.path != EXPECTED_SAMPLE_PATH
            or self.sha256 != EXPECTED_SAMPLE_SHA256
            or self.size_bytes != EXPECTED_SAMPLE_BYTES
            or self.codec != "h264"
        ):
            raise ValueError("C6-5B sample identity changed.")

        if not is_sha256_digest(self.sha256):
            raise ValueError("C6-5B sample SHA-256 is invalid.")


@dataclass(frozen=True)
class DeepStreamPluginIdentity:
    """Exact DeepStream GStreamer plugin identity."""

    version: str
    filename: str

    # ADD 2026-09-05: Plugin version과 shared-object path가 비어 있지 않은지 검증한다.
    def validate(self) -> None:
        if not self.version or not self.filename:
            raise ValueError("C6-5B plugin identity must be non-empty.")


@dataclass(frozen=True)
class NvmmPipelineContract:
    """Frozen C6-5B NVDEC to NVMM conversion path."""

    decoder: str
    decoder_output_memory: str
    decoder_output_format: str
    converter: str
    converter_output_memory: str
    converter_output_format: str
    sink: str
    sync: bool
    async_: bool
    timeout_seconds: int

    # ADD 2026-09-05: C6-5B pipeline을 NVDEC→NVMM NV12→RGBA path에 고정한다.
    def validate(self) -> None:
        if type(self.sync) is not bool or type(self.async_) is not bool:
            raise TypeError("C6-5B sink flags must be strict booleans.")
        if type(self.timeout_seconds) is not int or self.timeout_seconds <= 0:
            raise ValueError("C6-5B timeout_seconds must be positive.")

        if (
            self.decoder != "nvv4l2decoder"
            or self.decoder_output_memory != "NVMM"
            or self.decoder_output_format != "NV12"
            or self.converter != "nvvideoconvert"
            or self.converter_output_memory != "NVMM"
            or self.converter_output_format != "RGBA"
            or self.sink != "fakesink"
            or self.sync is not False
            or self.async_ is not False
            or self.timeout_seconds != 60
        ):
            raise ValueError("C6-5B NVMM pipeline contract changed.")


@dataclass(frozen=True)
class NvmmSmokePolicy:
    """Restrictions for the C6-5B no-inference GPU-path smoke."""

    require_clean_repository: bool
    network_allowed: bool
    inference_allowed: bool
    tensorrt_engine_used: bool
    dataset_used: bool
    validation_used: bool
    test_used: bool
    final_test_used: bool

    # ADD 2026-09-05: C6-5B를 no-network/no-inference/no-data smoke로 제한한다.
    def validate(self) -> None:
        values = (
            self.require_clean_repository,
            self.network_allowed,
            self.inference_allowed,
            self.tensorrt_engine_used,
            self.dataset_used,
            self.validation_used,
            self.test_used,
            self.final_test_used,
        )
        if any(type(value) is not bool for value in values):
            raise TypeError("C6-5B policy values must be strict booleans.")

        if (
            self.require_clean_repository is not True
            or self.network_allowed is not False
            or self.inference_allowed is not False
            or self.tensorrt_engine_used is not False
            or self.dataset_used is not False
            or self.validation_used is not False
            or self.test_used is not False
            or self.final_test_used is not False
        ):
            raise ValueError("C6-5B smoke policy changed.")


@dataclass(frozen=True)
class DeepStreamNvmmSmokeConfig:
    """Top-level frozen C6-5B smoke configuration."""

    schema_version: int
    smoke_id: str
    c6_5a_closure_commit: str
    runtime: DeepStreamNvmmRuntime
    host_dependency: NvdecHostDependency
    sample: NvmmSampleIdentity
    decoder_plugin: DeepStreamPluginIdentity
    converter_plugin: DeepStreamPluginIdentity
    pipeline: NvmmPipelineContract
    policy: NvmmSmokePolicy
    config_path: Path

    # ADD 2026-09-05: C6-5B config 검증 → MODIFY 2026-09-05: plugin identity까지 exact 고정한다.
    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported C6-5B config schema.")
        if self.smoke_id != EXPECTED_SMOKE_ID:
            raise ValueError("Unexpected C6-5B smoke_id.")
        if self.c6_5a_closure_commit != EXPECTED_C6_5A_CLOSURE_COMMIT:
            raise ValueError("C6-5B baseline commit changed.")

        self.runtime.validate()
        self.host_dependency.validate()
        self.sample.validate()
        self.decoder_plugin.validate()
        self.converter_plugin.validate()

        if (
            self.decoder_plugin.version != EXPECTED_DECODER_VERSION
            or self.decoder_plugin.filename != EXPECTED_DECODER_FILENAME
        ):
            raise ValueError("C6-5B decoder plugin identity changed.")

        if (
            self.converter_plugin.version != EXPECTED_CONVERTER_VERSION
            or self.converter_plugin.filename != EXPECTED_CONVERTER_FILENAME
        ):
            raise ValueError("C6-5B converter plugin identity changed.")

        self.pipeline.validate()
        self.policy.validate()


# ADD 2026-09-05: JSON object를 strict mapping으로 변환한다.
def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return cast(dict[str, Any], value)


# ADD 2026-09-05: JSON mapping이 허용된 field만 갖는지 검증한다.
def _require_fields(
    mapping: dict[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(mapping) != expected:
        raise ValueError(f"{label} fields do not match schema.")


# ADD 2026-09-05: JSON string을 strict str로 검증한다.
def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str.")
    return value


# ADD 2026-09-05: JSON bool을 strict bool로 검증한다.
def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be bool.")
    return cast(bool, value)


# ADD 2026-09-05: JSON integer를 strict int로 검증한다.
def _integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be int.")
    return cast(int, value)


# ADD 2026-09-05: Frozen C6-5B JSON config를 typed contract로 로드한다.
def load_deepstream_nvmm_smoke_config(
    path: Path = DEFAULT_DEEPSTREAM_NVMM_CONFIG,
) -> DeepStreamNvmmSmokeConfig:
    raw_object: object = json.loads(path.read_text(encoding="utf-8"))
    raw = _mapping(raw_object, label="C6-5B config")

    _require_fields(
        raw,
        {
            "schema_version",
            "smoke_id",
            "c6_5a_closure_commit",
            "runtime",
            "host_dependency",
            "sample",
            "plugins",
            "pipeline",
            "policy",
        },
        label="C6-5B config",
    )

    runtime = _mapping(raw["runtime"], label="runtime")
    host = _mapping(raw["host_dependency"], label="host_dependency")
    sample = _mapping(raw["sample"], label="sample")
    plugins = _mapping(raw["plugins"], label="plugins")
    pipeline = _mapping(raw["pipeline"], label="pipeline")
    policy = _mapping(raw["policy"], label="policy")

    _require_fields(
        runtime,
        {
            "image_tag",
            "image_id",
            "repo_digest",
            "gpu_name",
            "gpu_compute_capability",
            "driver_version",
            "deepstream_version",
            "gstreamer_version",
            "container_toolkit_version",
        },
        label="runtime",
    )
    _require_fields(
        host,
        {
            "package",
            "package_version",
            "required_library",
            "cdi_spec_path",
            "cdi_spec_sha256",
            "cdi_required_entry",
            "cdi_refresh_required",
        },
        label="host_dependency",
    )
    _require_fields(
        sample,
        {"path", "sha256", "size_bytes", "codec"},
        label="sample",
    )
    _require_fields(
        plugins,
        {"nvv4l2decoder", "nvvideoconvert"},
        label="plugins",
    )
    _require_fields(
        pipeline,
        {
            "decoder",
            "decoder_output_memory",
            "decoder_output_format",
            "converter",
            "converter_output_memory",
            "converter_output_format",
            "sink",
            "sync",
            "async",
            "timeout_seconds",
        },
        label="pipeline",
    )
    _require_fields(
        policy,
        {
            "require_clean_repository",
            "network_allowed",
            "inference_allowed",
            "tensorrt_engine_used",
            "dataset_used",
            "validation_used",
            "test_used",
            "final_test_used",
        },
        label="policy",
    )

    decoder = _mapping(plugins["nvv4l2decoder"], label="plugins.nvv4l2decoder")
    converter = _mapping(
        plugins["nvvideoconvert"],
        label="plugins.nvvideoconvert",
    )
    _require_fields(decoder, {"version", "filename"}, label="nvv4l2decoder")
    _require_fields(converter, {"version", "filename"}, label="nvvideoconvert")

    config = DeepStreamNvmmSmokeConfig(
        schema_version=_integer(raw["schema_version"], label="schema_version"),
        smoke_id=_string(raw["smoke_id"], label="smoke_id"),
        c6_5a_closure_commit=_string(
            raw["c6_5a_closure_commit"],
            label="c6_5a_closure_commit",
        ),
        runtime=DeepStreamNvmmRuntime(
            image_tag=_string(runtime["image_tag"], label="runtime.image_tag"),
            image_id=_string(runtime["image_id"], label="runtime.image_id"),
            repo_digest=_string(
                runtime["repo_digest"],
                label="runtime.repo_digest",
            ),
            gpu_name=_string(runtime["gpu_name"], label="runtime.gpu_name"),
            gpu_compute_capability=_string(
                runtime["gpu_compute_capability"],
                label="runtime.gpu_compute_capability",
            ),
            driver_version=_string(
                runtime["driver_version"],
                label="runtime.driver_version",
            ),
            deepstream_version=_string(
                runtime["deepstream_version"],
                label="runtime.deepstream_version",
            ),
            gstreamer_version=_string(
                runtime["gstreamer_version"],
                label="runtime.gstreamer_version",
            ),
            container_toolkit_version=_string(
                runtime["container_toolkit_version"],
                label="runtime.container_toolkit_version",
            ),
        ),
        host_dependency=NvdecHostDependency(
            package=_string(host["package"], label="host_dependency.package"),
            package_version=_string(
                host["package_version"],
                label="host_dependency.package_version",
            ),
            required_library=_string(
                host["required_library"],
                label="host_dependency.required_library",
            ),
            cdi_spec_path=Path(
                _string(
                    host["cdi_spec_path"],
                    label="host_dependency.cdi_spec_path",
                )
            ),
            cdi_spec_sha256=_string(
                host["cdi_spec_sha256"],
                label="host_dependency.cdi_spec_sha256",
            ),
            cdi_required_entry=_string(
                host["cdi_required_entry"],
                label="host_dependency.cdi_required_entry",
            ),
            cdi_refresh_required=_boolean(
                host["cdi_refresh_required"],
                label="host_dependency.cdi_refresh_required",
            ),
        ),
        sample=NvmmSampleIdentity(
            path=_string(sample["path"], label="sample.path"),
            sha256=_string(sample["sha256"], label="sample.sha256"),
            size_bytes=_integer(sample["size_bytes"], label="sample.size_bytes"),
            codec=_string(sample["codec"], label="sample.codec"),
        ),
        decoder_plugin=DeepStreamPluginIdentity(
            version=_string(decoder["version"], label="decoder.version"),
            filename=_string(decoder["filename"], label="decoder.filename"),
        ),
        converter_plugin=DeepStreamPluginIdentity(
            version=_string(converter["version"], label="converter.version"),
            filename=_string(converter["filename"], label="converter.filename"),
        ),
        pipeline=NvmmPipelineContract(
            decoder=_string(pipeline["decoder"], label="pipeline.decoder"),
            decoder_output_memory=_string(
                pipeline["decoder_output_memory"],
                label="pipeline.decoder_output_memory",
            ),
            decoder_output_format=_string(
                pipeline["decoder_output_format"],
                label="pipeline.decoder_output_format",
            ),
            converter=_string(
                pipeline["converter"],
                label="pipeline.converter",
            ),
            converter_output_memory=_string(
                pipeline["converter_output_memory"],
                label="pipeline.converter_output_memory",
            ),
            converter_output_format=_string(
                pipeline["converter_output_format"],
                label="pipeline.converter_output_format",
            ),
            sink=_string(pipeline["sink"], label="pipeline.sink"),
            sync=_boolean(pipeline["sync"], label="pipeline.sync"),
            async_=_boolean(pipeline["async"], label="pipeline.async"),
            timeout_seconds=_integer(
                pipeline["timeout_seconds"],
                label="pipeline.timeout_seconds",
            ),
        ),
        policy=NvmmSmokePolicy(
            require_clean_repository=_boolean(
                policy["require_clean_repository"],
                label="policy.require_clean_repository",
            ),
            network_allowed=_boolean(
                policy["network_allowed"],
                label="policy.network_allowed",
            ),
            inference_allowed=_boolean(
                policy["inference_allowed"],
                label="policy.inference_allowed",
            ),
            tensorrt_engine_used=_boolean(
                policy["tensorrt_engine_used"],
                label="policy.tensorrt_engine_used",
            ),
            dataset_used=_boolean(
                policy["dataset_used"],
                label="policy.dataset_used",
            ),
            validation_used=_boolean(
                policy["validation_used"],
                label="policy.validation_used",
            ),
            test_used=_boolean(policy["test_used"], label="policy.test_used"),
            final_test_used=_boolean(
                policy["final_test_used"],
                label="policy.final_test_used",
            ),
        ),
        config_path=path.resolve(),
    )
    config.validate()
    return config


# ADD 2026-09-05: Canonical gst-launch NVDEC/NVMM command를 deterministic tuple로 만든다.
def expected_gstreamer_pipeline(
    config: DeepStreamNvmmSmokeConfig,
) -> tuple[str, ...]:
    config.validate()
    return (
        "gst-launch-1.0",
        "-e",
        "-v",
        "filesrc",
        f"location={config.sample.path}",
        "!",
        "h264parse",
        "!",
        config.pipeline.decoder,
        "!",
        "video/x-raw(memory:NVMM)",
        "!",
        config.pipeline.converter,
        "!",
        (f"video/x-raw(memory:NVMM),format={config.pipeline.converter_output_format}"),
        "!",
        config.pipeline.sink,
        f"sync={str(config.pipeline.sync).lower()}",
        f"async={str(config.pipeline.async_).lower()}",
    )


# ADD 2026-09-05: Host repository SHA와 clean-tree canonical boundary를 확인한다.
def resolve_repository_identity(repo: Path) -> dict[str, object]:
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=repo,
        text=True,
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ("git", "status", "--porcelain"),
            cwd=repo,
            text=True,
        ).strip()
    )

    if len(commit) != 40:
        raise RuntimeError("C6-5B repository commit must be a full SHA.")
    if dirty:
        raise RuntimeError("C6-5B canonical smoke requires a clean repository.")

    return {
        "git_commit": commit,
        "working_tree_dirty": False,
    }


# ADD 2026-09-05: Exact DeepStream image ID와 repo digest가 local Docker에 존재하는지 검증한다.
def inspect_deepstream_image(
    config: DeepStreamNvmmSmokeConfig,
) -> dict[str, object]:
    completed = subprocess.run(
        (
            "sudo",
            "docker",
            "image",
            "inspect",
            config.runtime.repo_digest,
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    raw: object = json.loads(completed.stdout)
    if not isinstance(raw, list) or len(raw) != 1:
        raise RuntimeError("C6-5B Docker inspect returned unexpected payload.")

    item = _mapping(raw[0], label="docker image inspect")
    image_id = _string(item.get("Id"), label="docker image Id")
    repo_digests = item.get("RepoDigests")
    if not isinstance(repo_digests, list):
        raise RuntimeError("C6-5B Docker RepoDigests is invalid.")

    if image_id != config.runtime.image_id:
        raise RuntimeError("C6-5B DeepStream image ID changed.")
    if config.runtime.repo_digest not in repo_digests:
        raise RuntimeError("C6-5B DeepStream repo digest is missing.")

    return {
        "image_id": image_id,
        "repo_digest": config.runtime.repo_digest,
    }


# ADD 2026-09-05: Host NVDEC package, L4 identity, CDI spec와 libnvcuvid를 fail-closed 검증한다.
def validate_host_dependencies(
    config: DeepStreamNvmmSmokeConfig,
) -> dict[str, object]:
    host = config.host_dependency

    package = subprocess.run(
        (
            "dpkg-query",
            "-W",
            "-f=${Status}|${Version}",
            host.package,
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    expected_package = f"install ok installed|{host.package_version}"
    if package != expected_package:
        raise RuntimeError("C6-5B NVDEC package identity changed.")

    ldconfig = subprocess.run(
        ("ldconfig", "-p"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if host.required_library not in ldconfig:
        raise RuntimeError("C6-5B host libnvcuvid is unavailable.")

    toolkit_output = subprocess.run(
        ("nvidia-ctk", "--version"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if not toolkit_output:
        raise RuntimeError("C6-5B nvidia-ctk returned no version.")

    toolkit_prefix = "NVIDIA Container Toolkit CLI version "
    if not toolkit_output[0].startswith(toolkit_prefix):
        raise RuntimeError("C6-5B nvidia-ctk version output changed.")
    toolkit_version = toolkit_output[0][len(toolkit_prefix) :].strip()
    if toolkit_version != config.runtime.container_toolkit_version:
        raise RuntimeError("C6-5B NVIDIA Container Toolkit version changed.")

    gpu_line = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    gpu_parts = [part.strip() for part in gpu_line.split(",")]
    expected_gpu = [
        config.runtime.gpu_name,
        config.runtime.driver_version,
        config.runtime.gpu_compute_capability,
    ]
    if gpu_parts != expected_gpu:
        raise RuntimeError("C6-5B host GPU identity changed.")

    cdi_sha = subprocess.run(
        ("sudo", "sha256sum", str(host.cdi_spec_path)),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()[0]
    if cdi_sha != host.cdi_spec_sha256:
        raise RuntimeError("C6-5B CDI spec SHA-256 changed.")

    cdi_entry = subprocess.run(
        (
            "sudo",
            "grep",
            "-F",
            host.cdi_required_entry,
            str(host.cdi_spec_path),
        ),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if cdi_entry.returncode != 0:
        raise RuntimeError("C6-5B CDI spec does not contain libnvcuvid.")

    return {
        "nvdec_package": host.package,
        "nvdec_package_version": host.package_version,
        "required_library": host.required_library,
        "cdi_spec_path": str(host.cdi_spec_path),
        "cdi_spec_sha256": cdi_sha,
        "cdi_required_entry": host.cdi_required_entry,
        "cdi_refresh_required": host.cdi_refresh_required,
        "container_toolkit_version": toolkit_version,
        "gpu_name": gpu_parts[0],
        "driver_version": gpu_parts[1],
        "gpu_compute_capability": gpu_parts[2],
    }


# ADD 2026-09-05: Container에서 실행할 dependency-free NVDEC/NVMM probe source를 생성한다.
def build_container_probe_source(
    config: DeepStreamNvmmSmokeConfig,
) -> str:
    expected = {
        "runtime": {
            "gpu_name": config.runtime.gpu_name,
            "gpu_compute_capability": config.runtime.gpu_compute_capability,
            "driver_version": config.runtime.driver_version,
            "deepstream_version": config.runtime.deepstream_version,
            "gstreamer_version": config.runtime.gstreamer_version,
        },
        "sample": {
            "path": config.sample.path,
            "sha256": config.sample.sha256,
            "size_bytes": config.sample.size_bytes,
        },
        "plugins": {
            "nvv4l2decoder": asdict(config.decoder_plugin),
            "nvvideoconvert": asdict(config.converter_plugin),
        },
        "pipeline_command": list(expected_gstreamer_pipeline(config)),
        "timeout_seconds": config.pipeline.timeout_seconds,
        "scope": {
            "inference_executed": False,
            "tensorrt_engine_used": False,
            "dataset_used": False,
            "validation_used": False,
            "test_used": False,
            "final_test_used": False,
        },
    }
    expected_json = json.dumps(expected, sort_keys=True)

    source = r"""
import ctypes
import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

EXPECTED = json.loads(__EXPECTED_JSON__)
PREFIX = "C6_5B_RESULT_JSON="


# ADD 2026-09-05: Container subprocess를 bounded capture mode로 실행한다.
# MODIFY 2026-09-05: Generated probe 함수에 explicit typed contract를 추가한다.
def run(
    command: Sequence[str],
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ADD 2026-09-05: gst-inspect text field를 strict label로 추출한다.
# MODIFY 2026-09-05: Generated parser 함수의 str input/output contract를 명시한다.
def inspect_field(
    text: str,
    label: str,
) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith(label):
            value = line[len(label):].strip()
            if value:
                return value
    raise RuntimeError(f"Missing gst-inspect field: {label}")


# ADD 2026-09-05: File SHA-256을 chunked read로 계산한다.
# MODIFY 2026-09-05: Generated hashing 함수의 Path→str contract를 명시한다.
def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# ADD 2026-09-05: C6-5B result를 single-line JSON marker로 출력한다.
# MODIFY 2026-09-05: Generated marker writer의 payload/return contract를 명시한다.
def emit(payload: dict[str, object]) -> None:
    print(PREFIX + json.dumps(payload, sort_keys=True))


# ADD 2026-09-05: Exact runtime/sample/plugins를 검증하고 NVDEC→NVMM pipeline을 실행한다.
# MODIFY 2026-09-05: Generated entrypoint의 explicit return contract를 명시한다.
def main() -> None:
    try:
        gpu = run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader",
            ]
        )
        if gpu.returncode != 0:
            raise RuntimeError("nvidia-smi failed.")

        gpu_parts = [part.strip() for part in gpu.stdout.strip().split(",")]
        if gpu_parts != [
            EXPECTED["runtime"]["gpu_name"],
            EXPECTED["runtime"]["driver_version"],
            EXPECTED["runtime"]["gpu_compute_capability"],
        ]:
            raise RuntimeError("Container GPU identity changed.")

        deepstream = run(["deepstream-app", "--version-all"])
        deepstream_text = deepstream.stdout + "\n" + deepstream.stderr
        expected_ds = (
            "deepstream-app version "
            + EXPECTED["runtime"]["deepstream_version"]
        )
        if expected_ds not in deepstream_text:
            raise RuntimeError("DeepStream version changed.")

        gst_version = run(["gst-launch-1.0", "--version"])
        expected_gst = (
            "gst-launch-1.0 version "
            + EXPECTED["runtime"]["gstreamer_version"]
        )
        if expected_gst not in gst_version.stdout:
            raise RuntimeError("GStreamer version changed.")

        sample = Path(EXPECTED["sample"]["path"])
        if not sample.is_file():
            raise RuntimeError("C6-5B sample is missing.")
        if sample.stat().st_size != EXPECTED["sample"]["size_bytes"]:
            raise RuntimeError("C6-5B sample byte size changed.")
        if hash_file(sample) != EXPECTED["sample"]["sha256"]:
            raise RuntimeError("C6-5B sample SHA-256 changed.")

        plugin_results = {}
        for name in ("nvv4l2decoder", "nvvideoconvert"):
            inspected = run(["gst-inspect-1.0", name])
            if inspected.returncode != 0:
                raise RuntimeError(f"Missing required plugin: {name}")

            version = inspect_field(inspected.stdout, "Version")
            filename = inspect_field(inspected.stdout, "Filename")
            expected_plugin = EXPECTED["plugins"][name]

            if version != expected_plugin["version"]:
                raise RuntimeError(f"{name} version changed.")
            if filename != expected_plugin["filename"]:
                raise RuntimeError(f"{name} filename changed.")

            plugin_results[name] = {
                "version": version,
                "filename": filename,
            }

        ctypes.CDLL("libnvcuvid.so.1")

        pipeline = run(
            EXPECTED["pipeline_command"],
            timeout=EXPECTED["timeout_seconds"],
        )
        log = pipeline.stdout + "\n" + pipeline.stderr

        fatal_patterns = [
            "Caught SIGSEGV",
            "S_EXT_CTRLS for CUDA_GPU_ID failed",
            "Internal data stream error",
            "not-negotiated",
            "ERROR: pipeline",
            "ERROR from element",
        ]
        fatal_hits = [
            pattern
            for pattern in fatal_patterns
            if pattern.lower() in log.lower()
        ]

        decoder_lines = [
            line
            for line in log.splitlines()
            if "nvv4l2decoder" in line.lower()
            and "GstPad:src" in line
            and "memory:NVMM" in line
        ]
        converter_lines = [
            line
            for line in log.splitlines()
            if "nvvideoconvert" in line.lower()
            and "GstPad:src" in line
            and "memory:NVMM" in line
        ]

        decoder_ok = any(
            "format=(string)NV12" in line
            and "width=(int)1280" in line
            and "height=(int)720" in line
            and "framerate=(fraction)30/1" in line
            and "nvbuf-memory-type=(string)nvbuf-mem-cuda-device" in line
            and "gpu-id=(int)0" in line
            for line in decoder_lines
        )
        converter_ok = any(
            "format=(string)RGBA" in line
            and "width=(int)1280" in line
            and "height=(int)720" in line
            and "framerate=(fraction)30/1" in line
            and "nvbuf-memory-type=(string)nvbuf-mem-cuda-device" in line
            and "gpu-id=(int)0" in line
            for line in converter_lines
        )
        eos = "Got EOS from element" in log

        passed = (
            pipeline.returncode == 0
            and eos
            and decoder_ok
            and converter_ok
            and not fatal_hits
        )

        payload = {
            "status": "passed" if passed else "failed",
            "runtime": {
                "gpu_name": gpu_parts[0],
                "driver_version": gpu_parts[1],
                "gpu_compute_capability": gpu_parts[2],
                "deepstream_version": EXPECTED["runtime"]["deepstream_version"],
                "gstreamer_version": EXPECTED["runtime"]["gstreamer_version"],
                "driver_capabilities": os.environ.get(
                    "NVIDIA_DRIVER_CAPABILITIES",
                    "",
                ),
                "visible_devices": os.environ.get(
                    "NVIDIA_VISIBLE_DEVICES",
                    "",
                ),
            },
            "sample": {
                "path": str(sample),
                "sha256": hash_file(sample),
                "size_bytes": sample.stat().st_size,
            },
            "plugins": plugin_results,
            "nvdec": {
                "dynamic_load": True,
            },
            "pipeline": {
                "command": EXPECTED["pipeline_command"],
                "exit_code": pipeline.returncode,
                "eos": eos,
                "decoder_nvmm_caps": decoder_ok,
                "decoder_format": "NV12" if decoder_ok else None,
                "converter_nvmm_caps": converter_ok,
                "converter_format": "RGBA" if converter_ok else None,
                "fatal_patterns": fatal_hits,
                "decoder_caps_lines": decoder_lines,
                "converter_caps_lines": converter_lines,
                "log_sha256": hashlib.sha256(
                    log.encode("utf-8")
                ).hexdigest(),
                "log": log,
            },
            "scope": EXPECTED["scope"],
        }

        emit(payload)
        raise SystemExit(0 if passed else 10)

    except Exception as exc:
        emit(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise SystemExit(10)


main()
"""
    return source.replace("__EXPECTED_JSON__", repr(expected_json))


# ADD 2026-09-05: Exact DeepStream digest와 isolated video-capable Docker command를 만든다.
def build_docker_smoke_command(
    config: DeepStreamNvmmSmokeConfig,
) -> tuple[str, ...]:
    config.validate()
    return (
        "sudo",
        "docker",
        "run",
        "--rm",
        "--runtime=nvidia",
        "--network",
        "none",
        "--gpus",
        "all",
        "--interactive",
        "--label",
        CONTAINER_LABEL,
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "--entrypoint",
        "python3",
        config.runtime.repo_digest,
        "-",
    )


# ADD 2026-09-05: Container marker JSON을 strict mapping으로 복원한다.
def parse_container_smoke_payload(stdout: str) -> dict[str, Any]:
    markers = [
        line[len(CONTAINER_RESULT_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(CONTAINER_RESULT_PREFIX)
    ]
    if not markers:
        raise RuntimeError("C6-5B container did not emit its result marker.")
    if len(markers) != 1:
        raise RuntimeError("C6-5B container emitted multiple result markers.")

    raw: object = json.loads(markers[0])
    return _mapping(raw, label="C6-5B container result")


# ADD 2026-09-05: Container payload가 exact NVDEC/NVMM success contract인지 검증한다.
def validate_container_smoke_payload(
    payload: dict[str, Any],
    config: DeepStreamNvmmSmokeConfig,
) -> None:
    if payload.get("status") != "passed":
        raise ValueError("C6-5B container smoke did not pass.")

    runtime = _mapping(payload.get("runtime"), label="result.runtime")
    sample = _mapping(payload.get("sample"), label="result.sample")
    plugins = _mapping(payload.get("plugins"), label="result.plugins")
    nvdec = _mapping(payload.get("nvdec"), label="result.nvdec")
    pipeline = _mapping(payload.get("pipeline"), label="result.pipeline")
    scope = _mapping(payload.get("scope"), label="result.scope")

    if (
        runtime.get("gpu_name") != config.runtime.gpu_name
        or runtime.get("driver_version") != config.runtime.driver_version
        or runtime.get("gpu_compute_capability") != config.runtime.gpu_compute_capability
        or runtime.get("deepstream_version") != config.runtime.deepstream_version
        or runtime.get("gstreamer_version") != config.runtime.gstreamer_version
        or runtime.get("driver_capabilities") != "compute,utility,video"
    ):
        raise ValueError("C6-5B container runtime identity changed.")

    if (
        sample.get("path") != config.sample.path
        or sample.get("sha256") != config.sample.sha256
        or sample.get("size_bytes") != config.sample.size_bytes
    ):
        raise ValueError("C6-5B container sample identity changed.")

    decoder = _mapping(
        plugins.get("nvv4l2decoder"),
        label="result.plugins.nvv4l2decoder",
    )
    converter = _mapping(
        plugins.get("nvvideoconvert"),
        label="result.plugins.nvvideoconvert",
    )
    if decoder != asdict(config.decoder_plugin):
        raise ValueError("C6-5B decoder plugin identity changed.")
    if converter != asdict(config.converter_plugin):
        raise ValueError("C6-5B converter plugin identity changed.")

    if nvdec.get("dynamic_load") is not True:
        raise ValueError("C6-5B libnvcuvid dynamic load failed.")

    if pipeline.get("command") != list(expected_gstreamer_pipeline(config)):
        raise ValueError("C6-5B executed pipeline command changed.")
    if pipeline.get("exit_code") != 0:
        raise ValueError("C6-5B GStreamer pipeline failed.")
    if pipeline.get("eos") is not True:
        raise ValueError("C6-5B pipeline did not reach EOS.")
    if pipeline.get("decoder_nvmm_caps") is not True:
        raise ValueError("C6-5B decoder did not expose NVMM caps.")
    if pipeline.get("decoder_format") != "NV12":
        raise ValueError("C6-5B decoder output format changed.")
    if pipeline.get("converter_nvmm_caps") is not True:
        raise ValueError("C6-5B converter did not preserve NVMM.")
    if pipeline.get("converter_format") != "RGBA":
        raise ValueError("C6-5B converter output format changed.")
    if pipeline.get("fatal_patterns") != []:
        raise ValueError("C6-5B pipeline contains fatal runtime diagnostics.")

    log_sha = pipeline.get("log_sha256")
    if not isinstance(log_sha, str) or not is_sha256_digest(log_sha):
        raise ValueError("C6-5B pipeline log SHA-256 is invalid.")

    log = pipeline.get("log")
    if not isinstance(log, str):
        raise ValueError("C6-5B pipeline log is invalid.")

    observed_log_sha = sha256_bytes(log.encode("utf-8"))
    if observed_log_sha != log_sha:
        raise ValueError("C6-5B pipeline log SHA-256 does not match log content.")

    decoder_lines = pipeline.get("decoder_caps_lines")
    converter_lines = pipeline.get("converter_caps_lines")
    if not isinstance(decoder_lines, list) or not decoder_lines:
        raise ValueError("C6-5B decoder caps evidence is missing.")
    if not isinstance(converter_lines, list) or not converter_lines:
        raise ValueError("C6-5B converter caps evidence is missing.")
    if not all(isinstance(line, str) for line in decoder_lines):
        raise ValueError("C6-5B decoder caps evidence is invalid.")
    if not all(isinstance(line, str) for line in converter_lines):
        raise ValueError("C6-5B converter caps evidence is invalid.")

    expected_scope = {
        "inference_executed": False,
        "tensorrt_engine_used": False,
        "dataset_used": False,
        "validation_used": False,
        "test_used": False,
        "final_test_used": False,
    }
    if scope != expected_scope:
        raise ValueError("C6-5B smoke scope changed.")


# ADD 2026-09-05: 비정상 종료 후에도 C6-5B labeled container를 강제로 정리한다.
def cleanup_deepstream_nvmm_containers() -> None:
    discovery = subprocess.run(
        (
            "sudo",
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label={CONTAINER_LABEL}",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    if discovery.returncode != 0:
        raise RuntimeError(
            "C6-5B failed to enumerate labeled containers: " + discovery.stderr.strip()
        )

    container_ids = [line.strip() for line in discovery.stdout.splitlines() if line.strip()]

    if not container_ids:
        return

    removal = subprocess.run(
        (
            "sudo",
            "docker",
            "rm",
            "-f",
            *container_ids,
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    if removal.returncode != 0:
        raise RuntimeError("C6-5B failed to remove labeled containers: " + removal.stderr.strip())


# ADD 2026-09-05: Host preflight 후 DeepStream container에서 canonical NVMM smoke를 실행한다.
# MODIFY 2026-09-05: timeout/abnormal exit에서도 labeled container를 finally에서 정리한다.
def run_deepstream_nvmm_smoke(
    config: DeepStreamNvmmSmokeConfig,
) -> tuple[dict[str, Any], str, str]:
    config.validate()
    source = build_container_probe_source(config)
    command: Sequence[str] = build_docker_smoke_command(config)

    try:
        completed = subprocess.run(
            list(command),
            input=source,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.pipeline.timeout_seconds + 30,
        )
    finally:
        cleanup_deepstream_nvmm_containers()

    payload = parse_container_smoke_payload(completed.stdout)

    if completed.returncode != 0:
        error = payload.get("error", "container smoke failed")
        raise RuntimeError(f"C6-5B container exited {completed.returncode}: {error}")

    validate_container_smoke_payload(payload, config)
    return payload, completed.stdout, completed.stderr


# ADD 2026-09-05: Clean foundation commit에서 canonical C6-5B evidence JSON을 기록한다.
def write_deepstream_nvmm_evidence(
    *,
    output_path: Path,
    repo: Path,
    config_path: Path = DEFAULT_DEEPSTREAM_NVMM_CONFIG,
) -> Path:
    repo = repo.resolve()
    output_path = output_path.resolve()
    config = load_deepstream_nvmm_smoke_config(config_path)

    repository = resolve_repository_identity(repo)

    if output_path == repo or repo in output_path.parents:
        raise ValueError("C6-5B canonical evidence must be outside repository.")
    if output_path.exists():
        raise FileExistsError(f"C6-5B evidence already exists: {output_path}")

    image = inspect_deepstream_image(config)
    host = validate_host_dependencies(config)
    payload, container_stdout, container_stderr = run_deepstream_nvmm_smoke(config)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    evidence = {
        "schema_version": 1,
        "stage": "C6-5B",
        "state": SMOKE_STATE,
        "observed_at": datetime.now(UTC).isoformat(),
        "repository": repository,
        "config_sha256": sha256_file(config.config_path),
        "deepstream_image": image,
        "host_dependency": host,
        "runtime": payload["runtime"],
        "sample": payload["sample"],
        "plugins": payload["plugins"],
        "nvdec": payload["nvdec"],
        "pipeline": payload["pipeline"],
        "policy": asdict(config.policy),
        "container_stdout": container_stdout,
        "container_stderr": container_stderr,
    }

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
