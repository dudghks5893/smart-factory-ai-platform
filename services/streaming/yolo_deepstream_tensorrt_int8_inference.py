"""C6-5C canonical DeepStream TensorRT INT8 inference runtime."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from services.streaming.yolo_deepstream_tensorrt_int8 import (
    DEFAULT_DEEPSTREAM_TENSORRT_CONFIG,
    DeepStreamTensorRtConfig,
    load_deepstream_tensorrt_config,
)
from services.streaming.yolo_deepstream_tensorrt_int8_build import (
    DockerImageIdentity,
    HostGpuIdentity,
    RepositoryIdentity,
    inspect_deepstream_image,
    inspect_host_gpu,
    resolve_repository_identity,
)
from shared.hashing import is_sha256_digest, sha256_bytes, sha256_file

DEFAULT_DEEPSTREAM_INFERENCE_CONFIG = Path(
    "configs/streaming/yolo_deepstream_tensorrt_int8_inference.json"
)

EXPECTED_INFERENCE_ID = "c6_5c_deepstream_l4_tensorrt_int8_inference_v1"
EXPECTED_ENGINE_BUILD_FOUNDATION = "9fbb9bac9c35c74b6d0a8fe54fdfebc7b1e37e92"
EXPECTED_BUILD_CONFIG_SHA256 = "4a5ec9e5eb3ede384247ad46d0d4e90be732cf5b34b066ca71f469d7f5128fd5"
EXPECTED_GSTREAMER_VERSION = "1.24.2"

EXPECTED_PLAN_PATH = Path(
    "artifacts/deployment/yolo_segmentation/deepstream_l4/tensorrt_int8/"
    "c6_5c_deepstream_l4_yolo11n_seg_int8_qdq_v1/model.plan"
)
EXPECTED_PLAN_SHA256 = "97acd724809f4817ad4a95525a1bafae6294b1a7c99e04c12d451eeda878866e"
EXPECTED_PLAN_BYTES = 4_940_452

EXPECTED_SAMPLE_PATH = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264"
EXPECTED_SAMPLE_SHA256 = "5f29353a6ec4727bd49fb523efc207d643e6638f4e5c56f060e1b61291aa6ea2"
EXPECTED_SAMPLE_BYTES = 14_759_548

EXPECTED_REQUIRED_PLUGINS = (
    "filesrc",
    "h264parse",
    "nvv4l2decoder",
    "queue",
    "nvstreammux",
    "nvinfer",
    "identity",
    "fakesink",
)

INFERENCE_STATE = "DEEPSTREAM_TENSORRT_INT8_INFERENCE_COMPLETED"
CONTAINER_RESULT_PREFIX = "C6_5C_INFERENCE_RESULT="
CONTAINER_LABEL = "c6_5c_inference=1"


@dataclass(frozen=True)
class InferenceEngineIdentity:
    """Immutable C6-5C L4 TensorRT plan identity."""

    relative_path: Path
    sha256: str
    size_bytes: int
    immutable: bool

    # ADD 2026-09-05: Canonical L4 plan path/hash/bytes를 exact identity에 고정한다.
    def validate(self) -> None:
        if (
            self.relative_path != EXPECTED_PLAN_PATH
            or self.sha256 != EXPECTED_PLAN_SHA256
            or self.size_bytes != EXPECTED_PLAN_BYTES
            or self.immutable is not True
        ):
            raise ValueError("C6-5C inference engine identity changed.")

        if not is_sha256_digest(self.sha256):
            raise ValueError("C6-5C inference engine SHA-256 is invalid.")


@dataclass(frozen=True)
class InferenceSampleIdentity:
    """Exact DeepStream H264 sample identity."""

    path: str
    sha256: str
    size_bytes: int
    codec: str

    # ADD 2026-09-05: Canonical inference fixture를 exact DeepStream sample에 고정한다.
    def validate(self) -> None:
        if (
            self.path != EXPECTED_SAMPLE_PATH
            or self.sha256 != EXPECTED_SAMPLE_SHA256
            or self.size_bytes != EXPECTED_SAMPLE_BYTES
            or self.codec != "h264"
        ):
            raise ValueError("C6-5C inference sample identity changed.")

        if not is_sha256_digest(self.sha256):
            raise ValueError("C6-5C inference sample SHA-256 is invalid.")


@dataclass(frozen=True)
class InferencePipelineContract:
    """Frozen DeepStream post-nvinfer 30-frame pipeline contract."""

    required_plugins: tuple[str, ...]
    streammux_width: int
    streammux_height: int
    batch_size: int
    live_source: bool
    batched_push_timeout_us: int
    target_frames: int
    sink_sync: bool
    sink_async: bool
    timeout_seconds: int

    # ADD 2026-09-05: NVDEC→NVMM→nvinfer 30-frame pipeline boundary를 검증한다.
    def validate(self) -> None:
        bool_values = (
            self.live_source,
            self.sink_sync,
            self.sink_async,
        )

        if any(type(value) is not bool for value in bool_values):
            raise TypeError("C6-5C inference pipeline flags must be bool.")

        if (
            self.required_plugins != EXPECTED_REQUIRED_PLUGINS
            or self.streammux_width != 1280
            or self.streammux_height != 720
            or self.batch_size != 1
            or self.live_source is not False
            or self.batched_push_timeout_us != 40_000
            or self.target_frames != 30
            or self.sink_sync is not False
            or self.sink_async is not False
            or self.timeout_seconds != 180
        ):
            raise ValueError("C6-5C inference pipeline contract changed.")


@dataclass(frozen=True)
class NvInferContract:
    """Frozen Gst-nvinfer configuration for raw tensor inference."""

    gpu_id: int
    net_scale_factor: str
    model_color_format: int
    batch_size: int
    network_mode: int
    process_mode: int
    interval: int
    gie_unique_id: int
    network_type: int
    output_tensor_meta: bool
    maintain_aspect_ratio: bool
    symmetric_padding: bool

    # ADD 2026-09-05: nvinfer를 raw tensor meta/no-parser boundary에 고정한다.
    def validate(self) -> None:
        bool_values = (
            self.output_tensor_meta,
            self.maintain_aspect_ratio,
            self.symmetric_padding,
        )

        if any(type(value) is not bool for value in bool_values):
            raise TypeError("C6-5C nvinfer flags must be strict booleans.")

        if (
            self.gpu_id != 0
            or self.net_scale_factor != "0.00392156862745098"
            or self.model_color_format != 0
            or self.batch_size != 1
            or self.network_mode != 1
            or self.process_mode != 1
            or self.interval != 0
            or self.gie_unique_id != 1
            or self.network_type != 100
            or self.output_tensor_meta is not True
            or self.maintain_aspect_ratio is not True
            or self.symmetric_padding is not True
        ):
            raise ValueError("C6-5C nvinfer contract changed.")


@dataclass(frozen=True)
class InferencePolicy:
    """Scope restrictions for C6-5C inference acceptance."""

    require_clean_repository: bool
    network_allowed: bool
    engine_rebuild_allowed: bool
    application_inference: bool
    segmentation_decode_allowed: bool
    overlay_allowed: bool
    dataset_used: bool
    validation_used: bool
    test_used: bool
    final_test_used: bool

    # ADD 2026-09-05: C6-5C inference를 no-rebuild/no-decode/no-final-test로 제한한다.
    def validate(self) -> None:
        values = (
            self.require_clean_repository,
            self.network_allowed,
            self.engine_rebuild_allowed,
            self.application_inference,
            self.segmentation_decode_allowed,
            self.overlay_allowed,
            self.dataset_used,
            self.validation_used,
            self.test_used,
            self.final_test_used,
        )

        if any(type(value) is not bool for value in values):
            raise TypeError("C6-5C inference policy flags must be bool.")

        if (
            self.require_clean_repository is not True
            or self.network_allowed is not False
            or self.engine_rebuild_allowed is not False
            or self.application_inference is not True
            or self.segmentation_decode_allowed is not False
            or self.overlay_allowed is not False
            or self.dataset_used is not False
            or self.validation_used is not False
            or self.test_used is not False
            or self.final_test_used is not False
        ):
            raise ValueError("C6-5C inference policy changed.")


@dataclass(frozen=True)
class DeepStreamInferenceConfig:
    """Top-level canonical C6-5C inference contract."""

    schema_version: int
    inference_id: str
    engine_build_foundation_commit: str
    build_config_sha256: str
    gstreamer_version: str
    engine: InferenceEngineIdentity
    source: InferenceSampleIdentity
    pipeline: InferencePipelineContract
    nvinfer: NvInferContract
    policy: InferencePolicy
    config_path: Path

    # ADD 2026-09-05: Top-level inference identity와 nested contracts를 검증한다.
    def validate(self) -> None:
        if (
            self.schema_version != 1
            or self.inference_id != EXPECTED_INFERENCE_ID
            or self.engine_build_foundation_commit != EXPECTED_ENGINE_BUILD_FOUNDATION
            or self.build_config_sha256 != EXPECTED_BUILD_CONFIG_SHA256
            or self.gstreamer_version != EXPECTED_GSTREAMER_VERSION
        ):
            raise ValueError("C6-5C top-level inference contract changed.")

        self.engine.validate()
        self.source.validate()
        self.pipeline.validate()
        self.nvinfer.validate()
        self.policy.validate()


# ADD 2026-09-05: JSON object를 strict mapping으로 변환한다.
def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be JSON object.")

    return cast(dict[str, Any], value)


# ADD 2026-09-05: JSON array를 strict list로 변환한다.
def _array(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be JSON array.")

    return cast(list[Any], value)


# ADD 2026-09-05: JSON object field set을 exact schema와 대조한다.
def _require_fields(
    value: dict[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields do not match schema.")


# ADD 2026-09-05: JSON string scalar를 strict str로 변환한다.
def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str.")

    return value


# ADD 2026-09-05: JSON integer scalar를 strict int로 변환한다.
def _integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be int.")

    return cast(int, value)


# ADD 2026-09-05: JSON boolean scalar를 strict bool로 변환한다.
def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be bool.")

    return cast(bool, value)


# ADD 2026-09-05: Frozen inference JSON을 typed runtime contract로 로드한다.
def load_deepstream_inference_config(
    path: Path = DEFAULT_DEEPSTREAM_INFERENCE_CONFIG,
) -> DeepStreamInferenceConfig:
    raw_object: object = json.loads(path.read_text(encoding="utf-8"))
    raw = _mapping(
        raw_object,
        label="C6-5C inference config",
    )

    _require_fields(
        raw,
        {
            "schema_version",
            "inference_id",
            "engine_build_foundation_commit",
            "build_config_sha256",
            "gstreamer_version",
            "engine",
            "source",
            "pipeline",
            "nvinfer",
            "policy",
        },
        label="C6-5C inference config",
    )

    engine_raw = _mapping(
        raw["engine"],
        label="engine",
    )
    source_raw = _mapping(
        raw["source"],
        label="source",
    )
    pipeline_raw = _mapping(
        raw["pipeline"],
        label="pipeline",
    )
    nvinfer_raw = _mapping(
        raw["nvinfer"],
        label="nvinfer",
    )
    policy_raw = _mapping(
        raw["policy"],
        label="policy",
    )

    _require_fields(
        engine_raw,
        {
            "relative_path",
            "sha256",
            "size_bytes",
            "immutable",
        },
        label="engine",
    )

    _require_fields(
        source_raw,
        {
            "path",
            "sha256",
            "size_bytes",
            "codec",
        },
        label="source",
    )

    _require_fields(
        pipeline_raw,
        {
            "required_plugins",
            "streammux_width",
            "streammux_height",
            "batch_size",
            "live_source",
            "batched_push_timeout_us",
            "target_frames",
            "sink_sync",
            "sink_async",
            "timeout_seconds",
        },
        label="pipeline",
    )

    _require_fields(
        nvinfer_raw,
        {
            "gpu_id",
            "net_scale_factor",
            "model_color_format",
            "batch_size",
            "network_mode",
            "process_mode",
            "interval",
            "gie_unique_id",
            "network_type",
            "output_tensor_meta",
            "maintain_aspect_ratio",
            "symmetric_padding",
        },
        label="nvinfer",
    )

    _require_fields(
        policy_raw,
        {
            "require_clean_repository",
            "network_allowed",
            "engine_rebuild_allowed",
            "application_inference",
            "segmentation_decode_allowed",
            "overlay_allowed",
            "dataset_used",
            "validation_used",
            "test_used",
            "final_test_used",
        },
        label="policy",
    )

    required_plugins = tuple(
        _string(
            value,
            label="pipeline.required_plugins",
        )
        for value in _array(
            pipeline_raw["required_plugins"],
            label="pipeline.required_plugins",
        )
    )

    config = DeepStreamInferenceConfig(
        schema_version=_integer(
            raw["schema_version"],
            label="schema_version",
        ),
        inference_id=_string(
            raw["inference_id"],
            label="inference_id",
        ),
        engine_build_foundation_commit=_string(
            raw["engine_build_foundation_commit"],
            label="engine_build_foundation_commit",
        ),
        build_config_sha256=_string(
            raw["build_config_sha256"],
            label="build_config_sha256",
        ),
        gstreamer_version=_string(
            raw["gstreamer_version"],
            label="gstreamer_version",
        ),
        engine=InferenceEngineIdentity(
            relative_path=Path(
                _string(
                    engine_raw["relative_path"],
                    label="engine.relative_path",
                )
            ),
            sha256=_string(
                engine_raw["sha256"],
                label="engine.sha256",
            ),
            size_bytes=_integer(
                engine_raw["size_bytes"],
                label="engine.size_bytes",
            ),
            immutable=_boolean(
                engine_raw["immutable"],
                label="engine.immutable",
            ),
        ),
        source=InferenceSampleIdentity(
            path=_string(
                source_raw["path"],
                label="source.path",
            ),
            sha256=_string(
                source_raw["sha256"],
                label="source.sha256",
            ),
            size_bytes=_integer(
                source_raw["size_bytes"],
                label="source.size_bytes",
            ),
            codec=_string(
                source_raw["codec"],
                label="source.codec",
            ),
        ),
        pipeline=InferencePipelineContract(
            required_plugins=required_plugins,
            streammux_width=_integer(
                pipeline_raw["streammux_width"],
                label="pipeline.streammux_width",
            ),
            streammux_height=_integer(
                pipeline_raw["streammux_height"],
                label="pipeline.streammux_height",
            ),
            batch_size=_integer(
                pipeline_raw["batch_size"],
                label="pipeline.batch_size",
            ),
            live_source=_boolean(
                pipeline_raw["live_source"],
                label="pipeline.live_source",
            ),
            batched_push_timeout_us=_integer(
                pipeline_raw["batched_push_timeout_us"],
                label="pipeline.batched_push_timeout_us",
            ),
            target_frames=_integer(
                pipeline_raw["target_frames"],
                label="pipeline.target_frames",
            ),
            sink_sync=_boolean(
                pipeline_raw["sink_sync"],
                label="pipeline.sink_sync",
            ),
            sink_async=_boolean(
                pipeline_raw["sink_async"],
                label="pipeline.sink_async",
            ),
            timeout_seconds=_integer(
                pipeline_raw["timeout_seconds"],
                label="pipeline.timeout_seconds",
            ),
        ),
        nvinfer=NvInferContract(
            gpu_id=_integer(
                nvinfer_raw["gpu_id"],
                label="nvinfer.gpu_id",
            ),
            net_scale_factor=_string(
                nvinfer_raw["net_scale_factor"],
                label="nvinfer.net_scale_factor",
            ),
            model_color_format=_integer(
                nvinfer_raw["model_color_format"],
                label="nvinfer.model_color_format",
            ),
            batch_size=_integer(
                nvinfer_raw["batch_size"],
                label="nvinfer.batch_size",
            ),
            network_mode=_integer(
                nvinfer_raw["network_mode"],
                label="nvinfer.network_mode",
            ),
            process_mode=_integer(
                nvinfer_raw["process_mode"],
                label="nvinfer.process_mode",
            ),
            interval=_integer(
                nvinfer_raw["interval"],
                label="nvinfer.interval",
            ),
            gie_unique_id=_integer(
                nvinfer_raw["gie_unique_id"],
                label="nvinfer.gie_unique_id",
            ),
            network_type=_integer(
                nvinfer_raw["network_type"],
                label="nvinfer.network_type",
            ),
            output_tensor_meta=_boolean(
                nvinfer_raw["output_tensor_meta"],
                label="nvinfer.output_tensor_meta",
            ),
            maintain_aspect_ratio=_boolean(
                nvinfer_raw["maintain_aspect_ratio"],
                label="nvinfer.maintain_aspect_ratio",
            ),
            symmetric_padding=_boolean(
                nvinfer_raw["symmetric_padding"],
                label="nvinfer.symmetric_padding",
            ),
        ),
        policy=InferencePolicy(
            require_clean_repository=_boolean(
                policy_raw["require_clean_repository"],
                label="policy.require_clean_repository",
            ),
            network_allowed=_boolean(
                policy_raw["network_allowed"],
                label="policy.network_allowed",
            ),
            engine_rebuild_allowed=_boolean(
                policy_raw["engine_rebuild_allowed"],
                label="policy.engine_rebuild_allowed",
            ),
            application_inference=_boolean(
                policy_raw["application_inference"],
                label="policy.application_inference",
            ),
            segmentation_decode_allowed=_boolean(
                policy_raw["segmentation_decode_allowed"],
                label="policy.segmentation_decode_allowed",
            ),
            overlay_allowed=_boolean(
                policy_raw["overlay_allowed"],
                label="policy.overlay_allowed",
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
        ),
        config_path=path,
    )

    config.validate()
    return config


# ADD 2026-09-05: Canonical plan file의 exact hash와 bytes를 검증한다.
def validate_engine_file(
    plan_path: Path,
    config: DeepStreamInferenceConfig,
) -> dict[str, object]:
    if not plan_path.is_file():
        raise FileNotFoundError(f"C6-5C inference plan missing: {plan_path}")

    sha = sha256_file(plan_path)
    size = plan_path.stat().st_size

    if sha != config.engine.sha256 or size != config.engine.size_bytes:
        raise ValueError("C6-5C canonical inference plan identity changed.")

    return {
        "sha256": sha,
        "size_bytes": size,
    }


# ADD 2026-09-05: Frozen raw-output Gst-nvinfer config text를 생성한다.
def build_nvinfer_config_text(
    config: DeepStreamInferenceConfig,
) -> str:
    nvinfer = config.nvinfer

    return "\n".join(
        (
            "[property]",
            f"gpu-id={nvinfer.gpu_id}",
            f"net-scale-factor={nvinfer.net_scale_factor}",
            f"model-color-format={nvinfer.model_color_format}",
            "model-engine-file=/model/model.plan",
            f"batch-size={nvinfer.batch_size}",
            f"network-mode={nvinfer.network_mode}",
            f"process-mode={nvinfer.process_mode}",
            f"interval={nvinfer.interval}",
            f"gie-unique-id={nvinfer.gie_unique_id}",
            f"network-type={nvinfer.network_type}",
            "output-tensor-meta=1",
            "maintain-aspect-ratio=1",
            "symmetric-padding=1",
            "",
        )
    )


# ADD 2026-09-05: Canonical DeepStream inference container source를 생성한다.
def build_container_source(
    config: DeepStreamInferenceConfig,
    build_config: DeepStreamTensorRtConfig,
) -> str:
    nvinfer_text = build_nvinfer_config_text(config)

    template = """from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import tensorrt as trt

EXPECTED_TRT_VERSION = __TRT_VERSION__
EXPECTED_GSTREAMER_VERSION = __GST_VERSION__

PLAN_PATH = Path("/model/model.plan")
EXPECTED_PLAN_SHA = __PLAN_SHA__
EXPECTED_PLAN_BYTES = __PLAN_BYTES__

SAMPLE_PATH = Path(__SAMPLE_PATH__)
EXPECTED_SAMPLE_SHA = __SAMPLE_SHA__
EXPECTED_SAMPLE_BYTES = __SAMPLE_BYTES__

REQUIRED_PLUGINS = tuple(json.loads(__REQUIRED_PLUGINS__))
TARGET_FRAMES = __TARGET_FRAMES__
TIMEOUT_SECONDS = __TIMEOUT_SECONDS__

NVINFER_CONFIG = __NVINFER_CONFIG__
RESULT_PREFIX = __RESULT_PREFIX__


# ADD 2026-09-05: Container subprocess를 captured text contract로 실행한다.
def run_command(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(command),
        text=True,
        capture_output=True,
        check=False,
        env=env,
        timeout=timeout,
    )


# ADD 2026-09-05: Container 내부 artifact의 SHA-256을 계산한다.
def hash_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


# ADD 2026-09-05: Required GStreamer plugin availability를 확인한다.
def inspect_plugin(name: str) -> str:
    result = run_command(
        (
            "gst-inspect-1.0",
            name,
        )
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Required GStreamer plugin missing: {name}"
        )

    return result.stdout


# ADD 2026-09-05: 30 post-nvinfer buffers를 실행하고 structural inference evidence를 출력한다.
def main() -> None:
    if trt.__version__ != EXPECTED_TRT_VERSION:
        raise RuntimeError(
            "C6-5C TensorRT version changed."
        )

    gst_version = run_command(
        (
            "gst-launch-1.0",
            "--version",
        )
    )

    if gst_version.returncode != 0:
        raise RuntimeError(
            "C6-5C failed to inspect GStreamer version."
        )

    if EXPECTED_GSTREAMER_VERSION not in gst_version.stdout:
        raise RuntimeError(
            "C6-5C GStreamer version changed."
        )

    plan_sha = hash_file(
        PLAN_PATH
    )
    plan_bytes = PLAN_PATH.stat().st_size

    if (
        plan_sha != EXPECTED_PLAN_SHA
        or plan_bytes != EXPECTED_PLAN_BYTES
    ):
        raise RuntimeError(
            "C6-5C mounted TensorRT plan identity changed."
        )

    sample_sha = hash_file(
        SAMPLE_PATH
    )
    sample_bytes = SAMPLE_PATH.stat().st_size

    if (
        sample_sha != EXPECTED_SAMPLE_SHA
        or sample_bytes != EXPECTED_SAMPLE_BYTES
    ):
        raise RuntimeError(
            "C6-5C DeepStream sample identity changed."
        )

    plugin_output = {
        name: inspect_plugin(name)
        for name in REQUIRED_PLUGINS
    }

    if "output-tensor-meta" not in plugin_output["nvinfer"]:
        raise RuntimeError(
            "C6-5C nvinfer output-tensor-meta property missing."
        )

    if "eos-after" not in plugin_output["identity"]:
        raise RuntimeError(
            "C6-5C identity eos-after property missing."
        )

    config_path = Path(
        "/tmp/c6_5c_nvinfer.txt"
    )

    config_path.write_text(
        NVINFER_CONFIG,
        encoding="utf-8",
    )

    command = (
        "gst-launch-1.0",
        "-e",
        "-v",
        "nvstreammux",
        "name=mux",
        "batch-size=1",
        "width=1280",
        "height=720",
        "live-source=false",
        "batched-push-timeout=40000",
        "filesrc",
        f"location={SAMPLE_PATH}",
        "!",
        "h264parse",
        "!",
        "nvv4l2decoder",
        "!",
        "queue",
        "!",
        "mux.sink_0",
        "mux.",
        "!",
        "nvinfer",
        f"config-file-path={config_path}",
        "!",
        "identity",
        "name=postinfer",
        f"eos-after={TARGET_FRAMES}",
        "silent=false",
        "!",
        "fakesink",
        "sync=false",
        "async=false",
    )

    env = os.environ.copy()
    env["GST_DEBUG"] = "nvinfer:4"

    try:
        completed = run_command(
            command,
            env=env,
            timeout=TIMEOUT_SECONDS,
        )

        if completed.stdout:
            print(
                completed.stdout,
                end="",
            )

        if completed.stderr:
            print(
                completed.stderr,
                end="",
                file=sys.stderr,
            )

        combined = (
            completed.stdout
            + "\\n"
            + completed.stderr
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "C6-5C nvinfer pipeline returned nonzero."
            )

        eos_observed = (
            'Got EOS from element "pipeline0".'
            in combined
        )

        engine_deserialized = (
            "deserialized trt engine from :/model/model.plan"
            in combined
        )

        engine_model_used = (
            "Use deserialized engine model: /model/model.plan"
            in combined
        )

        nvmm_observed = (
            "video/x-raw(memory:NVMM)"
            in combined
            and "nvbuf-mem-cuda-device"
            in combined
        )

        if not eos_observed:
            raise RuntimeError(
                "C6-5C pipeline EOS was not observed."
            )

        if not engine_deserialized:
            raise RuntimeError(
                "C6-5C nvinfer deserialize evidence missing."
            )

        if not engine_model_used:
            raise RuntimeError(
                "C6-5C nvinfer engine-use evidence missing."
            )

        if not nvmm_observed:
            raise RuntimeError(
                "C6-5C NVMM CUDA-device caps evidence missing."
            )

        payload = {
            "status": "passed",
            "tensorrt_version": trt.__version__,
            "gstreamer_version": EXPECTED_GSTREAMER_VERSION,
            "plan_sha256": plan_sha,
            "plan_bytes": plan_bytes,
            "sample_sha256": sample_sha,
            "sample_bytes": sample_bytes,
            "required_plugins": list(REQUIRED_PLUGINS),
            "output_tensor_meta_property": True,
            "identity_eos_after_property": True,
            "pipeline_exit_code": completed.returncode,
            "eos_observed": eos_observed,
            "engine_deserialized": engine_deserialized,
            "engine_model_used": engine_model_used,
            "nvmm_observed": nvmm_observed,
            "post_inference_eos_after": TARGET_FRAMES,
            "target_frames_reached": True,
            "application_inference_executed": True,
            "segmentation_decode_executed": False,
            "overlay_executed": False,
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

    finally:
        config_path.unlink(
            missing_ok=True
        )


if __name__ == "__main__":
    main()
"""

    return (
        template.replace(
            "__TRT_VERSION__",
            repr(build_config.runtime.tensorrt_python_version),
        )
        .replace(
            "__GST_VERSION__",
            repr(config.gstreamer_version),
        )
        .replace(
            "__PLAN_SHA__",
            repr(config.engine.sha256),
        )
        .replace(
            "__PLAN_BYTES__",
            str(config.engine.size_bytes),
        )
        .replace(
            "__SAMPLE_PATH__",
            repr(config.source.path),
        )
        .replace(
            "__SAMPLE_SHA__",
            repr(config.source.sha256),
        )
        .replace(
            "__SAMPLE_BYTES__",
            str(config.source.size_bytes),
        )
        .replace(
            "__REQUIRED_PLUGINS__",
            repr(json.dumps(list(config.pipeline.required_plugins))),
        )
        .replace(
            "__TARGET_FRAMES__",
            str(config.pipeline.target_frames),
        )
        .replace(
            "__TIMEOUT_SECONDS__",
            str(config.pipeline.timeout_seconds),
        )
        .replace(
            "__NVINFER_CONFIG__",
            repr(nvinfer_text),
        )
        .replace(
            "__RESULT_PREFIX__",
            repr(CONTAINER_RESULT_PREFIX),
        )
    )


# ADD 2026-09-05: Exact digest/no-network/read-only-plan Docker command를 생성한다.
def build_docker_command(
    config: DeepStreamInferenceConfig,
    build_config: DeepStreamTensorRtConfig,
    *,
    plan_path: Path,
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
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "-v",
        f"{plan_path.resolve()}:/model/model.plan:ro",
        "--entrypoint",
        "python3",
        build_config.runtime.repo_digest,
        "-",
    )


# ADD 2026-09-05: Host subprocess 실행을 captured text contract로 통일한다.
def _run_command(
    command: Sequence[str],
    *,
    input_text: str | None = None,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(command),
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )


# ADD 2026-09-05: Labeled C6-5C inference container가 없는지 확인한다.
def assert_no_inference_containers() -> None:
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
        raise RuntimeError("C6-5C failed to inspect inference containers.")

    if result.stdout.strip():
        raise RuntimeError("C6-5C inference container already exists.")


# ADD 2026-09-05: 실패/timeout 이후 labeled inference container를 즉시 제거한다.
def cleanup_inference_containers() -> None:
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

    ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    if not ids:
        return

    removal = _run_command(
        (
            "sudo",
            "docker",
            "rm",
            "-f",
            *ids,
        )
    )

    if removal.returncode != 0:
        raise RuntimeError("C6-5C failed to remove inference containers.")


# ADD 2026-09-05: Container result line을 strict JSON payload로 변환한다.
def parse_container_inference_payload(
    stdout: str,
    *,
    prefix: str = CONTAINER_RESULT_PREFIX,
) -> dict[str, Any]:
    matches = [line[len(prefix) :] for line in stdout.splitlines() if line.startswith(prefix)]

    if len(matches) != 1:
        raise ValueError("C6-5C inference result payload count must be exactly one.")

    raw: object = json.loads(matches[0])

    return _mapping(
        raw,
        label="C6-5C inference result",
    )


# ADD 2026-09-05: Container inference payload를 frozen runtime contract와 대조한다.
def validate_container_inference_payload(
    payload: dict[str, Any],
    config: DeepStreamInferenceConfig,
    build_config: DeepStreamTensorRtConfig,
) -> None:
    _require_fields(
        payload,
        {
            "status",
            "tensorrt_version",
            "gstreamer_version",
            "plan_sha256",
            "plan_bytes",
            "sample_sha256",
            "sample_bytes",
            "required_plugins",
            "output_tensor_meta_property",
            "identity_eos_after_property",
            "pipeline_exit_code",
            "eos_observed",
            "engine_deserialized",
            "engine_model_used",
            "nvmm_observed",
            "post_inference_eos_after",
            "target_frames_reached",
            "application_inference_executed",
            "segmentation_decode_executed",
            "overlay_executed",
            "dataset_used",
            "validation_used",
            "test_used",
            "final_test_used",
        },
        label="C6-5C inference result",
    )

    plugins = tuple(
        _string(
            value,
            label="required_plugins",
        )
        for value in _array(
            payload["required_plugins"],
            label="required_plugins",
        )
    )

    if (
        _string(
            payload["status"],
            label="status",
        )
        != "passed"
        or _string(
            payload["tensorrt_version"],
            label="tensorrt_version",
        )
        != build_config.runtime.tensorrt_python_version
        or _string(
            payload["gstreamer_version"],
            label="gstreamer_version",
        )
        != config.gstreamer_version
        or _string(
            payload["plan_sha256"],
            label="plan_sha256",
        )
        != config.engine.sha256
        or _integer(
            payload["plan_bytes"],
            label="plan_bytes",
        )
        != config.engine.size_bytes
        or _string(
            payload["sample_sha256"],
            label="sample_sha256",
        )
        != config.source.sha256
        or _integer(
            payload["sample_bytes"],
            label="sample_bytes",
        )
        != config.source.size_bytes
        or plugins != config.pipeline.required_plugins
        or _boolean(
            payload["output_tensor_meta_property"],
            label="output_tensor_meta_property",
        )
        is not True
        or _boolean(
            payload["identity_eos_after_property"],
            label="identity_eos_after_property",
        )
        is not True
        or _integer(
            payload["pipeline_exit_code"],
            label="pipeline_exit_code",
        )
        != 0
        or _boolean(
            payload["eos_observed"],
            label="eos_observed",
        )
        is not True
        or _boolean(
            payload["engine_deserialized"],
            label="engine_deserialized",
        )
        is not True
        or _boolean(
            payload["engine_model_used"],
            label="engine_model_used",
        )
        is not True
        or _boolean(
            payload["nvmm_observed"],
            label="nvmm_observed",
        )
        is not True
        or _integer(
            payload["post_inference_eos_after"],
            label="post_inference_eos_after",
        )
        != config.pipeline.target_frames
        or _boolean(
            payload["target_frames_reached"],
            label="target_frames_reached",
        )
        is not True
        or _boolean(
            payload["application_inference_executed"],
            label="application_inference_executed",
        )
        is not True
        or _boolean(
            payload["segmentation_decode_executed"],
            label="segmentation_decode_executed",
        )
        is not False
        or _boolean(
            payload["overlay_executed"],
            label="overlay_executed",
        )
        is not False
        or _boolean(
            payload["dataset_used"],
            label="dataset_used",
        )
        is not False
        or _boolean(
            payload["validation_used"],
            label="validation_used",
        )
        is not False
        or _boolean(
            payload["test_used"],
            label="test_used",
        )
        is not False
        or _boolean(
            payload["final_test_used"],
            label="final_test_used",
        )
        is not False
    ):
        raise ValueError("C6-5C inference result contract changed.")


# ADD 2026-09-05: DeepStream container에서 canonical raw TensorRT inference를 실행한다.
def run_deepstream_tensorrt_inference(
    *,
    plan_path: Path,
    config: DeepStreamInferenceConfig,
    build_config: DeepStreamTensorRtConfig,
) -> tuple[dict[str, Any], str, str]:
    validate_engine_file(
        plan_path,
        config,
    )

    assert_no_inference_containers()

    source = build_container_source(
        config,
        build_config,
    )

    command = build_docker_command(
        config,
        build_config,
        plan_path=plan_path,
    )

    try:
        completed = _run_command(
            command,
            input_text=source,
            timeout_seconds=(config.pipeline.timeout_seconds + 30),
        )
    finally:
        cleanup_inference_containers()

    if completed.returncode != 0:
        raise RuntimeError(
            "C6-5C inference container failed "
            f"with exit {completed.returncode}: " + completed.stderr[-4000:]
        )

    payload = parse_container_inference_payload(completed.stdout)

    validate_container_inference_payload(
        payload,
        config,
        build_config,
    )

    validate_engine_file(
        plan_path,
        config,
    )

    return (
        payload,
        completed.stdout,
        completed.stderr,
    )


# ADD 2026-09-05: Canonical inference evidence JSON 구조를 구성한다.
def build_inference_evidence(
    *,
    config: DeepStreamInferenceConfig,
    build_config: DeepStreamTensorRtConfig,
    repository: RepositoryIdentity,
    image: DockerImageIdentity,
    gpu: HostGpuIdentity,
    payload: dict[str, Any],
    container_stdout: str,
    container_stderr: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "C6-5C",
        "state": INFERENCE_STATE,
        "created_at": datetime.now(UTC).isoformat(),
        "repository": {
            "git_commit": repository.git_commit,
            "working_tree_dirty": repository.working_tree_dirty,
        },
        "engine_build_foundation_commit": (config.engine_build_foundation_commit),
        "config_sha256": sha256_file(config.config_path),
        "build_config_sha256": sha256_file(build_config.config_path),
        "engine": {
            "relative_path": str(config.engine.relative_path),
            "sha256": config.engine.sha256,
            "size_bytes": config.engine.size_bytes,
            "immutable": config.engine.immutable,
        },
        "source": {
            "path": config.source.path,
            "sha256": config.source.sha256,
            "size_bytes": config.source.size_bytes,
            "codec": config.source.codec,
        },
        "runtime": {
            "image_tag": build_config.runtime.image_tag,
            "image_id": image.image_id,
            "repo_digests": list(image.repo_digests),
            "gpu_name": gpu.gpu_name,
            "driver_version": gpu.driver_version,
            "gpu_compute_capability": gpu.compute_capability,
            "deepstream_version": build_config.runtime.deepstream_version,
            "gstreamer_version": config.gstreamer_version,
            "tensorrt_version": build_config.runtime.tensorrt_python_version,
        },
        "pipeline": {
            "required_plugins": list(config.pipeline.required_plugins),
            "streammux_width": config.pipeline.streammux_width,
            "streammux_height": config.pipeline.streammux_height,
            "batch_size": config.pipeline.batch_size,
            "target_frames": config.pipeline.target_frames,
            "batched_push_timeout_us": (config.pipeline.batched_push_timeout_us),
        },
        "nvinfer": {
            "network_mode": config.nvinfer.network_mode,
            "network_type": config.nvinfer.network_type,
            "output_tensor_meta": config.nvinfer.output_tensor_meta,
            "maintain_aspect_ratio": config.nvinfer.maintain_aspect_ratio,
            "symmetric_padding": config.nvinfer.symmetric_padding,
        },
        "result": payload,
        "diagnostics": {
            "container_stdout_sha256": sha256_bytes(container_stdout.encode("utf-8")),
            "container_stderr_sha256": sha256_bytes(container_stderr.encode("utf-8")),
            "container_stdout": container_stdout,
            "container_stderr": container_stderr,
        },
        "scope": {
            "network_used": False,
            "engine_rebuilt": False,
            "application_inference_executed": True,
            "segmentation_decode_executed": False,
            "overlay_executed": False,
            "dataset_used": False,
            "validation_used": False,
            "test_used": False,
            "final_test_used": False,
        },
    }


# ADD 2026-09-05: Clean pushed foundation에서 canonical inference evidence를 기록한다.
def write_deepstream_tensorrt_inference_evidence(
    *,
    evidence_output: Path,
    repo: Path,
    inference_config_path: Path = DEFAULT_DEEPSTREAM_INFERENCE_CONFIG,
    build_config_path: Path = DEFAULT_DEEPSTREAM_TENSORRT_CONFIG,
) -> Path:
    repo = repo.resolve()
    evidence_output = evidence_output.resolve()

    config = load_deepstream_inference_config(inference_config_path)
    build_config = load_deepstream_tensorrt_config(build_config_path)

    actual_build_config_sha = sha256_file(build_config.config_path)

    if actual_build_config_sha != config.build_config_sha256:
        raise ValueError("C6-5C build config SHA-256 changed.")

    if evidence_output.is_relative_to(repo):
        raise ValueError("C6-5C canonical inference evidence must be outside repository.")

    if evidence_output.exists():
        raise FileExistsError(f"C6-5C inference evidence already exists: {evidence_output}")

    plan_path = (repo / config.engine.relative_path).resolve()

    if not plan_path.is_relative_to(repo):
        raise ValueError("C6-5C plan path escaped repository.")

    validate_engine_file(
        plan_path,
        config,
    )

    repository = resolve_repository_identity(repo)

    image = inspect_deepstream_image(build_config)

    gpu = inspect_host_gpu(build_config)

    try:
        payload, stdout, stderr = run_deepstream_tensorrt_inference(
            plan_path=plan_path,
            config=config,
            build_config=build_config,
        )

        evidence = build_inference_evidence(
            config=config,
            build_config=build_config,
            repository=repository,
            image=image,
            gpu=gpu,
            payload=payload,
            container_stdout=stdout,
            container_stderr=stderr,
        )

        evidence_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = (
            json.dumps(
                evidence,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

        with evidence_output.open("xb") as handle:
            handle.write(data)

        return evidence_output

    except Exception:
        if evidence_output.exists():
            evidence_output.unlink()

        raise
