"""C6-3B TensorRT INT8 streaming characterization on an exact T4 runtime."""

from __future__ import annotations

import importlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from ml.deployment.yolo_onnx import EXPECTED_CLASSES
from ml.deployment.yolo_tensorrt_int8_engine import (
    DEFAULT_TENSORRT_INT8_ENGINE_CONFIG,
    INT8_ENGINE_FILENAME,
    INT8_ENGINE_METADATA_FILENAME,
    Int8EngineMetadata,
    load_yolo_tensorrt_int8_engine_config,
)
from services.streaming.gstreamer_frames import (
    gst_sample_to_bgr_numpy,
    load_gstreamer_modules,
)
from shared.hashing import is_sha256_digest, sha256_file

DEFAULT_STREAMING_CHARACTERIZATION_CONFIG = Path(
    "configs/streaming/yolo_tensorrt_int8_streaming_characterization.yaml"
)
STREAMING_CHARACTERIZATION_STATE = "TENSORRT_INT8_STREAMING_METRICS_COLLECTED_ACCEPTANCE_PENDING"
EXPECTED_CHARACTERIZATION_ID = "c6_3b_yolo11n_seg_tensorrt_int8_streaming_v1"
EXPECTED_OUTPUT_ROOT = Path("outputs/streaming/yolo_gstreamer/c6_3_tensorrt_int8_characterization")

EXPECTED_PYTHON_FRAME_ACCEPTANCE_COMMIT = "1a7419ef4c074d4ac1e49fd3deba23922dd8504d"
EXPECTED_PYTHON_FRAME_VALIDATION_SHA256 = (
    "83db8f1d40bd03ab457ded7829e576e189dbbc1d3fb1fb8384be4976e71929fc"
)
EXPECTED_PYTHON_FRAME_CONFIG_SHA256 = (
    "4c20bfc683e0e20a6bdd015ddfd8ef6d24fceab3f102e390ad55facef8320fe8"
)
EXPECTED_PYTHON_FRAME_FRAME_SHA256 = (
    "e1851d821c8e04ae3f7e07e546e50a5b055b2a0ea38be00b9b4e2deac2bc852d"
)
EXPECTED_PYTHON_FRAME_ARCHIVE_SHA256 = (
    "c63860141627e2e0aa44a7cc897acb478352728fa22f7fd01f4ecd2ea087232a"
)

EXPECTED_C5_ACCEPTANCE_STATE = "TENSORRT_INT8_PARITY_ACCEPTED"
EXPECTED_C5_CLOSURE_COMMIT = "88e9b0b2440e99b6dfd2594bdc9a4947eff75187"
EXPECTED_C5_ACCEPTANCE_POLICY_SHA256 = (
    "938c06a099b681de9ac48d95132f423f5255ba4527f05d3f27f75d9eae5ad56c"
)
EXPECTED_ENGINE_EXPORT_ID = "c5_4b2_yolo11n_seg_tensorrt_int8_qdq"
EXPECTED_ENGINE_SHA256 = "4f397d59741f4efb7832087030b890a0fe059a657d074a3b07cdeb54493e8971"
EXPECTED_ENGINE_METADATA_SHA256 = "d44de78cc89fea67d6b351c2ba92f76dda0242386f4b6f14e216740ca682461e"
EXPECTED_ENGINE_CONFIG_SHA256 = "63eebcac04d11c9247bf7543fe18d0798758ab20cc734d2b18bfbece4eaf6b41"
EXPECTED_ENGINE_EVIDENCE_ZIP_SHA256 = (
    "0cba556981b12a95b25feb324d0ff02b9cadeda6bde056b46e27eb7698f66b00"
)
EXPECTED_ENGINE_BUILD_COMMIT = "7835291c8fb123eba6acfa839977f94093c2f3ac"

EXPECTED_TENSORRT_VERSION = "10.13.3.9.post1"
EXPECTED_CUDA_RUNTIME_VERSION = "12.8"
EXPECTED_GPU_NAME = "Tesla T4"
EXPECTED_GPU_COMPUTE_CAPABILITY = "7.5"
EXPECTED_TORCH_VERSION = "2.10.0+cu128"
EXPECTED_ULTRALYTICS_VERSION = "8.4.128"

EXPECTED_CHARACTERIZATION_ACCEPTANCE_STATE = "PENDING_TENSORRT_STREAMING_TOLERANCE_APPROVAL"


@dataclass(frozen=True)
class PythonFrameBoundaryIdentity:
    """Accepted C6-3 Python appsink-to-NumPy boundary identity."""

    acceptance_commit: str
    validation_sha256: str
    config_sha256: str
    frame_sha256: str
    evidence_archive_sha256: str

    # ADD 2026-09-04: TensorRT streaming이 accepted Python frame boundary를 정확히 상속하게 한다.
    def validate(self) -> None:
        expected = {
            "acceptance_commit": EXPECTED_PYTHON_FRAME_ACCEPTANCE_COMMIT,
            "validation_sha256": EXPECTED_PYTHON_FRAME_VALIDATION_SHA256,
            "config_sha256": EXPECTED_PYTHON_FRAME_CONFIG_SHA256,
            "frame_sha256": EXPECTED_PYTHON_FRAME_FRAME_SHA256,
            "evidence_archive_sha256": EXPECTED_PYTHON_FRAME_ARCHIVE_SHA256,
        }
        mismatches = [
            name
            for name, expected_value in expected.items()
            if getattr(self, name) != expected_value
        ]
        if mismatches:
            raise ValueError(
                "C6-3B Python frame boundary identity changed: " + ", ".join(mismatches)
            )
        for digest in (
            self.validation_sha256,
            self.config_sha256,
            self.frame_sha256,
            self.evidence_archive_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C6-3B Python frame identity contains invalid SHA-256.")


@dataclass(frozen=True)
class StreamingBackendIdentity:
    """Exact accepted C5-4 TensorRT INT8 backend used by C6-3B."""

    acceptance_state: str
    c5_closure_commit: str
    acceptance_policy_sha256: str
    engine_export_id: str
    engine_sha256: str
    engine_metadata_sha256: str
    engine_config_sha256: str
    engine_evidence_zip_sha256: str
    engine_build_commit: str
    engine_rebuild_allowed: bool

    # ADD 2026-09-04: C6-3B에서 engine rebuild나 accepted backend identity drift를 차단한다.
    def validate(self) -> None:
        expected = {
            "acceptance_state": EXPECTED_C5_ACCEPTANCE_STATE,
            "c5_closure_commit": EXPECTED_C5_CLOSURE_COMMIT,
            "acceptance_policy_sha256": EXPECTED_C5_ACCEPTANCE_POLICY_SHA256,
            "engine_export_id": EXPECTED_ENGINE_EXPORT_ID,
            "engine_sha256": EXPECTED_ENGINE_SHA256,
            "engine_metadata_sha256": EXPECTED_ENGINE_METADATA_SHA256,
            "engine_config_sha256": EXPECTED_ENGINE_CONFIG_SHA256,
            "engine_evidence_zip_sha256": EXPECTED_ENGINE_EVIDENCE_ZIP_SHA256,
            "engine_build_commit": EXPECTED_ENGINE_BUILD_COMMIT,
        }
        mismatches = [
            name
            for name, expected_value in expected.items()
            if getattr(self, name) != expected_value
        ]
        if mismatches:
            raise ValueError("C6-3B backend identity changed: " + ", ".join(mismatches))
        if self.engine_rebuild_allowed is not False:
            raise ValueError("C6-3B must restore the exact engine without rebuilding.")
        for digest in (
            self.acceptance_policy_sha256,
            self.engine_sha256,
            self.engine_metadata_sha256,
            self.engine_config_sha256,
            self.engine_evidence_zip_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C6-3B backend identity contains invalid SHA-256.")


@dataclass(frozen=True)
class StreamingRuntimeIdentity:
    """Exact NVIDIA runtime required by the accepted INT8 engine."""

    tensorrt_version: str
    cuda_runtime_version: str
    gpu_name: str
    gpu_compute_capability: str
    torch_version: str
    ultralytics_version: str
    device: int

    # ADD 2026-09-04: C6-3B TensorRT execution을 accepted C5-4 T4 runtime과 일치시킨다.
    def validate(self) -> None:
        expected = {
            "tensorrt_version": EXPECTED_TENSORRT_VERSION,
            "cuda_runtime_version": EXPECTED_CUDA_RUNTIME_VERSION,
            "gpu_name": EXPECTED_GPU_NAME,
            "gpu_compute_capability": EXPECTED_GPU_COMPUTE_CAPABILITY,
            "torch_version": EXPECTED_TORCH_VERSION,
            "ultralytics_version": EXPECTED_ULTRALYTICS_VERSION,
        }
        mismatches = [
            name
            for name, expected_value in expected.items()
            if getattr(self, name) != expected_value
        ]
        if mismatches:
            raise ValueError("C6-3B runtime identity changed: " + ", ".join(mismatches))
        if type(self.device) is not int or self.device != 0:
            raise ValueError("C6-3B must use CUDA device 0.")


@dataclass(frozen=True)
class StreamingSourceConfig:
    """Bounded live-like GStreamer source and backpressure settings."""

    source: str
    pattern: str
    is_live: bool
    do_timestamp: bool
    num_buffers: int
    width: int
    height: int
    framerate: int
    pixel_format: str
    appsink_name: str
    queue_max_buffers: int
    queue_leaky: str
    appsink_max_buffers: int
    appsink_drop: bool
    appsink_sync: bool

    # ADD 2026-09-04: Characterization source와 latest-frame-wins boundary를 고정한다.
    def validate(self) -> None:
        if (
            self.source != "videotestsrc"
            or self.pattern != "ball"
            or self.is_live is not True
            or self.do_timestamp is not True
            or self.num_buffers != 180
            or self.width != 640
            or self.height != 640
            or self.framerate != 30
            or self.pixel_format != "BGR"
            or self.appsink_name != "framesink"
            or self.queue_max_buffers != 1
            or self.queue_leaky != "downstream"
            or self.appsink_max_buffers != 1
            or self.appsink_drop is not True
            or self.appsink_sync is not False
        ):
            raise ValueError("C6-3B streaming source/backpressure contract changed.")


@dataclass(frozen=True)
class StreamingInferenceConfig:
    """Ultralytics TensorRT prediction settings inherited from C5."""

    imgsz: int
    conf: float
    iou: float
    max_det: int
    retina_masks: bool
    warmup_iterations: int

    # ADD 2026-09-04: C5 prediction settings과 a priori warmup boundary를 고정한다.
    def validate(self) -> None:
        if (
            self.imgsz != 640
            or self.conf != 0.001
            or self.iou != 0.7
            or self.max_det != 300
            or self.retina_masks is not False
            or self.warmup_iterations != 10
        ):
            raise ValueError("C6-3B inference settings changed without review.")


@dataclass(frozen=True)
class StreamingCharacterizationPolicy:
    """Threshold-free first characterization of the streaming TensorRT path."""

    scope: str
    metrics_only: bool
    numeric_thresholds: None
    acceptance_state: str
    dataset_used: bool
    validation_used: bool
    test_used: bool
    final_test_used: bool
    deepstream_used: bool

    # ADD 2026-09-04: 첫 streaming run을 metrics-only로 유지하고 dataset/final-test 사용을 금지한다.
    def validate(self) -> None:
        if (
            self.scope != "gstreamer_appsink_numpy_to_ultralytics_tensorrt_int8"
            or self.metrics_only is not True
            or self.numeric_thresholds is not None
            or self.acceptance_state != EXPECTED_CHARACTERIZATION_ACCEPTANCE_STATE
            or self.dataset_used is not False
            or self.validation_used is not False
            or self.test_used is not False
            or self.final_test_used is not False
            or self.deepstream_used is not False
        ):
            raise ValueError("C6-3B characterization policy changed.")


@dataclass(frozen=True)
class YoloTensorRtInt8StreamingCharacterizationConfig:
    """Top-level C6-3B TensorRT streaming characterization contract."""

    schema_version: int
    characterization_id: str
    output_root: Path
    python_frame_boundary: PythonFrameBoundaryIdentity
    backend: StreamingBackendIdentity
    runtime: StreamingRuntimeIdentity
    stream: StreamingSourceConfig
    inference: StreamingInferenceConfig
    characterization: StreamingCharacterizationPolicy
    config_path: Path

    # ADD 2026-09-04: C6-3B config 전체를 strict frozen foundation으로 검증한다.
    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported C6-3B config schema.")
        if self.characterization_id != EXPECTED_CHARACTERIZATION_ID:
            raise ValueError("Unexpected C6-3B characterization_id.")
        if self.output_root != EXPECTED_OUTPUT_ROOT:
            raise ValueError("C6-3B output_root changed.")
        self.python_frame_boundary.validate()
        self.backend.validate()
        self.runtime.validate()
        self.stream.validate()
        self.inference.validate()
        self.characterization.validate()


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return cast(dict[str, Any], value)


# ADD 2026-09-04: YAML을 strict typed C6-3B characterization config로 로드한다.
def load_streaming_characterization_config(
    path: Path,
) -> YoloTensorRtInt8StreamingCharacterizationConfig:
    raw_obj: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = _mapping(raw_obj, label="C6-3B config")
    expected = {
        "schema_version",
        "characterization_id",
        "output_root",
        "python_frame_boundary",
        "backend",
        "runtime",
        "stream",
        "inference",
        "characterization",
    }
    if set(raw) != expected:
        raise ValueError("C6-3B config fields do not match schema.")

    try:
        config = YoloTensorRtInt8StreamingCharacterizationConfig(
            schema_version=raw["schema_version"],
            characterization_id=str(raw["characterization_id"]),
            output_root=Path(str(raw["output_root"])),
            python_frame_boundary=PythonFrameBoundaryIdentity(
                **_mapping(raw["python_frame_boundary"], label="python_frame_boundary")
            ),
            backend=StreamingBackendIdentity(**_mapping(raw["backend"], label="backend")),
            runtime=StreamingRuntimeIdentity(**_mapping(raw["runtime"], label="runtime")),
            stream=StreamingSourceConfig(**_mapping(raw["stream"], label="stream")),
            inference=StreamingInferenceConfig(**_mapping(raw["inference"], label="inference")),
            characterization=StreamingCharacterizationPolicy(
                **_mapping(raw["characterization"], label="characterization")
            ),
            config_path=path.resolve(),
        )
    except TypeError as exc:
        raise ValueError("C6-3B config typed fields are invalid.") from exc
    config.validate()
    return config


# ADD 2026-09-04: Fixed live-like source를 C6-1 latest-frame-wins appsink pipeline으로 만든다.
def build_streaming_pipeline(
    config: YoloTensorRtInt8StreamingCharacterizationConfig,
) -> str:
    source = config.stream
    live = "true" if source.is_live else "false"
    timestamp = "true" if source.do_timestamp else "false"
    drop = "true" if source.appsink_drop else "false"
    sync = "true" if source.appsink_sync else "false"
    return (
        f"videotestsrc num-buffers={source.num_buffers} pattern={source.pattern} "
        f"is-live={live} do-timestamp={timestamp} "
        f"! video/x-raw,width={source.width},height={source.height},"
        f"framerate={source.framerate}/1 "
        "! videoconvert "
        f"! video/x-raw,format={source.pixel_format},width={source.width},"
        f"height={source.height},framerate={source.framerate}/1 "
        f"! queue max-size-buffers={source.queue_max_buffers} "
        "max-size-bytes=0 max-size-time=0 "
        f"leaky={source.queue_leaky} "
        f"! appsink name={source.appsink_name} emit-signals=false "
        f"max-buffers={source.appsink_max_buffers} drop={drop} sync={sync} "
        "wait-on-eos=false"
    )


# ADD 2026-09-04: Finite positive latency samples를 deterministic summary로 집계한다.
def summarize_latency_ms(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("C6-3B latency summary requires at least one value.")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or bool((array <= 0.0).any()):
        raise ValueError("C6-3B latency values must be finite and positive.")
    return {
        "count": len(values),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


# ADD 2026-09-04: Source-generated count에서 latest-frame-wins drop count/rate를 계산한다.
def summarize_frame_counts(*, source_buffers: int, processed_frames: int) -> dict[str, float | int]:
    if source_buffers <= 0 or processed_frames <= 0 or processed_frames > source_buffers:
        raise ValueError("C6-3B frame counts are invalid.")
    dropped = source_buffers - processed_frames
    return {
        "source_buffers": source_buffers,
        "processed_frames": processed_frames,
        "dropped_frames": dropped,
        "drop_rate": dropped / source_buffers,
    }


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=repo, text=True).strip()


# ADD 2026-09-04: Exact B2 engine bytes/metadata/config를 rebuild 없이 검증한다.
def verify_exact_engine_artifact(
    *,
    repo: Path,
    artifact_dir: Path,
    config: YoloTensorRtInt8StreamingCharacterizationConfig,
) -> tuple[Path, Int8EngineMetadata]:
    config.validate()
    engine_config_path = repo / DEFAULT_TENSORRT_INT8_ENGINE_CONFIG
    engine_config = load_yolo_tensorrt_int8_engine_config(engine_config_path)
    expected_dir = (repo / engine_config.output_root / engine_config.export_id).resolve()
    if artifact_dir.resolve() != expected_dir:
        raise ValueError("C6-3B engine must use the canonical ignored B2 artifact namespace.")

    engine_path = artifact_dir / INT8_ENGINE_FILENAME
    metadata_path = artifact_dir / INT8_ENGINE_METADATA_FILENAME
    if not engine_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("C6-3B exact B2 engine artifact is incomplete.")
    if sha256_file(engine_path) != config.backend.engine_sha256:
        raise ValueError("C6-3B exact engine SHA mismatch.")
    if sha256_file(metadata_path) != config.backend.engine_metadata_sha256:
        raise ValueError("C6-3B engine metadata SHA mismatch.")
    if sha256_file(engine_config_path) != config.backend.engine_config_sha256:
        raise ValueError("C6-3B engine config SHA mismatch.")

    raw_obj: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw = _mapping(raw_obj, label="C6-3B engine metadata")
    try:
        metadata = Int8EngineMetadata(**raw)
    except TypeError as exc:
        raise ValueError("C6-3B engine metadata fields are invalid.") from exc
    metadata.validate(config=engine_config)

    repository = metadata.repository
    expected_environment = {
        "tensorrt_version": config.runtime.tensorrt_version,
        "cuda_runtime_version": config.runtime.cuda_runtime_version,
        "gpu_name": config.runtime.gpu_name,
        "gpu_compute_capability": config.runtime.gpu_compute_capability,
        "torch_version": config.runtime.torch_version,
        "ultralytics_version": config.runtime.ultralytics_version,
    }
    mismatches = [
        name
        for name, expected_value in expected_environment.items()
        if metadata.environment.get(name) != expected_value
    ]
    if (
        metadata.export_id != config.backend.engine_export_id
        or metadata.engine_sha256 != config.backend.engine_sha256
        or repository.get("git_commit") != config.backend.engine_build_commit
        or repository.get("working_tree_dirty") is not False
        or metadata.validation_used is not False
        or metadata.test_used is not False
        or metadata.test_split_used is not False
        or mismatches
    ):
        raise ValueError("C6-3B exact B2 engine semantic identity changed.")
    return engine_path, metadata


# ADD 2026-09-04: Current CUDA/TensorRT runtime을 evidence용 mapping으로 수집한다.
def collect_runtime_environment(device: int) -> dict[str, str]:
    torch = importlib.import_module("torch")
    tensorrt = importlib.import_module("tensorrt")
    ultralytics = importlib.import_module("ultralytics")

    if not bool(torch.cuda.is_available()):
        raise RuntimeError("C6-3B requires CUDA.")
    if device < 0 or device >= int(torch.cuda.device_count()):
        raise RuntimeError("C6-3B CUDA device is unavailable.")
    props = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    cuda_runtime = torch.version.cuda
    if not cuda_runtime:
        raise RuntimeError("C6-3B PyTorch runtime does not expose CUDA version.")

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "python_implementation": sys.implementation.name,
        "torch_version": str(torch.__version__),
        "ultralytics_version": str(ultralytics.__version__),
        "tensorrt_version": str(tensorrt.__version__),
        "cuda_runtime_version": str(cuda_runtime),
        "gpu_name": str(props.name),
        "gpu_compute_capability": f"{capability[0]}.{capability[1]}",
        "device": f"cuda:{device}",
    }


# ADD 2026-09-04: Runtime identity가 accepted C5-4 engine environment와 정확히 일치하는지 검증한다.
def verify_runtime_environment(
    *,
    config: YoloTensorRtInt8StreamingCharacterizationConfig,
    environment: Mapping[str, str],
) -> None:
    expected = {
        "torch_version": config.runtime.torch_version,
        "ultralytics_version": config.runtime.ultralytics_version,
        "tensorrt_version": config.runtime.tensorrt_version,
        "cuda_runtime_version": config.runtime.cuda_runtime_version,
        "gpu_name": config.runtime.gpu_name,
        "gpu_compute_capability": config.runtime.gpu_compute_capability,
        "device": f"cuda:{config.runtime.device}",
    }
    mismatches = [
        name for name, expected_value in expected.items() if environment.get(name) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "C6-3B runtime differs from accepted C5-4 engine runtime: "
            + ", ".join(sorted(mismatches))
        )


# ADD 2026-09-04: Pinned Ultralytics TensorRT segmentation backend를 lazy-load한다.
def load_tensorrt_model(engine_path: Path) -> Any:
    ultralytics = importlib.import_module("ultralytics")
    model = ultralytics.YOLO(str(engine_path), task="segment")
    if model.task != "segment":
        raise RuntimeError("C6-3B engine did not load as segmentation task.")
    return model


def _predict_kwargs(
    config: YoloTensorRtInt8StreamingCharacterizationConfig,
    frame: np.ndarray,
) -> dict[str, object]:
    inference = config.inference
    return {
        "source": frame,
        "conf": inference.conf,
        "iou": inference.iou,
        "max_det": inference.max_det,
        "retina_masks": inference.retina_masks,
        "imgsz": inference.imgsz,
        "device": str(config.runtime.device),
        "save": False,
        "stream": False,
        "verbose": False,
    }


# ADD 2026-09-04: Backend initialization을 measured streaming window 밖에서 10회 warmup한다.
def warmup_tensorrt_model(
    model: Any,
    config: YoloTensorRtInt8StreamingCharacterizationConfig,
) -> None:
    torch = importlib.import_module("torch")
    frame = np.zeros(
        (config.stream.height, config.stream.width, 3),
        dtype=np.uint8,
    )
    for _ in range(config.inference.warmup_iterations):
        results = list(model.predict(**_predict_kwargs(config, frame)))
        if len(results) != 1:
            raise RuntimeError("C6-3B warmup requires exactly one result.")
    torch.cuda.synchronize(config.runtime.device)


# ADD 2026-09-04: Warmup 뒤 Ultralytics가 실제 TensorRT engine context를 사용했는지 검증한다.
def verify_loaded_tensorrt_backend(model: Any) -> None:
    names = {int(key): str(value) for key, value in model.names.items()}
    if names != EXPECTED_CLASSES:
        raise RuntimeError("C6-3B engine classes changed from bent/color/scratch.")
    backend = getattr(getattr(model, "predictor", None), "model", None)
    if backend is None:
        raise RuntimeError("C6-3B Ultralytics predictor backend was not initialized.")
    if getattr(backend, "format", None) != "engine":
        raise RuntimeError("C6-3B execution did not use TensorRT engine backend.")
    if getattr(backend, "context", None) is None:
        raise RuntimeError("C6-3B TensorRT execution context was not initialized.")


def _raise_bus_error(gst: Any, message: Any) -> None:
    error, debug = message.parse_error()
    raise RuntimeError(f"C6-3B GStreamer error: {error}; debug={debug}")


# ADD 2026-09-04: Live GStreamer appsink frames를 exact INT8 backend로 처리하고 metrics를 수집한다.
def characterize_stream(
    *,
    model: Any,
    config: YoloTensorRtInt8StreamingCharacterizationConfig,
) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    gst, _ = load_gstreamer_modules()
    gst.init(None)

    pipeline = gst.parse_launch(build_streaming_pipeline(config))
    appsink = pipeline.get_by_name(config.stream.appsink_name)
    if appsink is None:
        raise RuntimeError("C6-3B could not find configured appsink.")
    bus = pipeline.get_bus()
    if bus is None:
        raise RuntimeError("C6-3B could not access GStreamer bus.")

    state_result = pipeline.set_state(gst.State.PLAYING)
    if state_result == gst.StateChangeReturn.FAILURE:
        raise RuntimeError("C6-3B GStreamer pipeline failed to start.")

    adapter_ms: list[float] = []
    inference_ms: list[float] = []
    processing_ms: list[float] = []
    prediction_counts: list[int] = []
    stream_started = time.perf_counter()

    try:
        while True:
            sample = appsink.emit("try-pull-sample", 5 * gst.SECOND)
            if sample is None:
                message = bus.timed_pop_filtered(
                    0,
                    gst.MessageType.ERROR | gst.MessageType.EOS,
                )
                if message is not None and message.type == gst.MessageType.ERROR:
                    _raise_bus_error(gst, message)
                if bool(appsink.get_property("eos")) or (
                    message is not None and message.type == gst.MessageType.EOS
                ):
                    break
                raise RuntimeError("C6-3B appsink timed out before EOS.")

            adapter_started = time.perf_counter()
            frame = gst_sample_to_bgr_numpy(sample)
            adapter_elapsed = (time.perf_counter() - adapter_started) * 1000.0

            if (
                frame.shape != (config.stream.height, config.stream.width, 3)
                or frame.dtype != np.uint8
                or not frame.flags.c_contiguous
                or not frame.flags.owndata
            ):
                raise RuntimeError("C6-3B received a frame outside the accepted NumPy contract.")

            torch.cuda.synchronize(config.runtime.device)
            inference_started = time.perf_counter()
            results = list(model.predict(**_predict_kwargs(config, frame)))
            torch.cuda.synchronize(config.runtime.device)
            inference_elapsed = (time.perf_counter() - inference_started) * 1000.0

            if len(results) != 1:
                raise RuntimeError("C6-3B requires exactly one result per processed frame.")
            boxes = getattr(results[0], "boxes", None)
            prediction_counts.append(0 if boxes is None else len(boxes))

            adapter_ms.append(adapter_elapsed)
            inference_ms.append(inference_elapsed)
            processing_ms.append(adapter_elapsed + inference_elapsed)
    finally:
        pipeline.set_state(gst.State.NULL)

    elapsed_s = time.perf_counter() - stream_started
    processed = len(processing_ms)
    counts = summarize_frame_counts(
        source_buffers=config.stream.num_buffers,
        processed_frames=processed,
    )
    if elapsed_s <= 0.0:
        raise RuntimeError("C6-3B stream elapsed time is invalid.")

    processing_summary = summarize_latency_ms(processing_ms)
    return {
        "frame_counts": counts,
        "frame_adapter_latency_ms": summarize_latency_ms(adapter_ms),
        "inference_latency_ms": summarize_latency_ms(inference_ms),
        "processing_latency_ms": processing_summary,
        "stream_elapsed_seconds": elapsed_s,
        "observed_processed_fps": processed / elapsed_s,
        "processing_capacity_fps_from_mean": 1000.0 / float(processing_summary["mean"]),
        "source_frame_period_ms": 1000.0 / config.stream.framerate,
        "prediction_count": {
            "min": min(prediction_counts),
            "mean": float(np.mean(np.asarray(prediction_counts, dtype=np.float64))),
            "max": max(prediction_counts),
        },
    }


# ADD 2026-09-04: Clean commit에서 threshold-free streaming evidence를 생성한다.
def run_streaming_characterization(
    *,
    config: YoloTensorRtInt8StreamingCharacterizationConfig,
    repo: Path,
    engine_artifact_dir: Path,
) -> Path:
    config.validate()
    commit = _git_output(repo, "rev-parse", "HEAD")
    if _git_output(repo, "status", "--porcelain"):
        raise RuntimeError("C6-3B canonical characterization requires a clean working tree.")

    engine_path, _ = verify_exact_engine_artifact(
        repo=repo,
        artifact_dir=engine_artifact_dir,
        config=config,
    )
    engine_sha_before = sha256_file(engine_path)

    environment = collect_runtime_environment(config.runtime.device)
    verify_runtime_environment(config=config, environment=environment)

    gst, _ = load_gstreamer_modules()
    gst.init(None)
    gstreamer_version = gst.version_string()

    model = load_tensorrt_model(engine_path)
    warmup_tensorrt_model(model, config)
    verify_loaded_tensorrt_backend(model)

    metrics = characterize_stream(model=model, config=config)

    if sha256_file(engine_path) != engine_sha_before:
        raise RuntimeError("C6-3B changed exact accepted engine bytes.")

    output_dir = repo / config.output_root / config.characterization_id
    if output_dir.exists():
        raise FileExistsError("C6-3B output namespace already exists.")
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "characterization.json"

    payload = {
        "schema_version": 1,
        "stage": "C6-3B",
        "characterization_id": config.characterization_id,
        "state": STREAMING_CHARACTERIZATION_STATE,
        "created_at": datetime.now(UTC).isoformat(),
        "repository": {
            "git_commit": commit,
            "working_tree_dirty_before_run": False,
        },
        "python_frame_boundary": asdict(config.python_frame_boundary),
        "backend": {
            **asdict(config.backend),
            "engine_path": str(engine_path),
            "engine_rebuilt": False,
        },
        "runtime": {
            **environment,
            "gstreamer_version": gstreamer_version,
            "numpy_version": version("numpy"),
            "pygobject_version": version("PyGObject"),
            "pycairo_version": version("pycairo"),
        },
        "stream": asdict(config.stream),
        "inference": asdict(config.inference),
        "metrics": metrics,
        "characterization": asdict(config.characterization),
        "dataset_used": False,
        "validation_used": False,
        "test_used": False,
        "final_test_used": False,
        "deepstream_used": False,
        "engine_rebuilt": False,
    }

    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path
