"""C6-3C prospective acceptance policy for TensorRT INT8 streaming."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from shared.hashing import is_sha256_digest, sha256_file

DEFAULT_STREAMING_ACCEPTANCE_POLICY = Path(
    "configs/streaming/yolo_tensorrt_int8_streaming_acceptance.yaml"
)
ACCEPTANCE_OUTPUT_ROOT = Path(
    "outputs/streaming/yolo_gstreamer/c6_3_tensorrt_int8_streaming_acceptance"
)
EXPECTED_POLICY_ID = "c6_3c_yolo11n_seg_tensorrt_int8_streaming_acceptance_v1"
EXPECTED_CHARACTERIZATION_ID = "c6_3b_yolo11n_seg_tensorrt_int8_streaming_v1"
EXPECTED_CHARACTERIZATION_STATE = "TENSORRT_INT8_STREAMING_METRICS_COLLECTED_ACCEPTANCE_PENDING"
ACCEPTED_STATE = "TENSORRT_INT8_STREAMING_ACCEPTED"
REJECTED_STATE = "TENSORRT_INT8_STREAMING_REJECTED"

EXPECTED_CHARACTERIZATION_REPOSITORY_COMMIT = "8e982aeb011f6d1d92a90ad53e0a9541cd3c441a"
EXPECTED_CHARACTERIZATION_SHA256 = (
    "97a4c1b233354ed362d40499e9a8e4af1b678385a6ed63a3bb394d963eb5f627"
)
EXPECTED_CHARACTERIZATION_CONFIG_SHA256 = (
    "594acd505cf9ab1bdc8fbaf4028a50a8e3f475ded1f147783e844397fa3b3f8c"
)
EXPECTED_RUNTIME_PREFLIGHT_SHA256 = (
    "7d3a997c01e186121ccd5171400b83912c25ae5075e3a5ac1a56be632f54331a"
)
EXPECTED_RUN_SUMMARY_SHA256 = "f1c03e4c73a5c4dd75eb80158c5a565ace045bd1d81bdf5a1d8187e2fddad042"
EXPECTED_CHARACTERIZATION_ARCHIVE_SHA256 = (
    "f6a9f994c2efa7e38954a256fdfcdaef4792edfe46ba7c0580add59130677bb7"
)

EXPECTED_ENGINE_SHA256 = "4f397d59741f4efb7832087030b890a0fe059a657d074a3b07cdeb54493e8971"
EXPECTED_ENGINE_METADATA_SHA256 = "d44de78cc89fea67d6b351c2ba92f76dda0242386f4b6f14e216740ca682461e"
EXPECTED_ENGINE_CONFIG_SHA256 = "63eebcac04d11c9247bf7543fe18d0798758ab20cc734d2b18bfbece4eaf6b41"
EXPECTED_ENGINE_BUILD_COMMIT = "7835291c8fb123eba6acfa839977f94093c2f3ac"

EXPECTED_TENSORRT_VERSION = "10.13.3.9.post1"
EXPECTED_CUDA_RUNTIME_VERSION = "12.8"
EXPECTED_GPU_NAME = "Tesla T4"
EXPECTED_GPU_COMPUTE_CAPABILITY = "7.5"
EXPECTED_TORCH_VERSION = "2.10.0+cu128"
EXPECTED_ULTRALYTICS_VERSION = "8.4.128"


@dataclass(frozen=True)
class CharacterizationSourcePolicy:
    """Exact metrics-first C6-3B evidence used only to define thresholds."""

    characterization_id: str
    state: str
    repository_commit: str
    characterization_sha256: str
    config_sha256: str
    runtime_preflight_sha256: str
    run_summary_sha256: str
    evidence_archive_sha256: str

    # ADD 2026-09-04: C6-3C threshold provenance를 exact C6-3B evidence로 고정한다.
    def validate(self) -> None:
        expected = {
            "characterization_id": EXPECTED_CHARACTERIZATION_ID,
            "state": EXPECTED_CHARACTERIZATION_STATE,
            "repository_commit": EXPECTED_CHARACTERIZATION_REPOSITORY_COMMIT,
            "characterization_sha256": EXPECTED_CHARACTERIZATION_SHA256,
            "config_sha256": EXPECTED_CHARACTERIZATION_CONFIG_SHA256,
            "runtime_preflight_sha256": EXPECTED_RUNTIME_PREFLIGHT_SHA256,
            "run_summary_sha256": EXPECTED_RUN_SUMMARY_SHA256,
            "evidence_archive_sha256": EXPECTED_CHARACTERIZATION_ARCHIVE_SHA256,
        }
        mismatches = [name for name, value in expected.items() if getattr(self, name) != value]
        if mismatches:
            raise ValueError("C6-3C characterization provenance changed: " + ", ".join(mismatches))
        for digest in (
            self.characterization_sha256,
            self.config_sha256,
            self.runtime_preflight_sha256,
            self.run_summary_sha256,
            self.evidence_archive_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C6-3C characterization provenance contains invalid SHA-256.")


@dataclass(frozen=True)
class StreamingBackendPolicy:
    """Exact accepted TensorRT INT8 backend identity."""

    engine_sha256: str
    engine_metadata_sha256: str
    engine_config_sha256: str
    engine_build_commit: str
    engine_rebuild_required_false: bool

    # ADD 2026-09-04: Prospective streaming acceptance가 exact accepted C5 engine만 사용하게 한다.
    def validate(self) -> None:
        expected = {
            "engine_sha256": EXPECTED_ENGINE_SHA256,
            "engine_metadata_sha256": EXPECTED_ENGINE_METADATA_SHA256,
            "engine_config_sha256": EXPECTED_ENGINE_CONFIG_SHA256,
            "engine_build_commit": EXPECTED_ENGINE_BUILD_COMMIT,
        }
        mismatches = [name for name, value in expected.items() if getattr(self, name) != value]
        if mismatches:
            raise ValueError("C6-3C backend identity changed: " + ", ".join(mismatches))
        if self.engine_rebuild_required_false is not True:
            raise ValueError("C6-3C must require engine_rebuilt=false.")
        for digest in (
            self.engine_sha256,
            self.engine_metadata_sha256,
            self.engine_config_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C6-3C backend identity contains invalid SHA-256.")


@dataclass(frozen=True)
class StreamingRuntimePolicy:
    """Hardware/software identity covered by policy v1."""

    tensorrt_version: str
    cuda_runtime_version: str
    gpu_name: str
    gpu_compute_capability: str
    torch_version: str
    ultralytics_version: str

    # ADD 2026-09-04: C6-3C를 accepted C5-4 T4 runtime identity에 묶는다.
    def validate(self) -> None:
        expected = {
            "tensorrt_version": EXPECTED_TENSORRT_VERSION,
            "cuda_runtime_version": EXPECTED_CUDA_RUNTIME_VERSION,
            "gpu_name": EXPECTED_GPU_NAME,
            "gpu_compute_capability": EXPECTED_GPU_COMPUTE_CAPABILITY,
            "torch_version": EXPECTED_TORCH_VERSION,
            "ultralytics_version": EXPECTED_ULTRALYTICS_VERSION,
        }
        mismatches = [name for name, value in expected.items() if getattr(self, name) != value]
        if mismatches:
            raise ValueError("C6-3C runtime identity changed: " + ", ".join(mismatches))


@dataclass(frozen=True)
class StreamingSourcePolicy:
    """Synthetic live-like source contract for prospective verification."""

    source_buffers: int
    framerate: int
    width: int
    height: int
    pixel_format: str
    latest_frame_wins: bool

    # ADD 2026-09-04: C6-3D source/backpressure boundary를 동일하게 유지한다.
    def validate(self) -> None:
        if (
            self.source_buffers != 180
            or self.framerate != 30
            or self.width != 640
            or self.height != 640
            or self.pixel_format != "BGR"
            or self.latest_frame_wins is not True
        ):
            raise ValueError("C6-3C streaming source policy changed.")


@dataclass(frozen=True)
class StreamingStructuralPolicy:
    """Fail-closed non-performance gates for prospective execution."""

    require_clean_repository: bool
    require_evidence_commit_match_current_repository: bool
    require_dataset_used_false: bool
    require_validation_used_false: bool
    require_test_used_false: bool
    require_final_test_used_false: bool
    require_deepstream_used_false: bool
    require_engine_rebuilt_false: bool

    # ADD 2026-09-04: Fresh clean-commit과 sealed-data constraints를 고정한다.
    def validate(self) -> None:
        values = tuple(asdict(self).values())
        if any(type(value) is not bool or value is not True for value in values):
            raise ValueError("C6-3C structural requirements must all be true.")


@dataclass(frozen=True)
class StreamingPerformancePolicy:
    """Approved real-time thresholds defined after metrics-only C6-3B."""

    max_drop_rate: float
    min_processed_frames: int
    min_observed_processed_fps: float
    max_frame_adapter_p95_ms: float
    max_inference_mean_ms: float
    max_inference_p95_ms: float
    max_processing_mean_ms: float
    max_processing_p95_ms: float
    min_processing_capacity_fps: float
    require_processing_p95_below_source_period: bool

    # ADD 2026-09-04: C6-3B 관측치에 headroom을 둔 30 FPS real-time policy를 고정한다.
    def validate(self) -> None:
        numeric = (
            self.max_drop_rate,
            self.min_observed_processed_fps,
            self.max_frame_adapter_p95_ms,
            self.max_inference_mean_ms,
            self.max_inference_p95_ms,
            self.max_processing_mean_ms,
            self.max_processing_p95_ms,
            self.min_processing_capacity_fps,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric
        ):
            raise TypeError("C6-3C performance thresholds must be finite numbers.")
        if type(self.min_processed_frames) is not int:
            raise TypeError("C6-3C min_processed_frames must be int.")
        expected = (
            float(self.max_drop_rate) == 0.01
            and self.min_processed_frames == 179
            and float(self.min_observed_processed_fps) == 29.0
            and float(self.max_frame_adapter_p95_ms) == 1.5
            and float(self.max_inference_mean_ms) == 13.0
            and float(self.max_inference_p95_ms) == 15.0
            and float(self.max_processing_mean_ms) == 14.0
            and float(self.max_processing_p95_ms) == 16.0
            and float(self.min_processing_capacity_fps) == 70.0
            and self.require_processing_p95_below_source_period is True
        )
        if not expected:
            raise ValueError("C6-3C performance policy changed without review.")


@dataclass(frozen=True)
class StreamingAcceptancePolicy:
    """Top-level C6-3C frozen policy."""

    schema_version: int
    policy_id: str
    output_root: Path
    characterization_source: CharacterizationSourcePolicy
    backend: StreamingBackendPolicy
    runtime: StreamingRuntimePolicy
    stream: StreamingSourcePolicy
    structural: StreamingStructuralPolicy
    performance: StreamingPerformancePolicy
    config_path: Path

    # ADD 2026-09-04: Policy schema/identity와 모든 nested constraints를 검증한다.
    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported C6-3C policy schema.")
        if self.policy_id != EXPECTED_POLICY_ID:
            raise ValueError("Unexpected C6-3C policy_id.")
        if self.output_root != ACCEPTANCE_OUTPUT_ROOT:
            raise ValueError("C6-3C output_root changed.")
        self.characterization_source.validate()
        self.backend.validate()
        self.runtime.validate()
        self.stream.validate()
        self.structural.validate()
        self.performance.validate()


@dataclass(frozen=True)
class AcceptanceGate:
    """One prospective acceptance decision."""

    name: str
    passed: bool
    observed: float | int | str | bool
    requirement: str


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return cast(dict[str, Any], value)


# ADD 2026-09-04: Repository-owned YAML을 strict C6-3C policy로 로드한다.
def load_streaming_acceptance_policy(path: Path) -> StreamingAcceptancePolicy:
    raw_obj: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = _mapping(raw_obj, label="C6-3C policy")
    expected = {
        "schema_version",
        "policy_id",
        "output_root",
        "characterization_source",
        "backend",
        "runtime",
        "stream",
        "structural",
        "performance",
    }
    if set(raw) != expected:
        raise ValueError("C6-3C policy fields do not match schema.")

    try:
        policy = StreamingAcceptancePolicy(
            schema_version=raw["schema_version"],
            policy_id=str(raw["policy_id"]),
            output_root=Path(str(raw["output_root"])),
            characterization_source=CharacterizationSourcePolicy(
                **_mapping(raw["characterization_source"], label="characterization_source")
            ),
            backend=StreamingBackendPolicy(**_mapping(raw["backend"], label="backend")),
            runtime=StreamingRuntimePolicy(**_mapping(raw["runtime"], label="runtime")),
            stream=StreamingSourcePolicy(**_mapping(raw["stream"], label="stream")),
            structural=StreamingStructuralPolicy(**_mapping(raw["structural"], label="structural")),
            performance=StreamingPerformancePolicy(
                **_mapping(raw["performance"], label="performance")
            ),
            config_path=path.resolve(),
        )
    except TypeError as exc:
        raise ValueError("C6-3C policy typed fields are invalid.") from exc
    policy.validate()
    return policy


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=repo, text=True).strip()


def _metric(mapping: dict[str, Any], *path: str) -> float:
    value: object = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError("C6-3C evidence metric path is missing: " + ".".join(path))
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("C6-3C evidence metric must be numeric: " + ".".join(path))
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("C6-3C evidence metric must be finite: " + ".".join(path))
    return number


def _integer_metric(mapping: dict[str, Any], *path: str) -> int:
    value: object = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError("C6-3C evidence metric path is missing: " + ".".join(path))
        value = value[key]
    if type(value) is not int:
        raise ValueError("C6-3C evidence metric must be int: " + ".".join(path))
    return value


# ADD 2026-09-04: Fresh characterization을 structural/performance prospective gates로 평가한다.
def evaluate_streaming_acceptance(
    *,
    policy: StreamingAcceptancePolicy,
    evidence_path: Path,
    repo: Path,
) -> dict[str, Any]:
    policy.validate()
    if _git_output(repo, "status", "--porcelain"):
        raise RuntimeError("C6-3D acceptance requires a clean repository.")
    current_commit = _git_output(repo, "rev-parse", "HEAD")

    raw_obj: object = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence = _mapping(raw_obj, label="C6-3D characterization evidence")

    if evidence.get("characterization_id") != EXPECTED_CHARACTERIZATION_ID:
        raise ValueError("C6-3D characterization_id changed.")
    if evidence.get("state") != EXPECTED_CHARACTERIZATION_STATE:
        raise ValueError("C6-3D evidence state is not acceptance-pending.")

    repository = _mapping(evidence.get("repository"), label="repository")
    backend = _mapping(evidence.get("backend"), label="backend")
    runtime = _mapping(evidence.get("runtime"), label="runtime")
    stream = _mapping(evidence.get("stream"), label="stream")
    metrics = _mapping(evidence.get("metrics"), label="metrics")
    frame_counts = _mapping(metrics.get("frame_counts"), label="frame_counts")

    structural_checks = {
        "repository_clean_before_run": repository.get("working_tree_dirty_before_run") is False,
        "evidence_commit_matches_current_repository": repository.get("git_commit")
        == current_commit,
        "exact_engine_sha": backend.get("engine_sha256") == policy.backend.engine_sha256,
        "engine_not_rebuilt": evidence.get("engine_rebuilt") is False
        and backend.get("engine_rebuilt") is False,
        "dataset_unused": evidence.get("dataset_used") is False,
        "validation_unused": evidence.get("validation_used") is False,
        "test_unused": evidence.get("test_used") is False,
        "final_test_unused": evidence.get("final_test_used") is False,
        "deepstream_unused": evidence.get("deepstream_used") is False,
        "runtime_tensorrt": runtime.get("tensorrt_version") == policy.runtime.tensorrt_version,
        "runtime_cuda": runtime.get("cuda_runtime_version") == policy.runtime.cuda_runtime_version,
        "runtime_gpu": runtime.get("gpu_name") == policy.runtime.gpu_name,
        "runtime_compute_capability": runtime.get("gpu_compute_capability")
        == policy.runtime.gpu_compute_capability,
        "runtime_torch": runtime.get("torch_version") == policy.runtime.torch_version,
        "runtime_ultralytics": runtime.get("ultralytics_version")
        == policy.runtime.ultralytics_version,
        "source_buffer_count": frame_counts.get("source_buffers") == policy.stream.source_buffers,
        "stream_framerate": stream.get("framerate") == policy.stream.framerate,
        "stream_width": stream.get("width") == policy.stream.width,
        "stream_height": stream.get("height") == policy.stream.height,
        "stream_pixel_format": stream.get("pixel_format") == policy.stream.pixel_format,
        "latest_frame_wins": stream.get("queue_max_buffers") == 1
        and stream.get("queue_leaky") == "downstream"
        and stream.get("appsink_max_buffers") == 1
        and stream.get("appsink_drop") is True
        and stream.get("appsink_sync") is False,
    }

    gates: list[AcceptanceGate] = [
        AcceptanceGate(name, passed, passed, "must be true")
        for name, passed in structural_checks.items()
    ]

    perf = policy.performance
    drop_rate = _metric(metrics, "frame_counts", "drop_rate")
    processed = _integer_metric(metrics, "frame_counts", "processed_frames")
    observed_fps = _metric(metrics, "observed_processed_fps")
    adapter_p95 = _metric(metrics, "frame_adapter_latency_ms", "p95")
    inference_mean = _metric(metrics, "inference_latency_ms", "mean")
    inference_p95 = _metric(metrics, "inference_latency_ms", "p95")
    processing_mean = _metric(metrics, "processing_latency_ms", "mean")
    processing_p95 = _metric(metrics, "processing_latency_ms", "p95")
    capacity_fps = _metric(metrics, "processing_capacity_fps_from_mean")
    source_period = _metric(metrics, "source_frame_period_ms")

    gates.extend(
        (
            AcceptanceGate(
                "drop_rate",
                drop_rate <= perf.max_drop_rate,
                drop_rate,
                f"<= {perf.max_drop_rate}",
            ),
            AcceptanceGate(
                "processed_frames",
                processed >= perf.min_processed_frames,
                processed,
                f">= {perf.min_processed_frames}",
            ),
            AcceptanceGate(
                "observed_processed_fps",
                observed_fps >= perf.min_observed_processed_fps,
                observed_fps,
                f">= {perf.min_observed_processed_fps}",
            ),
            AcceptanceGate(
                "frame_adapter_p95_ms",
                adapter_p95 <= perf.max_frame_adapter_p95_ms,
                adapter_p95,
                f"<= {perf.max_frame_adapter_p95_ms}",
            ),
            AcceptanceGate(
                "inference_mean_ms",
                inference_mean <= perf.max_inference_mean_ms,
                inference_mean,
                f"<= {perf.max_inference_mean_ms}",
            ),
            AcceptanceGate(
                "inference_p95_ms",
                inference_p95 <= perf.max_inference_p95_ms,
                inference_p95,
                f"<= {perf.max_inference_p95_ms}",
            ),
            AcceptanceGate(
                "processing_mean_ms",
                processing_mean <= perf.max_processing_mean_ms,
                processing_mean,
                f"<= {perf.max_processing_mean_ms}",
            ),
            AcceptanceGate(
                "processing_p95_ms",
                processing_p95 <= perf.max_processing_p95_ms,
                processing_p95,
                f"<= {perf.max_processing_p95_ms}",
            ),
            AcceptanceGate(
                "processing_capacity_fps",
                capacity_fps >= perf.min_processing_capacity_fps,
                capacity_fps,
                f">= {perf.min_processing_capacity_fps}",
            ),
            AcceptanceGate(
                "processing_p95_below_source_period",
                processing_p95 < source_period,
                processing_p95,
                f"< source period {source_period}",
            ),
        )
    )

    accepted = all(gate.passed for gate in gates)
    payload = {
        "schema_version": 1,
        "stage": "C6-3D",
        "policy_id": policy.policy_id,
        "policy_sha256": sha256_file(policy.config_path),
        "state": ACCEPTED_STATE if accepted else REJECTED_STATE,
        "created_at": datetime.now(UTC).isoformat(),
        "repository": {
            "git_commit": current_commit,
            "working_tree_dirty": False,
        },
        "prospective_characterization_sha256": sha256_file(evidence_path),
        "characterization_source_for_threshold_design": asdict(policy.characterization_source),
        "engine_sha256": policy.backend.engine_sha256,
        "gates": [asdict(gate) for gate in gates],
        "passed_gate_count": sum(gate.passed for gate in gates),
        "total_gate_count": len(gates),
        "all_gates_passed": accepted,
        "dataset_used": False,
        "validation_used": False,
        "test_used": False,
        "final_test_used": False,
        "deepstream_used": False,
        "engine_rebuilt": False,
    }
    return payload


# ADD 2026-09-04: Acceptance decision을 ignored output namespace에 strict JSON으로 저장한다.
def write_streaming_acceptance(
    *,
    policy: StreamingAcceptancePolicy,
    evidence_path: Path,
    repo: Path,
) -> Path:
    result = evaluate_streaming_acceptance(
        policy=policy,
        evidence_path=evidence_path,
        repo=repo,
    )
    output_dir = repo / policy.output_root / policy.policy_id
    if output_dir.exists():
        raise FileExistsError("C6-3D acceptance output namespace already exists.")
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "acceptance.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path
