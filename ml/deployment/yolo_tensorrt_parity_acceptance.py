"""Frozen post-characterization acceptance gate for YOLO TensorRT FP16 parity."""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Never, cast

import yaml

from ml.evaluation.final_benchmark import RepositoryProvenance, resolve_repository_provenance
from ml.training.yolo_segmentation import validate_artifact_id
from shared.hashing import is_sha256_digest, sha256_file

ACCEPTANCE_SCHEMA_VERSION = 1
ACCEPTANCE_RESULT_FILENAME = "acceptance.json"
ACCEPTED_STATE = "TENSORRT_FP16_PARITY_ACCEPTED"
REJECTED_STATE = "TENSORRT_FP16_PARITY_REJECTED"
EXPECTED_EVIDENCE_STATE = "TENSORRT_FP16_METRICS_COLLECTED_ACCEPTANCE_PENDING"
EXPECTED_PENDING_STATE = "PENDING_TENSORRT_FP16_TOLERANCE_APPROVAL"
DEFAULT_ACCEPTANCE_POLICY = Path("configs/deployment/yolo_tensorrt_fp16_parity_acceptance.yaml")
EXPECTED_POLICY_ID = "c5_3_yolo_tensorrt_fp16_parity_v1"
EXPECTED_SOURCE_EXPERIMENT_ID = "c4_2c_yolo11n_seg_crop350_nomosaic_maskratio2_seed42"
EXPECTED_FROZEN_MANIFEST_SHA256 = "2a26b1bc03a1876f828e12a625c69c76af5e8c5713e3f64be699feffe2e8aa09"
EXPECTED_SOURCE_MODEL_SHA256 = "e3fd10cdd708d31421feacfc5d694cb638e0ea60672e08796391b33aecf67155"
EXPECTED_ONNX_SHA256 = "f916325bb126d174de9c1fdfc24802eec11c46014f723fbf3ba3b3c1755c1490"
EXPECTED_TENSORRT_CONFIG_SHA256 = "edc135932e9367f67b9179dbbd47b01da6fa07db878a7f8af73b491718b517c9"
EXPECTED_ENGINE_SHA256 = "9bbbe5297e6cc55bcea877a79f45485ee7e1e5e6a831ad5276aedc8e3d904037"
EXPECTED_ENGINE_BUILD_COMMIT = "5604219d07bf384f46f2827f4da999781832e183"
ACCEPTANCE_OUTPUT_ROOT = Path("outputs/deployment/yolo_segmentation/tensorrt_parity_acceptance")

Scalar = bool | int | float | str


@dataclass(frozen=True)
class TensorRtBackendPolicy:
    """Reference and candidate runtime identities covered by TensorRT policy v1."""

    reference: str
    candidate: str

    # ADD 2026-09-02: TensorRT acceptance backend pair를 GPU FP32↔FP16 contract로 고정한다.
    def validate(self) -> None:
        if self.reference != "pytorch_fp32_gpu" or self.candidate != "tensorrt_fp16_gpu":
            raise ValueError("C5-3 acceptance backend contract changed without review.")


@dataclass(frozen=True)
class TensorRtIdentityPolicy:
    """Exact frozen source, ONNX, TensorRT config, and engine identities."""

    source_experiment_id: str
    frozen_manifest_sha256: str
    source_model_sha256: str
    onnx_sha256: str
    tensorrt_config_sha256: str
    engine_sha256: str
    engine_build_repository_commit: str

    # ADD 2026-09-02: Characterization에 사용한 exact source/engine identity만 허용한다.
    def validate(self) -> None:
        expected = {
            "source_experiment_id": EXPECTED_SOURCE_EXPERIMENT_ID,
            "frozen_manifest_sha256": EXPECTED_FROZEN_MANIFEST_SHA256,
            "source_model_sha256": EXPECTED_SOURCE_MODEL_SHA256,
            "onnx_sha256": EXPECTED_ONNX_SHA256,
            "tensorrt_config_sha256": EXPECTED_TENSORRT_CONFIG_SHA256,
            "engine_sha256": EXPECTED_ENGINE_SHA256,
            "engine_build_repository_commit": EXPECTED_ENGINE_BUILD_COMMIT,
        }
        mismatches = [
            name
            for name, expected_value in expected.items()
            if getattr(self, name) != expected_value
        ]
        if mismatches:
            raise ValueError(
                "C5-3 acceptance identity changed without review: " + ", ".join(sorted(mismatches))
            )
        for digest in (
            self.frozen_manifest_sha256,
            self.source_model_sha256,
            self.onnx_sha256,
            self.tensorrt_config_sha256,
            self.engine_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C5-3 acceptance identity contains invalid SHA-256.")
        if len(self.engine_build_repository_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.engine_build_repository_commit
        ):
            raise ValueError("C5-3 engine build repository commit is invalid.")


@dataclass(frozen=True)
class TensorRtRuntimePolicy:
    """GPU/runtime identity required for hardware-specific TensorRT verification."""

    tensorrt_version: str
    cuda_runtime_version: str
    gpu_name: str
    gpu_compute_capability: str
    torch_version: str
    ultralytics_version: str

    # ADD 2026-09-02: Characterization runtime identity를 T4/CUDA13/TRT10.13으로 고정한다.
    def validate(self) -> None:
        expected = {
            "tensorrt_version": "10.13.3.9.post1",
            "cuda_runtime_version": "13.0",
            "gpu_name": "Tesla T4",
            "gpu_compute_capability": "7.5",
            "torch_version": "2.13.0+cu130",
            "ultralytics_version": "8.4.128",
        }
        mismatches = [
            name
            for name, expected_value in expected.items()
            if getattr(self, name) != expected_value
        ]
        if mismatches:
            raise ValueError(
                "C5-3 acceptance runtime changed without review: " + ", ".join(sorted(mismatches))
            )


@dataclass(frozen=True)
class TensorRtStructuralPolicy:
    """Fail-closed validation-only structural equivalence requirements."""

    required_split: str
    required_sample_count: int
    require_test_used_false: bool
    require_test_split_used_false: bool
    require_structural_gates_passed: bool
    require_prediction_count_match: bool
    require_zero_unmatched: bool
    require_class_agreement_rate: float
    require_all_tensors_finite: bool
    require_evidence_commit_match_policy: bool

    # ADD 2026-09-02: Validation-only structural requirements와 prospective boundary를 고정한다.
    def validate(self) -> None:
        boolean_values = (
            self.require_test_used_false,
            self.require_test_split_used_false,
            self.require_structural_gates_passed,
            self.require_prediction_count_match,
            self.require_zero_unmatched,
            self.require_all_tensors_finite,
            self.require_evidence_commit_match_policy,
        )
        if any(type(value) is not bool for value in boolean_values):
            raise TypeError("C5-3 acceptance structural flags must be booleans.")
        if type(self.required_sample_count) is not int:
            raise TypeError("C5-3 required sample count must be an integer.")
        if type(self.require_class_agreement_rate) not in {int, float}:
            raise TypeError("C5-3 class agreement requirement must be numeric.")
        if (
            self.required_split != "val"
            or self.required_sample_count != 28
            or any(value is not True for value in boolean_values)
            or float(self.require_class_agreement_rate) != 1.0
        ):
            raise ValueError("C5-3 acceptance structural policy changed without review.")


@dataclass(frozen=True)
class TensorRtNumericPolicy:
    """Approved FP16 parity tolerances defined after C5-3B characterization."""

    max_confidence_abs_error: float
    min_box_iou: float
    min_mask_iou: float

    # ADD 2026-09-02: Characterization 이후 승인한 FP16 parity tolerance를 고정한다.
    def validate(self) -> None:
        values = (
            self.max_confidence_abs_error,
            self.min_box_iou,
            self.min_mask_iou,
        )
        if any(
            type(value) not in {int, float} or not math.isfinite(float(value)) for value in values
        ):
            raise TypeError("C5-3 acceptance numeric tolerances must be finite numbers.")
        if (
            float(self.max_confidence_abs_error) != 0.01
            or float(self.min_box_iou) != 0.98
            or float(self.min_mask_iou) != 0.995
        ):
            raise ValueError("C5-3 acceptance numeric policy changed without review.")


@dataclass(frozen=True)
class TensorRtPerformancePolicy:
    """Deployment-performance requirement kept separate from numeric parity."""

    benchmark_scope: str
    required_warmup_iterations: int
    required_measured_iterations: int
    require_candidate_faster: bool
    min_speedup_ratio: float

    # ADD 2026-09-02: End-to-end benchmark scope와 최소 유의미 speedup을 고정한다.
    def validate(self) -> None:
        if (
            type(self.required_warmup_iterations) is not int
            or type(self.required_measured_iterations) is not int
            or type(self.require_candidate_faster) is not bool
            or type(self.min_speedup_ratio) not in {int, float}
        ):
            raise TypeError("C5-3 performance policy fields have invalid types.")
        if (
            self.benchmark_scope != "ultralytics_end_to_end_single_image"
            or self.required_warmup_iterations != 10
            or self.required_measured_iterations != 50
            or self.require_candidate_faster is not True
            or float(self.min_speedup_ratio) != 1.05
        ):
            raise ValueError("C5-3 performance policy changed without review.")


@dataclass(frozen=True)
class YoloTensorRtParityAcceptancePolicy:
    """Repository-owned TensorRT FP16 policy frozen after characterization."""

    schema_version: int
    policy_id: str
    backend: TensorRtBackendPolicy
    identity: TensorRtIdentityPolicy
    runtime: TensorRtRuntimePolicy
    structural: TensorRtStructuralPolicy
    numeric: TensorRtNumericPolicy
    performance: TensorRtPerformancePolicy
    output_root: Path
    config_path: Path

    # ADD 2026-09-02: Repository-owned TensorRT acceptance policy 전체 contract를 검증한다.
    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != ACCEPTANCE_SCHEMA_VERSION:
            raise ValueError("Unsupported C5-3 acceptance policy schema version.")
        validate_artifact_id(self.policy_id)
        if self.policy_id != EXPECTED_POLICY_ID:
            raise ValueError("C5-3 acceptance policy id changed without review.")
        if self.output_root != ACCEPTANCE_OUTPUT_ROOT:
            raise ValueError("C5-3 acceptance output_root must remain in ignored outputs/.")
        self.backend.validate()
        self.identity.validate()
        self.runtime.validate()
        self.structural.validate()
        self.numeric.validate()
        self.performance.validate()


@dataclass(frozen=True)
class AcceptanceCheck:
    """One explicit identity, runtime, structural, numeric, or performance decision."""

    name: str
    category: str
    comparator: str
    expected: Scalar
    observed: Scalar
    passed: bool

    # ADD 2026-09-02: 개별 acceptance check를 finite strict scalar evidence로 검증한다.
    def validate(self) -> None:
        categories = {"identity", "runtime", "structural", "numeric", "performance"}
        if not self.name or self.category not in categories or not self.comparator:
            raise ValueError("C5-3 acceptance check metadata is invalid.")
        if type(self.passed) is not bool:
            raise ValueError("C5-3 acceptance check result must be boolean.")
        for value in (self.expected, self.observed):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("C5-3 acceptance check contains NaN or Inf.")


@dataclass(frozen=True)
class YoloTensorRtParityAcceptanceResult:
    """Deterministic result of applying the committed policy to prospective evidence."""

    schema_version: int
    policy_id: str
    state: str
    accepted: bool
    parity_id: str
    policy_sha256: str
    parity_evidence_sha256: str
    source_experiment_id: str
    frozen_manifest_sha256: str
    source_model_sha256: str
    onnx_sha256: str
    tensorrt_config_sha256: str
    engine_sha256: str
    split: str
    test_used: bool
    test_split_used: bool
    checks: tuple[AcceptanceCheck, ...]
    evidence_repository: Mapping[str, str | bool]
    policy_repository: Mapping[str, str | bool]

    # ADD 2026-09-02: Acceptance result의 state/check/provenance 정합성을 검증한다.
    def validate(self) -> None:
        if self.schema_version != ACCEPTANCE_SCHEMA_VERSION:
            raise ValueError("C5-3 acceptance result schema is invalid.")
        if self.policy_id != EXPECTED_POLICY_ID:
            raise ValueError("C5-3 acceptance result policy id is invalid.")
        validate_artifact_id(self.parity_id)
        if self.state not in {ACCEPTED_STATE, REJECTED_STATE} or type(self.accepted) is not bool:
            raise ValueError("C5-3 acceptance result lifecycle is invalid.")
        if not self.checks:
            raise ValueError("C5-3 acceptance result requires explicit checks.")
        expected_accepted = all(check.passed for check in self.checks)
        if self.accepted != expected_accepted:
            raise ValueError("C5-3 acceptance result is inconsistent with its checks.")
        if self.state != (ACCEPTED_STATE if self.accepted else REJECTED_STATE):
            raise ValueError("C5-3 acceptance state is inconsistent with accepted flag.")
        for digest in (
            self.policy_sha256,
            self.parity_evidence_sha256,
            self.frozen_manifest_sha256,
            self.source_model_sha256,
            self.onnx_sha256,
            self.tensorrt_config_sha256,
            self.engine_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C5-3 acceptance result contains invalid SHA-256.")
        if self.split != "val" or self.test_used is not False or self.test_split_used is not False:
            raise ValueError("C5-3 acceptance result violated the validation-only test seal.")
        for mapping, label in (
            (self.evidence_repository, "evidence"),
            (self.policy_repository, "policy"),
        ):
            if set(mapping) != {"git_commit", "working_tree_dirty"}:
                raise ValueError(f"C5-3 {label} repository provenance fields are invalid.")
            if type(mapping["working_tree_dirty"]) is not bool:
                raise ValueError(f"C5-3 {label} working_tree_dirty must be boolean.")
        if cast(bool, self.policy_repository["working_tree_dirty"]):
            raise ValueError("Official C5-3 acceptance requires a clean policy repository.")
        for check in self.checks:
            check.validate()
        try:
            json.dumps(asdict(self), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("C5-3 acceptance result must be strict JSON data.") from exc

    # ADD 2026-09-02: Acceptance result를 deterministic strict JSON bytes로 직렬화한다.
    def to_json_bytes(self) -> bytes:
        self.validate()
        return (json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


# ADD 2026-09-02: Untrusted JSON/YAML object의 field set을 dataclass schema와 대조한다.
def _require_fields(
    raw: Mapping[str, Any],
    cls: type[Any],
    *,
    excluded: set[str] | None = None,
    label: str,
) -> None:
    excluded = excluded or set()
    expected = {field.name for field in fields(cls)} - excluded
    if set(raw) != expected:
        raise ValueError(f"{label} fields do not match the schema.")


# ADD 2026-09-02: Python JSON decoder의 NaN/Infinity extension을 fail closed한다.
def _reject_json_constant(value: str) -> Never:
    raise ValueError(f"Strict JSON rejects non-finite constant: {value}")


# ADD 2026-09-02: Saved TensorRT parity JSON을 strict object로 로드한다.
def _strict_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Cannot read strict C5-3 parity evidence JSON.") from exc
    if not isinstance(raw, dict):
        raise ValueError("C5-3 parity evidence JSON root must be an object.")
    return cast(dict[str, Any], raw)


# ADD 2026-09-02: Numeric evidence를 bool과 non-finite 값 없이 strict float로 읽는다.
def _number(value: object, *, label: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{label} must be numeric.")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


# ADD 2026-09-02: Count evidence를 strict non-negative integer로 읽는다.
def _count(value: object, *, label: str) -> int:
    if type(value) is not int or cast(int, value) < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return cast(int, value)


# ADD 2026-09-02: Mapping evidence를 typed mapping으로 제한한다.
def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    return cast(dict[str, Any], value)


# ADD 2026-09-02: Sequence evidence를 문자열과 혼동하지 않고 array로 제한한다.
def _array(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{label} must be an array.")
    return cast(Sequence[object], value)


# ADD 2026-09-02: Nested sample evidence의 count conservation과 finite tensor seal을 검증한다.
def _validate_samples(raw_samples: object, *, expected_split: str) -> tuple[int, bool]:
    samples = _array(raw_samples, label="C5-3 parity samples")
    all_finite = True
    for raw_sample in samples:
        sample = _mapping(raw_sample, label="C5-3 parity sample")
        expected_fields = {
            "sample_id",
            "split",
            "pytorch_prediction_count",
            "tensorrt_prediction_count",
            "matched_instance_count",
            "unmatched_pytorch_count",
            "unmatched_tensorrt_count",
            "pytorch_tensors",
            "tensorrt_tensors",
            "pytorch_predictions",
            "tensorrt_predictions",
            "matches",
        }
        if set(sample) != expected_fields:
            raise ValueError("C5-3 parity sample fields do not match the schema.")
        if not isinstance(sample["sample_id"], str) or not sample["sample_id"]:
            raise ValueError("C5-3 parity sample requires a non-empty sample_id.")
        if sample["split"] != expected_split:
            raise ValueError("C5-3 parity sample escaped the validation split.")
        pytorch_count = _count(
            sample["pytorch_prediction_count"],
            label="PyTorch sample prediction count",
        )
        tensorrt_count = _count(
            sample["tensorrt_prediction_count"],
            label="TensorRT sample prediction count",
        )
        matched_count = _count(
            sample["matched_instance_count"],
            label="Matched sample prediction count",
        )
        unmatched_pytorch = _count(
            sample["unmatched_pytorch_count"],
            label="Unmatched PyTorch sample count",
        )
        unmatched_tensorrt = _count(
            sample["unmatched_tensorrt_count"],
            label="Unmatched TensorRT sample count",
        )
        if (
            matched_count + unmatched_pytorch != pytorch_count
            or matched_count + unmatched_tensorrt != tensorrt_count
        ):
            raise ValueError("C5-3 parity sample counts are not conserved.")
        if len(_array(sample["pytorch_predictions"], label="PyTorch predictions")) != pytorch_count:
            raise ValueError("C5-3 PyTorch prediction observations do not match sample count.")
        if (
            len(_array(sample["tensorrt_predictions"], label="TensorRT predictions"))
            != tensorrt_count
        ):
            raise ValueError("C5-3 TensorRT prediction observations do not match sample count.")
        if len(_array(sample["matches"], label="Parity matches")) != matched_count:
            raise ValueError("C5-3 parity matches do not match sample count.")
        for tensor_group in ("pytorch_tensors", "tensorrt_tensors"):
            for raw_tensor in _array(sample[tensor_group], label=tensor_group):
                tensor = _mapping(raw_tensor, label="C5-3 tensor observation")
                finite = tensor.get("finite")
                if type(finite) is not bool:
                    raise ValueError("C5-3 tensor finite flag must be boolean.")
                all_finite = all_finite and cast(bool, finite)
    return len(samples), all_finite


# ADD 2026-09-02: Distribution evidence에서 count와 requested finite metric을 읽는다.
def _distribution(
    raw: object,
    *,
    metric: str,
    label: str,
) -> tuple[int, float]:
    values = _mapping(raw, label=label)
    expected = {"count", "min", "mean", "max"}
    if set(values) != expected:
        raise ValueError(f"{label} fields do not match the schema.")
    count = _count(values["count"], label=f"{label} count")
    metric_value = _number(values[metric], label=f"{label} {metric}")
    return count, metric_value


# ADD 2026-09-02: Latency evidence의 fixed measurement boundary와 summary를 strict하게 읽는다.
def _latency(raw: object) -> dict[str, float | int | str]:
    values = _mapping(raw, label="C5-3 latency evidence")
    expected = {
        "scope",
        "sample_id",
        "warmup_iterations",
        "measured_iterations",
        "pytorch_latency_ms",
        "tensorrt_latency_ms",
        "speedup_ratio",
    }
    if set(values) != expected:
        raise ValueError("C5-3 latency evidence fields do not match the schema.")
    pytorch = _mapping(values["pytorch_latency_ms"], label="PyTorch latency distribution")
    tensorrt = _mapping(values["tensorrt_latency_ms"], label="TensorRT latency distribution")
    latency_fields = {"count", "min", "mean", "p50", "p95", "max"}
    if set(pytorch) != latency_fields or set(tensorrt) != latency_fields:
        raise ValueError("C5-3 latency distribution fields do not match the schema.")
    for name, distribution in (("PyTorch", pytorch), ("TensorRT", tensorrt)):
        if _count(distribution["count"], label=f"{name} latency count") <= 0:
            raise ValueError("C5-3 latency count must be positive.")
        for metric in ("min", "mean", "p50", "p95", "max"):
            if _number(distribution[metric], label=f"{name} latency {metric}") <= 0.0:
                raise ValueError("C5-3 latency measurements must be positive.")
    scope = values["scope"]
    sample_id = values["sample_id"]
    if not isinstance(scope, str) or not isinstance(sample_id, str) or not sample_id:
        raise ValueError("C5-3 latency scope/sample id is invalid.")
    return {
        "scope": scope,
        "sample_id": sample_id,
        "warmup_iterations": _count(values["warmup_iterations"], label="Latency warmup iterations"),
        "measured_iterations": _count(
            values["measured_iterations"],
            label="Latency measured iterations",
        ),
        "pytorch_mean_ms": _number(pytorch["mean"], label="PyTorch latency mean"),
        "tensorrt_mean_ms": _number(tensorrt["mean"], label="TensorRT latency mean"),
        "speedup_ratio": _number(values["speedup_ratio"], label="TensorRT speedup ratio"),
    }


# ADD 2026-09-02: Repository-owned TensorRT FP16 acceptance YAML을 typed strict contract로 로드한다.
def load_yolo_tensorrt_parity_acceptance_policy(
    path: Path,
) -> YoloTensorRtParityAcceptancePolicy:
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("Cannot read C5-3 TensorRT acceptance policy.") from exc
    if not isinstance(raw, dict):
        raise ValueError("C5-3 TensorRT acceptance policy root must be a mapping.")
    values = cast(dict[str, Any], raw)
    expected = {
        "schema_version",
        "policy_id",
        "backend",
        "identity",
        "runtime",
        "structural",
        "numeric",
        "performance",
        "output_root",
    }
    if set(values) != expected:
        raise ValueError("C5-3 TensorRT acceptance policy fields do not match schema.")

    nested_types: tuple[tuple[str, type[Any]], ...] = (
        ("backend", TensorRtBackendPolicy),
        ("identity", TensorRtIdentityPolicy),
        ("runtime", TensorRtRuntimePolicy),
        ("structural", TensorRtStructuralPolicy),
        ("numeric", TensorRtNumericPolicy),
        ("performance", TensorRtPerformancePolicy),
    )
    nested: dict[str, Any] = {}
    for key, cls in nested_types:
        raw_nested = _mapping(values[key], label=f"C5-3 {key} policy")
        _require_fields(raw_nested, cls, label=f"C5-3 {key} policy")
        try:
            nested[key] = cls(**raw_nested)
        except TypeError as exc:
            raise ValueError(f"C5-3 {key} policy has invalid typed fields.") from exc

    policy = YoloTensorRtParityAcceptancePolicy(
        schema_version=values["schema_version"],
        policy_id=str(values["policy_id"]),
        backend=cast(TensorRtBackendPolicy, nested["backend"]),
        identity=cast(TensorRtIdentityPolicy, nested["identity"]),
        runtime=cast(TensorRtRuntimePolicy, nested["runtime"]),
        structural=cast(TensorRtStructuralPolicy, nested["structural"]),
        numeric=cast(TensorRtNumericPolicy, nested["numeric"]),
        performance=cast(TensorRtPerformancePolicy, nested["performance"]),
        output_root=Path(str(values["output_root"])),
        config_path=path.resolve(),
    )
    policy.validate()
    return policy


# ADD 2026-09-02: Equality acceptance check를 normalized scalar evidence로 만든다.
def _eq_check(
    *,
    name: str,
    category: str,
    expected: Scalar,
    observed: Scalar,
) -> AcceptanceCheck:
    return AcceptanceCheck(
        name=name,
        category=category,
        comparator="==",
        expected=expected,
        observed=observed,
        passed=observed == expected,
    )


# ADD 2026-09-02: Upper-bound numeric acceptance check를 만든다.
def _le_check(
    *,
    name: str,
    category: str,
    expected: float,
    observed: float,
) -> AcceptanceCheck:
    return AcceptanceCheck(
        name=name,
        category=category,
        comparator="<=",
        expected=expected,
        observed=observed,
        passed=observed <= expected,
    )


# ADD 2026-09-02: Lower-bound numeric/performance acceptance check를 만든다.
def _ge_check(
    *,
    name: str,
    category: str,
    expected: float,
    observed: float,
) -> AcceptanceCheck:
    return AcceptanceCheck(
        name=name,
        category=category,
        comparator=">=",
        expected=expected,
        observed=observed,
        passed=observed >= expected,
    )


# ADD 2026-09-02: Prospective TensorRT parity JSON에 frozen policy를 pure function으로 적용한다.
def build_yolo_tensorrt_parity_acceptance_result(
    *,
    evidence: Mapping[str, Any],
    policy: YoloTensorRtParityAcceptancePolicy,
    policy_sha256: str,
    parity_evidence_sha256: str,
    policy_repository: RepositoryProvenance,
) -> YoloTensorRtParityAcceptanceResult:
    policy.validate()
    policy_repository.validate()
    if policy_repository.working_tree_dirty:
        raise ValueError("Official C5-3 acceptance requires a clean policy repository.")

    expected_top_level = {
        "schema_version",
        "parity_id",
        "state",
        "created_at",
        "source_experiment_id",
        "frozen_manifest_sha256",
        "source_model_sha256",
        "source_onnx_sha256",
        "engine_sha256",
        "tensorrt_config_sha256",
        "split",
        "test_used",
        "test_split_used",
        "sample_count",
        "pytorch_prediction_count",
        "tensorrt_prediction_count",
        "matched_instance_count",
        "unmatched_pytorch_count",
        "unmatched_tensorrt_count",
        "class_agreement_count",
        "class_agreement_rate",
        "confidence_abs_error",
        "box_iou",
        "mask_iou",
        "latency",
        "structural_gates_passed",
        "numeric_acceptance",
        "samples",
        "environment",
        "repository",
    }
    if set(evidence) != expected_top_level:
        raise ValueError("C5-3 parity evidence fields do not match the schema.")

    parity_id = evidence["parity_id"]
    if not isinstance(parity_id, str):
        raise ValueError("C5-3 parity_id must be a string.")
    validate_artifact_id(parity_id)

    sample_count = _count(evidence["sample_count"], label="C5-3 parity sample count")
    nested_sample_count, all_tensors_finite = _validate_samples(
        evidence["samples"],
        expected_split=policy.structural.required_split,
    )
    pytorch_count = _count(
        evidence["pytorch_prediction_count"],
        label="C5-3 PyTorch prediction count",
    )
    tensorrt_count = _count(
        evidence["tensorrt_prediction_count"],
        label="C5-3 TensorRT prediction count",
    )
    unmatched_pytorch = _count(
        evidence["unmatched_pytorch_count"],
        label="C5-3 unmatched PyTorch count",
    )
    unmatched_tensorrt = _count(
        evidence["unmatched_tensorrt_count"],
        label="C5-3 unmatched TensorRT count",
    )
    matched_count = _count(
        evidence["matched_instance_count"],
        label="C5-3 matched instance count",
    )

    confidence_count, confidence_max = _distribution(
        evidence["confidence_abs_error"],
        metric="max",
        label="Confidence absolute error",
    )
    box_count, box_min = _distribution(
        evidence["box_iou"],
        metric="min",
        label="Box IoU",
    )
    mask_count, mask_min = _distribution(
        evidence["mask_iou"],
        metric="min",
        label="Mask IoU",
    )
    if (
        confidence_count != matched_count
        or box_count != matched_count
        or mask_count != matched_count
    ):
        raise ValueError("C5-3 parity metric distribution counts do not match matched instances.")

    latency = _latency(evidence["latency"])
    environment = _mapping(evidence["environment"], label="C5-3 parity environment")
    repository = _mapping(evidence["repository"], label="C5-3 evidence repository")
    if set(repository) != {"git_commit", "working_tree_dirty"}:
        raise ValueError("C5-3 evidence repository provenance fields are invalid.")
    if (
        not isinstance(repository["git_commit"], str)
        or type(repository["working_tree_dirty"]) is not bool
    ):
        raise ValueError("C5-3 evidence repository provenance types are invalid.")

    checks: list[AcceptanceCheck] = []

    for name, expected, observed in (
        (
            "evidence_state",
            EXPECTED_EVIDENCE_STATE,
            str(evidence["state"]),
        ),
        (
            "numeric_acceptance_pending",
            EXPECTED_PENDING_STATE,
            str(evidence["numeric_acceptance"]),
        ),
        (
            "source_experiment_id",
            policy.identity.source_experiment_id,
            str(evidence["source_experiment_id"]),
        ),
        (
            "frozen_manifest_sha256",
            policy.identity.frozen_manifest_sha256,
            str(evidence["frozen_manifest_sha256"]),
        ),
        (
            "source_model_sha256",
            policy.identity.source_model_sha256,
            str(evidence["source_model_sha256"]),
        ),
        (
            "onnx_sha256",
            policy.identity.onnx_sha256,
            str(evidence["source_onnx_sha256"]),
        ),
        (
            "tensorrt_config_sha256",
            policy.identity.tensorrt_config_sha256,
            str(evidence["tensorrt_config_sha256"]),
        ),
        (
            "engine_sha256",
            policy.identity.engine_sha256,
            str(evidence["engine_sha256"]),
        ),
    ):
        checks.append(
            _eq_check(
                name=name,
                category="identity",
                expected=expected,
                observed=observed,
            )
        )

    for name in (
        "tensorrt_version",
        "cuda_runtime_version",
        "gpu_name",
        "gpu_compute_capability",
        "torch_version",
        "ultralytics_version",
    ):
        expected = str(getattr(policy.runtime, name))
        observed_raw = environment.get(name)
        observed = str(observed_raw) if observed_raw is not None else "<missing>"
        checks.append(
            _eq_check(
                name=f"runtime_{name}",
                category="runtime",
                expected=expected,
                observed=observed,
            )
        )

    evidence_commit = cast(str, repository["git_commit"])
    policy_commit = policy_repository.git_commit
    structural_values: tuple[tuple[str, Scalar, Scalar], ...] = (
        ("split", policy.structural.required_split, str(evidence["split"])),
        ("sample_count", policy.structural.required_sample_count, sample_count),
        ("nested_sample_count", sample_count, nested_sample_count),
        ("test_used", False, cast(bool, evidence["test_used"])),
        ("test_split_used", False, cast(bool, evidence["test_split_used"])),
        (
            "structural_gates_passed",
            True,
            cast(bool, evidence["structural_gates_passed"]),
        ),
        ("prediction_count_match", True, pytorch_count == tensorrt_count),
        (
            "zero_unmatched",
            True,
            unmatched_pytorch == 0 and unmatched_tensorrt == 0,
        ),
        (
            "class_agreement_rate",
            policy.structural.require_class_agreement_rate,
            _number(evidence["class_agreement_rate"], label="Class agreement rate"),
        ),
        ("all_tensors_finite", True, all_tensors_finite),
        ("evidence_repository_clean", False, cast(bool, repository["working_tree_dirty"])),
        ("evidence_commit_matches_policy", policy_commit, evidence_commit),
    )
    for name, structural_expected, structural_observed in structural_values:
        checks.append(
            _eq_check(
                name=name,
                category="structural",
                expected=structural_expected,
                observed=structural_observed,
            )
        )

    checks.extend(
        (
            _le_check(
                name="confidence_abs_error_max",
                category="numeric",
                expected=policy.numeric.max_confidence_abs_error,
                observed=confidence_max,
            ),
            _ge_check(
                name="box_iou_min",
                category="numeric",
                expected=policy.numeric.min_box_iou,
                observed=box_min,
            ),
            _ge_check(
                name="mask_iou_min",
                category="numeric",
                expected=policy.numeric.min_mask_iou,
                observed=mask_min,
            ),
        )
    )

    performance_values: tuple[AcceptanceCheck, ...] = (
        _eq_check(
            name="benchmark_scope",
            category="performance",
            expected=policy.performance.benchmark_scope,
            observed=str(latency["scope"]),
        ),
        _eq_check(
            name="warmup_iterations",
            category="performance",
            expected=policy.performance.required_warmup_iterations,
            observed=cast(int, latency["warmup_iterations"]),
        ),
        _eq_check(
            name="measured_iterations",
            category="performance",
            expected=policy.performance.required_measured_iterations,
            observed=cast(int, latency["measured_iterations"]),
        ),
        _eq_check(
            name="candidate_mean_latency_faster",
            category="performance",
            expected=True,
            observed=cast(float, latency["tensorrt_mean_ms"])
            < cast(float, latency["pytorch_mean_ms"]),
        ),
        _ge_check(
            name="speedup_ratio",
            category="performance",
            expected=policy.performance.min_speedup_ratio,
            observed=cast(float, latency["speedup_ratio"]),
        ),
    )
    checks.extend(performance_values)

    accepted = all(check.passed for check in checks)
    result = YoloTensorRtParityAcceptanceResult(
        schema_version=ACCEPTANCE_SCHEMA_VERSION,
        policy_id=policy.policy_id,
        state=ACCEPTED_STATE if accepted else REJECTED_STATE,
        accepted=accepted,
        parity_id=parity_id,
        policy_sha256=policy_sha256,
        parity_evidence_sha256=parity_evidence_sha256,
        source_experiment_id=str(evidence["source_experiment_id"]),
        frozen_manifest_sha256=str(evidence["frozen_manifest_sha256"]),
        source_model_sha256=str(evidence["source_model_sha256"]),
        onnx_sha256=str(evidence["source_onnx_sha256"]),
        tensorrt_config_sha256=str(evidence["tensorrt_config_sha256"]),
        engine_sha256=str(evidence["engine_sha256"]),
        split=str(evidence["split"]),
        test_used=cast(bool, evidence["test_used"]),
        test_split_used=cast(bool, evidence["test_split_used"]),
        checks=tuple(checks),
        evidence_repository={
            "git_commit": evidence_commit,
            "working_tree_dirty": cast(bool, repository["working_tree_dirty"]),
        },
        policy_repository=policy_repository.to_json_dict(),
    )
    result.validate()
    return result


# ADD 2026-09-02: Repository-owned path가 repository_root 밖으로 이탈하지 않게 한다.
def _repository_path(repository_root: Path, path: Path, *, field: str) -> Path:
    root = repository_root.resolve()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"C5-3 {field} must remain inside repository_root.") from exc
    return resolved


# ADD 2026-09-02: Saved evidence와 committed policy로 acceptance artifact를 publish한다.
def evaluate_yolo_tensorrt_parity_acceptance(
    *,
    repository_root: Path,
    parity_evidence_path: Path,
    policy_path: Path = DEFAULT_ACCEPTANCE_POLICY,
    output_dir: Path | None = None,
) -> Path:
    root = repository_root.resolve()
    resolved_policy = _repository_path(root, policy_path, field="acceptance policy")
    resolved_evidence = _repository_path(
        root,
        parity_evidence_path,
        field="parity evidence",
    )
    policy = load_yolo_tensorrt_parity_acceptance_policy(resolved_policy)
    provenance = resolve_repository_provenance(root)
    provenance.validate()
    if provenance.working_tree_dirty:
        raise ValueError("Official C5-3 acceptance requires a clean committed repository.")

    evidence = _strict_json(resolved_evidence)
    result = build_yolo_tensorrt_parity_acceptance_result(
        evidence=evidence,
        policy=policy,
        policy_sha256=sha256_file(resolved_policy),
        parity_evidence_sha256=sha256_file(resolved_evidence),
        policy_repository=provenance,
    )

    destination = (
        _repository_path(root, output_dir, field="acceptance output")
        if output_dir is not None
        else root / policy.output_root / result.parity_id
    )
    staging = destination.parent / f".{destination.name}.staging"
    if destination.exists() or staging.exists():
        raise FileExistsError("C5-3 TensorRT acceptance output namespace already exists.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(exist_ok=False)
    try:
        result_path = staging / ACCEPTANCE_RESULT_FILENAME
        result_path.write_bytes(result.to_json_bytes())
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / ACCEPTANCE_RESULT_FILENAME
