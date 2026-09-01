"""Frozen post-characterization acceptance gate for YOLO ONNX FP32 parity."""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Never, cast

import yaml

from ml.deployment.yolo_onnx_parity import (
    NormalizedPredictionObservation,
    ParityInstanceMatch,
    RuntimeTensorObservation,
    SampleParityEvidence,
    YoloOnnxParityEvidence,
)
from ml.evaluation.final_benchmark import RepositoryProvenance, resolve_repository_provenance
from ml.training.yolo_segmentation import validate_artifact_id
from shared.hashing import is_sha256_digest, sha256_file

ACCEPTANCE_SCHEMA_VERSION = 1
ACCEPTANCE_RESULT_FILENAME = "acceptance.json"
ACCEPTED_STATE = "PARITY_ACCEPTED"
REJECTED_STATE = "PARITY_REJECTED"
DEFAULT_ACCEPTANCE_POLICY = Path("configs/deployment/yolo_onnx_fp32_parity_acceptance.yaml")
EXPECTED_POLICY_ID = "c5_2_yolo_onnx_fp32_parity_v1"
EXPECTED_SOURCE_EXPERIMENT_ID = "c4_2c_yolo11n_seg_crop350_nomosaic_maskratio2_seed42"
EXPECTED_FROZEN_MANIFEST_SHA256 = "2a26b1bc03a1876f828e12a625c69c76af5e8c5713e3f64be699feffe2e8aa09"
EXPECTED_SOURCE_MODEL_SHA256 = "e3fd10cdd708d31421feacfc5d694cb638e0ea60672e08796391b33aecf67155"
EXPECTED_EXPORT_CONFIG_SHA256 = "f1c2ef5045fdd89d964b2dc79c501580c9f55c2a1d38f38f13cf4794bafd0e85"
EXPECTED_ONNX_SHA256 = "f916325bb126d174de9c1fdfc24802eec11c46014f723fbf3ba3b3c1755c1490"
ACCEPTANCE_OUTPUT_ROOT = Path("outputs/deployment/yolo_segmentation/onnx_parity_acceptance")

Scalar = bool | int | float | str


@dataclass(frozen=True)
class ParityBackendPolicy:
    """Reference and candidate runtime identities covered by policy v1."""

    reference: str
    candidate: str

    # ADD 2026-09-02: Policy v1 backend pair를 exact FP32 contract로 고정한다.
    def validate(self) -> None:
        if self.reference != "pytorch_fp32" or self.candidate != "onnxruntime_fp32":
            raise ValueError("C5-2 acceptance backend contract changed without review.")


@dataclass(frozen=True)
class ParityIdentityPolicy:
    """Exact frozen source and exported ONNX identities covered by policy v1."""

    source_experiment_id: str
    frozen_manifest_sha256: str
    source_model_sha256: str
    export_config_sha256: str
    onnx_sha256: str

    # ADD 2026-09-02: Characterization에 사용한 frozen/export identity만 허용한다.
    def validate(self) -> None:
        expected = {
            "source_experiment_id": EXPECTED_SOURCE_EXPERIMENT_ID,
            "frozen_manifest_sha256": EXPECTED_FROZEN_MANIFEST_SHA256,
            "source_model_sha256": EXPECTED_SOURCE_MODEL_SHA256,
            "export_config_sha256": EXPECTED_EXPORT_CONFIG_SHA256,
            "onnx_sha256": EXPECTED_ONNX_SHA256,
        }
        mismatches = [
            name
            for name, expected_value in expected.items()
            if getattr(self, name) != expected_value
        ]
        if mismatches:
            raise ValueError(
                "C5-2 acceptance identity changed without review: " + ", ".join(sorted(mismatches))
            )
        for digest in (
            self.frozen_manifest_sha256,
            self.source_model_sha256,
            self.export_config_sha256,
            self.onnx_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C5-2 acceptance identity contains invalid SHA-256.")


@dataclass(frozen=True)
class ParityStructuralPolicy:
    """Fail-closed structural equivalence requirements."""

    required_split: str
    require_test_used_false: bool
    require_test_split_used_false: bool
    require_structural_gates_passed: bool
    require_finite_outputs: bool
    require_tensor_schema_match: bool
    require_prediction_count_match: bool
    require_zero_unmatched: bool
    required_class_agreement_rate: float

    # ADD 2026-09-02: Validation-only structural acceptance 조건을 고정한다.
    def validate(self) -> None:
        boolean_values = (
            self.require_test_used_false,
            self.require_test_split_used_false,
            self.require_structural_gates_passed,
            self.require_finite_outputs,
            self.require_tensor_schema_match,
            self.require_prediction_count_match,
            self.require_zero_unmatched,
        )
        if any(type(value) is not bool for value in boolean_values):
            raise TypeError("C5-2 acceptance structural flags must be booleans.")
        if (
            self.required_split != "val"
            or any(value is not True for value in boolean_values)
            or type(self.required_class_agreement_rate) not in {int, float}
            or float(self.required_class_agreement_rate) != 1.0
        ):
            raise ValueError("C5-2 acceptance structural policy changed without review.")


@dataclass(frozen=True)
class ParityNumericPolicy:
    """Approved FP32 export-equivalence tolerances defined after characterization."""

    max_confidence_abs_error: float
    min_box_iou: float
    min_mask_iou: float

    # ADD 2026-09-02: Characterization 이후 승인한 FP32 equivalence tolerance를 고정한다.
    def validate(self) -> None:
        values = (
            self.max_confidence_abs_error,
            self.min_box_iou,
            self.min_mask_iou,
        )
        if any(
            type(value) not in {int, float} or not math.isfinite(float(value)) for value in values
        ):
            raise TypeError("C5-2 acceptance numeric tolerances must be finite numbers.")
        if (
            float(self.max_confidence_abs_error) != 1e-4
            or float(self.min_box_iou) != 0.999
            or float(self.min_mask_iou) != 0.999
        ):
            raise ValueError("C5-2 acceptance numeric policy changed without review.")


@dataclass(frozen=True)
class YoloOnnxParityAcceptancePolicy:
    """Repository-owned policy frozen after the first characterization run."""

    schema_version: int
    policy_id: str
    backend: ParityBackendPolicy
    identity: ParityIdentityPolicy
    structural: ParityStructuralPolicy
    numeric: ParityNumericPolicy
    output_root: Path
    config_path: Path

    # ADD 2026-09-02: Repository-owned acceptance policy 전체 contract를 검증한다.
    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != ACCEPTANCE_SCHEMA_VERSION:
            raise ValueError("Unsupported C5-2 acceptance policy schema version.")
        validate_artifact_id(self.policy_id)
        if self.policy_id != EXPECTED_POLICY_ID:
            raise ValueError("C5-2 acceptance policy id changed without review.")
        if self.output_root != ACCEPTANCE_OUTPUT_ROOT:
            raise ValueError("C5-2 acceptance output_root must remain in ignored outputs/.")
        self.backend.validate()
        self.identity.validate()
        self.structural.validate()
        self.numeric.validate()


@dataclass(frozen=True)
class AcceptanceCheck:
    """One explicit structural, identity, or numeric acceptance decision."""

    name: str
    category: str
    comparator: str
    expected: Scalar
    observed: Scalar
    passed: bool

    # ADD 2026-09-02: 개별 acceptance check가 strict scalar evidence인지 검증한다.
    def validate(self) -> None:
        if not self.name or self.category not in {"identity", "structural", "numeric"}:
            raise ValueError("C5-2 acceptance check metadata is invalid.")
        if not self.comparator or type(self.passed) is not bool:
            raise ValueError("C5-2 acceptance check comparator/result is invalid.")
        for value in (self.expected, self.observed):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("C5-2 acceptance check contains NaN or Inf.")


@dataclass(frozen=True)
class YoloOnnxParityAcceptanceResult:
    """Deterministic result of applying one committed policy to parity evidence."""

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
    export_config_sha256: str
    onnx_sha256: str
    split: str
    test_used: bool
    test_split_used: bool
    checks: tuple[AcceptanceCheck, ...]
    evidence_repository: Mapping[str, str | bool]
    policy_repository: Mapping[str, str | bool]

    # ADD 2026-09-02: Acceptance result의 state/check/provenance 정합성을 검증한다.
    def validate(self) -> None:
        if self.schema_version != ACCEPTANCE_SCHEMA_VERSION or self.policy_id != EXPECTED_POLICY_ID:
            raise ValueError("C5-2 acceptance result schema/policy is invalid.")
        validate_artifact_id(self.parity_id)
        if self.state not in {ACCEPTED_STATE, REJECTED_STATE} or type(self.accepted) is not bool:
            raise ValueError("C5-2 acceptance result lifecycle is invalid.")
        if not self.checks:
            raise ValueError("C5-2 acceptance result requires explicit checks.")
        expected_accepted = all(check.passed for check in self.checks)
        if self.accepted != expected_accepted:
            raise ValueError("C5-2 acceptance result is inconsistent with its checks.")
        if self.state != (ACCEPTED_STATE if self.accepted else REJECTED_STATE):
            raise ValueError("C5-2 acceptance state is inconsistent with accepted flag.")
        for digest in (
            self.policy_sha256,
            self.parity_evidence_sha256,
            self.frozen_manifest_sha256,
            self.source_model_sha256,
            self.export_config_sha256,
            self.onnx_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C5-2 acceptance result contains invalid SHA-256.")
        if self.split != "val" or self.test_used is not False or self.test_split_used is not False:
            raise ValueError("C5-2 acceptance result violated the validation-only test seal.")
        if set(self.policy_repository) != {"git_commit", "working_tree_dirty"}:
            raise ValueError("C5-2 policy repository provenance fields are invalid.")
        policy_repository = RepositoryProvenance(
            git_commit=str(self.policy_repository["git_commit"]),
            working_tree_dirty=cast(bool, self.policy_repository["working_tree_dirty"]),
        )
        policy_repository.validate()
        if policy_repository.working_tree_dirty:
            raise ValueError("Official C5-2 acceptance requires a clean policy repository state.")
        if set(self.evidence_repository) != {"git_commit", "working_tree_dirty"}:
            raise ValueError("C5-2 evidence repository provenance fields are invalid.")
        for check in self.checks:
            check.validate()
        try:
            json.dumps(asdict(self), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("C5-2 acceptance result must be strict JSON data.") from exc

    # ADD 2026-09-02: Acceptance result를 deterministic strict JSON으로 직렬화한다.
    def to_json_bytes(self) -> bytes:
        self.validate()
        return (json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


# ADD 2026-09-02: Untrusted JSON/YAML object의 field set을 dataclass schema와 대조한다.
def _require_fields(raw: Mapping[str, Any], cls: type[Any], *, label: str) -> None:
    expected = {field.name for field in fields(cls)}
    if set(raw) != expected:
        raise ValueError(f"{label} fields do not match the schema.")


# ADD 2026-09-02: Python JSON decoder의 NaN/Infinity extension을 fail closed한다.
def _reject_json_constant(value: str) -> Never:
    raise ValueError(f"Strict JSON rejects non-finite constant: {value}")


# ADD 2026-09-02: Saved parity JSON을 non-finite 허용 없이 strict object로 로드한다.
def _strict_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Cannot read strict C5-2 parity evidence JSON.") from exc
    if not isinstance(raw, dict):
        raise ValueError("C5-2 parity evidence JSON root must be an object.")
    return cast(dict[str, Any], raw)


# ADD 2026-09-02: JSON shape array를 strict integer tuple로 복원한다.
def _tuple_of_ints(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list | tuple) or any(type(item) is not int for item in value):
        raise ValueError(f"{label} must be an integer array.")
    return tuple(cast(list[int] | tuple[int, ...], value))


# ADD 2026-09-02: Saved tensor observation을 typed parity evidence로 복원한다.
def _runtime_tensor(raw: object) -> RuntimeTensorObservation:
    if not isinstance(raw, dict):
        raise ValueError("Parity tensor entry must be an object.")
    values = cast(dict[str, Any], raw)
    _require_fields(values, RuntimeTensorObservation, label="Parity tensor")
    tensor = RuntimeTensorObservation(
        name=str(values["name"]),
        dtype=str(values["dtype"]),
        shape=_tuple_of_ints(values["shape"], label="Parity tensor shape"),
        finite=values["finite"],
    )
    tensor.validate()
    return tensor


# ADD 2026-09-02: Saved normalized prediction을 typed parity evidence로 복원한다.
def _prediction(raw: object) -> NormalizedPredictionObservation:
    if not isinstance(raw, dict):
        raise ValueError("Parity prediction entry must be an object.")
    values = cast(dict[str, Any], raw)
    _require_fields(values, NormalizedPredictionObservation, label="Parity prediction")
    box = values["box_xyxy"]
    if not isinstance(box, list | tuple) or len(box) != 4:
        raise ValueError("Parity prediction box_xyxy must contain four values.")
    mask_shape = _tuple_of_ints(values["mask_shape"], label="Parity prediction mask_shape")
    box_values = tuple(float(item) for item in box)
    prediction = NormalizedPredictionObservation(
        prediction_index=values["prediction_index"],
        class_id=values["class_id"],
        confidence=values["confidence"],
        box_xyxy=(box_values[0], box_values[1], box_values[2], box_values[3]),
        mask_shape=cast(tuple[int, int], mask_shape),
        mask_foreground_pixels=values["mask_foreground_pixels"],
        mask_sha256=str(values["mask_sha256"]),
    )
    prediction.validate()
    return prediction


# ADD 2026-09-02: Saved backend match를 typed parity evidence로 복원한다.
def _match(raw: object) -> ParityInstanceMatch:
    if not isinstance(raw, dict):
        raise ValueError("Parity match entry must be an object.")
    values = cast(dict[str, Any], raw)
    _require_fields(values, ParityInstanceMatch, label="Parity match")
    match = ParityInstanceMatch(**values)
    match.validate()
    return match


# ADD 2026-09-02: Nested parity JSON array의 container type을 검증한다.
def _sequence(raw: Mapping[str, Any], name: str) -> list[object]:
    value = raw[name]
    if not isinstance(value, list):
        raise ValueError(f"Parity sample {name} must be an array.")
    return cast(list[object], value)


# ADD 2026-09-02: Saved per-sample evidence를 typed validation-only record로 복원한다.
def _sample(raw: object) -> SampleParityEvidence:
    if not isinstance(raw, dict):
        raise ValueError("Parity sample entry must be an object.")
    values = cast(dict[str, Any], raw)
    _require_fields(values, SampleParityEvidence, label="Parity sample")
    sample = SampleParityEvidence(
        sample_id=str(values["sample_id"]),
        split=str(values["split"]),
        pytorch_prediction_count=values["pytorch_prediction_count"],
        onnx_prediction_count=values["onnx_prediction_count"],
        matched_instance_count=values["matched_instance_count"],
        unmatched_pytorch_count=values["unmatched_pytorch_count"],
        unmatched_onnx_count=values["unmatched_onnx_count"],
        pytorch_tensors=tuple(
            _runtime_tensor(item) for item in _sequence(values, "pytorch_tensors")
        ),
        onnx_tensors=tuple(_runtime_tensor(item) for item in _sequence(values, "onnx_tensors")),
        pytorch_predictions=tuple(
            _prediction(item) for item in _sequence(values, "pytorch_predictions")
        ),
        onnx_predictions=tuple(_prediction(item) for item in _sequence(values, "onnx_predictions")),
        matches=tuple(_match(item) for item in _sequence(values, "matches")),
    )
    sample.validate()
    return sample


# ADD 2026-09-02: Inference 없이 저장된 characterization JSON만 typed evidence로 로드한다.
def load_yolo_onnx_parity_evidence(path: Path) -> YoloOnnxParityEvidence:
    """Load the characterization JSON without opening any model or dataset content."""

    values = _strict_json(path)
    _require_fields(values, YoloOnnxParityEvidence, label="C5-2 parity evidence")
    samples_raw = values["samples"]
    if not isinstance(samples_raw, list):
        raise ValueError("C5-2 parity samples must be an array.")
    try:
        evidence = YoloOnnxParityEvidence(
            **{key: value for key, value in values.items() if key not in {"samples"}},
            samples=tuple(_sample(item) for item in samples_raw),
        )
    except TypeError as exc:
        raise ValueError("C5-2 parity evidence values do not match the typed schema.") from exc
    evidence.validate()
    return evidence


# ADD 2026-09-02: Characterization 이후 승인된 policy v1 YAML을 exact contract로 로드한다.
def load_yolo_onnx_parity_acceptance_policy(path: Path) -> YoloOnnxParityAcceptancePolicy:
    """Load policy v1 as a strict, exact repository contract."""

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("Cannot read C5-2 acceptance policy.") from exc
    if not isinstance(raw, dict):
        raise ValueError("C5-2 acceptance policy root must be a mapping.")
    values = cast(dict[str, Any], raw)
    expected = {
        "schema_version",
        "policy_id",
        "backend",
        "identity",
        "structural",
        "numeric",
        "output_root",
    }
    if set(values) != expected:
        raise ValueError("C5-2 acceptance policy fields do not match the schema.")
    for name in ("backend", "identity", "structural", "numeric"):
        if not isinstance(values[name], dict):
            raise ValueError(f"C5-2 acceptance {name} must be a mapping.")
    try:
        policy = YoloOnnxParityAcceptancePolicy(
            schema_version=values["schema_version"],
            policy_id=str(values["policy_id"]),
            backend=ParityBackendPolicy(**cast(dict[str, Any], values["backend"])),
            identity=ParityIdentityPolicy(**cast(dict[str, Any], values["identity"])),
            structural=ParityStructuralPolicy(**cast(dict[str, Any], values["structural"])),
            numeric=ParityNumericPolicy(**cast(dict[str, Any], values["numeric"])),
            output_root=Path(str(values["output_root"])),
            config_path=path.resolve(),
        )
    except TypeError as exc:
        raise ValueError("C5-2 acceptance policy nested fields do not match the schema.") from exc
    policy.validate()
    return policy


# ADD 2026-09-02: Aggregate parity distribution의 count와 finite numeric values를 검증한다.
def _metric_distribution(
    mapping: Mapping[str, float | int | None],
    *,
    label: str,
) -> dict[str, float]:
    if set(mapping) != {"count", "min", "mean", "max"}:
        raise ValueError(f"C5-2 {label} distribution fields are malformed.")
    count = mapping["count"]
    if type(count) is not int or count <= 0:
        raise ValueError(f"C5-2 {label} requires at least one matched observation.")
    result: dict[str, float] = {}
    for name in ("min", "mean", "max"):
        value = mapping[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"C5-2 {label} contains missing, NaN, or Inf metrics.")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"C5-2 {label} contains missing, NaN, or Inf metrics.")
        result[name] = numeric_value
    return result


# ADD 2026-09-02: Backend tensor name/dtype/shape를 order-independent signature로 만든다.
def _tensor_signature(
    tensors: tuple[RuntimeTensorObservation, ...],
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    return tuple(sorted((item.name, item.dtype, item.shape) for item in tensors))


# ADD 2026-09-02: Machine-readable expected/observed acceptance check를 생성한다.
def _check(
    *,
    name: str,
    category: str,
    comparator: str,
    expected: Scalar,
    observed: Scalar,
    passed: bool,
) -> AcceptanceCheck:
    check = AcceptanceCheck(
        name=name,
        category=category,
        comparator=comparator,
        expected=expected,
        observed=observed,
        passed=passed,
    )
    check.validate()
    return check


# ADD 2026-09-02: Frozen policy를 typed parity evidence에 적용해 ACCEPT/REJECT를 판정한다.
def assess_yolo_onnx_parity_acceptance(
    *,
    evidence: YoloOnnxParityEvidence,
    policy: YoloOnnxParityAcceptancePolicy,
    policy_sha256: str,
    parity_evidence_sha256: str,
    policy_provenance: RepositoryProvenance,
) -> YoloOnnxParityAcceptanceResult:
    """Apply the already-frozen policy without inference, dataset access, or threshold tuning."""

    evidence.validate()
    policy.validate()
    policy_provenance.validate()
    if policy_provenance.working_tree_dirty:
        raise ValueError("Official C5-2 acceptance requires a clean committed policy repository.")
    if not is_sha256_digest(policy_sha256) or not is_sha256_digest(parity_evidence_sha256):
        raise ValueError("C5-2 acceptance input provenance requires valid SHA-256 digests.")

    confidence = _metric_distribution(evidence.confidence_abs_error, label="confidence_abs_error")
    box = _metric_distribution(evidence.box_iou, label="box_iou")
    mask = _metric_distribution(evidence.mask_iou, label="mask_iou")

    tensor_schema_match = all(
        _tensor_signature(sample.pytorch_tensors) == _tensor_signature(sample.onnx_tensors)
        for sample in evidence.samples
    )
    finite_outputs = all(
        tensor.finite
        for sample in evidence.samples
        for tensor in (*sample.pytorch_tensors, *sample.onnx_tensors)
    )
    prediction_count_match = (
        evidence.pytorch_prediction_count == evidence.onnx_prediction_count
        and all(
            sample.pytorch_prediction_count == sample.onnx_prediction_count
            for sample in evidence.samples
        )
    )
    zero_unmatched = evidence.unmatched_pytorch_count == 0 and evidence.unmatched_onnx_count == 0
    class_rate_observed: Scalar = (
        float(evidence.class_agreement_rate)
        if evidence.class_agreement_rate is not None
        else "missing"
    )

    identity = policy.identity
    structural = policy.structural
    numeric = policy.numeric
    checks = (
        _check(
            name="source_experiment_id",
            category="identity",
            comparator="==",
            expected=identity.source_experiment_id,
            observed=evidence.source_experiment_id,
            passed=evidence.source_experiment_id == identity.source_experiment_id,
        ),
        _check(
            name="frozen_manifest_sha256",
            category="identity",
            comparator="==",
            expected=identity.frozen_manifest_sha256,
            observed=evidence.frozen_manifest_sha256,
            passed=evidence.frozen_manifest_sha256 == identity.frozen_manifest_sha256,
        ),
        _check(
            name="source_model_sha256",
            category="identity",
            comparator="==",
            expected=identity.source_model_sha256,
            observed=evidence.source_model_sha256,
            passed=evidence.source_model_sha256 == identity.source_model_sha256,
        ),
        _check(
            name="export_config_sha256",
            category="identity",
            comparator="==",
            expected=identity.export_config_sha256,
            observed=evidence.export_config_sha256,
            passed=evidence.export_config_sha256 == identity.export_config_sha256,
        ),
        _check(
            name="onnx_sha256",
            category="identity",
            comparator="==",
            expected=identity.onnx_sha256,
            observed=evidence.onnx_sha256,
            passed=evidence.onnx_sha256 == identity.onnx_sha256,
        ),
        _check(
            name="split",
            category="structural",
            comparator="==",
            expected=structural.required_split,
            observed=evidence.split,
            passed=evidence.split == structural.required_split,
        ),
        _check(
            name="test_used",
            category="structural",
            comparator="==",
            expected=False,
            observed=evidence.test_used,
            passed=evidence.test_used is False,
        ),
        _check(
            name="test_split_used",
            category="structural",
            comparator="==",
            expected=False,
            observed=evidence.test_split_used,
            passed=evidence.test_split_used is False,
        ),
        _check(
            name="structural_gates_passed",
            category="structural",
            comparator="==",
            expected=True,
            observed=evidence.structural_gates_passed,
            passed=evidence.structural_gates_passed is True,
        ),
        _check(
            name="finite_outputs",
            category="structural",
            comparator="==",
            expected=True,
            observed=finite_outputs,
            passed=finite_outputs,
        ),
        _check(
            name="tensor_schema_match",
            category="structural",
            comparator="==",
            expected=True,
            observed=tensor_schema_match,
            passed=tensor_schema_match,
        ),
        _check(
            name="prediction_count_match",
            category="structural",
            comparator="==",
            expected=True,
            observed=prediction_count_match,
            passed=prediction_count_match,
        ),
        _check(
            name="zero_unmatched",
            category="structural",
            comparator="==",
            expected=True,
            observed=zero_unmatched,
            passed=zero_unmatched,
        ),
        _check(
            name="class_agreement_rate",
            category="structural",
            comparator="==",
            expected=float(structural.required_class_agreement_rate),
            observed=class_rate_observed,
            passed=evidence.class_agreement_rate == float(structural.required_class_agreement_rate),
        ),
        _check(
            name="confidence_abs_error_max",
            category="numeric",
            comparator="<=",
            expected=float(numeric.max_confidence_abs_error),
            observed=confidence["max"],
            passed=confidence["max"] <= float(numeric.max_confidence_abs_error),
        ),
        _check(
            name="box_iou_min",
            category="numeric",
            comparator=">=",
            expected=float(numeric.min_box_iou),
            observed=box["min"],
            passed=box["min"] >= float(numeric.min_box_iou),
        ),
        _check(
            name="mask_iou_min",
            category="numeric",
            comparator=">=",
            expected=float(numeric.min_mask_iou),
            observed=mask["min"],
            passed=mask["min"] >= float(numeric.min_mask_iou),
        ),
    )
    accepted = all(check.passed for check in checks)
    result = YoloOnnxParityAcceptanceResult(
        schema_version=ACCEPTANCE_SCHEMA_VERSION,
        policy_id=policy.policy_id,
        state=ACCEPTED_STATE if accepted else REJECTED_STATE,
        accepted=accepted,
        parity_id=evidence.parity_id,
        policy_sha256=policy_sha256,
        parity_evidence_sha256=parity_evidence_sha256,
        source_experiment_id=evidence.source_experiment_id,
        frozen_manifest_sha256=evidence.frozen_manifest_sha256,
        source_model_sha256=evidence.source_model_sha256,
        export_config_sha256=evidence.export_config_sha256,
        onnx_sha256=evidence.onnx_sha256,
        split=evidence.split,
        test_used=evidence.test_used,
        test_split_used=evidence.test_split_used,
        checks=checks,
        evidence_repository=evidence.repository,
        policy_repository=policy_provenance.to_json_dict(),
    )
    result.validate()
    return result


# ADD 2026-09-02: Policy/output repository path가 repository_root 밖으로 이탈하지 않게 한다.
def _repository_path(repository_root: Path, path: Path, *, field: str) -> Path:
    root = repository_root.resolve()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"C5-2 {field} must remain inside repository_root.") from exc
    return resolved


# ADD 2026-09-02: Saved JSON과 committed policy만 사용해 별도 acceptance artifact를 publish한다.
def evaluate_yolo_onnx_parity_acceptance(
    *,
    repository_root: Path,
    parity_evidence_path: Path,
    policy_path: Path = DEFAULT_ACCEPTANCE_POLICY,
    output_dir: Path | None = None,
) -> Path:
    """Evaluate a saved parity JSON and publish a separate acceptance result."""

    root = repository_root.resolve()
    resolved_policy = _repository_path(root, policy_path, field="acceptance policy")
    policy = load_yolo_onnx_parity_acceptance_policy(resolved_policy)
    evidence_path = parity_evidence_path.resolve()
    evidence = load_yolo_onnx_parity_evidence(evidence_path)
    provenance = resolve_repository_provenance(root)
    provenance.validate()
    if provenance.working_tree_dirty:
        raise ValueError("Official C5-2 acceptance requires a clean committed repository state.")

    result = assess_yolo_onnx_parity_acceptance(
        evidence=evidence,
        policy=policy,
        policy_sha256=sha256_file(resolved_policy),
        parity_evidence_sha256=sha256_file(evidence_path),
        policy_provenance=provenance,
    )
    destination = (
        _repository_path(root, output_dir, field="acceptance output")
        if output_dir is not None
        else root / policy.output_root / f"{evidence.parity_id}--{policy.policy_id}"
    )
    if destination.exists():
        raise FileExistsError(f"C5-2 acceptance output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    result_path = destination / ACCEPTANCE_RESULT_FILENAME
    try:
        with result_path.open("xb") as handle:
            handle.write(result.to_json_bytes())
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return result_path
