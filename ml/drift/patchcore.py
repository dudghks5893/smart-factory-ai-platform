"""PatchCore score-distribution reference and batch drift contracts."""

from __future__ import annotations

import bisect
import json
import math
import re
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ml.datasets.manifest import ManifestRecord
from ml.evaluation.predictions import RawPredictionRecord
from ml.evaluation.thresholds import ThresholdArtifact
from shared.hashing import is_sha256_digest

DRIFT_SCHEMA_VERSION = 1
REFERENCE_SCHEMA_VERSION = 1
REFERENCE_FILENAME = "reference.json"
DRIFT_FILENAME = "drift.json"
REFERENCE_SPLIT = "validation"
REFERENCE_LABEL = "normal"
DEFAULT_PSI_BIN_COUNT = 10
DEFAULT_PSI_EPSILON = 1e-6
DEFAULT_MINIMUM_SAMPLE_COUNT = 30
DEFAULT_PSI_WARNING_THRESHOLD = 0.1
DEFAULT_PSI_DRIFT_THRESHOLD = 0.25
DEFAULT_ANOMALY_RATIO_WARNING_THRESHOLD = 0.1
DEFAULT_ANOMALY_RATIO_DRIFT_THRESHOLD = 0.2

_ARTIFACT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclass(frozen=True)
class DriftLineage:
    """Immutable model inputs that isolate one comparable score population."""

    model_sha256: str
    artifact_metadata_sha256: str
    manifest_sha256: str
    threshold_artifact_sha256: str

    # ADD 2026-08-21: Drift lineage를 stable JSON mapping으로 변환한다.
    def to_json_dict(self) -> dict[str, str]:
        """Return a stable JSON-compatible lineage mapping."""
        return asdict(self)

    # ADD 2026-08-21: JSON mapping의 full PatchCore lineage hash를 검증해 복원한다.
    @classmethod
    def from_json_dict(cls, raw: object) -> DriftLineage:
        """Validate and restore one complete drift lineage."""
        if not isinstance(raw, dict):
            raise ValueError("Drift lineage must be a JSON object.")
        expected = {
            "model_sha256",
            "artifact_metadata_sha256",
            "manifest_sha256",
            "threshold_artifact_sha256",
        }
        if set(raw) != expected:
            raise ValueError("Drift lineage fields do not match the schema.")
        lineage = cls(**{field: _required_string(raw[field], field) for field in expected})
        lineage.validate()
        return lineage

    # ADD 2026-08-21: Drift 비교에 필요한 모든 lineage 값이 SHA-256인지 검증한다.
    def validate(self) -> None:
        """Reject incomplete or malformed lineage digests."""
        for field, digest in asdict(self).items():
            if not is_sha256_digest(digest):
                raise ValueError(f"Drift lineage {field} must be a SHA-256 hex digest.")


@dataclass(frozen=True)
class ScoreSummary:
    """Finite descriptive statistics for one anomaly-score population."""

    mean: float
    std: float
    minimum: float
    maximum: float
    p50: float
    p90: float
    p95: float
    p99: float

    # ADD 2026-08-21: Score summary를 explicit min/max key의 JSON mapping으로 변환한다.
    def to_json_dict(self) -> dict[str, float]:
        """Return a stable JSON-compatible summary mapping."""
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
        }

    # ADD 2026-08-21: Persisted score summary의 field와 finite value를 검증한다.
    @classmethod
    def from_json_dict(cls, raw: object) -> ScoreSummary:
        """Validate and restore persisted descriptive statistics."""
        if not isinstance(raw, dict):
            raise ValueError("Score summary must be a JSON object.")
        expected = {"mean", "std", "min", "max", "p50", "p90", "p95", "p99"}
        if set(raw) != expected:
            raise ValueError("Score summary fields do not match the schema.")
        return cls(
            mean=_finite_float(raw["mean"], "mean"),
            std=_finite_float(raw["std"], "std"),
            minimum=_finite_float(raw["min"], "min"),
            maximum=_finite_float(raw["max"], "max"),
            p50=_finite_float(raw["p50"], "p50"),
            p90=_finite_float(raw["p90"], "p90"),
            p95=_finite_float(raw["p95"], "p95"),
            p99=_finite_float(raw["p99"], "p99"),
        )


@dataclass(frozen=True)
class DriftReference:
    """Validation-normal score baseline with fixed PSI bins and full lineage."""

    schema_version: int
    reference_id: str
    model_name: str
    category: str
    lineage: DriftLineage
    source_split: str
    source_label: str
    validation_predictions_sha256: str
    sample_count: int
    score_values: tuple[float, ...]
    summary: ScoreSummary
    image_threshold: float
    comparison_operator: str
    reference_anomaly_ratio: float
    psi_bin_count_requested: int
    psi_bin_edges: tuple[float, ...]
    reference_bin_counts: tuple[int, ...]
    psi_epsilon: float
    created_at: str

    # ADD 2026-08-21: Reference baseline을 inspectable schema-versioned JSON으로 변환한다.
    def to_json_dict(self) -> dict[str, Any]:
        """Return the complete reproducible reference artifact payload."""
        return {
            "schema_version": self.schema_version,
            "reference_id": self.reference_id,
            "model_name": self.model_name,
            "category": self.category,
            "lineage": self.lineage.to_json_dict(),
            "source": {
                "split": self.source_split,
                "label": self.source_label,
                "validation_predictions_sha256": self.validation_predictions_sha256,
            },
            "sample_count": self.sample_count,
            "score_values": list(self.score_values),
            "summary": self.summary.to_json_dict(),
            "threshold": {
                "image_threshold": self.image_threshold,
                "comparison_operator": self.comparison_operator,
            },
            "reference_anomaly_ratio": self.reference_anomaly_ratio,
            "psi": {
                "bin_count_requested": self.psi_bin_count_requested,
                "bin_edges": list(self.psi_bin_edges),
                "reference_bin_counts": list(self.reference_bin_counts),
                "epsilon": self.psi_epsilon,
            },
            "created_at": self.created_at,
        }

    # ADD 2026-08-21: JSON reference artifact를 typed validated contract로 복원한다.
    @classmethod
    def from_json_dict(cls, raw: object) -> DriftReference:
        """Validate and restore a reference artifact without source files."""
        if not isinstance(raw, dict):
            raise ValueError("Drift reference root must be a JSON object.")
        try:
            source = raw["source"]
            threshold = raw["threshold"]
            psi = raw["psi"]
            if not isinstance(source, dict) or not isinstance(threshold, dict):
                raise TypeError("reference source and threshold must be objects")
            if not isinstance(psi, dict):
                raise TypeError("reference psi must be an object")
            scores = _finite_float_tuple(raw["score_values"], "score_values")
            bin_edges = _finite_float_tuple(psi["bin_edges"], "psi.bin_edges")
            bin_counts = _integer_tuple(psi["reference_bin_counts"], "reference_bin_counts")
            reference = cls(
                schema_version=_required_integer(raw["schema_version"], "schema_version"),
                reference_id=_required_string(raw["reference_id"], "reference_id"),
                model_name=_required_string(raw["model_name"], "model_name"),
                category=_required_string(raw["category"], "category"),
                lineage=DriftLineage.from_json_dict(raw["lineage"]),
                source_split=_required_string(source["split"], "source.split"),
                source_label=_required_string(source["label"], "source.label"),
                validation_predictions_sha256=_required_string(
                    source["validation_predictions_sha256"],
                    "source.validation_predictions_sha256",
                ),
                sample_count=_required_integer(raw["sample_count"], "sample_count"),
                score_values=scores,
                summary=ScoreSummary.from_json_dict(raw["summary"]),
                image_threshold=_finite_float(
                    threshold["image_threshold"], "threshold.image_threshold"
                ),
                comparison_operator=_required_string(
                    threshold["comparison_operator"], "threshold.comparison_operator"
                ),
                reference_anomaly_ratio=_finite_float(
                    raw["reference_anomaly_ratio"], "reference_anomaly_ratio"
                ),
                psi_bin_count_requested=_required_integer(
                    psi["bin_count_requested"], "psi.bin_count_requested"
                ),
                psi_bin_edges=bin_edges,
                reference_bin_counts=bin_counts,
                psi_epsilon=_finite_float(psi["epsilon"], "psi.epsilon"),
                created_at=_required_string(raw["created_at"], "created_at"),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Drift reference is missing or contains invalid fields.") from exc
        reference.validate()
        return reference

    # ADD 2026-08-21: Validation-only source, score, PSI와 lineage reference invariant를 검증한다.
    def validate(self) -> None:
        """Reject a reference that cannot reproduce a valid comparison baseline."""
        if self.schema_version != REFERENCE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported drift reference schema: {self.schema_version}.")
        validate_artifact_id(self.reference_id)
        if not self.model_name or not self.category:
            raise ValueError("Drift reference model_name and category must not be empty.")
        self.lineage.validate()
        if self.source_split != REFERENCE_SPLIT or self.source_label != REFERENCE_LABEL:
            raise ValueError("Drift reference must use normal validation predictions only.")
        if not is_sha256_digest(self.validation_predictions_sha256):
            raise ValueError("Reference validation prediction hash must be SHA-256.")
        if self.sample_count <= 0 or self.sample_count != len(self.score_values):
            raise ValueError("Reference sample_count must match non-empty score_values.")
        _require_finite(self.score_values, "reference score")
        if self.summary != summarize_scores(self.score_values):
            raise ValueError("Reference score summary does not match score_values.")
        if self.comparison_operator != ">":
            raise ValueError("Drift reference comparison_operator must be '>'.")
        if self.summary.maximum != self.image_threshold:
            raise ValueError(
                "Drift reference maximum score must reproduce the calibrated image threshold."
            )
        expected_ratio = anomaly_ratio(
            self.score_values,
            threshold=self.image_threshold,
        )
        if self.reference_anomaly_ratio != expected_ratio:
            raise ValueError("Reference anomaly ratio does not match scores and threshold.")
        if self.psi_bin_count_requested < 2 or self.psi_epsilon <= 0:
            raise ValueError("Reference PSI bin count and epsilon must be positive.")
        if tuple(sorted(set(self.psi_bin_edges))) != self.psi_bin_edges:
            raise ValueError("Reference PSI bin edges must be finite, unique, and sorted.")
        _require_finite(self.psi_bin_edges, "reference PSI edge")
        if len(self.reference_bin_counts) != len(self.psi_bin_edges) + 1:
            raise ValueError("Reference PSI bin counts do not match the fixed bin edges.")
        if any(count < 0 for count in self.reference_bin_counts):
            raise ValueError("Reference PSI bin counts must be non-negative.")
        if sum(self.reference_bin_counts) != self.sample_count:
            raise ValueError("Reference PSI bin counts must sum to sample_count.")
        if self.reference_bin_counts != histogram_counts(self.score_values, self.psi_bin_edges):
            raise ValueError("Reference PSI bin counts do not match score_values.")
        _parse_aware_datetime(self.created_at, "reference created_at")


@dataclass(frozen=True)
class DriftObservation:
    """One persisted production score with comparable model lineage."""

    created_at: datetime
    model_name: str
    category: str
    anomaly_score: float
    is_anomaly: bool
    lineage: DriftLineage

    # ADD 2026-08-21: Production drift observation의 timestamp, score와 lineage를 검증한다.
    def validate(self) -> None:
        """Reject an invalid persisted inspection before statistics are computed."""
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("Drift observation created_at must be timezone-aware.")
        if not self.model_name or not self.category:
            raise ValueError("Drift observation model_name and category must not be empty.")
        _finite_float(self.anomaly_score, "anomaly_score")
        if type(self.is_anomaly) is not bool:
            raise TypeError("Drift observation is_anomaly must be boolean.")
        self.lineage.validate()


@dataclass(frozen=True)
class DriftPolicy:
    """Operational status thresholds kept separate from drift statistics."""

    minimum_sample_count: int = DEFAULT_MINIMUM_SAMPLE_COUNT
    psi_warning_threshold: float = DEFAULT_PSI_WARNING_THRESHOLD
    psi_drift_threshold: float = DEFAULT_PSI_DRIFT_THRESHOLD
    anomaly_ratio_warning_threshold: float = DEFAULT_ANOMALY_RATIO_WARNING_THRESHOLD
    anomaly_ratio_drift_threshold: float = DEFAULT_ANOMALY_RATIO_DRIFT_THRESHOLD

    # ADD 2026-08-21: Minimum sample과 ordered warning/drift threshold를 검증한다.
    def validate(self) -> None:
        """Reject policy values that make status classification ambiguous."""
        if self.minimum_sample_count <= 0:
            raise ValueError("Drift minimum_sample_count must be positive.")
        pairs = (
            (self.psi_warning_threshold, self.psi_drift_threshold, "PSI"),
            (
                self.anomaly_ratio_warning_threshold,
                self.anomaly_ratio_drift_threshold,
                "anomaly ratio",
            ),
        )
        for warning, drift, name in pairs:
            if not math.isfinite(warning) or not math.isfinite(drift):
                raise ValueError(f"Drift {name} thresholds must be finite.")
            if warning < 0 or warning >= drift:
                raise ValueError(f"Drift {name} thresholds require 0 <= warning < drift.")

    # ADD 2026-08-21: Drift status policy를 report JSON mapping으로 변환한다.
    def to_json_dict(self) -> dict[str, int | float]:
        """Return the operational thresholds used for one report."""
        return asdict(self)


# ADD 2026-08-21: CLI output identifier가 single safe path segment인지 검증한다.
def validate_artifact_id(value: str) -> None:
    """Reject traversal and ambiguous output identifiers."""
    if not _ARTIFACT_ID_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise ValueError("Drift artifact id must be one safe 1-128 character path segment.")


# ADD 2026-08-21: Expected manifest에 정확히 대응하는 normal validation score만 로드한다.
def load_validation_normal_scores(
    predictions_path: Path,
    *,
    expected_records: Sequence[ManifestRecord],
    category: str,
) -> tuple[float, ...]:
    """Load reference scores while forbidding test/anomaly leakage."""
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Validation prediction JSONL not found: {predictions_path}")
    if not expected_records:
        raise ValueError("Drift reference validation manifest must not be empty.")
    if any(
        record.split != REFERENCE_SPLIT
        or record.source_split != "train"
        or record.label != 0
        or record.category != category
        for record in expected_records
    ):
        raise ValueError("Drift reference manifest must contain validation-normal records only.")

    # JSONL record를 strict prediction schema로 복원하고 sample id 중복을 거부한다.
    try:
        lines = predictions_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read validation predictions: {predictions_path}") from exc
    records_by_id: dict[str, RawPredictionRecord] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"Validation prediction JSONL has a blank line at {line_number}.")
        try:
            record = RawPredictionRecord.from_json_dict(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid validation prediction JSON at line {line_number}.") from exc
        if record.sample_id in records_by_id:
            raise ValueError(f"Duplicate validation prediction sample_id: {record.sample_id}")
        records_by_id[record.sample_id] = record

    expected_by_id = {record.sample_id: record for record in expected_records}
    if len(expected_by_id) != len(expected_records) or set(records_by_id) != set(expected_by_id):
        raise ValueError("Reference prediction sample ids do not match the validation manifest.")

    # Manifest order를 고정하고 split/category/label metadata를 원본과 대조한다.
    scores: list[float] = []
    for expected in expected_records:
        record = records_by_id[expected.sample_id]
        if (
            record.split != REFERENCE_SPLIT
            or record.category != expected.category
            or record.defect_type != expected.defect_type
            or record.label != 0
        ):
            raise ValueError("Drift reference accepts only matching normal validation predictions.")
        scores.append(record.raw_anomaly_score)
    _require_finite(scores, "validation score")
    return tuple(scores)


# ADD 2026-08-21: Validated threshold provenance와 validation score로 reference를 생성한다.
def build_drift_reference(
    *,
    reference_id: str,
    thresholds: ThresholdArtifact,
    threshold_artifact_sha256: str,
    score_values: Sequence[float],
    psi_bin_count: int,
    psi_epsilon: float,
    created_at: str,
) -> DriftReference:
    """Build a reproducible validation-normal reference with fixed PSI bins."""
    validate_artifact_id(reference_id)
    thresholds.validate()
    scores = tuple(float(score) for score in score_values)
    _require_finite(scores, "reference score")
    if not scores:
        raise ValueError("Drift reference scores must not be empty.")
    if thresholds.validation_sample_count != len(scores):
        raise ValueError("Reference score count does not match threshold calibration count.")
    if max(scores) != thresholds.image_threshold:
        raise ValueError("Reference scores do not reproduce the calibrated image threshold.")
    if psi_bin_count < 2 or not math.isfinite(psi_epsilon) or psi_epsilon <= 0:
        raise ValueError("Reference PSI requires bin_count >= 2 and finite epsilon > 0.")
    if not is_sha256_digest(threshold_artifact_sha256):
        raise ValueError("Threshold artifact hash must be SHA-256.")

    edges = build_reference_bin_edges(scores, psi_bin_count)
    reference = DriftReference(
        schema_version=REFERENCE_SCHEMA_VERSION,
        reference_id=reference_id,
        model_name=thresholds.model_name,
        category=thresholds.category,
        lineage=DriftLineage(
            model_sha256=thresholds.model_sha256,
            artifact_metadata_sha256=thresholds.artifact_metadata_sha256,
            manifest_sha256=thresholds.manifest_sha256,
            threshold_artifact_sha256=threshold_artifact_sha256,
        ),
        source_split=REFERENCE_SPLIT,
        source_label=REFERENCE_LABEL,
        validation_predictions_sha256=thresholds.validation_predictions_sha256,
        sample_count=len(scores),
        score_values=scores,
        summary=summarize_scores(scores),
        image_threshold=thresholds.image_threshold,
        comparison_operator=thresholds.comparison_operator,
        reference_anomaly_ratio=anomaly_ratio(scores, threshold=thresholds.image_threshold),
        psi_bin_count_requested=psi_bin_count,
        psi_bin_edges=edges,
        reference_bin_counts=histogram_counts(scores, edges),
        psi_epsilon=psi_epsilon,
        created_at=created_at,
    )
    reference.validate()
    return reference


# ADD 2026-08-21: Finite score population의 summary와 linear-interpolated quantile을 계산한다.
def summarize_scores(scores: Sequence[float]) -> ScoreSummary:
    """Calculate deterministic population statistics for finite scores."""
    values = tuple(float(score) for score in scores)
    if not values:
        raise ValueError("Cannot summarize an empty score population.")
    _require_finite(values, "score")
    ordered = tuple(sorted(values))
    return ScoreSummary(
        mean=statistics.fmean(ordered),
        std=statistics.pstdev(ordered),
        minimum=ordered[0],
        maximum=ordered[-1],
        p50=_quantile(ordered, 0.50),
        p90=_quantile(ordered, 0.90),
        p95=_quantile(ordered, 0.95),
        p99=_quantile(ordered, 0.99),
    )


# ADD 2026-08-21: Reference quantile로 fixed PSI bin edge를 만들고 duplicate edge를 제거한다.
def build_reference_bin_edges(scores: Sequence[float], bin_count: int) -> tuple[float, ...]:
    """Build finite reference-derived edges including a stable constant-score fallback."""
    if bin_count < 2:
        raise ValueError("PSI bin_count must be at least 2.")
    ordered = tuple(sorted(float(score) for score in scores))
    if not ordered:
        raise ValueError("Cannot build PSI bins from empty reference scores.")
    _require_finite(ordered, "reference score")
    if ordered[0] == ordered[-1]:
        scale = max(abs(ordered[0]), 1.0) * 1e-9
        return (ordered[0] - scale, ordered[0] + scale)
    return tuple(sorted({_quantile(ordered, index / bin_count) for index in range(1, bin_count)}))


# ADD 2026-08-21: Fixed reference edge에 score를 배치해 underflow/overflow 포함 count를 계산한다.
def histogram_counts(scores: Sequence[float], bin_edges: Sequence[float]) -> tuple[int, ...]:
    """Count finite scores in fixed bins using right-open internal boundaries."""
    edges = tuple(float(edge) for edge in bin_edges)
    if tuple(sorted(set(edges))) != edges:
        raise ValueError("PSI bin edges must be finite, unique, and sorted.")
    _require_finite(edges, "PSI edge")
    counts = [0] * (len(edges) + 1)
    for score in scores:
        value = _finite_float(score, "score")
        counts[bisect.bisect_right(edges, value)] += 1
    return tuple(counts)


# ADD 2026-08-21: Additive smoothing을 적용한 fixed-bin Population Stability Index를 계산한다.
def calculate_psi(
    reference_counts: Sequence[int],
    current_counts: Sequence[int],
    *,
    epsilon: float,
) -> float:
    """Calculate finite PSI from aligned counts with symmetric additive smoothing."""
    if len(reference_counts) != len(current_counts) or not reference_counts:
        raise ValueError("PSI reference/current counts must be non-empty and aligned.")
    if any(type(count) is not int or count < 0 for count in (*reference_counts, *current_counts)):
        raise ValueError("PSI counts must be non-negative integers.")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("PSI epsilon must be finite and positive.")
    reference_total = sum(reference_counts)
    current_total = sum(current_counts)
    if reference_total <= 0 or current_total <= 0:
        raise ValueError("PSI requires non-empty reference and current populations.")

    bin_total = len(reference_counts)
    reference_denominator = reference_total + epsilon * bin_total
    current_denominator = current_total + epsilon * bin_total
    psi = 0.0
    for reference_count, current_count in zip(reference_counts, current_counts, strict=True):
        reference_ratio = (reference_count + epsilon) / reference_denominator
        current_ratio = (current_count + epsilon) / current_denominator
        psi += (current_ratio - reference_ratio) * math.log(current_ratio / reference_ratio)
    if not math.isfinite(psi):
        raise ValueError("PSI calculation produced a non-finite result.")
    return max(0.0, psi)


# ADD 2026-08-21: Strict score > threshold 기준 anomaly prediction ratio를 계산한다.
def anomaly_ratio(scores: Sequence[float], *, threshold: float) -> float:
    """Return the anomaly fraction without dividing by a reference ratio."""
    if not scores:
        raise ValueError("Anomaly ratio requires at least one score.")
    finite_threshold = _finite_float(threshold, "threshold")
    values = tuple(_finite_float(score, "score") for score in scores)
    return sum(score > finite_threshold for score in values) / len(values)


# ADD 2026-08-21: Statistics와 operational policy를 분리해 stable/warning/drift를 판정한다.
def classify_drift_status(
    *,
    sample_count: int,
    psi: float,
    anomaly_ratio_absolute_delta: float,
    policy: DriftPolicy,
) -> str:
    """Classify one current window using explicit operational thresholds."""
    policy.validate()
    finite_psi = _finite_float(psi, "PSI")
    finite_ratio_delta = _finite_float(anomaly_ratio_absolute_delta, "anomaly ratio absolute delta")
    if sample_count < policy.minimum_sample_count:
        return "insufficient_data"
    if (
        finite_psi >= policy.psi_drift_threshold
        or finite_ratio_delta >= policy.anomaly_ratio_drift_threshold
    ):
        return "drift"
    if (
        finite_psi >= policy.psi_warning_threshold
        or finite_ratio_delta >= policy.anomaly_ratio_warning_threshold
    ):
        return "warning"
    return "stable"


# ADD 2026-08-21: Current production window를 reference와 비교해 immutable drift payload를 만든다.
def build_drift_report(
    *,
    drift_id: str,
    reference: DriftReference,
    observations: Sequence[DriftObservation],
    since: datetime,
    until: datetime,
    policy: DriftPolicy,
    created_at: str,
) -> dict[str, Any]:
    """Validate lineage and calculate a deterministic batch drift report."""
    validate_artifact_id(drift_id)
    reference.validate()
    policy.validate()
    _validate_window(since, until)
    _parse_aware_datetime(created_at, "drift created_at")

    # DB row 전체가 requested window와 reference model lineage에 속하는지 먼저 검증한다.
    for observation in observations:
        observation.validate()
        if not since <= observation.created_at < until:
            raise ValueError("Drift observation falls outside the requested time window.")
        if observation.model_name != reference.model_name:
            raise ValueError("Drift observation model_name does not match the reference.")
        if observation.category != reference.category:
            raise ValueError("Drift observation category does not match the reference.")
        if observation.lineage != reference.lineage:
            raise ValueError("Mixed or mismatched drift model lineage detected.")
        expected_anomaly = observation.anomaly_score > reference.image_threshold
        if observation.is_anomaly is not expected_anomaly:
            raise ValueError("Persisted anomaly result violates the reference threshold contract.")

    scores = tuple(observation.anomaly_score for observation in observations)
    current_summary = summarize_scores(scores) if scores else None
    current_ratio = anomaly_ratio(scores, threshold=reference.image_threshold) if scores else 0.0
    current_counts = histogram_counts(scores, reference.psi_bin_edges)
    psi = (
        calculate_psi(
            reference.reference_bin_counts,
            current_counts,
            epsilon=reference.psi_epsilon,
        )
        if scores
        else 0.0
    )
    ratio_delta = abs(current_ratio - reference.reference_anomaly_ratio)
    status = classify_drift_status(
        sample_count=len(scores),
        psi=psi,
        anomaly_ratio_absolute_delta=ratio_delta,
        policy=policy,
    )

    # Descriptive statistics와 status policy를 별도 section으로 기록한다.
    statistics_payload: dict[str, Any] = {
        "psi": psi,
        "current_bin_counts": list(current_counts),
        "anomaly_ratio": current_ratio,
        "reference_anomaly_ratio": reference.reference_anomaly_ratio,
        "anomaly_ratio_absolute_delta": ratio_delta,
        "mean_delta": None,
        "p50_delta": None,
        "p95_delta": None,
    }
    if current_summary is not None:
        statistics_payload.update(
            {
                "mean_delta": current_summary.mean - reference.summary.mean,
                "p50_delta": current_summary.p50 - reference.summary.p50,
                "p95_delta": current_summary.p95 - reference.summary.p95,
            }
        )
    return {
        "schema_version": DRIFT_SCHEMA_VERSION,
        "drift_id": drift_id,
        "model_name": reference.model_name,
        "category": reference.category,
        "lineage": reference.lineage.to_json_dict(),
        "reference": {
            "reference_id": reference.reference_id,
            "sample_count": reference.sample_count,
            "source_split": reference.source_split,
            "source_label": reference.source_label,
            "summary": reference.summary.to_json_dict(),
        },
        "current_window": {
            "since": since.isoformat(),
            "until": until.isoformat(),
            "boundary": "since_inclusive_until_exclusive",
            "sample_count": len(scores),
            "summary": None if current_summary is None else current_summary.to_json_dict(),
        },
        "statistics": statistics_payload,
        "policy": policy.to_json_dict(),
        "status": status,
        "created_at": created_at,
    }


# ADD 2026-08-21: Drift reference JSON을 overwrite 없이 finite deterministic 형식으로 저장한다.
def write_drift_reference(reference: DriftReference, path: Path) -> None:
    """Persist one validated reference without overwriting existing evidence."""
    reference.validate()
    _write_json_artifact(path, reference.to_json_dict())


# ADD 2026-08-21: Persisted drift reference를 schema와 invariant 검증 후 로드한다.
def read_drift_reference(path: Path) -> DriftReference:
    """Read and validate one drift reference artifact."""
    raw = _read_json_artifact(path, "Drift reference")
    return DriftReference.from_json_dict(raw)


# ADD 2026-08-21: Drift report JSON을 overwrite 없이 finite deterministic 형식으로 저장한다.
def write_drift_report(report: dict[str, Any], path: Path) -> None:
    """Persist one drift report without overwriting existing operational evidence."""
    if report.get("schema_version") != DRIFT_SCHEMA_VERSION:
        raise ValueError("Drift report schema_version is invalid.")
    _write_json_artifact(path, report)


# ADD 2026-08-21: Linear interpolation 방식으로 ordered population quantile을 계산한다.
def _quantile(ordered: Sequence[float], probability: float) -> float:
    if not ordered or not 0 <= probability <= 1:
        raise ValueError("Quantile requires ordered values and probability in [0, 1].")
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(ordered[lower_index])
    weight = position - lower_index
    return float(ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight)


# ADD 2026-08-21: Drift report의 timezone-aware half-open time window를 검증한다.
def _validate_window(since: datetime, until: datetime) -> None:
    if since.tzinfo is None or since.utcoffset() is None:
        raise ValueError("Drift since must be timezone-aware.")
    if until.tzinfo is None or until.utcoffset() is None:
        raise ValueError("Drift until must be timezone-aware.")
    if since >= until:
        raise ValueError("Drift time window requires since < until.")


# ADD 2026-08-21: ISO timestamp string이 timezone-aware datetime인지 검증해 반환한다.
def _parse_aware_datetime(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 datetime.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone offset.")
    return parsed


# ADD 2026-08-21: JSON artifact를 external path에서 읽고 parse failure를 domain error로 변환한다.
def _read_json_artifact(path: Path, name: str) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {name.lower()}: {path}") from exc


# ADD 2026-08-21: JSON artifact를 parent 존재 계약과 overwrite 금지 하에 저장한다.
def _write_json_artifact(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"Drift artifact already exists: {path}")
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


# ADD 2026-08-21: Sequence의 모든 numeric value가 finite인지 검증한다.
def _require_finite(values: Sequence[float], field: str) -> None:
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError(f"{field} values must be finite.")


# ADD 2026-08-21: JSON scalar를 boolean 제외 finite float로 검증한다.
def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number.")
    return result


# ADD 2026-08-21: JSON array를 finite float tuple로 검증한다.
def _finite_float_tuple(value: object, field: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array.")
    return tuple(_finite_float(item, field) for item in value)


# ADD 2026-08-21: JSON array를 boolean 제외 non-negative integer tuple로 검증한다.
def _integer_tuple(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{field} must be a non-negative integer array.")
    return tuple(value)


# ADD 2026-08-21: JSON field를 boolean 제외 integer로 검증한다.
def _required_integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer.")
    return value


# ADD 2026-08-21: JSON field를 non-empty string으로 검증한다.
def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string.")
    return value
