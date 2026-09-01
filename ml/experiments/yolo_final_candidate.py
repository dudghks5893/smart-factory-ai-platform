"""Validation-only YOLO final-candidate selection and freeze contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from ml.experiments.yolo_segmentation import validate_experiment_result
from ml.training.yolo_segmentation import validate_artifact_id
from shared.hashing import is_sha256_digest, sha256_file

FINAL_CANDIDATE_SCHEMA_VERSION = 1
FINAL_CANDIDATE_STATE = "FINAL_CANDIDATE_FROZEN"
FINAL_CANDIDATE_SELECTION_BASIS = "VALIDATION_ONLY"
FINAL_TEST_STATE = "SEALED_NOT_USED"
OFFICIAL_EVIDENCE_SOURCE = "OFFICIAL_EXPERIMENT_PACKAGE"
OFFICIAL_MODEL_ENTRY = "model/model.pt"
OFFICIAL_METADATA_ENTRY = "model/metadata.json"
OFFICIAL_RESULT_ENTRY = "evidence/experiment_result.json"
OFFICIAL_EXPERIMENT_METADATA_ENTRY = "evidence/experiment_metadata.json"
OFFICIAL_VALIDATION_ENTRY = "evidence/validation_metrics.json"
OFFICIAL_ERROR_ANALYSIS_ENTRY = "evidence/error_analysis_summary.json"
OFFICIAL_COMPARISON_ENTRY = "evidence/comparison_to_baseline.json"
OFFICIAL_REGION_ENTRY = "evidence/region_coverage.json"
OFFICIAL_TRAIN_VIEW_ENTRY = "evidence/train_view_metadata.json"
OFFICIAL_VISUALIZATION_ENTRY = "evidence/visualization_manifest.json"
OFFICIAL_SHA_MANIFEST_ENTRY = "SHA256SUMS.txt"
REQUIRED_PRIMARY_CHECKS = frozenset(
    {
        "small_recall_above_floor",
        "mask_map50_95_floor",
        "multi_recall_floor",
        "good_negative_fp_guardrail",
    }
)
REQUIRED_VALIDATION_METRICS = frozenset(
    {
        "mask_map50_95",
        "mask_map50",
        "framework_mask_recall",
        "strict_precision",
        "strict_recall",
        "strict_f1",
        "strict_tp",
        "strict_fp",
        "strict_fn",
        "small_recall",
        "medium_recall",
        "large_recall",
        "multi_component_recall",
        "single_component_recall",
        "good_negative_fp_image_count",
        "complete_miss_sample_count",
        "wrong_class_sample_count",
        "gt_component_coverage_recall_at_50",
        "small_gt_coverage_recall_at_50",
        "class_aware_union_iou",
        "class_aware_union_gt_coverage",
        "class_aware_union_prediction_precision",
    }
)
COUNT_VALIDATION_METRICS = frozenset(
    {
        "strict_tp",
        "strict_fp",
        "strict_fn",
        "good_negative_fp_image_count",
        "complete_miss_sample_count",
        "wrong_class_sample_count",
    }
)


@dataclass(frozen=True)
class OfficialCandidateEvidence:
    """Validated identities and validation evidence read from one Official package."""

    experiment_id: str
    status: str
    decision: str
    decision_reason: str
    repository_git_commit: str
    dataset_manifest_sha256: str
    experiment_config_sha256: str
    official_package_sha256: str
    model_sha256: str
    metadata_sha256: str
    packaged_experiment_result_sha256: str
    model_size_bytes: int
    task: str
    model_family: str
    selected_model_name: str
    framework: str
    framework_version: str
    seed: int
    best_epoch: int
    validation_metrics: dict[str, float | int]
    primary_confirmation_checks: dict[str, bool]
    test_used: bool
    test_split_used: bool


@dataclass(frozen=True)
class FinalCandidateManifest:
    """Repository-owned immutable pointer to the validation-selected YOLO artifact."""

    schema_version: int
    task: str
    model_family: str
    selected_experiment_id: str
    selected_model_name: str
    selection_state: str
    selection_basis: str
    selected_at: str
    repository_git_commit: str
    dataset_manifest_sha256: str
    experiment_config_sha256: str
    official_package_sha256: str
    model_sha256: str
    metadata_sha256: str
    packaged_experiment_result_sha256: str
    model_size_bytes: int
    framework: str
    framework_version: str
    seed: int
    best_epoch: int
    validation_metrics: dict[str, float | int]
    primary_confirmation_checks: dict[str, bool]
    test_used: bool
    test_split_used: bool
    final_test_state: str
    evidence_source: str
    official_model_entry: str
    official_metadata_entry: str
    official_result_entry: str
    reason_for_selection: str

    # ADD 2026-09-01: Frozen pointer가 exact provenance와 sealed-test state를 유지하는지 검증한다.
    def validate(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != FINAL_CANDIDATE_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported YOLO final-candidate schema version.")
        required_strings = (
            "task",
            "model_family",
            "selected_experiment_id",
            "selected_model_name",
            "selection_state",
            "selection_basis",
            "selected_at",
            "repository_git_commit",
            "dataset_manifest_sha256",
            "experiment_config_sha256",
            "official_package_sha256",
            "model_sha256",
            "metadata_sha256",
            "packaged_experiment_result_sha256",
            "framework",
            "framework_version",
            "final_test_state",
            "evidence_source",
            "official_model_entry",
            "official_metadata_entry",
            "official_result_entry",
            "reason_for_selection",
        )
        if any(
            type(getattr(self, name)) is not str or not getattr(self, name)
            for name in required_strings
        ):
            raise ValueError("Frozen YOLO string field is missing or invalid.")
        if (
            self.selection_state != FINAL_CANDIDATE_STATE
            or self.selection_basis != FINAL_CANDIDATE_SELECTION_BASIS
            or self.final_test_state != FINAL_TEST_STATE
        ):
            raise ValueError("YOLO final-candidate lifecycle state is invalid.")
        if self.test_used is not False or self.test_split_used is not False:
            raise ValueError("Frozen YOLO candidate must keep the final test sealed and unused.")
        if self.evidence_source != OFFICIAL_EVIDENCE_SOURCE:
            raise ValueError("Frozen YOLO candidate must originate from an Official package.")
        if (
            self.official_model_entry != OFFICIAL_MODEL_ENTRY
            or self.official_metadata_entry != OFFICIAL_METADATA_ENTRY
            or self.official_result_entry != OFFICIAL_RESULT_ENTRY
        ):
            raise ValueError("Frozen YOLO package entry identity is invalid.")
        validate_artifact_id(self.selected_experiment_id)
        if not re.fullmatch(r"[0-9a-f]{40}", self.repository_git_commit):
            raise ValueError("Official experiment Git commit must be a full lowercase SHA-1.")
        for name in (
            "dataset_manifest_sha256",
            "experiment_config_sha256",
            "official_package_sha256",
            "model_sha256",
            "metadata_sha256",
            "packaged_experiment_result_sha256",
        ):
            if not is_sha256_digest(getattr(self, name)):
                raise ValueError(f"Frozen YOLO provenance field is not SHA-256: {name}")
        if (
            type(self.model_size_bytes) is not int
            or type(self.seed) is not int
            or type(self.best_epoch) is not int
            or self.model_size_bytes <= 0
            or self.seed < 0
            or self.best_epoch < 0
        ):
            raise ValueError("Frozen YOLO model size, seed, or best epoch is invalid.")
        if type(self.test_used) is not bool or type(self.test_split_used) is not bool:
            raise ValueError("Frozen YOLO test-seal fields must be booleans.")
        if not isinstance(self.validation_metrics, dict) or not isinstance(
            self.primary_confirmation_checks, dict
        ):
            raise ValueError("Frozen YOLO metrics/checks must be objects.")
        _validate_selected_at(self.selected_at)
        _validate_primary_checks(self.primary_confirmation_checks)
        _validate_metrics(self.validation_metrics)

    # ADD 2026-09-01: Manifest를 stable key order의 strict JSON bytes로 직렬화한다.
    def to_json_bytes(self) -> bytes:
        self.validate()
        return (json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


# ADD 2026-09-01: Official package의 고정 entry를 streaming SHA로 검증한다.
def _hash_zip_entry(archive: zipfile.ZipFile, entry: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with archive.open(entry, "r") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except KeyError as exc:
        raise FileNotFoundError(f"Official package entry is missing: {entry}") from exc
    return digest.hexdigest(), size


# ADD 2026-09-01: Official package의 고정 JSON evidence만 strict object로 읽는다.
def _read_zip_json(archive: zipfile.ZipFile, entry: str) -> tuple[dict[str, Any], str]:
    try:
        content = archive.read(entry)
    except KeyError as exc:
        raise FileNotFoundError(f"Official package entry is missing: {entry}") from exc
    try:
        payload: object = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Official package JSON is invalid: {entry}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Official package JSON root must be an object: {entry}")
    return payload, hashlib.sha256(content).hexdigest()


# ADD 2026-09-01: Required ZIP member가 정확히 한 번만 존재하도록 identity ambiguity를 차단한다.
def _validate_unique_zip_entries(archive: zipfile.ZipFile, entries: tuple[str, ...]) -> None:
    member_counts: dict[str, int] = {}
    for info in archive.infolist():
        member_counts[info.filename] = member_counts.get(info.filename, 0) + 1
    invalid = [entry for entry in entries if member_counts.get(entry) != 1]
    if invalid:
        raise ValueError(
            "Official package required entries must exist exactly once: " + ", ".join(invalid)
        )


# ADD 2026-09-01: Package SHA manifest를 strict unique entry-to-digest mapping으로 읽는다.
def _read_sha256_manifest(archive: zipfile.ZipFile) -> dict[str, str]:
    try:
        content = archive.read(OFFICIAL_SHA_MANIFEST_ENTRY).decode("ascii")
    except KeyError as exc:
        raise FileNotFoundError("Official package SHA256SUMS.txt is missing.") from exc
    except UnicodeDecodeError as exc:
        raise ValueError("Official package SHA256SUMS.txt must be ASCII.") from exc
    hashes: dict[str, str] = {}
    for line in content.splitlines():
        digest, separator, entry = line.partition("  ")
        if not separator or not entry or not is_sha256_digest(digest) or entry in hashes:
            raise ValueError("Official package SHA256SUMS.txt is malformed or ambiguous.")
        hashes[entry] = digest
    return hashes


# ADD 2026-09-01: Nested Official evidence object를 누락 없이 읽는다.
def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Official candidate evidence object is missing: {key}")
    return value


# ADD 2026-09-01: Official evidence의 required string field를 읽는다.
def _string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Official candidate evidence string is missing: {key}")
    return value


# ADD 2026-09-01: Official validation metric을 finite numeric value로 읽는다.
def _number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Official candidate metric is invalid: {key}")
    if not math.isfinite(float(value)):
        raise ValueError(f"Official candidate metric is invalid: {key}")
    return float(value)


# ADD 2026-09-01: Count/epoch/seed field를 non-negative integer로 읽는다.
def _integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"Official candidate integer is invalid: {key}")
    return value


# ADD 2026-09-01: Required seal flag가 명시적 boolean false인지 검증한다.
def _require_false(payload: dict[str, Any], key: str, *, context: str) -> None:
    if key not in payload or payload[key] is not False:
        raise ValueError(f"{context} must explicitly set {key}=false.")


# ADD 2026-09-01: Validation evidence가 explicit val/test-sealed protocol인지 검증한다.
def _validate_validation_seal(payload: dict[str, Any], *, context: str) -> None:
    if payload.get("split") != "val":
        raise ValueError(f"{context} must explicitly use split=val.")
    _require_false(payload, "test_split_used", context=context)
    if "test_used" in payload:
        _require_false(payload, "test_used", context=context)


# ADD 2026-09-01: Historical Baseline reference가 validation-only selection source인지 검증한다.
def _validate_historical_reference(payload: dict[str, Any], *, context: str) -> None:
    _require_false(payload, "derived_test_metrics_used_for_selection", context=context)
    validation = _mapping(payload, "validation_framework")
    _validate_validation_seal(validation, context=f"{context} validation framework")


# ADD 2026-09-01: Selection timestamp가 timezone-aware ISO-8601인지 검증한다.
def _validate_selected_at(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Final-candidate selected_at must be ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Final-candidate selected_at must include a timezone offset.")


# ADD 2026-09-01: C4-2C required Primary checks가 모두 존재하고 PASS인지 검증한다.
def _validate_primary_checks(checks: dict[str, bool]) -> None:
    if set(checks) != REQUIRED_PRIMARY_CHECKS or any(
        value is not True for value in checks.values()
    ):
        raise ValueError("Official candidate must pass every required Primary validation check.")


# ADD 2026-09-01: Frozen validation snapshot이 complete finite metric set인지 검증한다.
def _validate_metrics(metrics: dict[str, float | int]) -> None:
    if set(metrics) != REQUIRED_VALIDATION_METRICS:
        raise ValueError("Frozen YOLO validation metric set is incomplete.")
    if any(
        type(value) not in {int, float} or not math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise ValueError("Frozen YOLO validation metrics must be finite numbers.")
    if any(
        type(metrics[name]) is not int or metrics[name] < 0 for name in COUNT_VALIDATION_METRICS
    ):
        raise ValueError("Frozen YOLO validation count metrics must be non-negative integers.")


# ADD 2026-09-01: Official result에서 freeze에 필요한 validation snapshot만 추출한다.
def _validation_metrics(result: dict[str, Any]) -> dict[str, float | int]:
    quality = _mapping(result, "quality_after")
    ultralytics = _mapping(quality, "ultralytics")
    mask = _mapping(ultralytics, "mask")
    diagnostic = _mapping(quality, "diagnostic")
    failures = _mapping(quality, "failure_modes")
    region = _mapping(quality, "secondary_region_coverage")
    return {
        "mask_map50_95": _number(mask, "map50_95"),
        "mask_map50": _number(mask, "map50"),
        "framework_mask_recall": _number(mask, "recall"),
        "strict_precision": _number(diagnostic, "precision"),
        "strict_recall": _number(diagnostic, "recall"),
        "strict_f1": _number(diagnostic, "f1"),
        "strict_tp": _integer(diagnostic, "tp"),
        "strict_fp": _integer(diagnostic, "fp"),
        "strict_fn": _integer(diagnostic, "fn"),
        "small_recall": _number(failures, "small_recall"),
        "medium_recall": _number(failures, "medium_recall"),
        "large_recall": _number(failures, "large_recall"),
        "multi_component_recall": _number(failures, "multi_component_recall"),
        "single_component_recall": _number(failures, "single_component_recall"),
        "good_negative_fp_image_count": _integer(failures, "good_negative_fp_image_count"),
        "complete_miss_sample_count": _integer(failures, "complete_miss_sample_count"),
        "wrong_class_sample_count": _integer(failures, "wrong_class_sample_count"),
        "gt_component_coverage_recall_at_50": _number(region, "gt_component_coverage_recall_at_50"),
        "small_gt_coverage_recall_at_50": _number(region, "small_gt_coverage_recall_at_50"),
        "class_aware_union_iou": _number(region, "class_aware_union_iou"),
        "class_aware_union_gt_coverage": _number(region, "class_aware_union_gt_coverage"),
        "class_aware_union_prediction_precision": _number(
            region, "class_aware_union_pred_precision"
        ),
    }


# ADD 2026-09-01: Official package identity와 validation-only eligibility를 교차 검증한다.
def load_official_candidate_evidence(
    package_path: Path,
    *,
    expected_package_sha256: str,
) -> OfficialCandidateEvidence:
    if not is_sha256_digest(expected_package_sha256):
        raise ValueError("Expected Official package SHA-256 is invalid.")
    actual_package_sha256 = sha256_file(package_path)
    if actual_package_sha256 != expected_package_sha256:
        raise ValueError("Official candidate package SHA-256 does not match the trust anchor.")

    # 고정된 candidate evidence와 artifact entry만 읽고 dataset/test namespace는 탐색하지 않는다.
    with zipfile.ZipFile(package_path, "r") as archive:
        _validate_unique_zip_entries(
            archive,
            (OFFICIAL_RESULT_ENTRY, OFFICIAL_SHA_MANIFEST_ENTRY),
        )
        result, result_sha256 = _read_zip_json(archive, OFFICIAL_RESULT_ENTRY)
        validate_experiment_result(result)
        experiment_id = _string(result, "experiment_id")
        validate_artifact_id(experiment_id)
        config_entry = f"config/{experiment_id}.yaml"
        required_entries = (
            OFFICIAL_MODEL_ENTRY,
            OFFICIAL_METADATA_ENTRY,
            OFFICIAL_RESULT_ENTRY,
            OFFICIAL_EXPERIMENT_METADATA_ENTRY,
            OFFICIAL_VALIDATION_ENTRY,
            OFFICIAL_ERROR_ANALYSIS_ENTRY,
            OFFICIAL_COMPARISON_ENTRY,
            OFFICIAL_REGION_ENTRY,
            OFFICIAL_TRAIN_VIEW_ENTRY,
            OFFICIAL_VISUALIZATION_ENTRY,
            OFFICIAL_SHA_MANIFEST_ENTRY,
            config_entry,
        )
        _validate_unique_zip_entries(archive, required_entries)
        sha_manifest = _read_sha256_manifest(archive)
        config_sha256, _ = _hash_zip_entry(archive, config_entry)
        model_sha256, model_size_bytes = _hash_zip_entry(archive, OFFICIAL_MODEL_ENTRY)
        metadata, metadata_sha256 = _read_zip_json(archive, OFFICIAL_METADATA_ENTRY)
        experiment_metadata, experiment_metadata_sha256 = _read_zip_json(
            archive, OFFICIAL_EXPERIMENT_METADATA_ENTRY
        )
        validation, validation_sha256 = _read_zip_json(archive, OFFICIAL_VALIDATION_ENTRY)
        error_analysis, error_analysis_sha256 = _read_zip_json(
            archive, OFFICIAL_ERROR_ANALYSIS_ENTRY
        )
        comparison, comparison_sha256 = _read_zip_json(archive, OFFICIAL_COMPARISON_ENTRY)
        region_evidence, region_sha256 = _read_zip_json(archive, OFFICIAL_REGION_ENTRY)
        train_view_evidence, train_view_sha256 = _read_zip_json(archive, OFFICIAL_TRAIN_VIEW_ENTRY)
        visualization, visualization_sha256 = _read_zip_json(archive, OFFICIAL_VISUALIZATION_ENTRY)

    computed_hashes = {
        OFFICIAL_MODEL_ENTRY: model_sha256,
        OFFICIAL_METADATA_ENTRY: metadata_sha256,
        OFFICIAL_RESULT_ENTRY: result_sha256,
        OFFICIAL_EXPERIMENT_METADATA_ENTRY: experiment_metadata_sha256,
        OFFICIAL_VALIDATION_ENTRY: validation_sha256,
        OFFICIAL_ERROR_ANALYSIS_ENTRY: error_analysis_sha256,
        OFFICIAL_COMPARISON_ENTRY: comparison_sha256,
        OFFICIAL_REGION_ENTRY: region_sha256,
        OFFICIAL_TRAIN_VIEW_ENTRY: train_view_sha256,
        OFFICIAL_VISUALIZATION_ENTRY: visualization_sha256,
        config_entry: config_sha256,
    }
    if any(sha_manifest.get(entry) != digest for entry, digest in computed_hashes.items()):
        raise ValueError("Official package entry bytes do not match SHA256SUMS.txt.")

    if result.get("status") != "CONFIRMED_CANDIDATE" or result.get("decision") != (
        "CONFIRMED_CANDIDATE"
    ):
        raise ValueError("Only an Official CONFIRMED_CANDIDATE can be frozen.")
    _validate_validation_seal(result, context="Official experiment result")
    _require_false(result, "test_used", context="Official experiment result")
    quality_before = _mapping(result, "quality_before")
    quality = _mapping(result, "quality_after")
    _validate_validation_seal(quality_before, context="Baseline comparison evidence")
    _validate_validation_seal(quality, context="Candidate quality evidence")
    region = _mapping(quality, "secondary_region_coverage")
    _require_false(region, "test_used", context="Candidate Region evidence")
    if region_evidence != region:
        raise ValueError("Standalone and result-embedded Region evidence do not match.")
    _require_false(region_evidence, "test_used", context="Standalone Region evidence")
    result_train_view = _mapping(result, "train_view")
    _require_false(result_train_view, "test_used", context="Result train-view evidence")
    _require_false(
        result_train_view,
        "validation_used_for_sampling",
        context="Result train-view evidence",
    )
    if train_view_evidence != result_train_view:
        raise ValueError("Standalone and result-embedded train-view evidence do not match.")
    _require_false(train_view_evidence, "test_used", context="Standalone train-view evidence")
    _validate_validation_seal(validation, context="Framework validation evidence")
    if (
        _mapping(validation, "quality_before") != quality_before
        or _mapping(validation, "quality_after") != quality
    ):
        raise ValueError(
            "Framework validation and experiment result quality evidence do not match."
        )
    _validate_validation_seal(error_analysis, context="Strict diagnostic evidence")
    _validate_validation_seal(comparison, context="Comparison evidence")
    if (
        _mapping(comparison, "quality_before") != quality_before
        or _mapping(comparison, "quality_after") != quality
    ):
        raise ValueError("Comparison and experiment result quality evidence do not match.")
    confirmation = _mapping(result, "primary_confirmation")
    if confirmation.get("decision") != "CONFIRMED_CANDIDATE":
        raise ValueError("Official candidate Primary confirmation decision is invalid.")
    raw_checks = _mapping(confirmation, "checks")
    if any(type(value) is not bool for value in raw_checks.values()):
        raise ValueError("Official candidate Primary check values must be booleans.")
    checks = {key: bool(value) for key, value in raw_checks.items()}
    _validate_primary_checks(checks)
    if _mapping(comparison, "primary_confirmation") != confirmation:
        raise ValueError("Comparison and experiment result Primary confirmation do not match.")
    comparison_secondary = _mapping(comparison, "secondary_evidence")
    _require_false(comparison_secondary, "blocking", context="Comparison secondary evidence")
    if _mapping(comparison_secondary, "region_coverage") != region:
        raise ValueError("Comparison and experiment result Region evidence do not match.")

    repository = _mapping(result, "repository")
    git_commit = _string(repository, "git_commit")
    if not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise ValueError("Official candidate repository commit is invalid.")
    if repository.get("working_tree_dirty") is not False:
        raise ValueError("Official candidate must originate from a clean committed repository.")

    result_model_sha256 = _string(result, "model_sha256")
    result_metadata_sha256 = _string(result, "metadata_sha256")
    result_manifest_sha256 = _string(result, "manifest_sha256")
    result_config_sha256 = _string(result, "experiment_config_sha256")
    if (
        model_sha256 != result_model_sha256
        or metadata_sha256 != result_metadata_sha256
        or config_sha256 != result_config_sha256
    ):
        raise ValueError("Official package bytes do not match experiment-result provenance.")
    if _string(metadata, "checkpoint_sha256") != model_sha256:
        raise ValueError("Official model SHA does not match artifact metadata.")
    if _string(metadata, "dataset_manifest_sha256") != result_manifest_sha256:
        raise ValueError("Official Manifest SHA does not match artifact metadata.")
    if _integer(result, "model_size_bytes") != model_size_bytes:
        raise ValueError("Official model size does not match packaged model bytes.")
    constants = _mapping(result, "constants")
    if _string(metadata, "architecture") != _string(constants, "model") or _integer(
        metadata, "seed"
    ) != _integer(constants, "seed"):
        raise ValueError("Official artifact metadata does not match experiment model constants.")

    if (
        _string(experiment_metadata, "experiment_id") != experiment_id
        or experiment_metadata.get("status") != "CONFIRMED_CANDIDATE"
        or experiment_metadata.get("decision") != "CONFIRMED_CANDIDATE"
        or _string(experiment_metadata, "git_commit") != git_commit
        or experiment_metadata.get("working_tree_dirty") is not False
        or _string(experiment_metadata, "dataset_manifest_sha256") != result_manifest_sha256
        or _string(experiment_metadata, "experiment_config_sha256") != result_config_sha256
    ):
        raise ValueError("Official experiment metadata does not match result provenance.")
    _require_false(experiment_metadata, "test_used", context="Official experiment metadata")
    metadata_validation = _mapping(experiment_metadata, "validation_protocol")
    _validate_validation_seal(metadata_validation, context="Experiment validation protocol")
    metadata_train_view = _mapping(experiment_metadata, "train_view")
    if metadata_train_view != result_train_view:
        raise ValueError("Experiment metadata and result train-view evidence do not match.")
    if (
        _string(result_train_view, "canonical_manifest_sha256") != result_manifest_sha256
        or _string(train_view_evidence, "canonical_manifest_sha256") != result_manifest_sha256
    ):
        raise ValueError("Train-view evidence does not match the selected Dataset Manifest.")
    _validate_historical_reference(
        _mapping(experiment_metadata, "historical_baseline_evidence"),
        context="Experiment historical Baseline evidence",
    )
    _validate_historical_reference(
        _mapping(comparison, "historical_baseline_reference"),
        context="Comparison historical Baseline evidence",
    )

    if (
        visualization.get("split") != "train_val_only"
        or _string(visualization, "experiment_id") != experiment_id
        or _string(visualization, "dataset_manifest_sha256") != result_manifest_sha256
    ):
        raise ValueError("Visualization evidence identity is not train/validation-only.")
    _require_false(visualization, "test_split_used", context="Visualization evidence")
    visualization_repository = _mapping(visualization, "repository")
    if (
        _string(visualization_repository, "git_commit") != git_commit
        or visualization_repository.get("working_tree_dirty") is not False
    ):
        raise ValueError("Visualization evidence repository provenance is invalid.")
    visualization_entries = visualization.get("entries")
    if not isinstance(visualization_entries, list) or not visualization_entries:
        raise ValueError("Visualization evidence entries are missing.")
    for entry in visualization_entries:
        if not isinstance(entry, dict) or entry.get("source_split") not in {"val", "none"}:
            raise ValueError("Visualization evidence contains a non-validation source split.")

    validation_metrics = _validation_metrics(result)
    _validate_metrics(validation_metrics)
    return OfficialCandidateEvidence(
        experiment_id=experiment_id,
        status="CONFIRMED_CANDIDATE",
        decision="CONFIRMED_CANDIDATE",
        decision_reason=_string(result, "decision_reason"),
        repository_git_commit=git_commit,
        dataset_manifest_sha256=result_manifest_sha256,
        experiment_config_sha256=result_config_sha256,
        official_package_sha256=actual_package_sha256,
        model_sha256=model_sha256,
        metadata_sha256=metadata_sha256,
        packaged_experiment_result_sha256=result_sha256,
        model_size_bytes=model_size_bytes,
        task=_string(metadata, "task"),
        model_family=_string(metadata, "architecture"),
        selected_model_name=_string(metadata, "model_name"),
        framework=_string(metadata, "framework"),
        framework_version=_string(metadata, "framework_version"),
        seed=_integer(metadata, "seed"),
        best_epoch=_integer(metadata, "best_epoch"),
        validation_metrics=validation_metrics,
        primary_confirmation_checks=checks,
        test_used=False,
        test_split_used=False,
    )


# ADD 2026-09-02: Frozen pointer와 Official package evidence의 exact identity를 공유 검증한다.
def verify_official_candidate_identity(
    candidate: FinalCandidateManifest,
    evidence: OfficialCandidateEvidence,
) -> None:
    expected = {
        "experiment_id": candidate.selected_experiment_id,
        "repository_git_commit": candidate.repository_git_commit,
        "dataset_manifest_sha256": candidate.dataset_manifest_sha256,
        "experiment_config_sha256": candidate.experiment_config_sha256,
        "official_package_sha256": candidate.official_package_sha256,
        "model_sha256": candidate.model_sha256,
        "metadata_sha256": candidate.metadata_sha256,
        "packaged_experiment_result_sha256": candidate.packaged_experiment_result_sha256,
        "model_size_bytes": candidate.model_size_bytes,
        "task": candidate.task,
        "model_family": candidate.model_family,
        "selected_model_name": candidate.selected_model_name,
        "framework": candidate.framework,
        "framework_version": candidate.framework_version,
        "seed": candidate.seed,
        "best_epoch": candidate.best_epoch,
        "validation_metrics": candidate.validation_metrics,
        "primary_confirmation_checks": candidate.primary_confirmation_checks,
        "test_used": False,
        "test_split_used": False,
    }
    mismatches = [name for name, value in expected.items() if getattr(evidence, name) != value]
    if mismatches:
        raise ValueError(
            "Official package identity does not match the frozen candidate: "
            + ", ".join(sorted(mismatches))
        )


# ADD 2026-09-02: Verified package의 fixed model/metadata bytes만 runtime artifact로 복원한다.
def materialize_official_candidate_artifact(
    *,
    package_path: Path,
    candidate: FinalCandidateManifest,
    evidence: OfficialCandidateEvidence,
    artifact_dir: Path,
) -> Path:
    verify_official_candidate_identity(candidate, evidence)
    if sha256_file(package_path) != candidate.official_package_sha256:
        raise ValueError("Official candidate package bytes changed verified identity.")
    if artifact_dir.exists():
        raise FileExistsError(f"Official candidate artifact already exists: {artifact_dir}")
    model_dir = artifact_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(package_path, "r") as archive:
            for entry, destination in (
                (OFFICIAL_MODEL_ENTRY, model_dir / "model.pt"),
                (OFFICIAL_METADATA_ENTRY, model_dir / "metadata.json"),
            ):
                with archive.open(entry, "r") as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        if (
            sha256_file(model_dir / "model.pt") != candidate.model_sha256
            or sha256_file(model_dir / "metadata.json") != candidate.metadata_sha256
        ):
            raise RuntimeError("Materialized Official candidate bytes changed verified identity.")
    except Exception:
        shutil.rmtree(artifact_dir, ignore_errors=True)
        raise
    return artifact_dir


# ADD 2026-09-01: Eligible Official evidence를 validation-only frozen pointer로 승격한다.
def build_final_candidate_manifest(
    evidence: OfficialCandidateEvidence,
    *,
    selected_at: str,
) -> FinalCandidateManifest:
    if evidence.status != "CONFIRMED_CANDIDATE" or evidence.decision != "CONFIRMED_CANDIDATE":
        raise ValueError("Rejected or pending YOLO experiments cannot be frozen.")
    if evidence.test_used is not False or evidence.test_split_used is not False:
        raise ValueError("A YOLO candidate with derived-test access cannot be frozen.")
    _validate_primary_checks(evidence.primary_confirmation_checks)
    _validate_metrics(evidence.validation_metrics)
    manifest = FinalCandidateManifest(
        schema_version=FINAL_CANDIDATE_SCHEMA_VERSION,
        task=evidence.task,
        model_family=evidence.model_family,
        selected_experiment_id=evidence.experiment_id,
        selected_model_name=evidence.selected_model_name,
        selection_state=FINAL_CANDIDATE_STATE,
        selection_basis=FINAL_CANDIDATE_SELECTION_BASIS,
        selected_at=selected_at,
        repository_git_commit=evidence.repository_git_commit,
        dataset_manifest_sha256=evidence.dataset_manifest_sha256,
        experiment_config_sha256=evidence.experiment_config_sha256,
        official_package_sha256=evidence.official_package_sha256,
        model_sha256=evidence.model_sha256,
        metadata_sha256=evidence.metadata_sha256,
        packaged_experiment_result_sha256=evidence.packaged_experiment_result_sha256,
        model_size_bytes=evidence.model_size_bytes,
        framework=evidence.framework,
        framework_version=evidence.framework_version,
        seed=evidence.seed,
        best_epoch=evidence.best_epoch,
        validation_metrics=dict(evidence.validation_metrics),
        primary_confirmation_checks=dict(evidence.primary_confirmation_checks),
        test_used=False,
        test_split_used=False,
        final_test_state=FINAL_TEST_STATE,
        evidence_source=OFFICIAL_EVIDENCE_SOURCE,
        official_model_entry=OFFICIAL_MODEL_ENTRY,
        official_metadata_entry=OFFICIAL_METADATA_ENTRY,
        official_result_entry=OFFICIAL_RESULT_ENTRY,
        reason_for_selection=(
            "Official validation candidate passed every required Primary confirmation check "
            "with the derived test sealed; rejected, pending, and research-only runs are "
            "ineligible."
        ),
    )
    manifest.validate()
    return manifest


# ADD 2026-09-01: Frozen manifest를 overwrite 없이 repository metadata로 저장한다.
def write_final_candidate_manifest(manifest: FinalCandidateManifest, output_path: Path) -> Path:
    if output_path.exists():
        raise FileExistsError(f"YOLO final-candidate manifest already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(manifest.to_json_bytes())
    return output_path


# ADD 2026-09-01: C4-4가 사용할 frozen manifest를 strict schema로 복원한다.
def load_final_candidate_manifest(path: Path) -> FinalCandidateManifest:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("YOLO final-candidate manifest is not valid JSON.") from exc
    if not isinstance(raw, dict):
        raise ValueError("YOLO final-candidate manifest root must be an object.")
    expected_fields = {field.name for field in fields(FinalCandidateManifest)}
    if set(raw) != expected_fields:
        raise ValueError("YOLO final-candidate manifest fields do not match the schema.")
    metrics = raw.get("validation_metrics")
    checks = raw.get("primary_confirmation_checks")
    if not isinstance(metrics, dict) or not isinstance(checks, dict):
        raise ValueError("YOLO final-candidate manifest metrics/checks must be objects.")
    manifest = FinalCandidateManifest(
        **{
            **raw,
            "validation_metrics": dict(metrics),
            "primary_confirmation_checks": dict(checks),
        }
    )
    manifest.validate()
    return manifest
