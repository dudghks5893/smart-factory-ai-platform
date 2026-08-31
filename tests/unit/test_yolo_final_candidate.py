"""Focused C4-3 final-candidate eligibility, provenance, and seal tests."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ml.experiments.yolo_final_candidate import (
    FINAL_CANDIDATE_SELECTION_BASIS,
    FINAL_CANDIDATE_STATE,
    FINAL_TEST_STATE,
    build_final_candidate_manifest,
    load_final_candidate_manifest,
    load_official_candidate_evidence,
)
from pipelines.freeze_yolo_final_candidate import build_parser, freeze_yolo_final_candidate
from shared.hashing import sha256_bytes, sha256_file

SELECTED_AT = "2026-09-01T12:00:00+09:00"
EXPERIMENT_ID = "fixture_confirmed_candidate"
GIT_COMMIT = "4a5c9721214e48f9d25ab0fcc51d212b3bee0eb9"
MANIFEST_SHA256 = "a" * 64
FROZEN_MANIFEST_PATH = Path("configs/model/yolo_segmentation_final_candidate.json")
FROZEN_MANIFEST_SHA256 = "2a26b1bc03a1876f828e12a625c69c76af5e8c5713e3f64be699feffe2e8aa09"

type PayloadMutation = Callable[
    [dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]],
    None,
]


# ADD 2026-09-01: Portable Official package fixture를 deterministic bytes로 만든다.
def _write_official_package(
    tmp_path: Path,
    *,
    status: str = "CONFIRMED_CANDIDATE",
    decision: str = "CONFIRMED_CANDIDATE",
    test_used: bool = False,
    test_split_used: bool = False,
    mutation: PayloadMutation | None = None,
    checksum_override_entry: str | None = None,
    duplicate_entry: str | None = None,
) -> tuple[Path, dict[str, str]]:
    model_bytes = b"immutable-yolo-model"
    config_bytes = b"experiment:\n  experiment_id: fixture_confirmed_candidate\n"
    model_sha256 = sha256_bytes(model_bytes)
    config_sha256 = sha256_bytes(config_bytes)
    metadata: dict[str, Any] = {
        "architecture": "yolo11n-seg",
        "best_epoch": 88,
        "checkpoint_sha256": model_sha256,
        "dataset_manifest_sha256": MANIFEST_SHA256,
        "framework": "ultralytics",
        "framework_version": "8.4.128",
        "model_name": "yolo11n-seg.pt",
        "seed": 42,
        "task": "segment",
    }
    metadata_bytes = (json.dumps(metadata, sort_keys=True) + "\n").encode()
    metadata_sha256 = sha256_bytes(metadata_bytes)
    quality_before: dict[str, Any] = {
        "split": "val",
        "test_split_used": False,
    }
    region_evidence: dict[str, Any] = {
        "test_used": test_used,
        "gt_component_coverage_recall_at_50": 0.8260869565,
        "small_gt_coverage_recall_at_50": 0.625,
        "class_aware_union_iou": 0.7646175341,
        "class_aware_union_gt_coverage": 0.8391536014,
        "class_aware_union_pred_precision": 0.8959237129,
    }
    train_view_evidence: dict[str, Any] = {
        "canonical_manifest_sha256": MANIFEST_SHA256,
        "test_used": False,
        "validation_used_for_sampling": False,
    }
    quality_after: dict[str, Any] = {
        "split": "val",
        "test_split_used": test_split_used,
        "ultralytics": {"mask": {"map50": 0.78, "map50_95": 0.46, "recall": 0.75}},
        "diagnostic": {
            "precision": 0.89,
            "recall": 0.74,
            "f1": 0.81,
            "tp": 17,
            "fp": 2,
            "fn": 6,
        },
        "failure_modes": {
            "small_recall": 0.375,
            "medium_recall": 1.0,
            "large_recall": 0.875,
            "multi_component_recall": 0.5714285714,
            "single_component_recall": 1.0,
            "good_negative_fp_image_count": 0,
            "complete_miss_sample_count": 0,
            "wrong_class_sample_count": 0,
        },
        "secondary_region_coverage": region_evidence,
    }
    result: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "validation fixture",
        "controlled_change": {"field": "training.recipe", "before": "a", "after": "b"},
        "constants": {"model": "yolo11n-seg", "seed": 42},
        "split": "val",
        "test_used": test_used,
        "test_split_used": test_split_used,
        "quality_before": quality_before,
        "quality_after": quality_after,
        "resource_metrics": {},
        "failure_mode_metrics": {},
        "model_sha256": model_sha256,
        "metadata_sha256": metadata_sha256,
        "manifest_sha256": MANIFEST_SHA256,
        "experiment_config_sha256": config_sha256,
        "model_size_bytes": len(model_bytes),
        "decision": decision,
        "decision_reason": "All absolute validation gates passed.",
        "status": status,
        "repository": {"git_commit": GIT_COMMIT, "working_tree_dirty": False},
        "train_view": train_view_evidence,
        "primary_confirmation": {
            "decision": decision,
            "checks": {
                "small_recall_above_floor": True,
                "mask_map50_95_floor": True,
                "multi_recall_floor": True,
                "good_negative_fp_guardrail": True,
            },
        },
    }
    historical_baseline: dict[str, Any] = {
        "derived_test_metrics_used_for_selection": False,
        "validation_framework": {"split": "val", "test_split_used": False},
    }
    evidence_documents: dict[str, dict[str, Any]] = {
        "experiment_metadata": {
            "experiment_id": EXPERIMENT_ID,
            "status": status,
            "decision": decision,
            "git_commit": GIT_COMMIT,
            "working_tree_dirty": False,
            "dataset_manifest_sha256": MANIFEST_SHA256,
            "experiment_config_sha256": config_sha256,
            "test_used": False,
            "validation_protocol": {"split": "val", "test_split_used": False},
            "train_view": train_view_evidence,
            "historical_baseline_evidence": historical_baseline,
        },
        "validation": {
            "split": "val",
            "test_split_used": False,
            "quality_before": quality_before,
            "quality_after": quality_after,
        },
        "error_analysis": {"split": "val", "test_split_used": False},
        "comparison": {
            "split": "val",
            "test_split_used": False,
            "quality_before": quality_before,
            "quality_after": quality_after,
            "historical_baseline_reference": historical_baseline,
            "primary_confirmation": result["primary_confirmation"],
            "secondary_evidence": {
                "blocking": False,
                "region_coverage": region_evidence,
            },
        },
        "region": region_evidence,
        "train_view": train_view_evidence,
        "visualization": {
            "experiment_id": EXPERIMENT_ID,
            "dataset_manifest_sha256": MANIFEST_SHA256,
            "split": "train_val_only",
            "test_split_used": False,
            "repository": {"git_commit": GIT_COMMIT, "working_tree_dirty": False},
            "entries": [{"source_split": "val"}, {"source_split": "none"}],
        },
    }
    if mutation is not None:
        mutation(result, metadata, evidence_documents)
        metadata_bytes = (json.dumps(metadata, sort_keys=True) + "\n").encode()
        if result.get("metadata_sha256") == metadata_sha256:
            result["metadata_sha256"] = sha256_bytes(metadata_bytes)
    result_bytes = (json.dumps(result, sort_keys=True) + "\n").encode()
    package_entries = {
        "model/model.pt": model_bytes,
        "model/metadata.json": metadata_bytes,
        f"config/{EXPERIMENT_ID}.yaml": config_bytes,
        "evidence/experiment_result.json": result_bytes,
        "evidence/experiment_metadata.json": (
            json.dumps(evidence_documents["experiment_metadata"], sort_keys=True) + "\n"
        ).encode(),
        "evidence/validation_metrics.json": (
            json.dumps(evidence_documents["validation"], sort_keys=True) + "\n"
        ).encode(),
        "evidence/error_analysis_summary.json": (
            json.dumps(evidence_documents["error_analysis"], sort_keys=True) + "\n"
        ).encode(),
        "evidence/comparison_to_baseline.json": (
            json.dumps(evidence_documents["comparison"], sort_keys=True) + "\n"
        ).encode(),
        "evidence/region_coverage.json": (
            json.dumps(evidence_documents["region"], sort_keys=True) + "\n"
        ).encode(),
        "evidence/train_view_metadata.json": (
            json.dumps(evidence_documents["train_view"], sort_keys=True) + "\n"
        ).encode(),
        "evidence/visualization_manifest.json": (
            json.dumps(evidence_documents["visualization"], sort_keys=True) + "\n"
        ).encode(),
    }
    package_hashes = {entry: sha256_bytes(content) for entry, content in package_entries.items()}
    if checksum_override_entry is not None:
        package_hashes[checksum_override_entry] = "f" * 64
    checksum_bytes = (
        "\n".join(f"{digest}  {entry}" for entry, digest in sorted(package_hashes.items())) + "\n"
    ).encode()
    package_path = tmp_path / "candidate.zip"
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry, content in package_entries.items():
            archive.writestr(entry, content)
        archive.writestr("SHA256SUMS.txt", checksum_bytes)
        if duplicate_entry is not None:
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr(duplicate_entry, package_entries[duplicate_entry])
    return package_path, {
        "package": sha256_file(package_path),
        "model": model_sha256,
        "metadata": sha256_bytes(metadata_bytes),
        "config": config_sha256,
        "result": sha256_bytes(result_bytes),
    }


# ADD 2026-09-01: Confirmed Official candidate를 exact frozen pointer로 저장하고 복원한다.
def test_confirmed_candidate_can_be_frozen_with_exact_sha_and_sealed_test(tmp_path: Path) -> None:
    package_path, hashes = _write_official_package(tmp_path)
    output_path = tmp_path / "final_candidate.json"
    freeze_yolo_final_candidate(
        official_package_path=package_path,
        expected_package_sha256=hashes["package"],
        output_path=output_path,
        selected_at=SELECTED_AT,
    )
    manifest = load_final_candidate_manifest(output_path)
    assert manifest.selection_state == FINAL_CANDIDATE_STATE
    assert manifest.selection_basis == FINAL_CANDIDATE_SELECTION_BASIS
    assert manifest.final_test_state == FINAL_TEST_STATE
    assert manifest.test_used is False
    assert manifest.test_split_used is False
    assert manifest.official_package_sha256 == hashes["package"]
    assert manifest.model_sha256 == hashes["model"]
    assert manifest.metadata_sha256 == hashes["metadata"]
    assert manifest.experiment_config_sha256 == hashes["config"]
    assert manifest.packaged_experiment_result_sha256 == hashes["result"]
    assert str(tmp_path) not in output_path.read_text(encoding="utf-8")


# ADD 2026-09-01: Rejected/Pending candidate가 final freeze eligibility를 얻지 못하는지 검증한다.
@pytest.mark.parametrize(
    ("status", "decision"),
    (
        ("REJECTED", "REJECT"),
        ("COMPLETED", "PENDING"),
        ("COMPLETED", "CONFIRMED_CANDIDATE"),
        ("CONFIRMED_CANDIDATE", "PENDING"),
    ),
)
def test_rejected_or_pending_candidate_cannot_be_frozen(
    tmp_path: Path,
    status: str,
    decision: str,
) -> None:
    package_path, hashes = _write_official_package(
        tmp_path,
        status=status,
        decision=decision,
    )
    with pytest.raises(ValueError, match="CONFIRMED_CANDIDATE"):
        load_official_candidate_evidence(
            package_path,
            expected_package_sha256=hashes["package"],
        )


# ADD 2026-09-01: 어느 test-access flag든 true이면 freeze 전에 fail-fast하는지 검증한다.
@pytest.mark.parametrize(("test_used", "test_split_used"), ((True, False), (False, True)))
def test_candidate_with_test_access_cannot_be_frozen(
    tmp_path: Path,
    test_used: bool,
    test_split_used: bool,
) -> None:
    package_path, hashes = _write_official_package(
        tmp_path,
        test_used=test_used,
        test_split_used=test_split_used,
    )
    with pytest.raises(ValueError, match="validation-only|sealed|test"):
        load_official_candidate_evidence(
            package_path,
            expected_package_sha256=hashes["package"],
        )


# ADD 2026-09-01: Missing/incorrect model provenance를 package byte boundary에서 거부한다.
@pytest.mark.parametrize("mode", ("missing", "incorrect"))
def test_missing_or_incorrect_provenance_fails(tmp_path: Path, mode: str) -> None:
    def mutate(
        result: dict[str, Any],
        _: dict[str, Any],
        __: dict[str, dict[str, Any]],
    ) -> None:
        if mode == "missing":
            result.pop("model_sha256")
        else:
            result["model_sha256"] = "f" * 64

    package_path, hashes = _write_official_package(tmp_path, mutation=mutate)
    with pytest.raises(ValueError, match="missing|provenance"):
        load_official_candidate_evidence(
            package_path,
            expected_package_sha256=hashes["package"],
        )


# ADD 2026-09-01: External package trust-anchor mismatch를 internal evidence보다 먼저 거부한다.
def test_incorrect_package_sha_fails(tmp_path: Path) -> None:
    package_path, _ = _write_official_package(tmp_path)
    with pytest.raises(ValueError, match="package SHA-256"):
        load_official_candidate_evidence(
            package_path,
            expected_package_sha256="f" * 64,
        )


# ADD 2026-09-01: Metadata/config/result checksum과 duplicate member ambiguity를 거부한다.
@pytest.mark.parametrize(
    "mode",
    ("metadata", "config", "result_checksum", "duplicate_model", "duplicate_result"),
)
def test_package_member_identity_is_fail_closed(tmp_path: Path, mode: str) -> None:
    def mutate(
        result: dict[str, Any],
        _: dict[str, Any],
        __: dict[str, dict[str, Any]],
    ) -> None:
        if mode == "metadata":
            result["metadata_sha256"] = "f" * 64
        elif mode == "config":
            result["experiment_config_sha256"] = "f" * 64

    checksum_override = "evidence/experiment_result.json" if mode == "result_checksum" else None
    duplicate_entry = {
        "duplicate_model": "model/model.pt",
        "duplicate_result": "evidence/experiment_result.json",
    }.get(mode)
    package_path, hashes = _write_official_package(
        tmp_path,
        mutation=mutate,
        checksum_override_entry=checksum_override,
        duplicate_entry=duplicate_entry,
    )
    with pytest.raises(ValueError, match="provenance|SHA256SUMS|exactly once"):
        load_official_candidate_evidence(
            package_path,
            expected_package_sha256=hashes["package"],
        )


# ADD 2026-09-01: Dirty repository와 failed Primary check가 Official freeze를 차단하는지 검증한다.
@pytest.mark.parametrize("mode", ("dirty", "primary_failed", "primary_missing"))
def test_nonofficial_or_failed_primary_evidence_cannot_be_frozen(
    tmp_path: Path,
    mode: str,
) -> None:
    def mutate(
        result: dict[str, Any],
        _: dict[str, Any],
        __: dict[str, dict[str, Any]],
    ) -> None:
        if mode == "dirty":
            result["repository"]["working_tree_dirty"] = True
        elif mode == "primary_failed":
            result["primary_confirmation"]["checks"]["small_recall_above_floor"] = False
        else:
            result["primary_confirmation"]["checks"].pop("small_recall_above_floor")

    package_path, hashes = _write_official_package(tmp_path, mutation=mutate)
    with pytest.raises(ValueError, match="clean committed|Primary"):
        load_official_candidate_evidence(
            package_path,
            expected_package_sha256=hashes["package"],
        )


# ADD 2026-09-01: Root/nested evidence의 required seal field 누락을 fail-closed로 거부한다.
@pytest.mark.parametrize(
    "mode",
    (
        "root",
        "quality_before",
        "quality_after",
        "train_view",
        "experiment_metadata",
        "historical_derived_test",
        "visualization_test",
    ),
)
def test_missing_required_test_seal_field_fails(tmp_path: Path, mode: str) -> None:
    def mutate(
        result: dict[str, Any],
        _: dict[str, Any],
        documents: dict[str, dict[str, Any]],
    ) -> None:
        if mode == "root":
            result.pop("test_used")
        elif mode == "quality_before":
            result["quality_before"].pop("test_split_used")
        elif mode == "quality_after":
            result["quality_after"].pop("test_split_used")
        elif mode == "train_view":
            result["train_view"].pop("test_used")
        elif mode == "experiment_metadata":
            documents["experiment_metadata"].pop("test_used")
        elif mode == "historical_derived_test":
            documents["experiment_metadata"]["historical_baseline_evidence"][
                "derived_test_metrics_used_for_selection"
            ] = True
        else:
            documents["visualization"]["entries"][0]["source_split"] = "test"

    package_path, hashes = _write_official_package(tmp_path, mutation=mutate)
    with pytest.raises(ValueError, match="explicitly|validation-only|non-validation"):
        load_official_candidate_evidence(
            package_path,
            expected_package_sha256=hashes["package"],
        )


# ADD 2026-09-01: Same evidence/timestamp의 frozen serialization이 byte-identical한지 검증한다.
def test_final_candidate_serialization_is_deterministic(tmp_path: Path) -> None:
    package_path, hashes = _write_official_package(tmp_path)
    evidence = load_official_candidate_evidence(
        package_path,
        expected_package_sha256=hashes["package"],
    )
    first = build_final_candidate_manifest(evidence, selected_at=SELECTED_AT)
    second = build_final_candidate_manifest(replace(evidence), selected_at=SELECTED_AT)
    assert first.to_json_bytes() == second.to_json_bytes()


# ADD 2026-09-01: CLI regeneration이 explicit evidence timestamp 없이는 시작되지 않게 한다.
def test_freeze_cli_requires_deterministic_selected_at() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--official-package",
                "candidate.zip",
                "--expected-package-sha256",
                "a" * 64,
            ]
        )


# ADD 2026-09-01: Unknown/missing/wrong-typed frozen manifest field를 strict loader가 거부한다.
@pytest.mark.parametrize("mode", ("unknown", "missing", "wrong_type", "test_used"))
def test_malformed_freeze_manifest_fails(tmp_path: Path, mode: str) -> None:
    payload = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    if mode == "unknown":
        payload["unexpected"] = True
    elif mode == "missing":
        payload.pop("model_sha256")
    elif mode == "wrong_type":
        payload["model_size_bytes"] = True
    else:
        payload["test_used"] = True
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema|invalid|sealed"):
        load_final_candidate_manifest(path)


# ADD 2026-09-01: Repository freeze pointer가 approved C4-2C identity와 byte SHA를 유지한다.
def test_repository_frozen_manifest_matches_official_c4_2c_identity() -> None:
    manifest = load_final_candidate_manifest(FROZEN_MANIFEST_PATH)
    assert sha256_file(FROZEN_MANIFEST_PATH) == FROZEN_MANIFEST_SHA256
    assert manifest.selected_experiment_id == (
        "c4_2c_yolo11n_seg_crop350_nomosaic_maskratio2_seed42"
    )
    assert manifest.repository_git_commit == GIT_COMMIT
    assert manifest.dataset_manifest_sha256 == (
        "1746338c091c18e96a11399c81ea9be0d7350105c4860cfa6a4162144ddb9905"
    )
    assert manifest.experiment_config_sha256 == (
        "258bf33955c06c5dbbbbeb4d162d5a50d125ae594135ccefbbce9b7d324572a3"
    )
    assert manifest.official_package_sha256 == (
        "81c721ab6d34e5563e9f8907fe4c9914d50e48ef35aacfabb6f4ca745420cd76"
    )
    assert manifest.model_sha256 == (
        "e3fd10cdd708d31421feacfc5d694cb638e0ea60672e08796391b33aecf67155"
    )
    assert manifest.metadata_sha256 == (
        "2d301687f1ee025f367b536d052b55eeba507c40bc95d265821337489ddeca2b"
    )
    assert manifest.packaged_experiment_result_sha256 == (
        "17e9abb231f2eee8f08831e99e8d963b2ab190151d91e0ab7f146e2d62f3fdcb"
    )
    assert all(manifest.primary_confirmation_checks.values())
    assert manifest.final_test_state == FINAL_TEST_STATE
