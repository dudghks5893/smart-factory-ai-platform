"""C4-4 final-test seal, ordering, provenance, and one-time output tests."""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import ml.evaluation.yolo_final_test as final_test_module
from ml.evaluation.final_benchmark import RepositoryProvenance
from ml.evaluation.yolo_final_test import (
    EXPECTED_FINAL_CANDIDATE_MANIFEST_SHA256,
    FINAL_TEST_COMPLETED_STATE,
    FINAL_TEST_FAILED_STATE,
    FINAL_TEST_OUTPUT_ROOT,
    FINAL_TEST_PREPARATION_FAILED_STATE,
    FINAL_TEST_RUN_STATE_FILENAME,
    FinalTestMeasurements,
    FinalTestPreflight,
    execute_yolo_final_test,
    prepare_yolo_final_test,
)
from ml.experiments.yolo_final_candidate import (
    FinalCandidateManifest,
    OfficialCandidateEvidence,
    load_final_candidate_manifest,
)
from pipelines.evaluate_yolo_final_test import build_parser
from shared.hashing import sha256_file

FROZEN_MANIFEST = Path("configs/model/yolo_segmentation_final_candidate.json")
EXPERIMENT_CONFIG = Path(
    "configs/experiments/yolo_segmentation/"
    "c4_2c_yolo11n_seg_crop350_nomosaic_maskratio2_seed42.yaml"
)
BASELINE_CONFIG = Path("configs/model/yolo_segmentation_baseline.yaml")
PYPROJECT = Path("pyproject.toml")
EXECUTION_COMMIT = "d" * 40


# ADD 2026-09-01: Static frozen candidate를 matching Official evidence fixture로 변환한다.
def _official_evidence(candidate: FinalCandidateManifest) -> OfficialCandidateEvidence:
    return OfficialCandidateEvidence(
        experiment_id=candidate.selected_experiment_id,
        status="CONFIRMED_CANDIDATE",
        decision="CONFIRMED_CANDIDATE",
        decision_reason="fixture",
        repository_git_commit=candidate.repository_git_commit,
        dataset_manifest_sha256=candidate.dataset_manifest_sha256,
        experiment_config_sha256=candidate.experiment_config_sha256,
        official_package_sha256=candidate.official_package_sha256,
        model_sha256=candidate.model_sha256,
        metadata_sha256=candidate.metadata_sha256,
        packaged_experiment_result_sha256=candidate.packaged_experiment_result_sha256,
        model_size_bytes=candidate.model_size_bytes,
        task=candidate.task,
        model_family=candidate.model_family,
        selected_model_name=candidate.selected_model_name,
        framework=candidate.framework,
        framework_version=candidate.framework_version,
        seed=candidate.seed,
        best_epoch=candidate.best_epoch,
        validation_metrics=dict(candidate.validation_metrics),
        primary_confirmation_checks=dict(candidate.primary_confirmation_checks),
        test_used=False,
        test_split_used=False,
    )


# ADD 2026-09-01: Real test content 없이 complete typed measurement fixture를 만든다.
def _measurements() -> FinalTestMeasurements:
    component = {"precision": 0.8, "recall": 0.7, "map50": 0.75, "map50_95": 0.5}
    return FinalTestMeasurements(
        evaluation_split="test",
        framework_metric_source="ultralytics_model.val(split=test)",
        sample_count=1,
        framework_metrics={"box": dict(component), "mask": dict(component)},
        framework_per_class_metrics={
            class_name: {"box": dict(component), "mask": dict(component)}
            for class_name in ("bent", "color", "scratch")
        },
        diagnostic={
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "per_class": {},
        },
        failure_analysis={
            "size_analysis": {},
            "component_analysis": {},
            "negative_analysis": {},
            "error_taxonomy": {},
            "complete_miss_sample_count": 0,
            "wrong_class_sample_count": 0,
        },
        region_coverage={
            "strict_instance_gt_recall": 1.0,
            "gt_component_coverage_recall_at_50": 1.0,
            "small_gt_coverage_recall_at_50": 1.0,
            "class_aware_union_iou": 1.0,
            "class_aware_union_gt_coverage": 1.0,
            "class_aware_union_prediction_precision": 1.0,
            "near_miss_iou_030_to_050": 0,
            "covered50_but_strict_instance_fail": 0,
            "test_gt_total": 1,
            "small_gt_total": 1,
        },
        sample_analysis=({"sample_id": "synthetic-test-fixture"},),
        environment={
            "framework": "ultralytics",
            "framework_version": "8.4.128",
            "torch_version": "2.13.0",
            "device": "cpu",
        },
        resource_metrics={"evaluation_wall_time_seconds": 1.0},
        started_at="2026-09-01T00:00:00+00:00",
        completed_at="2026-09-01T00:00:01+00:00",
    )


# ADD 2026-09-01: Test evaluator 전에 synthetic artifact directory를 준비한다.
def _fixture_artifact_materializer(preflight: FinalTestPreflight) -> Path:
    artifact_dir = preflight.output_dir / "fixture-artifact"
    artifact_dir.mkdir()
    return artifact_dir


# ADD 2026-09-01: Synthetic artifact는 production verifier 없이 verified fixture로 취급한다.
def _fixture_artifact_verifier(_: FinalTestPreflight, __: Path) -> None:
    return None


# ADD 2026-09-01: Normal seal tests에 동일 fake materialization boundary를 적용한다.
def _execute_fixture(
    preflight: FinalTestPreflight,
    *,
    confirm_final_test: bool,
    evaluator: final_test_module.FinalTestEvaluator,
) -> final_test_module.FinalTestRunArtifacts | None:
    return execute_yolo_final_test(
        preflight,
        confirm_final_test=confirm_final_test,
        evaluator=evaluator,
        artifact_materializer=_fixture_artifact_materializer,
        artifact_verifier=_fixture_artifact_verifier,
    )


# ADD 2026-09-01: Namespace lifecycle JSON을 strict object로 읽는다.
def _run_state(preflight: FinalTestPreflight) -> dict[str, Any]:
    payload = json.loads(
        (preflight.output_dir / FINAL_TEST_RUN_STATE_FILENAME).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise AssertionError("Synthetic run state must be a JSON object.")
    return payload


# ADD 2026-09-01: Repository layout와 Manifest-byte-only dataset fixture를 준비한다.
@pytest.fixture
def preflight_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence]:
    repository_root = tmp_path / "repository"
    for source in (FROZEN_MANIFEST, EXPERIMENT_CONFIG, BASELINE_CONFIG, PYPROJECT):
        destination = repository_root / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    candidate = load_final_candidate_manifest(repository_root / FROZEN_MANIFEST)
    evidence = _official_evidence(candidate)
    package_path = tmp_path / "official.zip"
    package_path.write_bytes(b"package fixture is replaced by a verified evidence fake")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    dataset_manifest = dataset_root / "manifest.csv"
    dataset_manifest.write_text("lexical manifest bytes only\n", encoding="utf-8")
    real_sha256_file = sha256_file

    # ADD 2026-09-01: Fake Manifest만 approved byte hash로 해석한다.
    def fixture_sha256_file(path: Path) -> str:
        if path.resolve() == dataset_manifest.resolve():
            return candidate.dataset_manifest_sha256
        return real_sha256_file(path)

    # ADD 2026-09-01: Package I/O 없이 matching Official evidence를 반환한다.
    def fixture_package_loader(
        path: Path,
        *,
        expected_package_sha256: str,
    ) -> OfficialCandidateEvidence:
        assert path == package_path
        assert expected_package_sha256 == candidate.official_package_sha256
        return evidence

    monkeypatch.setattr(final_test_module, "sha256_file", fixture_sha256_file)
    monkeypatch.setattr(
        final_test_module,
        "load_official_candidate_evidence",
        fixture_package_loader,
    )
    return (
        {
            "repository_root": repository_root,
            "package_path": package_path,
            "dataset_root": dataset_root,
        },
        candidate,
        evidence,
    )


# ADD 2026-09-01: Fixture resources로 clean committed preflight를 수행한다.
def _prepare(paths: dict[str, Path]) -> FinalTestPreflight:
    return prepare_yolo_final_test(
        frozen_manifest_path=FROZEN_MANIFEST,
        official_package_path=paths["package_path"],
        dataset_root=paths["dataset_root"],
        repository_root=paths["repository_root"],
        provenance_resolver=lambda _: RepositoryProvenance(
            git_commit=EXECUTION_COMMIT,
            working_tree_dirty=False,
        ),
    )


# ADD 2026-09-01: Confirmation 부재가 evaluator와 namespace 모두 열지 않는지 검증한다.
def test_missing_confirmation_exits_sealed_before_test_evaluator(
    preflight_fixture: tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence],
) -> None:
    paths, _, _ = preflight_fixture
    preflight = _prepare(paths)
    called = False
    materializer_called = False

    # ADD 2026-09-01: Confirmation 부재에서 artifact preparation 호출 여부를 기록한다.
    def materializer(_: FinalTestPreflight) -> Path:
        nonlocal materializer_called
        materializer_called = True
        return preflight.output_dir / "must-not-exist"

    # ADD 2026-09-01: 호출 여부를 기록하는 synthetic evaluator를 제공한다.
    def evaluator(_: FinalTestPreflight, __: Path) -> FinalTestMeasurements:
        nonlocal called
        called = True
        return _measurements()

    result = execute_yolo_final_test(
        preflight,
        confirm_final_test=False,
        evaluator=evaluator,
        artifact_materializer=materializer,
        artifact_verifier=_fixture_artifact_verifier,
    )
    assert result is None
    assert called is False
    assert materializer_called is False
    assert not preflight.output_dir.exists()


# ADD 2026-09-01: Invalid frozen lifecycle/seal field가 evaluator 전에 fail-fast하는지 검증한다.
@pytest.mark.parametrize(
    "mutation",
    (
        {"selection_state": "NOT_FROZEN"},
        {"selection_basis": "TEST_SELECTED"},
        {"final_test_state": "FINAL_TEST_COMPLETED"},
        {"test_used": True},
        {"test_split_used": True},
    ),
)
def test_invalid_frozen_state_never_invokes_test_evaluator(
    preflight_fixture: tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence],
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, Any],
) -> None:
    paths, candidate, _ = preflight_fixture
    monkeypatch.setattr(
        final_test_module,
        "load_final_candidate_manifest",
        lambda _: replace(candidate, **mutation),
    )
    called = False

    # ADD 2026-09-01: Invalid lifecycle에서 호출되면 실패할 evaluator를 제공한다.
    def evaluator(_: FinalTestPreflight, __: Path) -> FinalTestMeasurements:
        nonlocal called
        called = True
        return _measurements()

    with pytest.raises(ValueError, match="eligible"):
        preflight = _prepare(paths)
        _execute_fixture(preflight, confirm_final_test=True, evaluator=evaluator)
    assert called is False


# ADD 2026-09-01: Missing strict seal field가 package/test resolver 전에 거부되는지 검증한다.
def test_missing_seal_field_never_invokes_test_evaluator(
    preflight_fixture: tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _, _ = preflight_fixture
    monkeypatch.setattr(
        final_test_module,
        "load_final_candidate_manifest",
        lambda _: (_ for _ in ()).throw(ValueError("manifest fields do not match schema")),
    )
    called = False

    # ADD 2026-09-01: Missing schema에서 호출되면 실패할 evaluator를 제공한다.
    def evaluator(_: FinalTestPreflight, __: Path) -> FinalTestMeasurements:
        nonlocal called
        called = True
        return _measurements()

    with pytest.raises(ValueError, match="schema"):
        preflight = _prepare(paths)
        _execute_fixture(preflight, confirm_final_test=True, evaluator=evaluator)
    assert called is False


# ADD 2026-09-01: Wrong frozen manifest SHA가 package/test resolver 전에 거부되는지 검증한다.
def test_wrong_frozen_manifest_sha_never_invokes_test_evaluator(
    preflight_fixture: tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _, _ = preflight_fixture
    monkeypatch.setattr(final_test_module, "EXPECTED_FINAL_CANDIDATE_MANIFEST_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="manifest SHA-256"):
        _prepare(paths)


# ADD 2026-09-01: Package trust-anchor failure가 test resolver 전에 전파되는지 검증한다.
def test_wrong_package_sha_never_invokes_test_evaluator(
    preflight_fixture: tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _, _ = preflight_fixture

    # ADD 2026-09-01: Package trust-anchor mismatch를 synthetic failure로 재현한다.
    def rejecting_loader(_: Path, *, expected_package_sha256: str) -> OfficialCandidateEvidence:
        raise ValueError(f"package SHA-256 mismatch: {expected_package_sha256}")

    monkeypatch.setattr(final_test_module, "load_official_candidate_evidence", rejecting_loader)
    with pytest.raises(ValueError, match="package SHA-256"):
        _prepare(paths)


# ADD 2026-09-01: Dirty repository provenance가 output/test boundary 전에 거부되는지 검증한다.
def test_dirty_git_never_reaches_test_evaluator(
    preflight_fixture: tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence],
) -> None:
    paths, _, _ = preflight_fixture
    with pytest.raises(ValueError, match="clean committed"):
        prepare_yolo_final_test(
            frozen_manifest_path=FROZEN_MANIFEST,
            official_package_path=paths["package_path"],
            dataset_root=paths["dataset_root"],
            repository_root=paths["repository_root"],
            provenance_resolver=lambda _: RepositoryProvenance(
                git_commit=EXECUTION_COMMIT,
                working_tree_dirty=True,
            ),
        )


# ADD 2026-09-01: Protocol-bearing config byte drift가 test evaluator 전에 거부되는지 검증한다.
def test_protocol_config_drift_never_reaches_test_evaluator(
    preflight_fixture: tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence],
) -> None:
    paths, _, _ = preflight_fixture
    config_path = paths["repository_root"] / EXPERIMENT_CONFIG
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "diagnostic_confidence: 0.25",
            "diagnostic_confidence: 0.20",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config SHA-256"):
        _prepare(paths)


# ADD 2026-09-01: Forged in-memory protocol override가 evaluator 전에 거부되는지 검증한다.
def test_in_memory_protocol_override_never_invokes_test_evaluator(
    preflight_fixture: tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence],
) -> None:
    paths, _, _ = preflight_fixture
    preflight = _prepare(paths)
    forged = replace(
        preflight,
        protocol=replace(preflight.protocol, diagnostic_confidence=0.20),
    )
    called = False

    # ADD 2026-09-01: Forged protocol에서 evaluator 호출 여부를 기록한다.
    def evaluator(_: FinalTestPreflight, __: Path) -> FinalTestMeasurements:
        nonlocal called
        called = True
        return _measurements()

    with pytest.raises(RuntimeError, match="protocol changed"):
        _execute_fixture(
            forged,
            confirm_final_test=True,
            evaluator=evaluator,
        )
    assert called is False
    assert not preflight.output_dir.exists()


# ADD 2026-09-01: Alternate output root로 same-candidate rerun namespace를 만들지 못하게 한다.
def test_alternate_output_root_is_rejected(
    preflight_fixture: tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence],
) -> None:
    paths, _, _ = preflight_fixture
    with pytest.raises(ValueError, match="output root"):
        prepare_yolo_final_test(
            frozen_manifest_path=FROZEN_MANIFEST,
            official_package_path=paths["package_path"],
            dataset_root=paths["dataset_root"],
            repository_root=paths["repository_root"],
            output_root=FINAL_TEST_OUTPUT_ROOT / "alternate",
            provenance_resolver=lambda _: RepositoryProvenance(
                git_commit=EXECUTION_COMMIT,
                working_tree_dirty=False,
            ),
        )


# ADD 2026-09-01: Official evidence의 지정 identity field를 불일치 fixture로 바꾼다.
def _mismatched_evidence(
    evidence: OfficialCandidateEvidence,
    *,
    field: str,
) -> OfficialCandidateEvidence:
    if field == "experiment_id":
        return replace(evidence, experiment_id="different_candidate")
    if field == "selected_model_name":
        return replace(evidence, selected_model_name="different-model.pt")
    if field == "best_epoch":
        return replace(evidence, best_epoch=evidence.best_epoch + 1)
    digest = "f" * 64
    if field == "model_sha256":
        return replace(evidence, model_sha256=digest)
    if field == "metadata_sha256":
        return replace(evidence, metadata_sha256=digest)
    if field == "experiment_config_sha256":
        return replace(evidence, experiment_config_sha256=digest)
    if field == "packaged_experiment_result_sha256":
        return replace(evidence, packaged_experiment_result_sha256=digest)
    if field == "dataset_manifest_sha256":
        return replace(evidence, dataset_manifest_sha256=digest)
    raise AssertionError(f"Unsupported mismatch fixture field: {field}")


# ADD 2026-09-01: Package/candidate identity mismatch가 test resolver 전에 거부되는지 검증한다.
@pytest.mark.parametrize(
    "field",
    (
        "model_sha256",
        "metadata_sha256",
        "experiment_config_sha256",
        "packaged_experiment_result_sha256",
        "dataset_manifest_sha256",
        "experiment_id",
        "selected_model_name",
        "best_epoch",
    ),
)
def test_official_identity_mismatch_never_invokes_test_evaluator(
    preflight_fixture: tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    paths, _, evidence = preflight_fixture
    mismatched = _mismatched_evidence(evidence, field=field)
    monkeypatch.setattr(
        final_test_module,
        "load_official_candidate_evidence",
        lambda *_args, **_kwargs: mismatched,
    )
    with pytest.raises(ValueError, match="does not match"):
        _prepare(paths)


# ADD 2026-09-01: Fully verified candidate와 explicit unlock만 injected evaluator를 호출한다.
def test_verified_candidate_with_confirmation_can_complete_once(
    preflight_fixture: tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence],
) -> None:
    paths, _, _ = preflight_fixture
    preflight = _prepare(paths)
    calls = 0

    # ADD 2026-09-01: Successful explicit unlock의 evaluator call count를 기록한다.
    def evaluator(_: FinalTestPreflight, __: Path) -> FinalTestMeasurements:
        nonlocal calls
        calls += 1
        return _measurements()

    result = _execute_fixture(
        preflight,
        confirm_final_test=True,
        evaluator=evaluator,
    )
    assert result is not None
    assert calls == 1
    assert result.evidence.final_test_state == FINAL_TEST_COMPLETED_STATE
    assert result.evidence.candidate_selection_changed is False
    assert result.evidence.threshold_tuned_on_test is False
    assert result.result_path.parent == preflight.output_dir / "completed"
    assert _run_state(preflight)["lifecycle_state"] == FINAL_TEST_COMPLETED_STATE
    assert preflight.output_dir.parent == paths["repository_root"] / FINAL_TEST_OUTPUT_ROOT
    with pytest.raises(FileExistsError, match="already exists"):
        _execute_fixture(
            preflight,
            confirm_final_test=True,
            evaluator=evaluator,
        )
    assert calls == 1


# ADD 2026-09-01: Final-test execution이 frozen candidate bytes를 변경하지 않는지 검증한다.
def test_final_test_cannot_change_frozen_candidate(
    preflight_fixture: tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence],
) -> None:
    paths, _, _ = preflight_fixture
    frozen_path = paths["repository_root"] / FROZEN_MANIFEST
    before = sha256_file(frozen_path)
    preflight = _prepare(paths)
    result = _execute_fixture(
        preflight,
        confirm_final_test=True,
        evaluator=lambda _preflight, _artifact_dir: _measurements(),
    )
    assert result is not None
    assert sha256_file(frozen_path) == before == EXPECTED_FINAL_CANDIDATE_MANIFEST_SHA256


# ADD 2026-09-01: Preflight 뒤 Manifest 변경이 unlock 직전 evaluator 호출을 차단하는지 검증한다.
def test_changed_dataset_manifest_is_rejected_before_test_evaluator(
    preflight_fixture: tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _, _ = preflight_fixture
    preflight = _prepare(paths)
    preflight_hasher = final_test_module.sha256_file
    called = False

    # ADD 2026-09-01: Unlock 직전 Dataset Manifest TOCTOU를 재현한다.
    def changed_manifest_hasher(path: Path) -> str:
        if path.resolve() == preflight.dataset_manifest_path.resolve():
            return "f" * 64
        return preflight_hasher(path)

    # ADD 2026-09-01: TOCTOU failure에서 호출 여부를 기록한다.
    def evaluator(_: FinalTestPreflight, __: Path) -> FinalTestMeasurements:
        nonlocal called
        called = True
        return _measurements()

    monkeypatch.setattr(final_test_module, "sha256_file", changed_manifest_hasher)
    with pytest.raises(RuntimeError, match="Dataset Manifest changed"):
        _execute_fixture(
            preflight,
            confirm_final_test=True,
            evaluator=evaluator,
        )
    assert called is False
    assert not preflight.output_dir.exists()


# ADD 2026-09-01: Materialized model mismatch가 evaluator 전 safe failure state를 남긴다.
def test_materialized_model_mismatch_never_invokes_evaluator(
    preflight_fixture: tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence],
) -> None:
    paths, _, _ = preflight_fixture
    preflight = _prepare(paths)
    called = False

    # ADD 2026-09-01: Wrong materialized SHA를 fail-fast하는 verifier를 재현한다.
    def rejecting_verifier(_: FinalTestPreflight, __: Path) -> None:
        raise RuntimeError("materialized artifact identity changed")

    # ADD 2026-09-01: Materialization failure에서 evaluator 호출 여부를 기록한다.
    def evaluator(_: FinalTestPreflight, __: Path) -> FinalTestMeasurements:
        nonlocal called
        called = True
        return _measurements()

    with pytest.raises(RuntimeError, match="materialized artifact"):
        execute_yolo_final_test(
            preflight,
            confirm_final_test=True,
            evaluator=evaluator,
            artifact_materializer=_fixture_artifact_materializer,
            artifact_verifier=rejecting_verifier,
        )
    state = _run_state(preflight)
    assert called is False
    assert state["lifecycle_state"] == FINAL_TEST_PREPARATION_FAILED_STATE
    assert state["test_access_started"] is False
    assert state["cleanup_permitted"] is True
    assert not (preflight.output_dir / "completed").exists()


# ADD 2026-09-01: Evaluator 중 materialized model drift가 completed evidence를 차단한다.
def test_materialized_model_drift_after_evaluator_is_not_completed(
    preflight_fixture: tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence],
) -> None:
    paths, _, _ = preflight_fixture
    preflight = _prepare(paths)
    verifier_calls = 0
    evaluator_calls = 0

    # ADD 2026-09-01: Unlock 전 검증은 통과시키고 post-evaluation recheck에서 거부한다.
    def drifting_verifier(_: FinalTestPreflight, __: Path) -> None:
        nonlocal verifier_calls
        verifier_calls += 1
        if verifier_calls == 2:
            raise RuntimeError("materialized artifact identity changed after evaluation")

    # ADD 2026-09-01: Authorized evaluator의 exact call count를 기록한다.
    def evaluator(_: FinalTestPreflight, __: Path) -> FinalTestMeasurements:
        nonlocal evaluator_calls
        evaluator_calls += 1
        return _measurements()

    with pytest.raises(RuntimeError, match="after evaluation"):
        execute_yolo_final_test(
            preflight,
            confirm_final_test=True,
            evaluator=evaluator,
            artifact_materializer=_fixture_artifact_materializer,
            artifact_verifier=drifting_verifier,
        )
    state = _run_state(preflight)
    assert evaluator_calls == 1
    assert verifier_calls == 2
    assert state["lifecycle_state"] == FINAL_TEST_FAILED_STATE
    assert state["test_access_started"] is True
    assert state["cleanup_permitted"] is False
    assert not (preflight.output_dir / "completed").exists()


# ADD 2026-09-01: Evaluator failure는 completed evidence 없이 non-rerunnable partial state를 남긴다.
def test_evaluator_failure_is_not_completed_or_automatically_rerunnable(
    preflight_fixture: tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence],
) -> None:
    paths, _, _ = preflight_fixture
    preflight = _prepare(paths)
    calls = 0

    # ADD 2026-09-01: Authorized evaluator가 시작된 뒤 failure를 재현한다.
    def failing_evaluator(_: FinalTestPreflight, __: Path) -> FinalTestMeasurements:
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic evaluator failure")

    with pytest.raises(RuntimeError, match="synthetic evaluator"):
        _execute_fixture(
            preflight,
            confirm_final_test=True,
            evaluator=failing_evaluator,
        )
    state = _run_state(preflight)
    assert calls == 1
    assert state["lifecycle_state"] == FINAL_TEST_FAILED_STATE
    assert state["test_access_started"] is True
    assert state["cleanup_permitted"] is False
    assert not (preflight.output_dir / "completed").exists()
    with pytest.raises(FileExistsError, match="already exists"):
        _execute_fixture(
            preflight,
            confirm_final_test=True,
            evaluator=failing_evaluator,
        )
    assert calls == 1


# ADD 2026-09-01: Identical completed evidence의 deterministic JSON/package를 검증한다.
def test_final_test_serialization_is_deterministic(
    preflight_fixture: tuple[dict[str, Path], FinalCandidateManifest, OfficialCandidateEvidence],
    tmp_path: Path,
) -> None:
    paths, _, _ = preflight_fixture
    preflight = _prepare(paths)
    result = _execute_fixture(
        preflight,
        confirm_final_test=True,
        evaluator=lambda _preflight, _artifact_dir: _measurements(),
    )
    assert result is not None
    first_bytes = result.evidence.to_json_bytes()
    second_bytes = result.evidence.to_json_bytes()
    first_package = tmp_path / "first.zip"
    second_package = tmp_path / "second.zip"
    first_sha = final_test_module._write_evidence_package(first_bytes, first_package)
    second_sha = final_test_module._write_evidence_package(second_bytes, second_package)
    assert first_bytes == second_bytes
    assert first_sha == second_sha
    with zipfile.ZipFile(first_package) as archive:
        assert archive.read("final_test_result.json") == first_bytes


# ADD 2026-09-01: CLI unlock은 opt-in store_true이고 기본값이 sealed인지 검증한다.
def test_final_test_cli_requires_explicit_confirmation() -> None:
    parser = build_parser()
    sealed = parser.parse_args(["--official-package", "candidate.zip", "--dataset", "dataset"])
    unlocked = parser.parse_args(
        [
            "--official-package",
            "candidate.zip",
            "--dataset",
            "dataset",
            "--confirm-final-test",
        ]
    )
    assert sealed.confirm_final_test is False
    assert unlocked.confirm_final_test is True


# ADD 2026-09-01: Validation-only metric snapshot이 completed test evidence로 오인되지 않게 한다.
def test_completed_measurements_reject_validation_metric_source() -> None:
    with pytest.raises(ValueError, match="test metric source"):
        replace(
            _measurements(),
            evaluation_split="val",
            framework_metric_source="ultralytics_model.val(split=val)",
        ).validate()
