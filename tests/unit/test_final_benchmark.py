"""Unit tests for final benchmark aggregation and evidence integrity."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ml.evaluation.final_benchmark import (
    API_V1_LABEL,
    RAG_DEMO_LABEL,
    FinalBenchmarkSources,
    RepositoryProvenance,
    build_final_benchmark,
    load_final_benchmark_artifact,
    resolve_repository_provenance,
)
from shared.hashing import sha256_file

_CREATED_AT = "2026-08-21T12:00:00+09:00"
_GIT_COMMIT = "c" * 40
_DIRTY_REPOSITORY = RepositoryProvenance(
    git_commit=_GIT_COMMIT,
    working_tree_dirty=True,
)


# ADD 2026-08-21: Repository의 versioned official evidence path를 반환한다.
def _official_root() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "benchmarks"


# ADD 2026-08-21: Loader contract를 만족하는 최소 immutable RAG evaluation을 만든다.
def _write_rag_artifact(root: Path) -> Path:
    artifact_dir = root / "rag-evaluation"
    artifact_dir.mkdir(parents=True)
    cases_path = artifact_dir / "cases.jsonl"
    cases_path.write_text('{"case_id":"case-001"}\n', encoding="utf-8")
    summary = {
        "schema_version": 1,
        "evaluation_id": "rag-evaluation-v1",
        "created_at": _CREATED_AT,
        "dataset": {"filename": "rag.jsonl", "sha256": "a" * 64, "case_count": 1},
        "index_lineage": {
            "index_id": "rag-index-v1",
            "metadata_sha256": "b" * 64,
            "document_count": 1,
            "chunk_count": 1,
        },
        "configuration": {
            "top_k_values": [1, 3, 5],
            "faithfulness_evaluator": "deterministic-extractive-support-v1",
        },
        "metrics": {
            "document_recall_at_k": {"1": 0.5, "3": 1.0, "5": 1.0},
            "chunk_recall_at_k": {"1": 0.25, "3": 0.75, "5": 1.0},
            "mean_reciprocal_rank": 0.625,
            "citation_precision": 0.25,
            "citation_recall": 1.0,
            "faithfulness": 1.0,
            "reference_fact_recall": 0.25,
            "unanswerable_abstention_accuracy": 1.0,
        },
        "score_analysis": {"policy": "analysis_only_no_automatic_threshold_update"},
        "artifacts": {"cases_sha256": sha256_file(cases_path)},
    }
    (artifact_dir / "evaluation.json").write_text(
        json.dumps(summary, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return artifact_dir


# ADD 2026-08-21: Default official sources와 test RAG artifact를 하나의 input contract로 묶는다.
def _sources(tmp_path: Path, *, api_v2_path: Path | None = None) -> FinalBenchmarkSources:
    benchmark_root = _official_root()
    return FinalBenchmarkSources(
        vision_quality_path=benchmark_root / "official" / "vision_quality_step3.json",
        model_runtime_path=benchmark_root / "official" / "model_runtime_step3_t4.json",
        api_v1_path=benchmark_root / "official" / "api_http_step4_v1_t4.json",
        api_v2_path=api_v2_path,
        rag_evaluation_dir=_write_rag_artifact(tmp_path),
        platform_verification_path=benchmark_root / "platform_verification.json",
    )


# ADD 2026-08-21: Test에서 source path 하나만 교체할 수 있도록 immutable sources를 복사한다.
def _replace_source(
    sources: FinalBenchmarkSources,
    **changes: Path | None,
) -> FinalBenchmarkSources:
    values: dict[str, Any] = {
        "vision_quality_path": sources.vision_quality_path,
        "model_runtime_path": sources.model_runtime_path,
        "api_v1_path": sources.api_v1_path,
        "api_v2_path": sources.api_v2_path,
        "rag_evaluation_dir": sources.rag_evaluation_dir,
        "platform_verification_path": sources.platform_verification_path,
    }
    values.update(changes)
    return FinalBenchmarkSources(
        vision_quality_path=Path(values["vision_quality_path"]),
        model_runtime_path=Path(values["model_runtime_path"]),
        api_v1_path=Path(values["api_v1_path"]),
        api_v2_path=(Path(values["api_v2_path"]) if values["api_v2_path"] else None),
        rag_evaluation_dir=Path(values["rag_evaluation_dir"]),
        platform_verification_path=Path(values["platform_verification_path"]),
    )


# ADD 2026-08-21: Current API schema v2 optional artifact의 실제 shape를 생성한다.
def _write_api_v2(path: Path) -> Path:
    quality = json.loads(
        (_official_root() / "official" / "vision_quality_step3.json").read_text(encoding="utf-8")
    )
    provenance = quality["provenance"]
    payload = {
        "schema_version": 2,
        "benchmark_name": "patchcore_fastapi_http_e2e",
        "category": quality["category"],
        "device": "cuda",
        "runtime": {"accelerator_name": "Tesla T4", "python_version": "3.12.13"},
        "provenance": {
            "manifest_sha256": provenance["manifest_sha256"],
            "artifact_metadata_sha256": "d" * 64,
            "model_sha256": provenance["model_sha256"],
            "threshold_artifact_sha256": provenance["threshold_artifact_sha256"],
        },
        "conditions": {
            "transport": "in_process_asgi_testclient",
            "request_batch_size": 1,
            "warmup_count": 10,
            "measured_count": 115,
        },
        "latency_definition": "request through PostgreSQL INSERT/COMMIT and response",
        "inspection_persistence_included": True,
        "metrics": {"p50_ms": 50.0, "throughput_requests_per_second": 20.0},
    }
    path.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")
    return path


# ADD 2026-08-21: Valid aggregation이 source hashes, labels와 independent environments를 보존한다.
def test_build_final_benchmark_aggregates_valid_sources(tmp_path: Path) -> None:
    sources = _sources(tmp_path)

    result = build_final_benchmark(
        sources=sources,
        output_root=tmp_path / "final",
        benchmark_id="final-v1",
        created_at=_CREATED_AT,
        repository_provenance=_DIRTY_REPOSITORY,
    )

    sections = result.payload["sections"]
    assert sections["api_application_performance_v1"]["label"] == API_V1_LABEL
    assert sections["rag_quality"]["label"] == RAG_DEMO_LABEL
    assert sections["api_application_performance_v2"]["status"] == "not_available"
    assert result.payload["repository"] == {
        "git_commit": _GIT_COMMIT,
        "working_tree_dirty": True,
    }
    assert sections["vision_quality"]["results"]["auroc"] == pytest.approx(0.9975562072336266)
    assert result.payload["sources"]["rag_evaluation"]["sha256"] == sha256_file(
        sources.rag_evaluation_dir / "evaluation.json"
    )
    environment_ids = {
        row["environment"]["environment_id"]
        for row in result.payload["environment_matrix"]
        if row["environment"] is not None
    }
    assert "kaggle-t4-step3-model-runtime" in environment_ids
    assert "local-deterministic-rag-step14" in environment_ids


# ADD 2026-08-21: Required artifact 부재와 model lineage mismatch를 fail-fast 처리한다.
def test_builder_rejects_missing_source_and_lineage_mismatch(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    missing = _replace_source(sources, model_runtime_path=tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="not found"):
        build_final_benchmark(
            sources=missing,
            output_root=tmp_path / "missing-output",
            benchmark_id="missing",
            created_at=_CREATED_AT,
            repository_provenance=_DIRTY_REPOSITORY,
        )

    mismatched_path = tmp_path / "model-mismatch.json"
    shutil.copyfile(sources.model_runtime_path, mismatched_path)
    mismatched = json.loads(mismatched_path.read_text(encoding="utf-8"))
    mismatched["provenance"]["model_sha256"] = "f" * 64
    mismatched_path.write_text(json.dumps(mismatched) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lineage mismatch"):
        build_final_benchmark(
            sources=_replace_source(sources, model_runtime_path=mismatched_path),
            output_root=tmp_path / "mismatch-output",
            benchmark_id="mismatch",
            created_at=_CREATED_AT,
            repository_provenance=_DIRTY_REPOSITORY,
        )


# ADD 2026-08-21: Real persistence contract를 가진 optional API v2 source만 available로 집계한다.
def test_builder_accepts_optional_persistence_api_v2(tmp_path: Path) -> None:
    api_v2_path = _write_api_v2(tmp_path / "api-v2.json")
    sources = _sources(tmp_path, api_v2_path=api_v2_path)

    result = build_final_benchmark(
        sources=sources,
        output_root=tmp_path / "final",
        benchmark_id="with-api-v2",
        created_at=_CREATED_AT,
        repository_provenance=_DIRTY_REPOSITORY,
    )

    api_v2 = result.payload["sections"]["api_application_performance_v2"]
    assert api_v2["status"] == "available"
    assert api_v2["label"] == "API Benchmark — schema v2 / persistence-inclusive"
    assert result.payload["sources"]["api_v2"]["sha256"] == sha256_file(api_v2_path)


# ADD 2026-08-21: Source metric의 NaN/Inf가 final artifact로 전파되지 않도록 거부한다.
def test_builder_rejects_non_finite_source_metric(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    invalid_path = tmp_path / "model-nan.json"
    invalid = json.loads(sources.model_runtime_path.read_text(encoding="utf-8"))
    invalid["results"]["latency_ms"]["p50"] = float("nan")
    invalid_path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite"):
        build_final_benchmark(
            sources=_replace_source(sources, model_runtime_path=invalid_path),
            output_root=tmp_path / "final",
            benchmark_id="non-finite",
            created_at=_CREATED_AT,
            repository_provenance=_DIRTY_REPOSITORY,
        )


# ADD 2026-08-21: Same inputs/time은 byte-identical하고 existing output overwrite는 거부한다.
def test_final_artifact_is_deterministic_and_immutable(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    first = build_final_benchmark(
        sources=sources,
        output_root=tmp_path / "first",
        benchmark_id="deterministic",
        created_at=_CREATED_AT,
        repository_provenance=_DIRTY_REPOSITORY,
    )
    second = build_final_benchmark(
        sources=sources,
        output_root=tmp_path / "second",
        benchmark_id="deterministic",
        created_at=_CREATED_AT,
        repository_provenance=_DIRTY_REPOSITORY,
    )

    assert first.benchmark_path.read_bytes() == second.benchmark_path.read_bytes()
    with pytest.raises(FileExistsError):
        build_final_benchmark(
            sources=sources,
            output_root=tmp_path / "first",
            benchmark_id="deterministic",
            created_at=_CREATED_AT,
            repository_provenance=_DIRTY_REPOSITORY,
        )


# ADD 2026-08-21: Generated artifact loader가 mandatory taxonomy label 변조를 거부한다.
def test_final_artifact_loader_rejects_ambiguous_label(tmp_path: Path) -> None:
    result = build_final_benchmark(
        sources=_sources(tmp_path),
        output_root=tmp_path / "final",
        benchmark_id="labels",
        created_at=_CREATED_AT,
        repository_provenance=_DIRTY_REPOSITORY,
    )
    raw: dict[str, Any] = json.loads(result.benchmark_path.read_text(encoding="utf-8"))
    raw["sections"]["api_application_performance_v1"]["label"] = "API benchmark"
    result.benchmark_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Cannot load"):
        load_final_benchmark_artifact(result.benchmark_path)


# ADD 2026-08-21: Repository dirty flag가 문자열로 변조된 artifact를 거부한다.
def test_final_artifact_loader_rejects_invalid_repository_provenance(tmp_path: Path) -> None:
    result = build_final_benchmark(
        sources=_sources(tmp_path),
        output_root=tmp_path / "final",
        benchmark_id="repository-provenance",
        created_at=_CREATED_AT,
        repository_provenance=_DIRTY_REPOSITORY,
    )
    raw = json.loads(result.benchmark_path.read_text(encoding="utf-8"))
    raw["repository"]["working_tree_dirty"] = "true"
    result.benchmark_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Cannot load"):
        load_final_benchmark_artifact(result.benchmark_path)


# ADD 2026-08-21: Injected Git runner로 clean/dirty repository provenance 판정을 검증한다.
@pytest.mark.parametrize(
    ("status_output", "expected_dirty"),
    [("", False), (" M README.md\n?? new-file.txt\n", True)],
)
def test_repository_provenance_resolver_tracks_dirty_state(
    status_output: str,
    expected_dirty: bool,
) -> None:
    commands: list[tuple[str, ...]] = []

    def run_git(arguments: Sequence[str], repository_root: Path) -> str:
        command = tuple(arguments)
        commands.append(command)
        assert repository_root == Path("/test/repository")
        return _GIT_COMMIT if command == ("rev-parse", "HEAD") else status_output

    provenance = resolve_repository_provenance(
        Path("/test/repository"),
        run_git=run_git,
    )

    assert provenance == RepositoryProvenance(
        git_commit=_GIT_COMMIT,
        working_tree_dirty=expected_dirty,
    )
    assert commands == [
        ("rev-parse", "HEAD"),
        ("status", "--porcelain", "--untracked-files=normal"),
    ]
