"""Unit tests for RAG evaluation datasets, lineage, and immutable artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.evaluation.rag import (
    CASES_FILENAME,
    EVALUATION_FILENAME,
    RagEvaluationConfig,
    RagEvaluationResult,
    evaluate_rag,
    load_rag_evaluation_artifact,
    load_rag_evaluation_dataset,
    validate_evaluation_references,
)
from ml.evaluation.rag_baseline import (
    DemoSemanticEmbeddingProvider,
    ExtractiveCitationGenerator,
)
from ml.rag.chunking import ChunkingConfig
from services.rag.index import build_rag_index
from shared.hashing import sha256_file


# ADD 2026-08-21: Actual demo corpus와 dataset path를 반환한다.
def _project_inputs() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    return root / "manuals" / "demo", root / "configs" / "evaluation" / "rag_demo.jsonl"


# ADD 2026-08-21: Artifact test용 actual demo immutable index를 생성한다.
def _build_demo_index(tmp_path: Path) -> Path:
    manuals, _ = _project_inputs()
    return build_rag_index(
        corpus_dir=manuals,
        output_root=tmp_path / "indexes",
        index_id="demo-index",
        embedding_provider=DemoSemanticEmbeddingProvider(),
        chunking_config=ChunkingConfig(),
        created_at="2026-08-21T00:00:00+00:00",
    ).index_dir


# ADD 2026-08-21: Fixed config/time의 deterministic demo evaluation을 실행한다.
def _evaluate_demo(
    *,
    index_dir: Path,
    dataset_path: Path,
    output_root: Path,
    evaluation_id: str,
) -> RagEvaluationResult:
    return evaluate_rag(
        index_dir=index_dir,
        dataset_path=dataset_path,
        output_root=output_root,
        evaluation_id=evaluation_id,
        embedding_provider=DemoSemanticEmbeddingProvider(),
        answer_generator=ExtractiveCitationGenerator(),
        config=RagEvaluationConfig(),
        created_at="2026-08-21T01:00:00+00:00",
    )


# ADD 2026-08-21: Actual demo dataset ordering, case types와 SHA contract를 검증한다.
def test_demo_evaluation_dataset_is_versioned_and_deterministic() -> None:
    _, dataset_path = _project_inputs()

    dataset = load_rag_evaluation_dataset(dataset_path)

    assert dataset.sha256 == sha256_file(dataset_path)
    assert len(dataset.cases) == 9
    assert [case.case_id for case in dataset.cases] == sorted(
        case.case_id for case in dataset.cases
    )
    assert {case.language for case in dataset.cases} == {"en", "ko"}
    assert sum(not case.answerable for case in dataset.cases) == 1


# ADD 2026-08-21: Duplicate case ID와 malformed answerability evidence를 거부한다.
def test_dataset_rejects_duplicate_and_corrupt_cases(tmp_path: Path) -> None:
    _, source = _project_inputs()
    first = source.read_text(encoding="utf-8").splitlines()[0]
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(f"{first}\n{first}\n", encoding="utf-8")
    corrupt = tmp_path / "corrupt.jsonl"
    raw = json.loads(first)
    raw["answerable"] = False
    corrupt.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        load_rag_evaluation_dataset(duplicate)
    with pytest.raises(ValueError, match="Invalid"):
        load_rag_evaluation_dataset(corrupt)


# ADD 2026-08-21: Dataset expected document/chunk reference가 index에 없으면 거부한다.
def test_dataset_rejects_unknown_index_reference(tmp_path: Path) -> None:
    index_dir = _build_demo_index(tmp_path)
    _, source = _project_inputs()
    raw = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
    raw["expected_chunk_ids"] = ["chunk-unknown"]
    dataset_path = tmp_path / "unknown.jsonl"
    dataset_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    dataset = load_rag_evaluation_dataset(dataset_path)
    from services.rag.index import load_rag_index

    with pytest.raises(ValueError, match="unknown"):
        validate_evaluation_references(dataset, load_rag_index(index_dir))


# ADD 2026-08-21: Same input/time/config가 byte-identical evaluation artifact를 생성한다.
def test_evaluation_artifact_is_deterministic_and_rejects_overwrite(tmp_path: Path) -> None:
    index_dir = _build_demo_index(tmp_path)
    _, dataset_path = _project_inputs()
    first = _evaluate_demo(
        index_dir=index_dir,
        dataset_path=dataset_path,
        output_root=tmp_path / "first",
        evaluation_id="deterministic",
    )
    second = _evaluate_demo(
        index_dir=index_dir,
        dataset_path=dataset_path,
        output_root=tmp_path / "second",
        evaluation_id="deterministic",
    )

    assert first.evaluation_path.read_bytes() == second.evaluation_path.read_bytes()
    assert first.cases_path.read_bytes() == second.cases_path.read_bytes()
    with pytest.raises(FileExistsError):
        _evaluate_demo(
            index_dir=index_dir,
            dataset_path=dataset_path,
            output_root=tmp_path / "first",
            evaluation_id="deterministic",
        )


# ADD 2026-08-21: Artifact lineage/hash와 non-finite/corrupt output validation을 검증한다.
def test_evaluation_artifact_validates_lineage_hash_and_finite_metrics(tmp_path: Path) -> None:
    index_dir = _build_demo_index(tmp_path)
    _, dataset_path = _project_inputs()
    result = _evaluate_demo(
        index_dir=index_dir,
        dataset_path=dataset_path,
        output_root=tmp_path / "evaluations",
        evaluation_id="integrity",
    )
    summary, records = load_rag_evaluation_artifact(result.output_dir)

    assert summary["index_lineage"]["metadata_sha256"] == sha256_file(index_dir / "metadata.json")
    assert len(records) == 9

    evaluation_path = result.output_dir / EVALUATION_FILENAME
    raw = json.loads(evaluation_path.read_text(encoding="utf-8"))
    raw["metrics"]["faithfulness"] = float("nan")
    evaluation_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot load"):
        load_rag_evaluation_artifact(result.output_dir)

    raw["metrics"]["faithfulness"] = 1.0
    evaluation_path.write_text(json.dumps(raw), encoding="utf-8")
    cases_path = result.output_dir / CASES_FILENAME
    cases_path.write_text(cases_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot load"):
        load_rag_evaluation_artifact(result.output_dir)
