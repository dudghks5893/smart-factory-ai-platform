"""End-to-end deterministic evaluation over the actual public demo corpus."""

from __future__ import annotations

from pathlib import Path

from ml.evaluation.rag import RagEvaluationConfig, evaluate_rag, load_rag_evaluation_artifact
from ml.evaluation.rag_baseline import (
    DemoSemanticEmbeddingProvider,
    ExtractiveCitationGenerator,
)
from ml.rag.chunking import ChunkingConfig
from services.rag.index import build_rag_index


# ADD 2026-08-21: Actual demo corpus/index/dataset 전체 evaluation artifact round-trip을 검증한다.
def test_actual_demo_rag_evaluation_pipeline(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    provider = DemoSemanticEmbeddingProvider()

    # Evaluation과 분리된 단계에서 actual demo corpus의 immutable index를 먼저 생성한다.
    index = build_rag_index(
        corpus_dir=project_root / "manuals" / "demo",
        output_root=tmp_path / "indexes",
        index_id="demo-evaluation-index",
        embedding_provider=provider,
        chunking_config=ChunkingConfig(),
        created_at="2026-08-21T00:00:00+00:00",
    )

    # Existing index와 versioned QA dataset으로 retrieval/generation metrics를 저장한다.
    result = evaluate_rag(
        index_dir=index.index_dir,
        dataset_path=project_root / "configs" / "evaluation" / "rag_demo.jsonl",
        output_root=tmp_path / "evaluations",
        evaluation_id="demo-evaluation",
        embedding_provider=provider,
        answer_generator=ExtractiveCitationGenerator(),
        config=RagEvaluationConfig(),
        created_at="2026-08-21T01:00:00+00:00",
    )
    summary, cases = load_rag_evaluation_artifact(result.output_dir)

    assert summary["metrics"]["document_recall_at_k"]["5"] == 1.0
    assert summary["metrics"]["chunk_recall_at_k"]["5"] == 1.0
    assert summary["metrics"]["faithfulness"] == 1.0
    assert summary["metrics"]["unanswerable_abstention_accuracy"] == 1.0
    unanswerable = next(case for case in cases if not case["answerable"])
    assert unanswerable["answer_status"] == "insufficient_context"
    assert unanswerable["generator_called"] is False
    assert unanswerable["citations"] == []
