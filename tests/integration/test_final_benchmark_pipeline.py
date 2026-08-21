"""Integration test for the final benchmark CLI over an actual RAG evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.evaluation.final_benchmark import (
    RepositoryProvenance,
    load_final_benchmark_artifact,
)
from ml.evaluation.rag import RagEvaluationConfig, evaluate_rag
from ml.evaluation.rag_baseline import (
    DemoSemanticEmbeddingProvider,
    ExtractiveCitationGenerator,
)
from ml.rag.chunking import ChunkingConfig
from pipelines.build_final_benchmark import main
from services.rag.index import build_rag_index


# ADD 2026-08-21: Actual demo RAG artifact를 official evidence와 CLI로 end-to-end 집계한다.
def test_final_benchmark_cli_aggregates_actual_rag_evaluation(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    index = build_rag_index(
        corpus_dir=project_root / "manuals" / "demo",
        output_root=tmp_path / "indexes",
        index_id="final-benchmark-index",
        embedding_provider=DemoSemanticEmbeddingProvider(),
        chunking_config=ChunkingConfig(),
        created_at="2026-08-21T00:00:00+00:00",
    )
    evaluation = evaluate_rag(
        index_dir=index.index_dir,
        dataset_path=project_root / "configs" / "evaluation" / "rag_demo.jsonl",
        output_root=tmp_path / "evaluations",
        evaluation_id="final-benchmark-evaluation",
        embedding_provider=DemoSemanticEmbeddingProvider(),
        answer_generator=ExtractiveCitationGenerator(),
        config=RagEvaluationConfig(),
        created_at="2026-08-21T01:00:00+00:00",
    )
    official = project_root / "configs" / "benchmarks" / "official"
    output_root = tmp_path / "final"

    exit_code = main(
        [
            "--vision-evaluation",
            str(official / "vision_quality_step3.json"),
            "--model-benchmark",
            str(official / "model_runtime_step3_t4.json"),
            "--api-benchmark-v1",
            str(official / "api_http_step4_v1_t4.json"),
            "--rag-evaluation-dir",
            str(evaluation.output_dir),
            "--platform-verification",
            str(project_root / "configs" / "benchmarks" / "platform_verification.json"),
            "--output-root",
            str(output_root),
            "--benchmark-id",
            "cli-integration",
        ],
        provenance_resolver=lambda _: RepositoryProvenance(
            git_commit="c" * 40,
            working_tree_dirty=True,
        ),
    )

    artifact = load_final_benchmark_artifact(output_root / "cli-integration" / "benchmark.json")
    assert exit_code == 0
    assert artifact["repository"] == {
        "git_commit": "c" * 40,
        "working_tree_dirty": True,
    }
    assert artifact["sections"]["rag_quality"]["results"]["citation_precision"] == pytest.approx(
        0.25625
    )
    assert (
        artifact["sections"]["model_runtime_performance"]["results"]["throughput_images_per_second"]
        == 45.11436063599353
    )
