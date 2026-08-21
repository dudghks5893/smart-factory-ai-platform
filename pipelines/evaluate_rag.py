"""Evaluate an existing immutable RAG index with the versioned offline dataset."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from ml.evaluation.rag import (
    DEFAULT_RAG_EVALUATION_ROOT,
    RagEvaluationConfig,
    evaluate_rag,
)
from ml.evaluation.rag_baseline import (
    DemoSemanticEmbeddingProvider,
    ExtractiveCitationGenerator,
)

DEFAULT_RAG_EVALUATION_DATASET = Path("configs/evaluation/rag_demo.jsonl")


# ADD 2026-08-21: Existing index, dataset, retrieval policy와 immutable output CLI를 정의한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing demo RAG index without external provider calls."
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-dataset",
        type=Path,
        default=DEFAULT_RAG_EVALUATION_DATASET,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RAG_EVALUATION_ROOT)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--min-score", type=float, default=0.2)
    return parser


# ADD 2026-08-21: Evaluation-only providers로 existing index를 평가하고 핵심 metric을 출력한다.
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RagEvaluationConfig(
        top_k_values=tuple(args.top_k),
        minimum_retrieval_score=args.min_score,
    )

    # Index는 재생성하지 않고 explicit path의 artifact와 versioned dataset만 평가한다.
    result = evaluate_rag(
        index_dir=args.index_dir,
        dataset_path=args.evaluation_dataset,
        output_root=args.output_root,
        evaluation_id=args.evaluation_id,
        embedding_provider=DemoSemanticEmbeddingProvider(),
        answer_generator=ExtractiveCitationGenerator(),
        config=config,
        created_at=datetime.now(UTC).isoformat(),
    )
    metrics = result.metrics
    print(f"evaluation_dir={result.output_dir}")
    print(f"document_recall_at_k={metrics['document_recall_at_k']}")
    print(f"chunk_recall_at_k={metrics['chunk_recall_at_k']}")
    print(f"citation_precision={metrics['citation_precision']}")
    print(f"citation_recall={metrics['citation_recall']}")
    print(f"faithfulness={metrics['faithfulness']}")
    print(f"unanswerable_abstention_accuracy={metrics['unanswerable_abstention_accuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
