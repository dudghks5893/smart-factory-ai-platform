"""Build the explicit deterministic index used by the public demo RAG evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from ml.evaluation.rag_baseline import DemoSemanticEmbeddingProvider
from ml.rag.chunking import ChunkingConfig
from pipelines.build_rag_index import (
    DEFAULT_MANUALS_DIR,
    DEFAULT_RAG_OUTPUT_ROOT,
    RagIndexBuildConfig,
    run_rag_index_build,
)


# ADD 2026-08-21: Demo evaluation index의 explicit source/output CLI arguments를 정의한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the deterministic public demo RAG evaluation index."
    )
    parser.add_argument("--manuals-dir", type=Path, default=DEFAULT_MANUALS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RAG_OUTPUT_ROOT)
    parser.add_argument("--index-id", required=True)
    parser.add_argument("--max-characters", type=int, default=1200)
    parser.add_argument("--overlap-paragraphs", type=int, default=1)
    return parser


# ADD 2026-08-21: Evaluation-only provider로 demo index를 별도 immutable artifact로 생성한다.
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RagIndexBuildConfig(
        manuals_dir=args.manuals_dir,
        output_root=args.output_root,
        index_id=args.index_id,
        chunking=ChunkingConfig(
            max_characters=args.max_characters,
            overlap_paragraphs=args.overlap_paragraphs,
        ),
    )

    # Evaluation 실행과 분리된 명령에서 고정 provider/config로 index를 먼저 생성한다.
    result = run_rag_index_build(
        config,
        embedding_provider=DemoSemanticEmbeddingProvider(),
        created_at=datetime.now(UTC).isoformat(),
    )
    print(f"index_dir={result.index_dir}")
    print(f"chunks={len(result.index.chunks)}")
    print(f"embedding_dimension={result.index.embedding_dimension}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
