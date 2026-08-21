"""Build an immutable SOP/manual RAG index with a configured embedding provider."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ml.rag.chunking import (
    DEFAULT_MAX_CHARACTERS,
    DEFAULT_OVERLAP_PARAGRAPHS,
    ChunkingConfig,
)
from services.rag.config import ExternalProviderSettings
from services.rag.index import IndexBuildResult, build_rag_index
from services.rag.providers import (
    EmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)

DEFAULT_MANUALS_DIR = Path("manuals/demo")
DEFAULT_RAG_OUTPUT_ROOT = Path("artifacts/rag/manuals")


@dataclass(frozen=True)
class RagIndexBuildConfig:
    """Offline source/output identifiers and deterministic chunk configuration."""

    manuals_dir: Path
    output_root: Path
    index_id: str
    chunking: ChunkingConfig


# ADD 2026-08-21: Offline config와 injected embedding provider로 immutable index를 생성한다.
def run_rag_index_build(
    config: RagIndexBuildConfig,
    *,
    embedding_provider: EmbeddingProvider,
    created_at: str,
) -> IndexBuildResult:
    """Build one index while keeping external provider construction outside domain logic."""
    return build_rag_index(
        corpus_dir=config.manuals_dir,
        output_root=config.output_root,
        index_id=config.index_id,
        embedding_provider=embedding_provider,
        chunking_config=config.chunking,
        created_at=created_at,
    )


# ADD 2026-08-21: RAG index builder의 source/output/chunk CLI arguments를 정의한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an immutable SOP/manual RAG index.")
    parser.add_argument("--manuals-dir", type=Path, default=DEFAULT_MANUALS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RAG_OUTPUT_ROOT)
    parser.add_argument("--index-id", required=True)
    parser.add_argument("--max-characters", type=int, default=DEFAULT_MAX_CHARACTERS)
    parser.add_argument("--overlap-paragraphs", type=int, default=DEFAULT_OVERLAP_PARAGRAPHS)
    return parser


# ADD 2026-08-21: CLI config와 production embedding provider를 로드해 build summary를 출력한다.
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    provider_settings = ExternalProviderSettings.from_environment(require_generation=False)
    provider = OpenAICompatibleEmbeddingProvider(
        model_name=provider_settings.embedding_model,
        client_config=provider_settings.client_config(),
    )
    config = RagIndexBuildConfig(
        manuals_dir=args.manuals_dir,
        output_root=args.output_root,
        index_id=args.index_id,
        chunking=ChunkingConfig(
            max_characters=args.max_characters,
            overlap_paragraphs=args.overlap_paragraphs,
        ),
    )

    # Source parsing, embedding과 artifact integrity validation을 거쳐 atomic index를 commit한다.
    result = run_rag_index_build(
        config,
        embedding_provider=provider,
        created_at=datetime.now(UTC).isoformat(),
    )
    print(f"index_dir={result.index_dir}")
    print(f"documents={len({chunk.document_id for chunk in result.index.chunks})}")
    print(f"chunks={len(result.index.chunks)}")
    print(f"embedding_dimension={result.index.embedding_dimension}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
