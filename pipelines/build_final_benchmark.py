"""Build the final cross-domain benchmark evidence artifact from existing sources."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from ml.evaluation.final_benchmark import (
    DEFAULT_FINAL_BENCHMARK_ROOT,
    FinalBenchmarkSources,
    RepositoryProvenance,
    build_final_benchmark,
    resolve_repository_provenance,
)

DEFAULT_OFFICIAL_ROOT = Path("configs/benchmarks/official")
DEFAULT_PLATFORM_VERIFICATION = Path("configs/benchmarks/platform_verification.json")
ProvenanceResolver = Callable[[Path], RepositoryProvenance]


# ADD 2026-08-21: Official evidence, RAG artifact와 optional API v2 input CLI를 정의한다.
# MODIFY 2026-08-21: Manual SHA를 제거하고 repository state 자동 탐지를 추가한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate existing official and deterministic final benchmark evidence."
    )
    parser.add_argument(
        "--vision-evaluation",
        type=Path,
        default=DEFAULT_OFFICIAL_ROOT / "vision_quality_step3.json",
    )
    parser.add_argument(
        "--model-benchmark",
        type=Path,
        default=DEFAULT_OFFICIAL_ROOT / "model_runtime_step3_t4.json",
    )
    parser.add_argument(
        "--api-benchmark-v1",
        type=Path,
        default=DEFAULT_OFFICIAL_ROOT / "api_http_step4_v1_t4.json",
    )
    parser.add_argument("--api-benchmark-v2", type=Path)
    parser.add_argument("--rag-evaluation-dir", type=Path, required=True)
    parser.add_argument(
        "--platform-verification",
        type=Path,
        default=DEFAULT_PLATFORM_VERIFICATION,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FINAL_BENCHMARK_ROOT)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    return parser


# ADD 2026-08-21: Existing source artifacts만 읽어 final benchmark path와 section 상태를 출력한다.
# MODIFY 2026-08-21: Build 시점 Git HEAD와 working-tree dirty state를 자동 주입한다.
def main(
    argv: Sequence[str] | None = None,
    *,
    provenance_resolver: ProvenanceResolver = resolve_repository_provenance,
) -> int:
    args = build_parser().parse_args(argv)

    # Git HEAD와 dirty state를 자동 판정한 뒤 measurement artifact를 aggregation한다.
    repository_provenance = provenance_resolver(args.repository_root)
    result = build_final_benchmark(
        sources=FinalBenchmarkSources(
            vision_quality_path=args.vision_evaluation,
            model_runtime_path=args.model_benchmark,
            api_v1_path=args.api_benchmark_v1,
            api_v2_path=args.api_benchmark_v2,
            rag_evaluation_dir=args.rag_evaluation_dir,
            platform_verification_path=args.platform_verification,
        ),
        output_root=args.output_root,
        benchmark_id=args.benchmark_id,
        created_at=datetime.now(UTC).isoformat(),
        repository_provenance=repository_provenance,
    )
    api_v2 = result.payload["sections"]["api_application_performance_v2"]
    print(f"benchmark={result.benchmark_path}")
    print(f"api_schema_v2_status={api_v2['status']}")
    print(f"source_count={len(result.payload['sources'])}")
    print(f"working_tree_dirty={repository_provenance.working_tree_dirty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
