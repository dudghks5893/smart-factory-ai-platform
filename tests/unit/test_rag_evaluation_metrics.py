"""Unit tests for deterministic RAG evaluation metric definitions."""

from __future__ import annotations

import numpy as np
import pytest

from ml.evaluation.rag import (
    ReferenceFact,
    calculate_citation_metrics,
    calculate_deterministic_faithfulness,
    calculate_recall_at_k,
    calculate_reference_fact_recall,
)
from ml.rag.chunking import ManualChunk
from services.rag.generation import RagCitation
from services.rag.index import RagIndex


# ADD 2026-08-21: Metric test용 chunk provenance를 생성한다.
def _chunk(chunk_id: str, document_id: str, text: str) -> ManualChunk:
    return ManualChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title="Demo",
        source_path=f"{document_id}.md",
        section="Procedure",
        page=None,
        text=text,
        chunk_index=0,
        source_sha256="a" * 64,
    )


# ADD 2026-08-21: Metric test용 structured citation을 생성한다.
def _citation(citation_id: str, chunk: ManualChunk) -> RagCitation:
    return RagCitation(
        citation_id=citation_id,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        title=chunk.title,
        section=chunk.section,
        page=None,
        source_path=chunk.source_path,
        retrieval_score=0.9,
    )


# ADD 2026-08-21: Perfect, missing과 multi-evidence fractional Recall@K를 검증한다.
def test_recall_at_k_handles_perfect_missing_and_multi_evidence() -> None:
    expected = ("chunk-a", "chunk-b")

    assert calculate_recall_at_k(expected, ("chunk-a", "chunk-b"), 2) == 1.0
    assert calculate_recall_at_k(expected, ("chunk-x", "chunk-a"), 1) == 0.0
    assert calculate_recall_at_k(expected, ("chunk-a", "chunk-x"), 2) == 0.5
    with pytest.raises(ValueError, match="expected"):
        calculate_recall_at_k((), (), 1)


# ADD 2026-08-21: Citation precision/recall과 no/unsupported citation을 검증한다.
def test_citation_metrics_count_relevant_missing_and_unsupported_sources() -> None:
    expected_a = _chunk("chunk-a", "doc-a", "Supported fact.")
    expected_b = _chunk("chunk-b", "doc-b", "Second fact.")
    unrelated = _chunk("chunk-x", "doc-x", "Unrelated fact.")

    partial = calculate_citation_metrics(
        (expected_a.chunk_id, expected_b.chunk_id),
        (_citation("C1", expected_a), _citation("C2", unrelated)),
    )
    none = calculate_citation_metrics((expected_a.chunk_id,), ())

    assert partial.precision == 0.5
    assert partial.recall == 0.5
    assert none.precision == 0.0
    assert none.recall == 0.0


# ADD 2026-08-21: Supported claim과 unsupported/unmarked claim의 faithfulness를 검증한다.
def test_faithfulness_requires_claim_text_in_its_cited_chunk() -> None:
    chunk = _chunk("chunk-a", "doc-a", "Clean the camera lens before inspection.")
    embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    embeddings.setflags(write=False)
    index = RagIndex("index", "test", "test-v1", 2, (chunk,), embeddings)
    citations = (_citation("C1", chunk),)

    faithful = calculate_deterministic_faithfulness(
        "Clean the camera lens before inspection. [C1]",
        citations,
        index,
    )
    unsupported = calculate_deterministic_faithfulness(
        "Set the motor torque to 40 Nm. [C1]\nUnmarked claim.",
        citations,
        index,
    )

    assert faithful.score == 1.0
    assert faithful.supported_claims == 1
    assert unsupported.score == 0.0
    assert unsupported.total_claims == 2


# ADD 2026-08-21: Reference fact coverage가 faithfulness와 별도 correctness metric인지 검증한다.
def test_reference_fact_recall_uses_all_required_terms() -> None:
    facts = (
        ReferenceFact("camera", "Check camera.", ("camera lens", "lighting")),
        ReferenceFact("record", "Keep record.", ("inspection record",)),
    )

    value = calculate_reference_fact_recall(
        "Check the camera lens and lighting before use.",
        facts,
    )

    assert value == 0.5
