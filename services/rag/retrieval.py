"""Exact deterministic cosine retrieval over an immutable in-memory RAG index."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.rag.chunking import ManualChunk
from services.rag.index import RagIndex
from services.rag.providers import EmbeddingProvider


@dataclass(frozen=True)
class RetrievalResult:
    """Ranked chunk and exact cosine score retained for evaluation evidence."""

    rank: int
    score: float
    chunk: ManualChunk


class ExactCosineRetriever:
    """Read-only brute-force retriever for a small manual corpus."""

    # ADD 2026-08-21: Immutable index/provider identity와 bounded retrieval policy를 검증한다.
    def __init__(
        self,
        *,
        index: RagIndex,
        embedding_provider: EmbeddingProvider,
        max_top_k: int,
        minimum_score: float,
    ) -> None:
        if not 1 <= max_top_k <= 100:
            raise ValueError("RAG max_top_k must be in [1, 100].")
        if not -1.0 <= minimum_score <= 1.0:
            raise ValueError("RAG minimum retrieval score must be in [-1, 1].")
        if (
            index.embedding_provider != embedding_provider.provider_name
            or index.embedding_model != embedding_provider.model_name
        ):
            raise ValueError("Embedding provider/model does not match the loaded RAG index.")
        self._index = index
        self._embedding_provider = embedding_provider
        self._max_top_k = max_top_k
        self._minimum_score = minimum_score

    # ADD 2026-08-21: Query vector를 normalize하고 exact cosine top-k를 deterministic하게 반환한다.
    def retrieve(self, question: str, *, top_k: int) -> tuple[RetrievalResult, ...]:
        """Rank score descending then chunk_id ascending and apply the operational threshold."""
        if not question.strip():
            raise ValueError("RAG question must not be empty.")
        if not 1 <= top_k <= self._max_top_k:
            raise ValueError(f"RAG top_k must be in [1, {self._max_top_k}].")
        if not self._index.chunks:
            return ()
        query = _normalized_query_vector(
            self._embedding_provider.embed_query(question),
            expected_dimension=self._index.embedding_dimension,
        )
        scores = self._index.embeddings @ query
        if not np.isfinite(scores).all():
            raise ValueError("RAG cosine similarity produced non-finite scores.")
        ordered = sorted(
            zip(self._index.chunks, scores.tolist(), strict=True),
            key=lambda item: (-item[1], item[0].chunk_id),
        )
        selected = [item for item in ordered if item[1] >= self._minimum_score][:top_k]
        return tuple(
            RetrievalResult(rank=rank, score=float(score), chunk=chunk)
            for rank, (chunk, score) in enumerate(selected, start=1)
        )


# ADD 2026-08-21: Query embedding을 finite non-zero normalized float32 vector로 변환한다.
def _normalized_query_vector(
    raw_vector: object,
    *,
    expected_dimension: int,
) -> np.ndarray:
    try:
        vector = np.asarray(raw_vector, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("RAG query embedding must be a dense numeric vector.") from exc
    if vector.shape != (expected_dimension,) or not np.isfinite(vector).all():
        raise ValueError("RAG query embedding dimension or finite contract failed.")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("RAG query embedding must not be a zero vector.")
    return np.asarray(vector / norm, dtype=np.float32)
