"""Deterministic test-only embedding and generation providers for RAG contracts."""

from __future__ import annotations

import re
from collections.abc import Sequence

from services.rag.providers import (
    GeneratedAnswer,
    GenerationContext,
    ProviderError,
)


class KeywordEmbeddingProvider:
    """Small deterministic lexical vectorizer used only in tests and local smoke."""

    provider_name = "test-keyword"
    model_name = "test-keyword-v1"

    # ADD 2026-08-21: Test document texts를 fixed keyword-count vectors로 변환한다.
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self._vector(text) for text in texts]

    # ADD 2026-08-21: Test query를 document와 같은 deterministic vector space로 변환한다.
    def embed_query(self, text: str) -> Sequence[float]:
        return self._vector(text)

    # ADD 2026-08-21: Stable keyword features와 non-zero bias를 계산한다.
    def _vector(self, text: str) -> tuple[float, ...]:
        tokens = re.findall(r"[a-z]+", text.lower())
        groups = (
            {"camera", "lens", "lighting", "image", "blurred"},
            {"quarantine", "containment", "release", "item"},
            {"escalate", "escalation", "reviewer", "maintenance"},
            {"threshold", "score", "model", "inspection"},
        )
        return (0.1, *(float(sum(token in group for token in tokens)) for group in groups))


class FixedQueryEmbeddingProvider:
    """Test embedding provider returning caller-controlled query values."""

    provider_name = "test"
    model_name = "test-v1"

    # ADD 2026-08-21: Controlled query vector와 optional document vectors를 초기화한다.
    def __init__(self, query_vector: Sequence[float]) -> None:
        self.query_vector = query_vector

    # ADD 2026-08-21: Index unit tests가 제공한 document vectors를 그대로 반환한다.
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self.query_vector for _ in texts]

    # ADD 2026-08-21: Retrieval boundary test용 controlled query vector를 반환한다.
    def embed_query(self, text: str) -> Sequence[float]:
        return self.query_vector


class GroundedFakeGenerator:
    """Test-only answer generator citing configured retrieved contexts."""

    provider_name = "test-generator"
    model_name = "test-generator-v1"

    # ADD 2026-08-21: Test answer에서 사용할 citation count를 설정한다.
    def __init__(self, citation_count: int = 1) -> None:
        self.citation_count = citation_count
        self.last_contexts: tuple[GenerationContext, ...] = ()

    # ADD 2026-08-21: Retrieved allow-list 안에서 deterministic grounded answer를 생성한다.
    def generate(
        self,
        question: str,
        contexts: Sequence[GenerationContext],
    ) -> GeneratedAnswer:
        self.last_contexts = tuple(contexts)
        citation_ids = tuple(context.citation_id for context in contexts[: self.citation_count])
        markers = " ".join(f"[{citation_id}]" for citation_id in citation_ids)
        return GeneratedAnswer(answer=f"Grounded demo answer. {markers}", citation_ids=citation_ids)


class InvalidCitationGenerator:
    """Test generator that violates the citation allow-list."""

    provider_name = "test-generator"
    model_name = "test-generator-invalid"

    # ADD 2026-08-21: Unknown citation을 반환해 validation failure를 유도한다.
    def generate(
        self,
        question: str,
        contexts: Sequence[GenerationContext],
    ) -> GeneratedAnswer:
        return GeneratedAnswer(answer="Unsupported claim. [C99]", citation_ids=("C99",))


class FailingGenerator:
    """Test generator representing a safe external provider failure."""

    provider_name = "test-generator"
    model_name = "test-generator-failure"

    # ADD 2026-08-21: Provider credential/detail을 포함한 internal failure를 발생시킨다.
    def generate(
        self,
        question: str,
        contexts: Sequence[GenerationContext],
    ) -> GeneratedAnswer:
        raise ProviderError("secret provider detail")
