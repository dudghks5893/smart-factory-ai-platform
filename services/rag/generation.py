"""Grounded RAG orchestration, abstention, and controlled citation validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from services.rag.index import RagIndex
from services.rag.providers import (
    AnswerGenerator,
    GeneratedAnswer,
    GenerationContext,
    InvalidProviderOutputError,
)
from services.rag.retrieval import ExactCosineRetriever, RetrievalResult

_CITATION_MARKER = re.compile(r"\[(C[1-9][0-9]*)\]")


@dataclass(frozen=True)
class RagCitation:
    """Structured citation controlled by retrieved chunk metadata."""

    citation_id: str
    chunk_id: str
    document_id: str
    title: str
    section: str
    page: int | None
    source_path: str
    retrieval_score: float


@dataclass(frozen=True)
class RetrievalEvidence:
    """Compact retrieval evidence preserved for STEP 14 evaluation."""

    rank: int
    chunk_id: str
    document_id: str
    score: float


@dataclass(frozen=True)
class RagQueryResult:
    """Grounded answer or explicit abstention with citations and retrieval evidence."""

    status: Literal["answered", "insufficient_context"]
    answer: str
    citations: tuple[RagCitation, ...]
    retrieval: tuple[RetrievalEvidence, ...]


class RagRuntime:
    """Immutable retrieval runtime plus one configured generation provider."""

    # ADD 2026-08-21: Retriever와 provider를 reusable process-local runtime으로 결합한다.
    def __init__(
        self,
        *,
        index: RagIndex,
        retriever: ExactCosineRetriever,
        answer_generator: AnswerGenerator,
    ) -> None:
        self.index = index
        self._retriever = retriever
        self._answer_generator = answer_generator

    # ADD 2026-08-21: Retrieve, abstain 또는 grounded generation/citation validation을 조율한다.
    def query(self, question: str, *, top_k: int) -> RagQueryResult:
        """Return evidence-rich output without persisting sensitive query text."""
        retrieved = self._retriever.retrieve(question, top_k=top_k)
        evidence = tuple(_retrieval_evidence(item) for item in retrieved)
        if not retrieved:
            return RagQueryResult(
                status="insufficient_context",
                answer=_abstention_answer(question),
                citations=(),
                retrieval=evidence,
            )
        contexts = tuple(
            GenerationContext(
                citation_id=f"C{index}",
                chunk_id=item.chunk.chunk_id,
                title=item.chunk.title,
                section=item.chunk.section,
                source_path=item.chunk.source_path,
                page=item.chunk.page,
                text=item.chunk.text,
            )
            for index, item in enumerate(retrieved, start=1)
        )
        generated = self._answer_generator.generate(question, contexts)
        citation_ids = validate_generated_answer(generated, contexts)
        by_citation_id = dict(
            zip((context.citation_id for context in contexts), retrieved, strict=True)
        )
        citations = tuple(
            _citation(citation_id, by_citation_id[citation_id]) for citation_id in citation_ids
        )
        return RagQueryResult(
            status="answered",
            answer=generated.answer,
            citations=citations,
            retrieval=evidence,
        )


# ADD 2026-08-21: Answer marker와 citation list를 retrieved context allow-list로 검증한다.
def validate_generated_answer(
    generated: GeneratedAnswer,
    contexts: tuple[GenerationContext, ...],
) -> tuple[str, ...]:
    """Reject unknown, duplicate, absent, or marker/list-mismatched citations."""
    allowed = {context.citation_id for context in contexts}
    citations = generated.citation_ids
    markers = tuple(_CITATION_MARKER.findall(generated.answer))
    if not generated.answer.strip() or not citations:
        raise InvalidProviderOutputError("Grounded answer must include at least one citation.")
    if len(set(citations)) != len(citations):
        raise InvalidProviderOutputError("Generated citations must be unique.")
    if not set(citations) <= allowed or not set(markers) <= allowed:
        raise InvalidProviderOutputError("Generated answer referenced an unknown citation.")
    if set(markers) != set(citations):
        raise InvalidProviderOutputError("Answer markers do not match structured citations.")
    return citations


# ADD 2026-08-21: Retrieved chunk를 public compact evaluation evidence로 변환한다.
def _retrieval_evidence(result: RetrievalResult) -> RetrievalEvidence:
    return RetrievalEvidence(
        rank=result.rank,
        chunk_id=result.chunk.chunk_id,
        document_id=result.chunk.document_id,
        score=result.score,
    )


# ADD 2026-08-21: Controlled citation id를 retrieved provenance와 score에 연결한다.
def _citation(citation_id: str, result: RetrievalResult) -> RagCitation:
    return RagCitation(
        citation_id=citation_id,
        chunk_id=result.chunk.chunk_id,
        document_id=result.chunk.document_id,
        title=result.chunk.title,
        section=result.chunk.section,
        page=result.chunk.page,
        source_path=result.chunk.source_path,
        retrieval_score=result.score,
    )


# ADD 2026-08-21: Question script에 맞춘 고정 abstention message를 반환한다.
def _abstention_answer(question: str) -> str:
    if re.search(r"[가-힣]", question):
        return "제공된 SOP에서 해당 내용을 확인할 수 없습니다."
    return "The supplied SOP does not contain enough information to answer this question."
