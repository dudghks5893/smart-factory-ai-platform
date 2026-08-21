"""Pydantic HTTP schemas for the independent RAG service."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from services.rag.config import MAX_QUESTION_CHARACTERS


class HealthResponse(BaseModel):
    """RAG process liveness independent of index/provider readiness."""

    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    """Loaded index and configured provider identity without network checks."""

    status: Literal["ready"]
    index_id: str
    embedding_provider: str
    embedding_model: str
    generation_provider: str
    generation_model: str


class RagQueryRequest(BaseModel):
    """Bounded user question and optional per-request retrieval size."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARACTERS)
    top_k: int | None = Field(default=None, ge=1)

    # ADD 2026-08-21: Whitespace-only question을 Pydantic length validation 뒤에도 거부한다.
    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Strip surrounding whitespace while treating question content only as user data."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be empty")
        return normalized


class CitationResponse(BaseModel):
    """Structured citation linked to one retrieved manual chunk."""

    model_config = ConfigDict(allow_inf_nan=False)

    citation_id: str
    chunk_id: str
    document_id: str
    title: str
    section: str
    page: int | None
    source_path: str
    retrieval_score: float


class RetrievalEvidenceResponse(BaseModel):
    """Compact ranked evidence retained for deterministic evaluation."""

    model_config = ConfigDict(allow_inf_nan=False)

    rank: int
    chunk_id: str
    document_id: str
    score: float


class RagQueryResponse(BaseModel):
    """Grounded answer or abstention with citations and retrieval evidence."""

    status: Literal["answered", "insufficient_context"]
    answer: str
    citations: list[CitationResponse]
    retrieval: list[RetrievalEvidenceResponse]


class ErrorDetail(BaseModel):
    """Stable public RAG error code and non-sensitive message."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """RAG service error envelope."""

    error: ErrorDetail
