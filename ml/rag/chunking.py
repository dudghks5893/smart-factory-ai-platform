"""Paragraph-aware deterministic manual chunking contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass

from ml.rag.documents import ManualDocument

DEFAULT_MAX_CHARACTERS = 1200
DEFAULT_OVERLAP_PARAGRAPHS = 1
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(frozen=True)
class ChunkingConfig:
    """Explicit maximum size and paragraph-overlap policy."""

    max_characters: int = DEFAULT_MAX_CHARACTERS
    overlap_paragraphs: int = DEFAULT_OVERLAP_PARAGRAPHS

    # ADD 2026-08-21: Chunk size와 bounded overlap configuration을 검증한다.
    def validate(self) -> None:
        """Reject sizes that fragment context or create unbounded repetition."""
        if not 100 <= self.max_characters <= 20_000:
            raise ValueError("Chunk max_characters must be in [100, 20000].")
        if not 0 <= self.overlap_paragraphs <= 10:
            raise ValueError("Chunk overlap_paragraphs must be in [0, 10].")

    # ADD 2026-08-21: Chunking configuration을 stable artifact mapping으로 변환한다.
    def to_json_dict(self) -> dict[str, int]:
        """Return deterministic JSON-compatible configuration."""
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ManualChunk:
    """One deterministic retrieval unit with complete source provenance."""

    chunk_id: str
    document_id: str
    title: str
    source_path: str
    section: str
    page: int | None
    text: str
    chunk_index: int
    source_sha256: str

    # ADD 2026-08-21: Chunk를 stable JSONL mapping으로 변환한다.
    def to_json_dict(self) -> dict[str, str | int | None]:
        """Return all citation and integrity fields without absolute paths."""
        return asdict(self)

    # ADD 2026-08-21: JSONL chunk field와 provenance invariant를 검증해 복원한다.
    @classmethod
    def from_json_dict(cls, raw: object) -> ManualChunk:
        """Restore one strict chunk artifact record."""
        if not isinstance(raw, dict):
            raise ValueError("RAG chunk must be a JSON object.")
        expected = {
            "chunk_id",
            "document_id",
            "title",
            "source_path",
            "section",
            "page",
            "text",
            "chunk_index",
            "source_sha256",
        }
        if set(raw) != expected:
            raise ValueError("RAG chunk fields do not match the schema.")
        page = raw["page"]
        if page is not None and (type(page) is not int or page <= 0):
            raise ValueError("RAG chunk page must be null or positive integer.")
        chunk_index = raw["chunk_index"]
        if type(chunk_index) is not int or chunk_index < 0:
            raise ValueError("RAG chunk_index must be non-negative.")
        chunk = cls(
            chunk_id=_required_string(raw["chunk_id"], "chunk_id"),
            document_id=_required_string(raw["document_id"], "document_id"),
            title=_required_string(raw["title"], "title"),
            source_path=_required_string(raw["source_path"], "source_path"),
            section=_required_string(raw["section"], "section"),
            page=page,
            text=_required_string(raw["text"], "text"),
            chunk_index=chunk_index,
            source_sha256=_required_string(raw["source_sha256"], "source_sha256"),
        )
        if (
            chunk.source_path.startswith("/")
            or "\\" in chunk.source_path
            or re.match(r"^[A-Za-z]:", chunk.source_path)
            or ".." in chunk.source_path.split("/")
        ):
            raise ValueError("RAG chunk source_path must be corpus-relative.")
        return chunk


# ADD 2026-08-21: Parsed documents를 source order와 section boundary를 보존해 chunking한다.
def chunk_documents(
    documents: tuple[ManualDocument, ...],
    config: ChunkingConfig,
) -> tuple[ManualChunk, ...]:
    """Return deterministic chunks with paragraph overlap and stable IDs."""
    config.validate()
    if not documents:
        raise ValueError("Cannot chunk an empty manual corpus.")
    chunks: list[ManualChunk] = []
    for document in documents:
        document_chunks: list[ManualChunk] = []
        chunk_index = 0
        for section in document.sections:
            units = tuple(
                unit
                for paragraph in section.paragraphs
                for unit in _split_oversized_paragraph(paragraph, config.max_characters)
            )
            for text in _pack_units(units, config):
                chunk_id = _chunk_id(document, section.heading, section.page, chunk_index, text)
                document_chunks.append(
                    ManualChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        title=document.title,
                        source_path=document.source_path,
                        section=section.heading,
                        page=section.page,
                        text=text,
                        chunk_index=chunk_index,
                        source_sha256=document.source_sha256,
                    )
                )
                chunk_index += 1
        if not document_chunks:
            raise ValueError(f"Manual produced no chunks: {document.source_path}")
        chunks.extend(document_chunks)
    return tuple(chunks)


# ADD 2026-08-21: Oversized paragraph를 sentence 우선, word fallback으로 bounded unit화한다.
def _split_oversized_paragraph(paragraph: str, max_characters: int) -> tuple[str, ...]:
    if len(paragraph) <= max_characters:
        return (paragraph,)
    sentences = tuple(part.strip() for part in _SENTENCE_BOUNDARY.split(paragraph) if part.strip())
    units: list[str] = []
    current = ""
    for sentence in sentences:
        for part in _split_oversized_sentence(sentence, max_characters):
            candidate = part if not current else f"{current} {part}"
            if len(candidate) <= max_characters:
                current = candidate
            else:
                units.append(current)
                current = part
    if current:
        units.append(current)
    return tuple(units)


# ADD 2026-08-21: Single oversized sentence를 whitespace boundary에서만 분할한다.
def _split_oversized_sentence(sentence: str, max_characters: int) -> tuple[str, ...]:
    if len(sentence) <= max_characters:
        return (sentence,)
    words = sentence.split()
    if any(len(word) > max_characters for word in words):
        raise ValueError("Manual contains a token longer than chunk max_characters.")
    parts: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_characters:
            current = candidate
        else:
            parts.append(current)
            current = word
    if current:
        parts.append(current)
    return tuple(parts)


# ADD 2026-08-21: Paragraph units를 overlap policy로 bounded chunk text에 packing한다.
def _pack_units(units: tuple[str, ...], config: ChunkingConfig) -> tuple[str, ...]:
    if not units:
        return ()
    packed: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = "\n\n".join([*current, unit])
        if current and len(candidate) > config.max_characters:
            packed.append("\n\n".join(current))
            current = current[-config.overlap_paragraphs :] if config.overlap_paragraphs else []
            while current and len("\n\n".join([*current, unit])) > config.max_characters:
                current.pop(0)
        current.append(unit)
    if current:
        text = "\n\n".join(current)
        if not packed or text != packed[-1]:
            packed.append(text)
    if any(len(text) > config.max_characters for text in packed):
        raise RuntimeError("Chunk packing exceeded max_characters.")
    return tuple(packed)


# ADD 2026-08-21: Source/config/content provenance에서 deterministic chunk digest를 생성한다.
def _chunk_id(
    document: ManualDocument,
    section: str,
    page: int | None,
    chunk_index: int,
    text: str,
) -> str:
    payload = json.dumps(
        {
            "schema": 1,
            "document_id": document.document_id,
            "source_sha256": document.source_sha256,
            "section": section,
            "page": page,
            "chunk_index": chunk_index,
            "text": text,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "chunk-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ADD 2026-08-21: Required chunk JSON string을 검증한다.
def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"RAG chunk {field} must be a non-empty string.")
    return value
