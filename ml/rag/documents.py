"""Deterministic Markdown and plain-text manual discovery and parsing."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from shared.hashing import sha256_file

SUPPORTED_MANUAL_SUFFIXES = {".md": "markdown", ".markdown": "markdown", ".txt": "text"}
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class DocumentSection:
    """One heading/page-preserving group of source paragraphs."""

    heading: str
    paragraphs: tuple[str, ...]
    page: int | None = None


@dataclass(frozen=True)
class ManualDocument:
    """Parsed manual with repository-relative provenance and stable identity."""

    document_id: str
    title: str
    source_path: str
    source_sha256: str
    source_type: str
    sections: tuple[DocumentSection, ...]


# ADD 2026-08-21: Corpus root에서 supported manual을 relative path 순으로 발견한다.
def discover_manual_paths(corpus_dir: Path) -> tuple[Path, ...]:
    """Return deterministic Markdown/TXT paths and reject unsupported visible files."""
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Manual corpus directory not found: {corpus_dir}")
    paths: list[Path] = []
    unsupported: list[str] = []
    for path in corpus_dir.rglob("*"):
        relative = path.relative_to(corpus_dir)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Manual corpus must not contain symlinks: {relative.as_posix()}")
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_MANUAL_SUFFIXES:
            unsupported.append(relative.as_posix())
        else:
            paths.append(path)
    if unsupported:
        raise ValueError("Unsupported manual source files: " + ", ".join(sorted(unsupported)))
    if not paths:
        raise ValueError("Manual corpus contains no supported documents.")
    return tuple(sorted(paths, key=lambda path: path.relative_to(corpus_dir).as_posix()))


# ADD 2026-08-21: Supported source를 stable provenance와 section metadata로 parsing한다.
def parse_manual(path: Path, *, corpus_dir: Path) -> ManualDocument:
    """Parse one Markdown or TXT document without storing an absolute local path."""
    try:
        relative_path = path.resolve().relative_to(corpus_dir.resolve())
    except ValueError as exc:
        raise ValueError("Manual source must be inside the configured corpus directory.") from exc
    source_path = relative_path.as_posix()
    source_type = SUPPORTED_MANUAL_SUFFIXES.get(path.suffix.lower())
    if source_type is None:
        raise ValueError(f"Unsupported manual source type: {path.suffix}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot read UTF-8 manual source: {source_path}") from exc
    if not text.strip():
        raise ValueError(f"Manual source is empty: {source_path}")

    if source_type == "markdown":
        title, sections = _parse_markdown(text, fallback_title=path.stem)
    else:
        title, sections = _parse_text(text, fallback_title=path.stem)
    document_id = "doc-" + hashlib.sha256(f"manual-document-v1\0{source_path}".encode()).hexdigest()
    return ManualDocument(
        document_id=document_id,
        title=title,
        source_path=source_path,
        source_sha256=sha256_file(path),
        source_type=source_type,
        sections=sections,
    )


# ADD 2026-08-21: Discovered corpus 전체를 deterministic document tuple로 parsing한다.
def load_manual_corpus(corpus_dir: Path) -> tuple[ManualDocument, ...]:
    """Load every supported manual in stable relative-path order."""
    return tuple(
        parse_manual(path, corpus_dir=corpus_dir) for path in discover_manual_paths(corpus_dir)
    )


# ADD 2026-08-21: Markdown heading hierarchy와 paragraph boundary를 section으로 보존한다.
def _parse_markdown(text: str, *, fallback_title: str) -> tuple[str, tuple[DocumentSection, ...]]:
    title = fallback_title.replace("_", " ").strip()
    headings: list[str] = []
    current_heading = "Document"
    paragraph_lines: list[str] = []
    paragraphs: list[str] = []
    sections: list[DocumentSection] = []

    # Heading 전환 전에 pending paragraph/section을 flush해 source order를 보존한다.
    for raw_line in text.splitlines():
        heading_match = _MARKDOWN_HEADING.match(raw_line)
        if heading_match:
            _flush_paragraph(paragraph_lines, paragraphs)
            _flush_section(current_heading, paragraphs, sections)
            level = len(heading_match.group(1))
            heading_text = _normalize_text(heading_match.group(2))
            if level == 1 and not sections and current_heading == "Document":
                title = heading_text
            headings = headings[: level - 1]
            headings.append(heading_text)
            current_heading = " > ".join(headings)
        elif raw_line.strip():
            paragraph_lines.append(raw_line.strip())
        else:
            _flush_paragraph(paragraph_lines, paragraphs)
    _flush_paragraph(paragraph_lines, paragraphs)
    _flush_section(current_heading, paragraphs, sections)
    if not sections:
        raise ValueError("Markdown manual contains no readable paragraphs.")
    return title, tuple(sections)


# ADD 2026-08-21: Plain text의 첫 줄 title과 blank-line paragraph boundary를 보존한다.
def _parse_text(text: str, *, fallback_title: str) -> tuple[str, tuple[DocumentSection, ...]]:
    raw_paragraphs = re.split(r"\n\s*\n", text.strip())
    paragraphs = tuple(
        normalized for paragraph in raw_paragraphs if (normalized := _normalize_text(paragraph))
    )
    if not paragraphs:
        raise ValueError("Text manual contains no readable paragraphs.")
    first_line = _normalize_text(text.splitlines()[0])
    title = first_line if len(first_line) <= 160 else fallback_title.replace("_", " ")
    return title, (DocumentSection(heading="Document", paragraphs=paragraphs),)


# ADD 2026-08-21: Accumulated source lines를 normalized paragraph 하나로 flush한다.
def _flush_paragraph(lines: list[str], paragraphs: list[str]) -> None:
    if lines:
        paragraphs.append(_normalize_text(" ".join(lines)))
        lines.clear()


# ADD 2026-08-21: Non-empty paragraph group을 immutable section으로 flush한다.
def _flush_section(
    heading: str,
    paragraphs: list[str],
    sections: list[DocumentSection],
) -> None:
    if paragraphs:
        sections.append(DocumentSection(heading=heading, paragraphs=tuple(paragraphs)))
        paragraphs.clear()


# ADD 2026-08-21: Source whitespace를 content-preserving single spaces로 정규화한다.
def _normalize_text(value: str) -> str:
    return " ".join(value.split())
