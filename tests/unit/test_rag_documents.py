"""Unit tests for deterministic manual parsing and paragraph-aware chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.rag.chunking import ChunkingConfig, chunk_documents
from ml.rag.documents import discover_manual_paths, load_manual_corpus, parse_manual
from shared.hashing import sha256_file


# ADD 2026-08-21: Markdown title/heading/paragraph와 relative provenance parsing을 검증한다.
def test_markdown_parsing_preserves_sections_and_source_sha(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manual.md"
    path.parent.mkdir()
    path.write_text(
        "# Demo Manual\n\nIntro text.\n\n## Camera Check\n\nClean lens.\n", encoding="utf-8"
    )

    document = parse_manual(path, corpus_dir=tmp_path)

    assert document.title == "Demo Manual"
    assert document.source_path == "nested/manual.md"
    assert document.source_sha256 == sha256_file(path)
    assert [section.heading for section in document.sections] == [
        "Demo Manual",
        "Demo Manual > Camera Check",
    ]
    assert document.sections[1].paragraphs == ("Clean lens.",)


# ADD 2026-08-21: Plain text paragraph parsing과 deterministic relative-path discovery를 검증한다.
def test_text_parsing_and_document_order_are_deterministic(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("B title\n\nSecond paragraph.", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A title\n\nFirst paragraph.", encoding="utf-8")

    paths = discover_manual_paths(tmp_path)
    documents = load_manual_corpus(tmp_path)

    assert [path.name for path in paths] == ["a.md", "b.txt"]
    assert [document.source_path for document in documents] == ["a.md", "b.txt"]
    assert documents[1].sections[0].paragraphs[-1] == "Second paragraph."


# ADD 2026-08-21: Empty source와 unsupported visible extension을 명시적으로 거부한다.
def test_empty_and_unsupported_manual_sources_are_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.md"
    empty.write_text(" \n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        parse_manual(empty, corpus_dir=tmp_path)

    empty.unlink()
    (tmp_path / "manual.pdf").write_bytes(b"not-supported")
    with pytest.raises(ValueError, match="Unsupported"):
        discover_manual_paths(tmp_path)


# ADD 2026-08-21: Same source/config가 stable IDs, size, section과 metadata를 보존하는지 검증한다.
def test_chunking_is_deterministic_bounded_and_provenance_preserving(tmp_path: Path) -> None:
    path = tmp_path / "manual.md"
    path.write_text(
        "# Demo\n\n## Camera\n\nFirst camera paragraph has details.\n\n"
        "Second camera paragraph has more details.\n\nThird camera paragraph ends here.\n",
        encoding="utf-8",
    )
    documents = load_manual_corpus(tmp_path)
    config = ChunkingConfig(max_characters=100, overlap_paragraphs=1)

    first = chunk_documents(documents, config)
    second = chunk_documents(documents, config)

    assert first == second
    assert all(len(chunk.text) <= 100 for chunk in first)
    assert all(chunk.section == "Demo > Camera" for chunk in first)
    assert all(chunk.source_path == "manual.md" for chunk in first)
    assert all(chunk.source_sha256 == documents[0].source_sha256 for chunk in first)
    assert len({chunk.chunk_id for chunk in first}) == len(first)


# ADD 2026-08-21: Chunk overflow가 paragraph overlap을 다음 chunk에 보존하는지 검증한다.
def test_chunking_preserves_configured_paragraph_overlap(tmp_path: Path) -> None:
    path = tmp_path / "manual.txt"
    paragraphs = [f"Paragraph {index} contains repeated detail words." for index in range(4)]
    path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    chunks = chunk_documents(
        load_manual_corpus(tmp_path),
        ChunkingConfig(max_characters=100, overlap_paragraphs=1),
    )

    assert len(chunks) >= 2
    first_last_paragraph = chunks[0].text.split("\n\n")[-1]
    assert chunks[1].text.startswith(first_last_paragraph)
