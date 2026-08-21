"""Unit tests for RAG artifact integrity and exact cosine retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.rag.chunking import ChunkingConfig, ManualChunk
from pipelines.build_rag_index import RagIndexBuildConfig, run_rag_index_build
from services.rag.index import CorruptIndexError, RagIndex, build_rag_index, load_rag_index
from services.rag.retrieval import ExactCosineRetriever
from shared.hashing import sha256_file
from tests.rag_helpers import FixedQueryEmbeddingProvider, KeywordEmbeddingProvider


# ADD 2026-08-21: Small index test용 manual corpus를 생성한다.
def _manual_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "manuals"
    corpus.mkdir(parents=True)
    (corpus / "camera.md").write_text(
        "# Demo Camera\n\n## Lens\n\nClean the camera lens before inspection.\n",
        encoding="utf-8",
    )
    (corpus / "response.txt").write_text(
        "Demo Response\n\nPlace anomaly items in quarantine and request reviewer escalation.",
        encoding="utf-8",
    )
    return corpus


# ADD 2026-08-21: Retrieval unit test용 minimal chunk를 생성한다.
def _chunk(chunk_id: str, index: int) -> ManualChunk:
    return ManualChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{index}",
        title=f"Document {index}",
        source_path=f"manual-{index}.md",
        section="Section",
        page=None,
        text=f"Text {index}",
        chunk_index=0,
        source_sha256=f"{index + 1:064x}",
    )


# ADD 2026-08-21: Index count, dimension, hash와 relative provenance를 검증한다.
def test_index_build_round_trip_and_overwrite_rejection(tmp_path: Path) -> None:
    corpus = _manual_corpus(tmp_path)
    result = build_rag_index(
        corpus_dir=corpus,
        output_root=tmp_path / "artifacts",
        index_id="demo-index",
        embedding_provider=KeywordEmbeddingProvider(),
        chunking_config=ChunkingConfig(max_characters=200, overlap_paragraphs=1),
        created_at="2026-08-21T00:00:00+00:00",
    )
    metadata = json.loads((result.index_dir / "metadata.json").read_text(encoding="utf-8"))

    assert len(result.index.chunks) == metadata["chunk_count"]
    assert result.index.embeddings.shape == (
        metadata["chunk_count"],
        metadata["embedding"]["dimension"],
    )
    assert metadata["artifacts"]["chunks_sha256"] == sha256_file(result.index_dir / "chunks.jsonl")
    assert str(tmp_path) not in json.dumps(metadata)
    assert all(not chunk.source_path.startswith("/") for chunk in result.index.chunks)
    with pytest.raises(FileExistsError):
        build_rag_index(
            corpus_dir=corpus,
            output_root=tmp_path / "artifacts",
            index_id="demo-index",
            embedding_provider=KeywordEmbeddingProvider(),
            chunking_config=ChunkingConfig(),
            created_at="2026-08-21T00:00:00+00:00",
        )


# ADD 2026-08-21: Same corpus/config의 stable chunk ordering/IDs를 검증한다.
def test_index_chunks_are_deterministic_across_builds(tmp_path: Path) -> None:
    corpus = _manual_corpus(tmp_path)
    config = ChunkingConfig(max_characters=200, overlap_paragraphs=1)
    first = build_rag_index(
        corpus_dir=corpus,
        output_root=tmp_path / "artifacts",
        index_id="first",
        embedding_provider=KeywordEmbeddingProvider(),
        chunking_config=config,
        created_at="2026-08-21T00:00:00+00:00",
    )
    second = build_rag_index(
        corpus_dir=corpus,
        output_root=tmp_path / "artifacts",
        index_id="second",
        embedding_provider=KeywordEmbeddingProvider(),
        chunking_config=config,
        created_at="2026-08-21T00:00:00+00:00",
    )

    assert [chunk.chunk_id for chunk in first.index.chunks] == [
        chunk.chunk_id for chunk in second.index.chunks
    ]
    assert np.array_equal(first.index.embeddings, second.index.embeddings)


# ADD 2026-08-21: Non-finite document embedding이 partial artifact commit 전에 거부되는지 검증한다.
def test_index_build_rejects_nonfinite_embeddings(tmp_path: Path) -> None:
    corpus = _manual_corpus(tmp_path)
    provider = FixedQueryEmbeddingProvider([float("nan"), 1.0])

    with pytest.raises(ValueError, match="finite"):
        build_rag_index(
            corpus_dir=corpus,
            output_root=tmp_path / "artifacts",
            index_id="invalid",
            embedding_provider=provider,
            chunking_config=ChunkingConfig(),
            created_at="2026-08-21T00:00:00+00:00",
        )
    assert not (tmp_path / "artifacts" / "invalid").exists()


# ADD 2026-08-21: Chunks tampering과 embedding shape/hash inconsistency를 fail-fast 검증한다.
def test_index_load_rejects_corrupt_hash_and_shape(tmp_path: Path) -> None:
    result = build_rag_index(
        corpus_dir=_manual_corpus(tmp_path),
        output_root=tmp_path / "artifacts",
        index_id="corrupt",
        embedding_provider=KeywordEmbeddingProvider(),
        chunking_config=ChunkingConfig(),
        created_at="2026-08-21T00:00:00+00:00",
    )
    chunks_path = result.index_dir / "chunks.jsonl"
    chunks_path.write_text(chunks_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(CorruptIndexError, match="hash"):
        load_rag_index(result.index_dir)

    result2 = build_rag_index(
        corpus_dir=_manual_corpus(tmp_path / "second"),
        output_root=tmp_path / "artifacts",
        index_id="shape",
        embedding_provider=KeywordEmbeddingProvider(),
        chunking_config=ChunkingConfig(),
        created_at="2026-08-21T00:00:00+00:00",
    )
    embedding_path = result2.index_dir / "embeddings.npy"
    with embedding_path.open("wb") as file:
        np.save(file, np.ones((1, 2), dtype=np.float32), allow_pickle=False)
    metadata_path = result2.index_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["artifacts"]["embeddings_sha256"] = sha256_file(embedding_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(CorruptIndexError, match="shape"):
        load_rag_index(result2.index_dir)


# ADD 2026-08-21: Invalid metadata schema와 rehashed chunk provenance tampering을 거부한다.
def test_index_load_rejects_corrupt_metadata_and_provenance(tmp_path: Path) -> None:
    schema_result = build_rag_index(
        corpus_dir=_manual_corpus(tmp_path / "schema"),
        output_root=tmp_path / "artifacts",
        index_id="schema",
        embedding_provider=KeywordEmbeddingProvider(),
        chunking_config=ChunkingConfig(),
        created_at="2026-08-21T00:00:00+00:00",
    )
    schema_path = schema_result.index_dir / "metadata.json"
    schema_metadata = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_metadata["schema_version"] = True
    schema_path.write_text(json.dumps(schema_metadata), encoding="utf-8")
    with pytest.raises(CorruptIndexError, match="schema_version"):
        load_rag_index(schema_result.index_dir)

    provenance_result = build_rag_index(
        corpus_dir=_manual_corpus(tmp_path / "provenance"),
        output_root=tmp_path / "artifacts",
        index_id="provenance",
        embedding_provider=KeywordEmbeddingProvider(),
        chunking_config=ChunkingConfig(),
        created_at="2026-08-21T00:00:00+00:00",
    )
    chunks_path = provenance_result.index_dir / "chunks.jsonl"
    chunk_lines = chunks_path.read_text(encoding="utf-8").splitlines()
    first_chunk = json.loads(chunk_lines[0])
    first_chunk["title"] = "Tampered title"
    chunk_lines[0] = json.dumps(first_chunk)
    chunks_path.write_text("\n".join(chunk_lines) + "\n", encoding="utf-8")
    metadata_path = provenance_result.index_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["artifacts"]["chunks_sha256"] = sha256_file(chunks_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(CorruptIndexError, match="provenance"):
        load_rag_index(provenance_result.index_dir)


# ADD 2026-08-21: Exact cosine top-k, score ordering과 chunk-id tie break를 검증한다.
def test_exact_retrieval_orders_scores_and_ties_deterministically() -> None:
    chunks = (_chunk("chunk-b", 0), _chunk("chunk-a", 1), _chunk("chunk-c", 2))
    matrix = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    matrix.setflags(write=False)
    index = RagIndex("test", "test", "test-v1", 2, chunks, matrix)
    retriever = ExactCosineRetriever(
        index=index,
        embedding_provider=FixedQueryEmbeddingProvider([1.0, 0.0]),
        max_top_k=3,
        minimum_score=-1,
    )

    results = retriever.retrieve("camera", top_k=3)

    assert [result.chunk.chunk_id for result in results] == ["chunk-a", "chunk-b", "chunk-c"]
    assert [result.rank for result in results] == [1, 2, 3]


# ADD 2026-08-21: Retrieval threshold/bounds, invalid vector와 empty index를 검증한다.
def test_retrieval_threshold_bounds_and_invalid_vectors() -> None:
    chunk = _chunk("chunk-a", 0)
    index = RagIndex(
        "test",
        "test",
        "test-v1",
        2,
        (chunk,),
        np.asarray([[1, 0]], dtype=np.float32),
    )
    thresholded = ExactCosineRetriever(
        index=index,
        embedding_provider=FixedQueryEmbeddingProvider([0.0, 1.0]),
        max_top_k=2,
        minimum_score=0.5,
    )
    assert thresholded.retrieve("unrelated", top_k=1) == ()
    with pytest.raises(ValueError, match="top_k"):
        thresholded.retrieve("question", top_k=3)

    for invalid in ([0.0, 0.0], [float("inf"), 0.0]):
        retriever = ExactCosineRetriever(
            index=index,
            embedding_provider=FixedQueryEmbeddingProvider(invalid),
            max_top_k=1,
            minimum_score=-1,
        )
        with pytest.raises(ValueError):
            retriever.retrieve("question", top_k=1)

    empty = RagIndex(
        "empty",
        "test",
        "test-v1",
        2,
        (),
        np.empty((0, 2), dtype=np.float32),
    )
    empty_retriever = ExactCosineRetriever(
        index=empty,
        embedding_provider=FixedQueryEmbeddingProvider([1.0, 0.0]),
        max_top_k=1,
        minimum_score=-1,
    )
    assert empty_retriever.retrieve("question", top_k=1) == ()


# ADD 2026-08-21: Pipeline wrapper가 actual demo corpus를 index하고 camera evidence를 검색한다.
def test_demo_manual_retrieval_smoke(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    provider = KeywordEmbeddingProvider()
    result = run_rag_index_build(
        RagIndexBuildConfig(
            manuals_dir=project_root / "manuals" / "demo",
            output_root=tmp_path / "artifacts",
            index_id="demo-smoke",
            chunking=ChunkingConfig(max_characters=500, overlap_paragraphs=1),
        ),
        embedding_provider=provider,
        created_at="2026-08-21T00:00:00+00:00",
    )
    retriever = ExactCosineRetriever(
        index=result.index,
        embedding_provider=provider,
        max_top_k=5,
        minimum_score=0.1,
    )

    results = retriever.retrieve("What camera lens check is required?", top_k=3)

    assert results
    assert any("camera" in item.chunk.text.lower() for item in results)
    assert all(not item.chunk.source_path.startswith("/") for item in results)
