"""Immutable RAG index artifact construction, integrity validation, and loading."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml.rag.chunking import ChunkingConfig, ManualChunk, chunk_documents
from ml.rag.documents import ManualDocument, load_manual_corpus
from services.rag.providers import EmbeddingProvider
from shared.hashing import is_sha256_digest, sha256_file

INDEX_SCHEMA_VERSION = 1
METADATA_FILENAME = "metadata.json"
CHUNKS_FILENAME = "chunks.jsonl"
EMBEDDINGS_FILENAME = "embeddings.npy"
_INDEX_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class CorruptIndexError(ValueError):
    """Index artifact integrity or schema validation failed."""


@dataclass(frozen=True)
class RagIndex:
    """Process-local immutable chunks and normalized embedding matrix."""

    index_id: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    chunks: tuple[ManualChunk, ...]
    embeddings: NDArray[np.float32]


@dataclass(frozen=True)
class IndexBuildResult:
    """Committed index directory and validated in-memory artifact."""

    index_dir: Path
    index: RagIndex


# ADD 2026-08-21: Corpus를 parse/chunk/embed해 atomic immutable RAG index로 commit한다.
def build_rag_index(
    *,
    corpus_dir: Path,
    output_root: Path,
    index_id: str,
    embedding_provider: EmbeddingProvider,
    chunking_config: ChunkingConfig,
    created_at: str,
) -> IndexBuildResult:
    """Build, validate, and atomically publish one non-overwriting index artifact."""
    _validate_index_id(index_id)
    _parse_aware_datetime(created_at)
    chunking_config.validate()
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / index_id
    if final_dir.exists():
        raise FileExistsError(f"RAG index artifact already exists: {final_dir}")

    # Source parsing과 embedding을 final path 밖에서 수행해 partial success를 노출하지 않는다.
    documents = load_manual_corpus(corpus_dir)
    chunks = chunk_documents(documents, chunking_config)
    raw_embeddings = embedding_provider.embed_documents([chunk.text for chunk in chunks])
    embeddings = _normalized_embedding_matrix(
        raw_embeddings,
        expected_count=len(chunks),
    )
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{index_id}.tmp-", dir=output_root))
    try:
        _write_index_files(
            temp_dir,
            index_id=index_id,
            documents=documents,
            chunks=chunks,
            embeddings=embeddings,
            embedding_provider=embedding_provider,
            chunking_config=chunking_config,
            created_at=created_at,
        )
        loaded = load_rag_index(temp_dir)
        temp_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return IndexBuildResult(index_dir=final_dir, index=loaded)


# ADD 2026-08-21: Metadata/chunks/embeddings hash와 shape를 검증해 process-local index를 복원한다.
def load_rag_index(index_dir: Path) -> RagIndex:
    """Fail fast on missing, corrupt, non-finite, or inconsistent index artifacts."""
    metadata_path = index_dir / METADATA_FILENAME
    chunks_path = index_dir / CHUNKS_FILENAME
    embeddings_path = index_dir / EMBEDDINGS_FILENAME
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        root = _metadata_mapping(metadata)
        if sha256_file(chunks_path) != root["artifacts"]["chunks_sha256"]:
            raise CorruptIndexError("RAG chunks artifact hash mismatch.")
        if sha256_file(embeddings_path) != root["artifacts"]["embeddings_sha256"]:
            raise CorruptIndexError("RAG embeddings artifact hash mismatch.")
        chunks = _read_chunks(chunks_path)
        embeddings = np.load(embeddings_path, allow_pickle=False)
    except CorruptIndexError:
        raise
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CorruptIndexError("Cannot load RAG index artifact.") from exc

    chunk_count = root["chunk_count"]
    dimension = root["embedding"]["dimension"]
    if len(chunks) != chunk_count:
        raise CorruptIndexError("RAG chunk count does not match metadata.")
    if embeddings.shape != (chunk_count, dimension):
        raise CorruptIndexError("RAG embedding matrix shape does not match metadata.")
    if embeddings.dtype != np.float32 or not np.isfinite(embeddings).all():
        raise CorruptIndexError("RAG embeddings must be finite float32 values.")
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms <= 0) or not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise CorruptIndexError("RAG document embeddings must use L2 normalization.")
    if root["document_count"] != len({chunk.document_id for chunk in chunks}):
        raise CorruptIndexError("RAG document count does not match chunks.")

    # Chunk citation provenance가 metadata의 canonical document record와 일치하는지 확인한다.
    documents_by_id = {document["document_id"]: document for document in root["documents"]}
    if len(documents_by_id) != root["document_count"]:
        raise CorruptIndexError("RAG document provenance IDs must be unique.")
    for chunk in chunks:
        document = documents_by_id.get(chunk.document_id)
        if document is None or (
            chunk.title,
            chunk.source_path,
            chunk.source_sha256,
        ) != (
            document["title"],
            document["source_path"],
            document["source_sha256"],
        ):
            raise CorruptIndexError("RAG chunk provenance does not match metadata.")
    embeddings.setflags(write=False)
    return RagIndex(
        index_id=root["index_id"],
        embedding_provider=root["embedding"]["provider"],
        embedding_model=root["embedding"]["model"],
        embedding_dimension=dimension,
        chunks=chunks,
        embeddings=embeddings,
    )


# ADD 2026-08-21: Raw provider vectors를 finite non-zero normalized float32 matrix로 변환한다.
def _normalized_embedding_matrix(
    raw_embeddings: Any,
    *,
    expected_count: int,
) -> NDArray[np.float32]:
    try:
        matrix = np.asarray(raw_embeddings, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Document embeddings must form a dense numeric matrix.") from exc
    if matrix.ndim != 2 or matrix.shape[0] != expected_count or matrix.shape[1] <= 0:
        raise ValueError("Document embedding count or dimension is invalid.")
    if not np.isfinite(matrix).all():
        raise ValueError("Document embeddings must be finite.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Document embeddings must not contain zero vectors.")
    return np.asarray(matrix / norms, dtype=np.float32)


# ADD 2026-08-21: Stable JSONL, NPY와 provenance metadata files를 temporary directory에 기록한다.
def _write_index_files(
    output_dir: Path,
    *,
    index_id: str,
    documents: tuple[ManualDocument, ...],
    chunks: tuple[ManualChunk, ...],
    embeddings: NDArray[np.float32],
    embedding_provider: EmbeddingProvider,
    chunking_config: ChunkingConfig,
    created_at: str,
) -> None:
    chunks_path = output_dir / CHUNKS_FILENAME
    embeddings_path = output_dir / EMBEDDINGS_FILENAME
    chunks_path.write_text(
        "".join(
            json.dumps(
                chunk.to_json_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for chunk in chunks
        ),
        encoding="utf-8",
    )
    with embeddings_path.open("wb") as file:
        np.save(file, embeddings, allow_pickle=False)
    metadata = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "index_id": index_id,
        "created_at": created_at,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "embedding": {
            "provider": embedding_provider.provider_name,
            "model": embedding_provider.model_name,
            "dimension": int(embeddings.shape[1]),
            "normalization": "l2",
        },
        "chunking": chunking_config.to_json_dict(),
        "documents": [
            {
                "document_id": document.document_id,
                "title": document.title,
                "source_path": document.source_path,
                "source_type": document.source_type,
                "source_sha256": document.source_sha256,
            }
            for document in documents
        ],
        "artifacts": {
            "chunks_sha256": sha256_file(chunks_path),
            "embeddings_sha256": sha256_file(embeddings_path),
        },
    }
    (output_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


# ADD 2026-08-21: Metadata schema, provenance path, hashes와 scalar types를 검증한다.
def _metadata_mapping(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise CorruptIndexError("RAG metadata must be a JSON object.")
    try:
        if type(raw["schema_version"]) is not int or raw["schema_version"] != INDEX_SCHEMA_VERSION:
            raise CorruptIndexError("Unsupported RAG index schema_version.")
        _validate_index_id(raw["index_id"])
        _parse_aware_datetime(raw["created_at"])
        for field in ("document_count", "chunk_count"):
            if type(raw[field]) is not int or raw[field] <= 0:
                raise CorruptIndexError(f"RAG metadata {field} must be positive.")
        embedding = raw["embedding"]
        artifacts = raw["artifacts"]
        documents = raw["documents"]
        if not isinstance(embedding, dict) or not isinstance(artifacts, dict):
            raise TypeError("embedding/artifacts must be objects")
        if (
            embedding["normalization"] != "l2"
            or type(embedding["dimension"]) is not int
            or embedding["dimension"] <= 0
        ):
            raise CorruptIndexError("RAG embedding metadata is invalid.")
        for field in ("provider", "model"):
            if not isinstance(embedding[field], str) or not embedding[field]:
                raise CorruptIndexError("RAG embedding identity is invalid.")
        for field in ("chunks_sha256", "embeddings_sha256"):
            if not is_sha256_digest(artifacts[field]):
                raise CorruptIndexError("RAG artifact digest is invalid.")
        if not isinstance(documents, list) or len(documents) != raw["document_count"]:
            raise CorruptIndexError("RAG document provenance count is invalid.")
        for document in documents:
            if not isinstance(document, dict):
                raise TypeError("document provenance must be object")
            for field in ("document_id", "title"):
                if not isinstance(document[field], str) or not document[field]:
                    raise CorruptIndexError("RAG document provenance identity is invalid.")
            if document["source_type"] not in {"markdown", "text"}:
                raise CorruptIndexError("RAG document source_type is invalid.")
            source_path = document["source_path"]
            if (
                not isinstance(source_path, str)
                or not source_path
                or source_path.startswith("/")
                or "\\" in source_path
                or re.match(r"^[A-Za-z]:", source_path)
                or ".." in source_path.split("/")
            ):
                raise CorruptIndexError("RAG source_path must be corpus-relative.")
            if not is_sha256_digest(document["source_sha256"]):
                raise CorruptIndexError("RAG source digest is invalid.")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, CorruptIndexError):
            raise
        raise CorruptIndexError("RAG metadata fields are invalid.") from exc
    return raw


# ADD 2026-08-21: JSONL chunk records를 blank-line 없이 strict schema로 복원한다.
def _read_chunks(path: Path) -> tuple[ManualChunk, ...]:
    chunks: list[ManualChunk] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise CorruptIndexError(f"RAG chunks JSONL has a blank line at {line_number}.")
        try:
            chunks.append(ManualChunk.from_json_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise CorruptIndexError(f"Invalid RAG chunk at line {line_number}.") from exc
    if not chunks or len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise CorruptIndexError("RAG chunks must be non-empty with unique IDs.")
    return tuple(chunks)


# ADD 2026-08-21: Index identifier가 safe single path segment인지 검증한다.
def _validate_index_id(index_id: object) -> None:
    if (
        not isinstance(index_id, str)
        or not _INDEX_ID_PATTERN.fullmatch(index_id)
        or index_id in {".", ".."}
    ):
        raise ValueError("RAG index_id must be one safe 1-128 character path segment.")


# ADD 2026-08-21: Artifact created_at이 timezone-aware ISO-8601인지 검증한다.
def _parse_aware_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("RAG created_at must be an ISO-8601 string.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("RAG created_at must be ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("RAG created_at must include a timezone offset.")
    return parsed
