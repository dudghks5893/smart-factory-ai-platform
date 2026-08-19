"""Streaming file hashing helpers."""

import hashlib
from pathlib import Path


# ADD 2026-08-19: Calculate a file digest without loading the full file into memory.
def sha256_file(path: Path) -> str:
    """Calculate a file digest without loading the full file into memory."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found for SHA-256 calculation: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
