"""Streaming file hashing helpers."""

import hashlib
from pathlib import Path


# ADD 2026-08-20: In-memory bytes의 SHA-256 digest를 disk I/O 없이 계산한다.
def sha256_bytes(content: bytes) -> str:
    """Return the SHA-256 hexadecimal digest of immutable bytes."""
    return hashlib.sha256(content).hexdigest()


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


# ADD 2026-08-20: 문자열이 SHA-256 hexadecimal digest 형식인지 반환한다.
def is_sha256_digest(value: str) -> bool:
    """Return whether a string is exactly one SHA-256 hexadecimal digest."""
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
