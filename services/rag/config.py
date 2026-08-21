"""Environment-backed configuration for offline/online RAG providers and service."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from services.rag.providers import OpenAICompatibleClientConfig

SUPPORTED_PROVIDER = "openai-compatible"
DEFAULT_PROVIDER_API_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TOP_K = 5
DEFAULT_MAX_TOP_K = 10
DEFAULT_MINIMUM_RETRIEVAL_SCORE = 0.2
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
MAX_QUESTION_CHARACTERS = 2000


@dataclass(frozen=True)
class ExternalProviderSettings:
    """Embedding/generation identity and secret-bearing HTTP runtime settings."""

    embedding_provider: str
    embedding_model: str
    generation_provider: str | None
    generation_model: str | None
    api_base_url: str
    api_key: str = field(repr=False)
    request_timeout_seconds: float

    # ADD 2026-08-21: Offline/online provider environment를 explicit required fields로 로드한다.
    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        require_generation: bool,
    ) -> ExternalProviderSettings:
        """Load provider configuration without logging or persisting the API key."""
        values = os.environ if environ is None else environ
        required = ["RAG_EMBEDDING_PROVIDER", "RAG_EMBEDDING_MODEL", "RAG_PROVIDER_API_KEY"]
        if require_generation:
            required.extend(["RAG_GENERATION_PROVIDER", "RAG_GENERATION_MODEL"])
        missing = [name for name in required if not values.get(name, "").strip()]
        if missing:
            raise ValueError("Missing required RAG provider variables: " + ", ".join(missing))
        try:
            timeout = float(
                values.get(
                    "RAG_REQUEST_TIMEOUT_SECONDS",
                    str(DEFAULT_REQUEST_TIMEOUT_SECONDS),
                )
            )
        except ValueError as exc:
            raise ValueError("RAG_REQUEST_TIMEOUT_SECONDS must be numeric.") from exc
        settings = cls(
            embedding_provider=values["RAG_EMBEDDING_PROVIDER"].strip(),
            embedding_model=values["RAG_EMBEDDING_MODEL"].strip(),
            generation_provider=values.get("RAG_GENERATION_PROVIDER", "").strip() or None,
            generation_model=values.get("RAG_GENERATION_MODEL", "").strip() or None,
            api_base_url=values.get(
                "RAG_PROVIDER_API_BASE_URL",
                DEFAULT_PROVIDER_API_BASE_URL,
            ).strip(),
            api_key=values["RAG_PROVIDER_API_KEY"].strip(),
            request_timeout_seconds=timeout,
        )
        settings.validate(require_generation=require_generation)
        return settings

    # ADD 2026-08-21: Supported provider identity, URL, timeout과 generation pairing을 검증한다.
    def validate(self, *, require_generation: bool) -> None:
        """Reject unsupported or incomplete production provider configuration."""
        if self.embedding_provider != SUPPORTED_PROVIDER:
            raise ValueError(f"RAG embedding provider must be {SUPPORTED_PROVIDER}.")
        if not self.embedding_model or not self.api_key:
            raise ValueError("RAG embedding model and provider API key are required.")
        if require_generation and (
            self.generation_provider != SUPPORTED_PROVIDER or not self.generation_model
        ):
            raise ValueError(f"RAG generation provider must be {SUPPORTED_PROVIDER}.")
        parsed = urlsplit(self.api_base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("RAG_PROVIDER_API_BASE_URL must be an absolute HTTP(S) URL.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("RAG_PROVIDER_API_BASE_URL must not embed credentials.")
        if parsed.query or parsed.fragment:
            raise ValueError("RAG_PROVIDER_API_BASE_URL must not contain query or fragment.")
        if (
            not math.isfinite(self.request_timeout_seconds)
            or not 0 < self.request_timeout_seconds <= 120
        ):
            raise ValueError("RAG_REQUEST_TIMEOUT_SECONDS must be in (0, 120].")

    # ADD 2026-08-21: Provider adapter가 사용할 bounded HTTP configuration을 생성한다.
    def client_config(self) -> OpenAICompatibleClientConfig:
        """Return runtime-only HTTP settings including the non-persisted secret."""
        return OpenAICompatibleClientConfig(
            api_base_url=self.api_base_url.rstrip("/"),
            api_key=self.api_key,
            timeout_seconds=self.request_timeout_seconds,
        )


@dataclass(frozen=True)
class RagSettings:
    """Validated process settings for one loaded RAG index and retrieval policy."""

    index_dir: Path
    top_k: int
    max_top_k: int
    minimum_retrieval_score: float
    provider: ExternalProviderSettings

    # ADD 2026-08-21: RAG service environment와 provider settings를 한 번 로드한다.
    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> RagSettings:
        """Load required index, retrieval policy, and production provider configuration."""
        values = os.environ if environ is None else environ
        index_dir = values.get("RAG_INDEX_DIR", "").strip()
        if not index_dir:
            raise ValueError("RAG_INDEX_DIR is required.")
        try:
            top_k = int(values.get("RAG_TOP_K", str(DEFAULT_TOP_K)))
            max_top_k = int(values.get("RAG_MAX_TOP_K", str(DEFAULT_MAX_TOP_K)))
            minimum_score = float(
                values.get(
                    "RAG_MIN_RETRIEVAL_SCORE",
                    str(DEFAULT_MINIMUM_RETRIEVAL_SCORE),
                )
            )
        except ValueError as exc:
            raise ValueError("RAG retrieval configuration contains invalid numbers.") from exc
        settings = cls(
            index_dir=Path(index_dir),
            top_k=top_k,
            max_top_k=max_top_k,
            minimum_retrieval_score=minimum_score,
            provider=ExternalProviderSettings.from_environment(
                values,
                require_generation=True,
            ),
        )
        settings.validate()
        return settings

    # ADD 2026-08-21: Default/maximum top-k와 operational cosine threshold를 검증한다.
    def validate(self) -> None:
        """Reject retrieval settings that can create unbounded context."""
        if not 1 <= self.top_k <= self.max_top_k <= 100:
            raise ValueError("RAG top_k/max_top_k must satisfy 1 <= top_k <= max_top_k <= 100.")
        if (
            not math.isfinite(self.minimum_retrieval_score)
            or not -1 <= self.minimum_retrieval_score <= 1
        ):
            raise ValueError("RAG_MIN_RETRIEVAL_SCORE must be in [-1, 1].")
        self.provider.validate(require_generation=True)
