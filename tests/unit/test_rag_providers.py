"""Unit tests for production provider adapters and environment configuration."""

from __future__ import annotations

from typing import Any

import pytest

from services.rag import providers
from services.rag.config import ExternalProviderSettings, RagSettings
from services.rag.providers import (
    GenerationContext,
    InvalidProviderOutputError,
    OpenAICompatibleAnswerGenerator,
    OpenAICompatibleClientConfig,
    OpenAICompatibleEmbeddingProvider,
)


# ADD 2026-08-21: Production adapter test용 non-secret HTTP configuration을 생성한다.
def _client_config() -> OpenAICompatibleClientConfig:
    return OpenAICompatibleClientConfig(
        api_base_url="https://provider.example.test/v1",
        api_key="test-only-secret",
        timeout_seconds=2.0,
    )


# ADD 2026-08-21: Embedding adapter의 provider index ordering과 finite vectors를 검증한다.
def test_openai_compatible_embedding_adapter_parses_ordered_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[object, str, object]] = []

    def fake_post(config: object, path: str, payload: object) -> object:
        captured.append((config, path, payload))
        return {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }

    monkeypatch.setattr(providers, "_post_json", fake_post)
    provider = OpenAICompatibleEmbeddingProvider(
        model_name="embedding-model",
        client_config=_client_config(),
    )

    vectors = provider.embed_documents(["first", "second"])

    assert vectors == ((1.0, 0.0), (0.0, 1.0))
    assert captured[0][1] == "/embeddings"


# ADD 2026-08-21: Embedding adapter가 malformed/non-finite provider output을 거부하는지 검증한다.
@pytest.mark.parametrize(
    "payload",
    [
        {"data": [{"index": 0, "embedding": [float("nan"), 1.0]}]},
        {"data": [{"index": 1, "embedding": [1.0, 0.0]}]},
    ],
)
def test_embedding_adapter_rejects_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
) -> None:
    monkeypatch.setattr(providers, "_post_json", lambda *_args, **_kwargs: payload)
    provider = OpenAICompatibleEmbeddingProvider(
        model_name="embedding-model",
        client_config=_client_config(),
    )

    with pytest.raises(InvalidProviderOutputError):
        provider.embed_query("question")


# ADD 2026-08-21: Generation adapter가 strict JSON answer와 citation IDs를 parsing하는지 검증한다.
def test_openai_compatible_generation_adapter_parses_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        providers,
        "_post_json",
        lambda *_args, **_kwargs: {
            "choices": [
                {
                    "message": {
                        "content": '{"answer":"Use the procedure. [C1]","citation_ids":["C1"]}'
                    }
                }
            ]
        },
    )
    generator = OpenAICompatibleAnswerGenerator(
        model_name="generation-model",
        client_config=_client_config(),
    )
    context = GenerationContext("C1", "chunk", "Title", "Section", "demo.md", None, "Text")

    generated = generator.generate("Question", (context,))

    assert generated.answer == "Use the procedure. [C1]"
    assert generated.citation_ids == ("C1",)


# ADD 2026-08-21: Generation adapter가 malformed structured provider output을 거부한다.
def test_generation_adapter_rejects_malformed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        providers,
        "_post_json",
        lambda *_args, **_kwargs: {
            "choices": [{"message": {"content": '{"answer":"Unsupported"}'}}]
        },
    )
    generator = OpenAICompatibleAnswerGenerator(
        model_name="generation-model",
        client_config=_client_config(),
    )
    context = GenerationContext("C1", "chunk", "Title", "Section", "demo.md", None, "Text")

    with pytest.raises(InvalidProviderOutputError):
        generator.generate("Question", (context,))


# ADD 2026-08-21: RAG environment가 required secret/config와 bounded retrieval policy를 검증한다.
def test_rag_settings_load_without_exposing_provider_secret() -> None:
    values = {
        "RAG_INDEX_DIR": "artifacts/rag/manuals/demo",
        "RAG_EMBEDDING_PROVIDER": "openai-compatible",
        "RAG_EMBEDDING_MODEL": "embedding-model",
        "RAG_GENERATION_PROVIDER": "openai-compatible",
        "RAG_GENERATION_MODEL": "generation-model",
        "RAG_PROVIDER_API_KEY": "secret-value",
        "RAG_TOP_K": "3",
        "RAG_MAX_TOP_K": "5",
        "RAG_MIN_RETRIEVAL_SCORE": "0.3",
    }

    settings = RagSettings.from_environment(values)

    assert settings.top_k == 3
    assert settings.max_top_k == 5
    assert settings.provider.api_key == "secret-value"
    assert "secret-value" not in repr(settings.provider)
    assert "secret-value" not in repr(settings.provider.client_config())


# ADD 2026-08-21: Offline provider settings가 generation config 없이도 load 가능한지 검증한다.
def test_offline_embedding_settings_do_not_require_generation_model() -> None:
    settings = ExternalProviderSettings.from_environment(
        {
            "RAG_EMBEDDING_PROVIDER": "openai-compatible",
            "RAG_EMBEDDING_MODEL": "embedding-model",
            "RAG_PROVIDER_API_KEY": "secret-value",
        },
        require_generation=False,
    )

    assert settings.generation_model is None
