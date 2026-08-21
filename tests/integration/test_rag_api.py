"""Integration tests for the independent FastAPI RAG service contract."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml.rag.chunking import ManualChunk
from services.rag.app import create_app
from services.rag.config import ExternalProviderSettings, RagSettings
from services.rag.generation import RagRuntime
from services.rag.index import RagIndex
from services.rag.providers import AnswerGenerator
from services.rag.retrieval import ExactCosineRetriever
from tests.rag_helpers import (
    FailingGenerator,
    FixedQueryEmbeddingProvider,
    GroundedFakeGenerator,
    InvalidCitationGenerator,
)


# ADD 2026-08-21: API test용 provider/retrieval settings를 생성한다.
def _settings(tmp_path: Path, *, minimum_score: float = -1.0) -> RagSettings:
    return RagSettings(
        index_dir=tmp_path / "unused-index",
        top_k=1,
        max_top_k=2,
        minimum_retrieval_score=minimum_score,
        provider=ExternalProviderSettings(
            embedding_provider="openai-compatible",
            embedding_model="embedding-model",
            generation_provider="openai-compatible",
            generation_model="generation-model",
            api_base_url="https://provider.example.test/v1",
            api_key="test-only-secret",
            request_timeout_seconds=1.0,
        ),
    )


# ADD 2026-08-21: API test용 one-chunk immutable index를 생성한다.
def _index(index_id: str = "api-index") -> RagIndex:
    chunk = ManualChunk(
        chunk_id="chunk-api",
        document_id="doc-api",
        title="Demo SOP",
        source_path="demo.md",
        section="Camera Check",
        page=None,
        text="Clean the camera lens before inspection.",
        chunk_index=0,
        source_sha256="a" * 64,
    )
    embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    embeddings.setflags(write=False)
    return RagIndex(index_id, "test", "test-v1", 2, (chunk,), embeddings)


# ADD 2026-08-21: Fake providers를 결합한 API runtime을 생성한다.
def _runtime(
    generator: AnswerGenerator,
    *,
    minimum_score: float = -1.0,
    index_id: str = "api-index",
    query_vector: tuple[float, float] = (1.0, 0.0),
) -> RagRuntime:
    index = _index(index_id)
    retriever = ExactCosineRetriever(
        index=index,
        embedding_provider=FixedQueryEmbeddingProvider(query_vector),
        max_top_k=2,
        minimum_score=minimum_score,
    )
    return RagRuntime(index=index, retriever=retriever, answer_generator=generator)


# ADD 2026-08-21: Injected runtime을 반환하고 startup load count를 기록하는 loader를 생성한다.
def _runtime_loader(
    runtime: RagRuntime,
    calls: list[RagSettings] | None = None,
) -> Callable[[RagSettings], RagRuntime]:
    # ADD 2026-08-21: App lifespan이 runtime을 정확히 한 번 요청했는지 기록한다.
    def load(settings: RagSettings) -> RagRuntime:
        if calls is not None:
            calls.append(settings)
        return runtime

    return load


# ADD 2026-08-21: Health, network-free readiness와 valid query/citation/evidence 응답을 검증한다.
def test_rag_health_ready_and_valid_query(tmp_path: Path) -> None:
    calls: list[RagSettings] = []
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(_runtime(GroundedFakeGenerator()), calls),
    )

    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        response = client.post("/v1/rag/query", json={"question": "Check the camera"})

    assert health.json() == {"status": "ok"}
    assert ready.json()["index_id"] == "api-index"
    assert ready.json()["generation_model"] == "generation-model"
    assert response.status_code == 200
    assert response.json()["status"] == "answered"
    assert response.json()["citations"][0]["citation_id"] == "C1"
    assert response.json()["retrieval"][0]["chunk_id"] == "chunk-api"
    assert "text" not in response.json()["retrieval"][0]
    assert len(calls) == 1


# ADD 2026-08-21: Empty/oversized question과 excessive top-k가 stable invalid_request를 반환한다.
@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "x" * 2001},
        {"question": "valid", "top_k": 3},
    ],
)
def test_rag_query_validation_errors_are_stable(tmp_path: Path, payload: dict[str, object]) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(_runtime(GroundedFakeGenerator())),
    )
    with TestClient(app) as client:
        response = client.post("/v1/rag/query", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


# ADD 2026-08-21: Insufficient retrieval context가 200 abstention과 empty citation을 반환한다.
def test_rag_query_insufficient_context_abstains(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(
            _runtime(
                GroundedFakeGenerator(),
                minimum_score=0.5,
                query_vector=(0.0, 1.0),
            )
        ),
    )
    with TestClient(app) as client:
        response = client.post("/v1/rag/query", json={"question": "unknown procedure"})

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_context"
    assert response.json()["citations"] == []


# ADD 2026-08-21: Provider failure가 secret/detail 없는 safe provider_error로 변환되는지 검증한다.
def test_rag_provider_failure_is_safe(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(_runtime(FailingGenerator())),
    )
    with TestClient(app) as client:
        response = client.post("/v1/rag/query", json={"question": "camera"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "provider_error"
    assert "secret provider detail" not in response.text


# ADD 2026-08-21: Unknown citation이 safe invalid_provider_output response로 변환되는지 검증한다.
def test_rag_invalid_provider_output_is_safe(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(_runtime(InvalidCitationGenerator())),
    )
    with TestClient(app) as client:
        response = client.post("/v1/rag/query", json={"question": "camera"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_provider_output"
    assert "C99" not in response.text


# ADD 2026-08-21: Lifespan 밖 ready가 live process와 별개인 rag_not_ready를 반환한다.
def test_rag_ready_without_lifespan_is_not_ready(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(_runtime(GroundedFakeGenerator())),
    )
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "rag_not_ready"


# ADD 2026-08-21: Multiple app factory instances가 runtime/index state를 공유하지 않는지 검증한다.
def test_multiple_rag_apps_keep_runtime_state_isolated(tmp_path: Path) -> None:
    first = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(_runtime(GroundedFakeGenerator(), index_id="first")),
    )
    second = create_app(
        settings=_settings(tmp_path),
        runtime_loader=_runtime_loader(_runtime(GroundedFakeGenerator(), index_id="second")),
    )

    with TestClient(first) as first_client, TestClient(second) as second_client:
        assert first_client.get("/ready").json()["index_id"] == "first"
        assert second_client.get("/ready").json()["index_id"] == "second"
