"""Independent FastAPI lifecycle and endpoints for the SOP/manual RAG assistant."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool

from services.rag.config import RagSettings
from services.rag.errors import RagApiError, install_rag_exception_handlers
from services.rag.generation import RagQueryResult, RagRuntime
from services.rag.index import load_rag_index
from services.rag.providers import (
    InvalidProviderOutputError,
    OpenAICompatibleAnswerGenerator,
    OpenAICompatibleEmbeddingProvider,
    ProviderError,
)
from services.rag.retrieval import ExactCosineRetriever
from services.rag.schemas import (
    CitationResponse,
    HealthResponse,
    RagQueryRequest,
    RagQueryResponse,
    ReadinessResponse,
    RetrievalEvidenceResponse,
)

type RuntimeLoader = Callable[[RagSettings], RagRuntime]


# ADD 2026-08-21: Validated settings에서 index/providers/retriever를 startup runtime으로 생성한다.
def load_rag_runtime(settings: RagSettings) -> RagRuntime:
    """Load the immutable index once and construct configured production adapters."""
    settings.validate()
    index = load_rag_index(settings.index_dir)
    client_config = settings.provider.client_config()
    embedding_provider = OpenAICompatibleEmbeddingProvider(
        model_name=settings.provider.embedding_model,
        client_config=client_config,
    )
    generation_model = settings.provider.generation_model
    if generation_model is None:
        raise ValueError("RAG generation model is required.")
    answer_generator = OpenAICompatibleAnswerGenerator(
        model_name=generation_model,
        client_config=client_config,
    )
    retriever = ExactCosineRetriever(
        index=index,
        embedding_provider=embedding_provider,
        max_top_k=settings.max_top_k,
        minimum_score=settings.minimum_retrieval_score,
    )
    return RagRuntime(index=index, retriever=retriever, answer_generator=answer_generator)


# ADD 2026-08-21: Service-local settings와 injectable runtime loader를 가진 FastAPI app을 생성한다.
def create_app(
    *,
    settings: RagSettings | None = None,
    runtime_loader: RuntimeLoader = load_rag_runtime,
) -> FastAPI:
    """Create an isolated RAG service that loads its index exactly once at startup."""

    # ADD 2026-08-21: Startup에서 settings/index/providers를 한 번 load하고 shutdown에서 해제한다.
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        active_settings = settings or RagSettings.from_environment()
        active_settings.validate()
        runtime = runtime_loader(active_settings)
        application.state.rag_settings = active_settings
        application.state.rag_runtime = runtime
        try:
            yield
        finally:
            application.state.rag_runtime = None
            application.state.rag_settings = None

    app = FastAPI(title="SmartFactory SOP/Manual RAG API", version="0.1.0", lifespan=lifespan)
    app.state.rag_settings = None
    app.state.rag_runtime = None
    install_rag_exception_handlers(app)

    # ADD 2026-08-21: RAG process liveness를 index/provider lifecycle과 독립적으로 반환한다.
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    # ADD 2026-08-21: Loaded index와 configured provider identity를 network call 없이 반환한다.
    @app.get("/ready", response_model=ReadinessResponse)
    async def ready(request: Request) -> ReadinessResponse:
        runtime, active_settings = _runtime_and_settings(request)
        index = runtime.index
        generation_provider = active_settings.provider.generation_provider
        generation_model = active_settings.provider.generation_model
        if generation_provider is None or generation_model is None:
            raise RagApiError(503, "rag_not_ready", "RAG runtime is not ready.")
        return ReadinessResponse(
            status="ready",
            index_id=index.index_id,
            embedding_provider=index.embedding_provider,
            embedding_model=index.embedding_model,
            generation_provider=generation_provider,
            generation_model=generation_model,
        )

    # ADD 2026-08-21: Bounded question을 retrieve/generate하고 structured evidence를 반환한다.
    @app.post("/v1/rag/query", response_model=RagQueryResponse)
    async def query_rag(request: Request, body: RagQueryRequest) -> RagQueryResponse:
        runtime, active_settings = _runtime_and_settings(request)
        top_k = active_settings.top_k if body.top_k is None else body.top_k
        if top_k > active_settings.max_top_k:
            raise RagApiError(422, "invalid_request", "Requested top_k exceeds the maximum.")
        try:
            result = await run_in_threadpool(runtime.query, body.question, top_k=top_k)
        except ProviderError as exc:
            raise RagApiError(502, "provider_error", "RAG provider request failed.") from exc
        except (InvalidProviderOutputError, ValueError) as exc:
            raise RagApiError(
                502,
                "invalid_provider_output",
                "RAG provider returned invalid output.",
            ) from exc
        return _query_response(result)

    return app


# ADD 2026-08-21: Request state에서 ready runtime/settings pair를 가져온다.
def _runtime_and_settings(request: Request) -> tuple[RagRuntime, RagSettings]:
    runtime = getattr(request.app.state, "rag_runtime", None)
    settings = getattr(request.app.state, "rag_settings", None)
    if not isinstance(runtime, RagRuntime) or not isinstance(settings, RagSettings):
        raise RagApiError(503, "rag_not_ready", "RAG runtime is not ready.")
    return runtime, settings


# ADD 2026-08-21: Domain query result를 Pydantic response contract로 변환한다.
def _query_response(result: RagQueryResult) -> RagQueryResponse:
    return RagQueryResponse(
        status=result.status,
        answer=result.answer,
        citations=[CitationResponse(**asdict(citation)) for citation in result.citations],
        retrieval=[RetrievalEvidenceResponse(**asdict(item)) for item in result.retrieval],
    )


app = create_app()
