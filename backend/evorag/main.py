from asyncio import Lock, Semaphore
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import version
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .analyzer import TextAnalyzer
from .config import Settings, load_settings
from .conversation import ConversationStore
from .coordinator import Coordinator
from .dataset_registry import DatasetRegistry
from .document_api import router as document_router
from .embedding_engine import EmbeddingEngine
from .errors import IngestionError, ResourceUnavailable
from .experiment_runner import ExperimentRunner
from .experiment_store import ExperimentStore
from .graph_api import router as graph_router
from .hybrid_retrieval import HybridRetriever
from .inference_api import InferenceServices, InferenceStatus, inference_status
from .inference_api import router as inference_router
from .inference_types import ProviderError
from .ingestion import IngestionService
from .inventory import ModelAvailability, inspect_models
from .observability import application_logger
from .ollama import Ollama, is_loopback
from .rag_service import RAGService
from .resources import ResourceManager
from .retrieval import DenseRetriever
from .rsi_engine import RSIEngine
from .rsi_interfaces import RSIFeedbackInterface
from .rsi_store import RSIStore
from .sparse_index import SparseIndex
from .sparse_retrieval import SparseRetriever
from .tracing import TraceStore


@dataclass(frozen=True)
class Services:
    settings: Settings
    resources: ResourceManager
    http: httpx.AsyncClient
    inventory_lock: Lock


def get_services(request: Request) -> Services | None:
    services = getattr(request.app.state, "services", None)
    return services if isinstance(services, Services) else None


class Readiness(BaseModel):
    application: Literal["ready", "initializing"]
    models: ModelAvailability | None = None


class DetailedReadiness(Readiness):
    inference: InferenceStatus


def create_app(
    settings: Settings | None = None, *, transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active = settings or load_settings()
        logger = application_logger()
        async with httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(
                active.provider.read_timeout_seconds,
                connect=active.provider.connect_timeout_seconds,
                pool=active.provider.connect_timeout_seconds,
            ),
            limits=httpx.Limits(
                max_connections=active.provider.max_connections,
                max_keepalive_connections=active.provider.max_connections,
            ),
        ) as client:
            resources = ResourceManager(active)
            app.state.services = Services(active, resources, client, Lock())
            analyzer = TextAnalyzer.from_config(active.sparse)
            ingestion = IngestionService(active, resources)
            sparse_index = SparseIndex(ingestion.repository, analyzer)
            ingestion.sparse_index = sparse_index
            app.state.ingestion = ingestion
            provider = Ollama(client, active)
            coordinator = Coordinator(active, provider)
            embeddings = EmbeddingEngine(
                active, provider, coordinator, app.state.ingestion.repository
            )
            dense_retriever = DenseRetriever(active, embeddings)
            sparse_retriever = SparseRetriever(active, sparse_index, analyzer)
            hybrid_retriever = HybridRetriever(active, dense_retriever, sparse_retriever, analyzer)
            trace_store = TraceStore(
                active.storage.database_path, active.tracing, active.storage.busy_timeout_ms
            )
            conversation_store = ConversationStore(active.conversation)
            rag_service = RAGService(
                active, hybrid_retriever, coordinator, trace_store, conversation_store
            )
            dataset_registry = DatasetRegistry()
            experiment_store = ExperimentStore(
                active.storage.database_path,
                active.evaluation.storage,
                active.storage.busy_timeout_ms,
            )
            experiment_runner = ExperimentRunner(
                active, dataset_registry, experiment_store, rag_service
            )
            rsi_store = RSIStore(
                active.storage.database_path, active.rsi.storage, active.storage.busy_timeout_ms
            )
            rsi_feedback = RSIFeedbackInterface(active, trace_store)
            rsi_engine = RSIEngine(active, rsi_store, rsi_feedback, experiment_runner)
            app.state.inference = InferenceServices(
                active,
                provider,
                coordinator,
                embeddings,
                dense_retriever,
                Semaphore(active.provider.max_connections + 1),
                sparse_index,
                hybrid_retriever,
                rag_service,
                dataset_registry,
                experiment_runner,
                rsi_engine,
            )
            logger.info("application.started")
            try:
                yield
            finally:
                app.state.services = None
                app.state.ingestion = None
                app.state.inference = None
                logger.info("application.stopped")

    app = FastAPI(title="EvoRag", version=version("evorag"), lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(document_router)
    app.include_router(inference_router)
    app.include_router(graph_router)

    @app.exception_handler(ProviderError)
    async def provider_error(request: Request, exc: ProviderError) -> JSONResponse:
        return JSONResponse({"error": exc.code}, status_code=exc.status)

    @app.exception_handler(IngestionError)
    async def ingestion_error(request: Request, exc: IngestionError) -> JSONResponse:
        return JSONResponse({"error": exc.code, "job_id": exc.job_id}, status_code=exc.status)

    @app.exception_handler(ResourceUnavailable)
    async def capacity_error(request: Request, exc: ResourceUnavailable) -> JSONResponse:
        return JSONResponse({"error": "resource_unavailable", "job_id": None}, status_code=503)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"error": "invalid_request", "job_id": None}, status_code=422)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", responses={503: {"model": Readiness}})
    async def ready(
        request: Request,
        response: Response,
        services: Annotated[Services | None, Depends(get_services)],
        inference: bool = False,
    ) -> DetailedReadiness | Readiness:
        if services is None:
            response.status_code = 503
            return Readiness(application="initializing")
        if not is_loopback(services.settings.runtime.base_url.host):
            models = ModelAvailability(inventory="unsupported")
        elif services.inventory_lock.locked():
            models = ModelAvailability(inventory="busy")
        else:
            async with services.inventory_lock:
                models = await inspect_models(services.http, services.settings.runtime)
        if inference:
            return DetailedReadiness(
                application="ready",
                models=models,
                inference=await inference_status(request.app.state.inference),
            )
        return Readiness(application="ready", models=models)

    return app


app = create_app()
