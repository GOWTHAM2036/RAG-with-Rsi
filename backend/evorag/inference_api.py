"""Local inference endpoints with bounded JSON bodies and explicit NDJSON stream endings."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Annotated, Any, TypeVar

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from starlette.requests import ClientDisconnect
from starlette.types import Message, Receive, Scope, Send

from .config import Settings
from .coordinator import Coordinator
from .embedding_engine import EmbeddingEngine, EmbeddingJob, EmbeddingRequest, EmbeddingResponse
from .errors import IngestionError
from .hybrid_retrieval import HybridRetriever, RetrieveRequest, RetrieveResponse
from .inference_types import (
    GenerationEvent,
    GenerationRequest,
    GenerationResponse,
    ModelInfo,
    ProviderError,
    ProviderHealth,
)
from .ollama import Ollama
from .retrieval import DenseRetriever
from .sparse_index import SparseIndex, SparseIndexJob, SparseIndexStatus


@dataclass(frozen=True)
class InferenceServices:
    settings: Settings
    provider: Ollama
    coordinator: Coordinator
    embeddings: EmbeddingEngine
    retriever: DenseRetriever
    readers: asyncio.Semaphore
    sparse_index: SparseIndex
    hybrid_retriever: HybridRetriever


async def inference_services(request: Request) -> InferenceServices:
    services = getattr(request.app.state, "inference", None)
    if not isinstance(services, InferenceServices):
        raise IngestionError("inference_unavailable", 503)
    return services


Engine = Annotated[InferenceServices, Depends(inference_services)]
Revision = Annotated[int | None, Query(ge=1, le=2_147_483_647)]


class JsonRoute(APIRoute):
    """Count ASGI frames before FastAPI buffers and decodes a typed JSON request."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handle = super().get_route_handler()

        async def bounded(request: Request) -> Response:
            if request.method != "POST" or self.body_field is None:
                return await handle(request)
            services = await inference_services(request)
            maximum = services.settings.provider.max_request_bytes
            if request.headers.get("content-encoding", "identity").lower() != "identity":
                raise IngestionError("unsupported_content_encoding", 415)
            length = request.headers.get("content-length")
            if length is not None:
                if len(length) > 16 or not length.isascii() or not length.isdecimal():
                    raise IngestionError("invalid_content_length", 400)
                if int(length) > maximum:
                    raise IngestionError("request_too_large", 413)
            if services.readers.locked():
                raise IngestionError("request_capacity_exhausted", 503)
            async with services.readers:
                data = bytearray()
                try:
                    async with asyncio.timeout(30):
                        async for block in request.stream():
                            if len(block) > maximum - len(data):
                                raise IngestionError("request_too_large", 413)
                            data.extend(block)
                except ClientDisconnect:
                    raise IngestionError("upload_interrupted", 400) from None
                except TimeoutError:
                    raise IngestionError("upload_timeout", 408) from None
                if length is not None and len(data) != int(length):
                    raise IngestionError("content_length_mismatch", 400)
                sent = False

                async def receive() -> Message:
                    nonlocal sent
                    if sent:
                        return await request.receive()
                    sent = True
                    return {"type": "http.request", "body": bytes(data), "more_body": False}

                return await handle(Request(request.scope, receive))

        return bounded


router = APIRouter(prefix="/api/v1", route_class=JsonRoute)


class InferenceStatus(ProviderHealth):
    provider: str
    resident: tuple[ModelInfo, ...] | None
    generation_admission: dict[str, object] | None
    embedding_admission: dict[str, object] | None
    error: str | None = None


async def inference_status(engine: InferenceServices) -> InferenceStatus:
    runtime = engine.settings.runtime
    generation_installed = embedding_installed = None
    resident = None
    generation_admission = embedding_admission = None
    error = None
    connected = False
    try:
        names = {model.name.removesuffix(":latest") for model in await engine.provider.models()}
        connected = True
        generation_installed = runtime.generation_model.removesuffix(":latest") in names
        embedding_installed = runtime.embedding_model.removesuffix(":latest") in names
        resident = await engine.provider.resident()
        generation_admission = await engine.coordinator.admission(runtime.generation_model)
        embedding_admission = await engine.coordinator.admission(runtime.embedding_model)
    except ProviderError as exc:
        error = exc.code
    return InferenceStatus(
        provider=runtime.provider,
        reachable=connected,
        generation_installed=generation_installed,
        embedding_installed=embedding_installed,
        resident=resident,
        generation_admission=generation_admission,
        embedding_admission=embedding_admission,
        error=error,
    )


@router.get("/models")
async def models(engine: Engine) -> tuple[ModelInfo, ...]:
    return await engine.provider.models()


@router.get("/models/status")
async def status(engine: Engine) -> InferenceStatus:
    return await inference_status(engine)


T = TypeVar("T")


async def _while_connected(request: Request, operation: Awaitable[T]) -> T:
    """Cancel inference if the client disconnects while waiting for the first response."""

    async def disconnected() -> None:
        while (await request.receive())["type"] != "http.disconnect":
            continue

    worker = asyncio.ensure_future(operation)
    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((worker, watcher), return_when=asyncio.FIRST_COMPLETED)
        if worker in done:
            return worker.result()
        raise ProviderError("request_cancelled")
    finally:
        worker.cancel()
        watcher.cancel()
        await asyncio.gather(worker, watcher, return_exceptions=True)


@router.post("/generate")
async def generate(request: Request, body: GenerationRequest, engine: Engine) -> GenerationResponse:
    return await _while_connected(request, engine.coordinator.generate(body))


class NdjsonResponse(StreamingResponse):
    media_type = "application/x-ndjson"

    def __init__(self, first: GenerationEvent, events: AsyncGenerator[GenerationEvent, None]):
        self.events = events
        super().__init__(
            self.lines(first), media_type=self.media_type, headers={"Cache-Control": "no-store"}
        )

    async def lines(self, first: GenerationEvent) -> AsyncIterator[bytes]:
        yield first.model_dump_json().encode() + b"\n"
        try:
            async for item in self.events:
                yield item.model_dump_json().encode() + b"\n"
        except ProviderError as exc:
            yield json.dumps({"error": exc.code}).encode() + b"\n"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.events.aclose()


@router.post("/generate/stream", response_class=NdjsonResponse, status_code=200)
async def stream(request: Request, body: GenerationRequest, engine: Engine) -> Response:
    events = engine.coordinator.stream(body)
    try:
        first = await _while_connected(request, anext(events))
    except StopAsyncIteration:
        raise ProviderError("invalid_response", 502) from None
    return NdjsonResponse(first, events)


@router.post("/embeddings")
async def embed(body: EmbeddingRequest, engine: Engine) -> EmbeddingResponse:
    return await engine.embeddings.embed(body)


@router.get("/embeddings/cache/stats")
def cache_stats(engine: Engine) -> dict[str, int]:
    return engine.embeddings.stats()


@router.post("/documents/{document_id}/embed")
async def document_embedding(
    document_id: str, engine: Engine, version: Revision = None, prepare_only: bool = False
) -> EmbeddingJob:
    if prepare_only:
        return await engine.embeddings.prepare_job(document_id, version)
    return await engine.embeddings.run_job(document_id, version)


@router.get("/embedding-jobs/{job_id}")
def embedding_job(job_id: str, engine: Engine) -> EmbeddingJob:
    return engine.embeddings.status(job_id)


@router.post("/embedding-jobs/{job_id}/cancel")
def cancel(job_id: str, engine: Engine) -> EmbeddingJob:
    return engine.embeddings.cancel(job_id)


@router.post("/retrieve")
async def retrieve(body: RetrieveRequest, engine: Engine) -> RetrieveResponse:
    return await engine.hybrid_retriever.search(body)


@router.get("/index/status")
def index_status(engine: Engine) -> SparseIndexStatus:
    return engine.sparse_index.status()


@router.post("/index/rebuild")
async def index_rebuild(engine: Engine, force: bool = True) -> SparseIndexJob:
    return await asyncio.to_thread(engine.sparse_index.rebuild_all, force=force)


@router.get("/index/jobs/{job_id}")
def index_job(job_id: str, engine: Engine) -> SparseIndexJob:
    return engine.sparse_index.job_status(job_id)
