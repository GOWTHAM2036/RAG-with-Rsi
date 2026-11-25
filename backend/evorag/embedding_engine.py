"""Embedding validation, query formatting, batch admission and durable revision progress."""

import asyncio
import hashlib
import ipaddress
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Annotated, Literal, Protocol
from uuid import uuid4

import numpy as np
from pydantic import Field

from .chunking import EstimatedTokens
from .config import Settings
from .contracts import Contract
from .documents import digest
from .embedding_store import EmbeddingJob as EmbeddingJob
from .embedding_store import EmbeddingSpace, EmbeddingStore
from .errors import EvoRagError
from .inference_types import ModelInfo, ProviderError
from .repository import Repository

Role = Literal["query", "document"]
Text = Annotated[str, Field(min_length=1, max_length=65536, pattern=r"\S")]
QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)


class EmbeddingRequest(Contract):
    texts: Annotated[tuple[Text, ...], Field(min_length=1, max_length=128)]
    role: Role = "document"


class EmbeddingResponse(Contract):
    embeddings: tuple[tuple[float, ...], ...]
    model: str
    digest: str
    dimensions: int
    normalized: bool
    role: Role
    cache_hits: int
    truncated: Literal[False] = False


class EmbeddingRuntime(Protocol):
    async def models(self) -> tuple[ModelInfo, ...]: ...
    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]: ...


class Admission(Protocol):
    def reserve(
        self, model_name: str, capability: Literal["generation", "embedding"]
    ) -> AbstractAsyncContextManager[ModelInfo]: ...


def encode_vector(row: Sequence[float], width: int, normalize: bool) -> bytes:
    if len(row) != width or not row or any(type(x) not in (float, int) for x in row):
        raise ProviderError("invalid_response")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            vector = np.asarray(row, dtype="<f4")
    except (OverflowError, ValueError):
        raise ProviderError("invalid_response") from None
    if not np.isfinite(vector).all():
        raise ProviderError("invalid_response")
    if normalize:
        # Float64 accumulator avoids overflow for values near the Float32 range limit.
        precise = vector.astype(np.float64)
        norm = float(np.linalg.norm(precise))
        if norm == 0:
            raise ProviderError("invalid_response")
        vector = (precise / norm).astype("<f4")
    return vector.tobytes()


def _valid_cache(blob: bytes, space: EmbeddingSpace) -> bool:
    if len(blob) != space.dimensions * 4:
        return False
    values = np.frombuffer(blob, dtype="<f4")
    return bool(
        np.isfinite(values).all()
        and (
            not space.normalized
            or abs(float(np.linalg.norm(values.astype(np.float64))) - 1.0) <= 1e-5
        )
    )


class EmbeddingEngine:
    def __init__(
        self,
        settings: Settings,
        provider: EmbeddingRuntime,
        coordinator: Admission,
        repository: Repository,
    ):
        self.settings = settings
        self.provider = provider
        self.coordinator = coordinator
        self.repository = repository
        self.store = EmbeddingStore(repository, settings.embedding_cache.max_size_mb * 1024**2)
        self._hits = self._misses = 0

    def stats(self) -> dict[str, int]:
        return {**self.store.stats(), "hits": self._hits, "misses": self._misses}

    async def _space(self) -> EmbeddingSpace:
        model = self.settings.runtime.embedding_model
        canonical = model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"
        match = next(
            (m for m in await self.provider.models() if m.name in (canonical, model)), None
        )
        if match is None:
            raise ProviderError("model_not_installed")
        if not match.digest:
            raise ProviderError("invalid_response")
        return EmbeddingSpace(
            provider=self.settings.runtime.provider,
            endpoint=str(self.settings.runtime.base_url),
            model=match.name,
            digest=match.digest,
            dimensions=self.settings.embeddings.dimensions,
            normalized=self.settings.embeddings.normalize,
            formatting_version="qwen-query-v1"
            if match.name.rsplit("/", 1)[-1].split(":")[0] == "qwen3-embedding"
            else "plain-v1",
        )

    async def _batch(
        self, texts: tuple[str, ...], role: Role, space: EmbeddingSpace
    ) -> tuple[list[bytes], int]:
        formatted = tuple(
            QUERY_PREFIX + t
            if role == "query" and space.formatting_version == "qwen-query-v1"
            else t
            for t in texts
        )
        if any(
            EstimatedTokens().count(text) > self.settings.runtime.context_window
            for text in formatted
        ):
            raise ProviderError("context_overflow")
        keys = [
            digest(space.model_dump_json(), role, hashlib.sha256(t.encode()).hexdigest())
            for t in texts
        ]
        inputs = dict(zip(keys, formatted, strict=True))

        async def lookup() -> dict[str, bytes]:
            if not self.settings.embedding_cache.enabled:
                return {}
            cached = await asyncio.to_thread(self.store.cached, list(inputs), space.dimensions)
            return {k: b for k, b in cached.items() if _valid_cache(b, space)}

        hits = await lookup()
        if len(hits) < len(inputs):
            async with self.coordinator.reserve(
                self.settings.runtime.embedding_model, "embedding"
            ) as live:
                if live.digest != space.digest:
                    raise ProviderError("embedding_model_changed", 409)
                # Another previously admitted request may have populated these keys.
                hits = await lookup()
                missing = [k for k in inputs if k not in hits]
                if missing:
                    remote = await self.provider.embed(tuple(inputs[k] for k in missing))
                    if len(remote) != len(missing):
                        raise ProviderError("invalid_response")
                    result = {
                        k: encode_vector(row, space.dimensions, space.normalized)
                        for k, row in zip(missing, remote, strict=True)
                    }
                    # Tags are mutable. Never label returned vectors with a superseded digest.
                    if (await self._space()).digest != space.digest:
                        raise ProviderError("embedding_model_changed", 409)
                    if self.settings.embedding_cache.enabled:
                        await asyncio.to_thread(self.store.save_cache, result)
                    self._misses += len(missing)
                else:
                    result = {}
        else:
            result = {}
        hit_count = sum(k in hits for k in keys)
        self._hits += hit_count
        return [(hits | result)[k] for k in keys], hit_count

    async def embed_with_space(
        self, request: EmbeddingRequest
    ) -> tuple[EmbeddingResponse, EmbeddingSpace]:
        space = await self._space()
        size = self.settings.embeddings.batch_size
        vectors: list[tuple[float, ...]] = []
        hits = 0
        for start in range(0, len(request.texts), size):
            rows, new_hits = await self._batch(
                request.texts[start : start + size], request.role, space
            )
            vectors.extend(tuple(float(x) for x in np.frombuffer(row, dtype="<f4")) for row in rows)
            hits += new_hits
        return (
            EmbeddingResponse(
                model=space.model,
                digest=space.digest,
                role=request.role,
                dimensions=space.dimensions,
                normalized=space.normalized,
                embeddings=tuple(vectors),
                cache_hits=hits,
            ),
            space,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        response, _ = await self.embed_with_space(request)
        return response

    async def prepare_job(self, doc_id: str, version: int | None = None) -> EmbeddingJob:
        # A provider used for document chunks must be local, even for custom operator settings.
        host = (self.settings.runtime.base_url.host or "").strip("[]")
        if host != "localhost":
            try:
                local = ipaddress.ip_address(host).is_loopback
            except ValueError:
                local = False
            if not local:
                raise ProviderError("private_document_endpoint", 403)
        space = await self._space()
        return await asyncio.to_thread(self.store.prepare, doc_id, version, space)

    def status(self, job_id: str) -> EmbeddingJob:
        return self.store.status(job_id)

    def cancel(self, job_id: str) -> EmbeddingJob:
        return self.store.stop(job_id, None, "cancelled", "request_cancelled")

    async def run_job(self, doc_id: str, version: int | None = None) -> EmbeddingJob:
        job = await self.prepare_job(doc_id, version)
        owner = uuid4().hex
        lease = max(300.0, self.settings.provider.read_timeout_seconds * 4)
        if not await asyncio.to_thread(self.store.claim, job.job_id, owner, lease):
            return await asyncio.to_thread(self.status, job.job_id)
        try:
            while job.completed < job.total:
                chunks = await asyncio.to_thread(
                    self.repository.chunks,
                    doc_id,
                    job.version,
                    min(100, self.settings.embeddings.batch_size),
                    job.completed,
                )
                if not chunks:
                    raise ProviderError("invalid_revision")
                rows, _ = await self._batch(
                    tuple(chunk.text for chunk in chunks), "document", job.space
                )
                previous = job.completed
                job = await asyncio.to_thread(
                    self.store.commit_batch,
                    job.job_id,
                    owner,
                    previous,
                    lease,
                    [(c.chunk_id, b) for c, b in zip(chunks, rows, strict=True)],
                )
                if job.status != "running" or job.completed == previous:
                    break
            return job
        except asyncio.CancelledError:
            await asyncio.to_thread(
                self.store.stop, job.job_id, owner, "cancelled", "request_cancelled"
            )
            raise
        except EvoRagError as exc:
            code = exc.code if isinstance(exc, ProviderError) else "storage_or_resource_unavailable"
            await asyncio.to_thread(self.store.stop, job.job_id, owner, "failed", code)
            raise
