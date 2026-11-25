"""Offline provider fixtures for cache integrity, admission and durable revision progress."""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
from evorag.config import (
    Chunking,
    EmbeddingCache,
    Embeddings,
    ModelConfiguration,
    Settings,
    Storage,
)
from evorag.embedding_engine import QUERY_PREFIX, EmbeddingEngine, EmbeddingRequest, encode_vector
from evorag.embedding_store import EmbeddingStore
from evorag.inference_types import ModelInfo, ProviderError
from evorag.ingestion import IngestionService
from evorag.resources import GIB, ResourceManager

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Runtime:
    def __init__(self) -> None:
        self.model = ModelInfo(name="qwen3-embedding:0.6b", digest="digest1", size_bytes=100)
        self.calls: list[tuple[str, ...]] = []
        self.fail_at = 0
        self.absent = False
        self.rotate = False
        self.paused = False
        self.entered = asyncio.Event()
        self.resume = asyncio.Event()
        self.lock = asyncio.Lock()

    async def models(self) -> tuple[ModelInfo, ...]:
        return () if self.absent else (self.model,)

    @asynccontextmanager
    async def reserve(
        self, model_name: str, capability: Literal["generation", "embedding"]
    ) -> AsyncIterator[ModelInfo]:
        assert capability == "embedding"
        async with self.lock:
            yield self.model

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        self.calls.append(texts)
        if len(self.calls) == self.fail_at:
            raise ProviderError("read_timeout")
        if self.paused:
            self.entered.set()
            await self.resume.wait()
        if self.rotate:
            self.model = self.model.model_copy(update={"digest": "rotated"})
        return tuple((3.0, 4.0) for _ in texts)


def create(
    folder: Path, *, enabled: bool = True, settings: Settings | None = None
) -> tuple[EmbeddingEngine, Runtime, IngestionService]:
    settings = settings or Settings(
        storage=Storage(database_path=folder / "db.sqlite3"),
        embeddings=Embeddings(dimensions=2, batch_size=2),
        chunking=Chunking(chunk_size=1, overlap=0, strategy="fixed"),
        embedding_cache=EmbeddingCache(enabled=enabled),
    )
    ingestion = IngestionService(settings, ResourceManager(settings, lambda: 8 * GIB))
    runtime = Runtime()
    return EmbeddingEngine(settings, runtime, runtime, ingestion.repository), runtime, ingestion


async def test_batch_order_role_revision_and_cache(tmp_path: Path) -> None:
    engine, runtime, _ = create(tmp_path)
    request = EmbeddingRequest(texts=("a", "b", "a"))
    result = await engine.embed(request)
    assert runtime.calls == [("a", "b")]
    assert result.embeddings[2] == pytest.approx((0.6, 0.8))
    assert result.cache_hits == 1
    assert (await engine.embed(request)).cache_hits == 3
    assert (await engine.embed(EmbeddingRequest(texts=("a",), role="query"))).cache_hits == 0
    assert runtime.calls[-1] == (QUERY_PREFIX + "a",)
    runtime.model = runtime.model.model_copy(update={"digest": "changed"})
    assert (await engine.embed(request)).digest == "changed"
    assert len(runtime.calls) == 3
    assert engine.stats()["entries"] == 5
    assert engine.stats()["misses"] == 5


async def test_corruption_and_eviction(tmp_path: Path) -> None:
    engine, runtime, _ = create(tmp_path)
    engine.store.max_cache_bytes = 264
    req = EmbeddingRequest(texts=("x",))
    await engine.embed(req)
    for blob, checksum in [
        (b"short", "anything"),
        (b"a" * 8, "wrong"),
        (b"\0" * 8, hashlib.sha256(b"\0" * 8).hexdigest()),
    ]:
        with engine.store.transaction(write=True) as db:
            db.execute("UPDATE embedding_cache SET vector=?,checksum=?", (blob, checksum))
        assert (await engine.embed(req)).cache_hits == 0
    await engine.embed(EmbeddingRequest(texts=("y",)))
    assert (await engine.embed(req)).cache_hits == 0
    assert engine.stats()["entries"] == 1
    engine.store.max_cache_bytes = 1
    await engine.embed(EmbeddingRequest(texts=("z",)))
    assert engine.stats()["entries"] == 0
    assert len(runtime.calls) == 7


async def test_concurrent_duplicate_request_and_disabled_cache(tmp_path: Path) -> None:
    engine, runtime, _ = create(tmp_path)
    runtime.paused = True
    req = EmbeddingRequest(texts=("same", "same"))
    first = asyncio.create_task(engine.embed(req))
    await asyncio.wait_for(runtime.entered.wait(), 5)
    second = asyncio.create_task(engine.embed(req))
    runtime.resume.set()
    values = await asyncio.gather(first, second)
    assert sum(item.cache_hits for item in values) == 2
    assert runtime.calls == [("same",)]
    engine, runtime, _ = create(tmp_path / "disabled", enabled=False)
    await engine.embed(req)
    await engine.embed(req)
    assert len(runtime.calls) == 2
    assert engine.stats()["entries"] == 0


@pytest.mark.parametrize(
    "row",
    [
        (),
        (3.0,),
        (3.0, 4.0, 5.0),
        (True, 2.0),
        ("1", 2.0),
        (float("inf"), 0.0),
        (float("nan"), 1.0),
        (0.0, 0.0),
        (1e99, 2.0),
        (10**1000, 2.0),
    ],
)
def test_bad_vector(row: tuple[float, ...]) -> None:
    with pytest.raises(ProviderError, match="invalid_response"):
        encode_vector(row, 2, True)
    assert encode_vector((0.0, 0.0), 2, False) == b"\0" * 8


async def test_model_errors(tmp_path: Path) -> None:
    engine, runtime, _ = create(tmp_path)
    req = EmbeddingRequest(texts=("ok",))
    runtime.absent = True
    with pytest.raises(ProviderError, match="model_not_installed"):
        await engine.embed(req)
    runtime.absent = False
    with pytest.raises(ProviderError, match="context_overflow"):
        await engine.embed(EmbeddingRequest(texts=("b" * 20000,)))
    runtime.rotate = True
    with pytest.raises(ProviderError, match="embedding_model_changed"):
        await engine.embed(req)
    with patch.object(runtime, "embed", return_value=()):
        with pytest.raises(ProviderError, match="invalid_response"):
            await engine.embed(req)
    runtime.model = runtime.model.model_copy(update={"digest": ""})
    with pytest.raises(ProviderError, match="invalid_response"):
        await engine.embed(req)


async def test_failure_resume_exact_revision(tmp_path: Path) -> None:
    engine, runtime, ingestion = create(tmp_path)
    revision = ingestion.ingest("x.txt", b"abcdefghijklmnopqrst")
    ingestion.ingest("x.txt", b"newer")
    job = await engine.prepare_job(revision.document_id, 1)
    runtime.fail_at = 2
    with pytest.raises(ProviderError, match="read_timeout"):
        await engine.run_job(revision.document_id, 1)
    state = engine.status(job.job_id)
    assert (state.status, state.completed, state.total, state.error) == (
        "failed",
        2,
        5,
        "read_timeout",
    )
    restarted, new_runtime, _ = create(tmp_path)
    result = await restarted.run_job(revision.document_id, 1)
    assert (result.job_id, result.status, result.completed) == (job.job_id, "complete", 5)
    assert new_runtime.calls == [("ijkl", "mnop"), ("qrst",)]
    assert await restarted.run_job(revision.document_id, 1) == result
    assert restarted.cancel(job.job_id) == result
    with restarted.store.transaction() as db:
        ids = {row[0] for row in db.execute("SELECT chunk_id FROM chunk_embeddings")}
    chunks = ingestion.repository.chunks(revision.document_id, 1, 10, 0)
    assert ids == {c.chunk_id for c in chunks}
    with pytest.raises(ProviderError, match="document_revision_not_found"):
        await engine.prepare_job("missing", 1)
    with pytest.raises(ProviderError, match="embedding_job_not_found"):
        engine.status("missing")


async def test_cancel_active_then_resume(tmp_path: Path) -> None:
    engine, runtime, ingestion = create(tmp_path)
    doc = ingestion.ingest("x.txt", b"word")
    job = await engine.prepare_job(doc.document_id)
    runtime.paused = True
    task = asyncio.create_task(engine.run_job(doc.document_id))
    await asyncio.wait_for(runtime.entered.wait(), 5)
    assert (await engine.run_job(doc.document_id)).status == "running"
    assert engine.cancel(job.job_id).status == "cancelled"
    runtime.resume.set()
    stopped = await task
    assert stopped.completed == 0 and stopped.status == "cancelled"
    assert (await engine.run_job(doc.document_id)).status == "complete"


async def test_abrupt_cancel_and_expired_lease(tmp_path: Path) -> None:
    engine, runtime, ingestion = create(tmp_path)
    doc = ingestion.ingest("x.txt", b"word")
    job = await engine.prepare_job(doc.document_id)
    runtime.paused = True
    task = asyncio.create_task(engine.run_job(doc.document_id))
    await asyncio.wait_for(runtime.entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.status(job.job_id).status == "cancelled"
    assert engine.store.claim(job.job_id, "old", -1)
    assert engine.store.claim(job.job_id, "new", 300)
    assert not engine.store.claim(job.job_id, "late", 300)


async def test_schema_and_local_boundary(tmp_path: Path) -> None:
    engine, _, _ = create(tmp_path)
    assert engine.stats()["entries"] == 0
    with engine.store.transaction(write=True) as db:
        db.execute("UPDATE embedding_schema SET version=99")
    with pytest.raises(ProviderError, match="unsupported_embedding_schema"):
        EmbeddingStore(engine.repository, 1000).stats()
    for host in ["example.invalid", "192.0.2.1"]:
        settings = Settings(
            runtime=ModelConfiguration.model_validate({"base_url": "http://" + host})
        )
        other, _, _ = create(tmp_path, settings=settings)
        with pytest.raises(ProviderError, match="private_document_endpoint"):
            await other.prepare_job("missing", 1)
