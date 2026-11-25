"""Unit tests for DenseRetriever and exact nearest-neighbor search over ingested chunks."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pytest
from evorag.config import Chunking, Embeddings, Retrieval, Settings, Storage
from evorag.contracts import QueryRequest
from evorag.embedding_engine import QUERY_PREFIX, EmbeddingEngine
from evorag.inference_types import ModelInfo, ProviderError
from evorag.ingestion import IngestionService
from evorag.interfaces import Retriever
from evorag.resources import GIB, ResourceManager
from evorag.retrieval import DenseRetriever, DenseRetrieveRequest

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _VectorRuntime:
    def __init__(self, vectors: dict[str, tuple[float, float]]) -> None:
        self.model = ModelInfo(name="qwen3-embedding:0.6b", digest="d1", size_bytes=64)
        self.vectors = vectors
        self.model_calls = 0

    async def models(self) -> tuple[ModelInfo, ...]:
        self.model_calls += 1
        return (self.model,)

    @asynccontextmanager
    async def reserve(self, model_name: str, capability: Literal["generation", "embedding"]):  # type: ignore[no-untyped-def]
        yield self.model

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(self.vectors[t.removeprefix(QUERY_PREFIX)] for t in texts)


async def test_dense_retriever_exact_ranking_and_filters(tmp_path: Path) -> None:
    vectors = {
        "alpha": (1.0, 0.0),
        "bravo": (0.0, 1.0),
        "charl": (0.6, 0.8),
        "delta": (1.0, 0.0),  # v2 of doc1
        "find alpha": (1.0, 0.0),
        "find bravo": (0.0, 1.0),
    }
    settings = Settings(
        storage=Storage(database_path=tmp_path / "db.sqlite3"),
        embeddings=Embeddings(dimensions=2, batch_size=2, normalize=True),
        chunking=Chunking(chunk_size=5, overlap=0, strategy="fixed"),
        retrieval=Retrieval(dense_top_k=2, sparse_top_k=2, final_top_k=2),
    )
    ingestion = IngestionService(settings, ResourceManager(settings, lambda: 8 * GIB))
    doc1_v1 = ingestion.ingest("doc1.txt", b"alpha")
    doc1_v2 = ingestion.ingest("doc1.txt", b"delta")
    doc2_v1 = ingestion.ingest("doc2.txt", b"bravo")
    doc3_v1 = ingestion.ingest("doc3.txt", b"charl")

    runtime = _VectorRuntime(vectors)
    engine = EmbeddingEngine(settings, runtime, runtime, ingestion.repository)
    retriever: Retriever = DenseRetriever(settings, engine)
    assert isinstance(retriever, DenseRetriever)

    # Empty corpus returns no results.
    calls_before = runtime.model_calls
    empty = await retriever.retrieve(QueryRequest(query_id="q0", question="find alpha"))
    assert empty == ()
    # Search resolves model space once (plus once inside reserve() when cache misses).
    assert runtime.model_calls - calls_before == 2

    for doc_id, ver in [
        (doc1_v1.document_id, 1),
        (doc1_v2.document_id, 2),
        (doc2_v1.document_id, 1),
        (doc3_v1.document_id, 1),
    ]:
        await engine.run_job(doc_id, ver)

    # Default retrieve (latest_only=True, top_k=2): doc1_v2 ("delta"), doc3_v1 ("charl").
    res = await retriever.retrieve(QueryRequest(query_id="q1", question="find alpha"))
    assert [r.chunk.text for r in res] == ["delta", "charl"]
    assert [r.rank for r in res] == [1, 2]
    assert res[0].score == pytest.approx(1.0)
    assert res[1].score == pytest.approx(0.6)

    # latest_only=False includes doc1_v1 ("alpha") and doc1_v2 ("delta").
    all_vers = (
        await retriever.search(
            DenseRetrieveRequest(
                question="find alpha", top_k=4, latest_only=False, metric="euclidean"
            )
        )
    ).results
    assert [r.chunk.text for r in all_vers[:2]] == ["alpha", "delta"]
    assert all_vers[0].score == pytest.approx(0.0)
    assert all_vers[1].score == pytest.approx(0.0)

    # Filter by specific document_id and version=1.
    v1_only = (
        await retriever.search(
            DenseRetrieveRequest(
                question="find alpha",
                top_k=5,
                document_id=doc1_v1.document_id,
                version=1,
                metric="dot",
            )
        )
    ).results
    assert len(v1_only) == 1
    assert v1_only[0].chunk.text == "alpha"
    assert v1_only[0].score == pytest.approx(1.0)


async def test_dense_retriever_zero_query_norm_unnormalized(tmp_path: Path) -> None:
    vectors = {
        "alpha": (1.0, 0.0),
        "zero": (0.0, 0.0),
    }
    settings = Settings(
        storage=Storage(database_path=tmp_path / "db.sqlite3"),
        embeddings=Embeddings(dimensions=2, batch_size=2, normalize=False),
        chunking=Chunking(chunk_size=5, overlap=0, strategy="fixed"),
    )
    ingestion = IngestionService(settings, ResourceManager(settings, lambda: 8 * GIB))
    doc = ingestion.ingest("doc.txt", b"alpha")
    runtime = _VectorRuntime(vectors)
    engine = EmbeddingEngine(settings, runtime, runtime, ingestion.repository)
    await engine.run_job(doc.document_id, 1)
    retriever = DenseRetriever(settings, engine)

    with pytest.raises(ProviderError, match="invalid_response"):
        await retriever.search(DenseRetrieveRequest(question="zero", metric="cosine"))
