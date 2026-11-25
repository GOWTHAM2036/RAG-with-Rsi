"""Unit tests for HybridRetriever, metadata filtering, fallback and fail-closed integrity."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pytest
from evorag.analyzer import TextAnalyzer
from evorag.config import Chunking, Embeddings, Retrieval, Settings, Storage
from evorag.contracts import QueryRequest
from evorag.embedding_engine import QUERY_PREFIX, EmbeddingEngine
from evorag.filtering import MetadataFilter
from evorag.hybrid_retrieval import HybridRetriever, RetrieveRequest
from evorag.inference_types import ModelInfo, ProviderError
from evorag.ingestion import IngestionService
from evorag.resources import GIB, ResourceManager
from evorag.retrieval import DenseRetriever
from evorag.sparse_index import SparseIndex
from evorag.sparse_retrieval import SparseRetriever

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _StubRuntime:
    def __init__(self, vectors: dict[str, tuple[float, float]], *, fail_code: str | None = None):
        self.model = ModelInfo(name="qwen3-embedding:0.6b", digest="d1", size_bytes=64)
        self.vectors = vectors
        self.fail_code = fail_code

    async def models(self) -> tuple[ModelInfo, ...]:
        if self.fail_code is not None:
            raise ProviderError(self.fail_code)
        return (self.model,)

    @asynccontextmanager
    async def reserve(self, model_name: str, capability: Literal["generation", "embedding"]):  # type: ignore[no-untyped-def]
        yield self.model

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        if self.fail_code is not None:
            raise ProviderError(self.fail_code)
        return tuple(self.vectors.get(t.removeprefix(QUERY_PREFIX), (0.5, 0.5)) for t in texts)


async def test_hybrid_modes_filters_and_explanations(tmp_path: Path) -> None:
    vectors = {
        "HTTP 404 error guide": (1.0, 0.0),
        "RTX 3050 hardware spec": (0.0, 1.0),
        "HTTP 404": (1.0, 0.0),
        "RTX 3050": (0.0, 1.0),
    }
    settings = Settings(
        storage=Storage(database_path=tmp_path / "hybrid.sqlite3"),
        embeddings=Embeddings(dimensions=2, batch_size=2, normalize=True),
        chunking=Chunking(chunk_size=128, overlap=0, strategy="fixed"),
        retrieval=Retrieval(dense_top_k=5, sparse_top_k=5, final_top_k=5),
    )
    ingestion = IngestionService(settings, ResourceManager(settings, lambda: 8 * GIB))
    analyzer = TextAnalyzer.from_config(settings.sparse)
    sparse_index = SparseIndex(ingestion.repository, analyzer)
    ingestion.sparse_index = sparse_index

    d1 = ingestion.ingest("errors.md", b"HTTP 404 error guide")
    d2 = ingestion.ingest("hardware.txt", b"RTX 3050 hardware spec")

    runtime = _StubRuntime(vectors)
    engine = EmbeddingEngine(settings, runtime, runtime, ingestion.repository)
    await engine.run_job(d1.document_id, 1)
    await engine.run_job(d2.document_id, 1)

    dense = DenseRetriever(settings, engine)
    sparse = SparseRetriever(settings, sparse_index, analyzer)
    hybrid = HybridRetriever(settings, dense, sparse, analyzer)

    # Hybrid mode with explanations.
    resp = await hybrid.search(
        RetrieveRequest(query="HTTP 404", mode="hybrid", include_explanations=True)
    )
    assert resp.mode == "hybrid" and resp.effective_mode == "hybrid" and not resp.degraded
    assert resp.results[0].chunk.document_id == d1.document_id
    assert resp.explanations is not None
    exp0 = resp.explanations[0]
    assert exp0.dense_rank == 1 and exp0.sparse_rank == 1
    assert exp0.rrf_score is not None and exp0.bm25_score is not None
    assert "http" in exp0.matched_terms and "404" in exp0.matched_terms
    assert str(tmp_path) not in resp.model_dump_json()

    # Metadata filter by file_type=".txt" finds only hardware.txt even if query mentions HTTP.
    filtered = await hybrid.search(
        RetrieveRequest(
            query="RTX 3050",
            mode="hybrid",
            filters=MetadataFilter(file_type=".txt"),
        )
    )
    assert len(filtered.results) == 1
    assert filtered.results[0].chunk.document_id == d2.document_id

    # Retriever protocol compatibility.
    proto_res = await hybrid.retrieve(QueryRequest(query_id="q1", question="RTX 3050"))
    assert proto_res[0].chunk.document_id == d2.document_id
    sparse_proto = await sparse.retrieve(QueryRequest(query_id="q2", question="HTTP 404"))
    assert sparse_proto[0].chunk.document_id == d1.document_id
    # Punctuation-only query returns empty.
    assert await sparse.retrieve(QueryRequest(query_id="q3", question="...")) == ()


async def test_sparse_fallback_and_fail_closed_corruption(tmp_path: Path) -> None:
    settings = Settings(
        storage=Storage(database_path=tmp_path / "fallback.sqlite3"),
        embeddings=Embeddings(dimensions=2, batch_size=2, normalize=True),
        chunking=Chunking(chunk_size=128, overlap=0, strategy="fixed"),
        retrieval=Retrieval(allow_sparse_fallback=True),
    )
    ingestion = IngestionService(settings, ResourceManager(settings, lambda: 8 * GIB))
    analyzer = TextAnalyzer.from_config(settings.sparse)
    sparse_index = SparseIndex(ingestion.repository, analyzer)
    ingestion.sparse_index = sparse_index

    doc = ingestion.ingest("notes.md", b"EVORAG_RUNTIME__BASE_URL configuration")

    # Provider unavailable -> degrades cleanly to sparse-only when allow_sparse_fallback=True.
    down_runtime = _StubRuntime({}, fail_code="provider_unavailable")
    down_engine = EmbeddingEngine(settings, down_runtime, down_runtime, ingestion.repository)
    hybrid_down = HybridRetriever(
        settings,
        DenseRetriever(settings, down_engine),
        SparseRetriever(settings, sparse_index, analyzer),
        analyzer,
    )
    degraded = await hybrid_down.search(
        RetrieveRequest(query="EVORAG_RUNTIME__BASE_URL", mode="hybrid", include_explanations=True)
    )
    assert degraded.degraded is True
    assert degraded.effective_mode == "sparse"
    assert degraded.degradation_reason == "provider_unavailable"
    assert len(degraded.results) == 1
    assert degraded.explanations is not None
    assert degraded.explanations[0].degraded is True
    assert degraded.explanations[0].version == 1

    # Sparse mode also populates document revision in explanations.
    sparse_only = await hybrid_down.search(
        RetrieveRequest(query="EVORAG_RUNTIME__BASE_URL", mode="sparse", include_explanations=True)
    )
    assert sparse_only.explanations is not None
    assert sparse_only.explanations[0].version == 1

    # Vector corruption MUST fail closed (never degrade to sparse).
    ok_runtime = _StubRuntime(
        {"EVORAG_RUNTIME__BASE_URL configuration": (1.0, 0.0), "q": (1.0, 0.0)}
    )
    ok_engine = EmbeddingEngine(settings, ok_runtime, ok_runtime, ingestion.repository)
    await ok_engine.run_job(doc.document_id, 1)

    def _remove_vectors() -> None:
        for vec_file in tmp_path.glob("*.vectors.f32"):
            vec_file.unlink()

    await asyncio.to_thread(_remove_vectors)
    hybrid_corrupt = HybridRetriever(
        settings,
        DenseRetriever(settings, ok_engine),
        SparseRetriever(settings, sparse_index, analyzer),
        analyzer,
    )
    with pytest.raises(ProviderError, match="vector_store_corrupted"):
        await hybrid_corrupt.search(RetrieveRequest(query="q", mode="hybrid"))


async def test_hybrid_top_k_expands_candidate_budget(tmp_path: Path) -> None:
    settings = Settings(
        storage=Storage(database_path=tmp_path / "topk.sqlite3"),
        embeddings=Embeddings(dimensions=2, batch_size=8, normalize=True),
        chunking=Chunking(chunk_size=128, overlap=0, strategy="fixed"),
        retrieval=Retrieval(dense_top_k=2, sparse_top_k=2, final_top_k=2),
    )
    ingestion = IngestionService(settings, ResourceManager(settings, lambda: 8 * GIB))
    analyzer = TextAnalyzer.from_config(settings.sparse)
    sparse_index = SparseIndex(ingestion.repository, analyzer)
    ingestion.sparse_index = sparse_index

    vectors: dict[str, tuple[float, float]] = {"alpha": (1.0, 0.0)}
    for i in range(4):
        text = f"alpha beta item {i}"
        vectors[text] = (1.0, 0.0)
        doc = ingestion.ingest(f"d{i}.txt", text.encode())
        await EmbeddingEngine(
            settings, _StubRuntime(vectors), _StubRuntime(vectors), ingestion.repository
        ).run_job(doc.document_id, 1)

    runtime = _StubRuntime(vectors)
    engine = EmbeddingEngine(settings, runtime, runtime, ingestion.repository)
    hybrid = HybridRetriever(
        settings,
        DenseRetriever(settings, engine),
        SparseRetriever(settings, sparse_index, analyzer),
        analyzer,
    )
    resp = await hybrid.search(RetrieveRequest(query="alpha", mode="hybrid", top_k=4))
    assert len(resp.results) == 4
