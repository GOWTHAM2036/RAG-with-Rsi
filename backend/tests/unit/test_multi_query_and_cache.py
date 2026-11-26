"""Unit tests for multi-query RRF, MultiQueryRetriever, RetrievalCache and consolidation."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock

import httpx
import pytest
from evorag.analyzer import TextAnalyzer
from evorag.config import RetrievalCacheSettings, Settings, Storage
from evorag.contracts import Chunk, RetrievalResult
from evorag.coordinator import Coordinator
from evorag.embedding_engine import EmbeddingEngine
from evorag.errors import IngestionError
from evorag.evidence_consolidation import consolidate_evidence
from evorag.filtering import MetadataFilter
from evorag.hybrid_retrieval import HybridRetriever, RetrieveResponse
from evorag.ingestion import IngestionService
from evorag.multi_query_retrieval import MultiQueryRetriever, multi_query_rrf
from evorag.ollama import Ollama
from evorag.query_planner import AdaptiveQueryPlanner
from evorag.query_types import QueryVariant, RetrievalCacheEntry
from evorag.rag_types import RAGRequest
from evorag.resources import ResourceManager
from evorag.retrieval import DenseRetriever
from evorag.retrieval_cache import RetrievalCache, build_cache_key, build_config_fingerprint
from evorag.sparse_index import SparseIndex
from evorag.sparse_retrieval import SparseRetriever


def _make_chunk(cid: str, doc_id: str, text: str, start: int = 0) -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id=doc_id,
        text=text,
        start=start,
        end=start + len(text),
    )


def _make_resp(
    query_id: str,
    chunks: list[Chunk],
    *,
    degraded: bool = False,
    model: str | None = "qwen3-embedding:0.6b",
    digest: str | None = "d1",
) -> RetrieveResponse:
    mode: Literal["sparse", "hybrid"] = "sparse" if degraded else "hybrid"
    results = tuple(
        RetrievalResult(chunk=c, score=1.0 / r, rank=r) for r, c in enumerate(chunks, start=1)
    )
    return RetrieveResponse(
        query_id=query_id,
        mode="hybrid",
        effective_mode=mode,
        degraded=degraded,
        degradation_reason="provider_unavailable" if degraded else None,
        model=model,
        digest=digest,
        results=results,
        explanations=None,
    )


def test_multi_query_rrf_weights_and_tie_breaking() -> None:
    settings = Settings()
    c1 = _make_chunk("c1", "d1", "Vector storage uses append-only float32 files.")
    c2 = _make_chunk("c2", "d2", "SQLite WAL uses write-ahead logging.")
    c3 = _make_chunk("c3", "d3", "Tie candidate alpha.")
    c4 = _make_chunk("c4", "d4", "Tie candidate beta.")

    v_orig = QueryVariant(variant_id="q_orig", kind="original", text="vector storage", weight=1.0)
    v_sub1 = QueryVariant(
        variant_id="sq_1", kind="subquery", text="sq1", weight=0.7, source_subquery_id="sq_1"
    )
    v_sub2 = QueryVariant(
        variant_id="sq_2", kind="subquery", text="sq2", weight=0.7, source_subquery_id="sq_2"
    )
    v_sub3 = QueryVariant(
        variant_id="sq_3", kind="subquery", text="sq3", weight=0.7, source_subquery_id="sq_3"
    )

    # Normalized variant weights (cap 1.5) prevent 3 subqueries (sum=2.1) from overwhelming c1.
    cands, details = multi_query_rrf(
        [
            (v_orig, _make_resp("0", [c1, c3])),
            (v_sub1, _make_resp("1", [c2, c4])),
            (v_sub2, _make_resp("2", [c2])),
            (v_sub3, _make_resp("3", [c2])),
        ],
        mq_settings=settings.multi_query,
        max_candidates=10,
    )
    assert len(cands) == 4
    by_id = {d.chunk_id: d for d in details}
    assert by_id["c1"].original_rank == 1
    assert by_id["c2"].original_rank is None
    assert "q_orig" in by_id["c1"].contributing_query_ids
    assert by_id["c3"].contributing_query_ids == ("q_orig",)

    # Deterministic tie-breaking: identical single-variant rank -> sorted by chunk_id ascending.
    tie_cands, _ = multi_query_rrf(
        [(v_orig, _make_resp("0", [c4])), (v_sub1, _make_resp("1", [c3]))],
        mq_settings=settings.multi_query,
        max_candidates=10,
    )
    # c4 has higher weight (1.0 vs 0.7), so c4 first; test exact tie with equal weights:
    v_eq1 = QueryVariant(variant_id="sq_a", kind="subquery", text="a", weight=0.7)
    v_eq2 = QueryVariant(variant_id="sq_b", kind="subquery", text="b", weight=0.7)
    eq_cands, _ = multi_query_rrf(
        [(v_eq1, _make_resp("a", [c4])), (v_eq2, _make_resp("b", [c3]))],
        mq_settings=settings.multi_query,
        max_candidates=10,
    )
    assert [c.result.chunk.chunk_id for c in eq_cands] == ["c3", "c4"]


def test_retrieval_cache_isolation_invalidation_and_ttl() -> None:
    cache = RetrievalCache(RetrievalCacheSettings(enabled=True, max_entries=2, ttl_seconds=60.0))
    c1 = _make_chunk("c1", "d1", "Hello world")
    resp = _make_resp("q1", [c1])
    degraded_resp = _make_resp("q1", [c1], degraded=True)
    cfg_fp = build_config_fingerprint(Settings())

    k_hybrid = build_cache_key(
        normalized_query="hello",
        retrieval_mode="hybrid",
        config_fingerprint=cfg_fp,
        analyzer_version="v1",
        index_generation=1,
        embedding_space_id="m:1024",
        filters=MetadataFilter(),
        document_revision_token="1:1:v1",
        reranking_enabled=False,
    )
    k_sparse = build_cache_key(
        normalized_query="hello",
        retrieval_mode="sparse",
        config_fingerprint=cfg_fp,
        analyzer_version="v1",
        index_generation=1,
        embedding_space_id=None,
        filters=MetadataFilter(),
        document_revision_token="1:1:v1",
        reranking_enabled=False,
    )
    k_rev2 = build_cache_key(
        normalized_query="hello",
        retrieval_mode="hybrid",
        config_fingerprint=cfg_fp,
        analyzer_version="v1",
        index_generation=2,
        embedding_space_id="m:1024",
        filters=MetadataFilter(),
        document_revision_token="2:2:v2",
        reranking_enabled=False,
    )
    assert k_hybrid != k_sparse
    assert k_hybrid != k_rev2

    entry = RetrievalCacheEntry(
        cache_key=k_hybrid,
        normalized_query="hello",
        retrieval_mode="hybrid",
        config_fingerprint=cfg_fp,
        analyzer_version="v1",
        index_generation=1,
        embedding_space_id="m:1024",
        document_revision_token="1:1:v1",
        reranking_enabled=False,
        created_at_monotonic=time.monotonic(),
        ttl_seconds=60.0,
    )
    cache.put(entry, degraded_resp)
    assert cache.get(k_hybrid) is None  # Never caches degraded responses!

    cache.put(entry, resp)
    assert cache.get(k_hybrid) is resp
    assert cache.get(k_sparse) is None
    assert cache.stats()["hits"] == 1

    # LRU eviction when max_entries=2
    for idx in (2, 3):
        k = f"k{idx}"
        cache.put(entry.model_copy(update={"cache_key": k}), resp)
    assert cache.stats()["entries"] == 2
    assert cache.get(k_hybrid) is None  # Evicted

    # TTL expiration
    expired_cache = RetrievalCache(
        RetrievalCacheSettings(enabled=True, max_entries=4, ttl_seconds=0.01)
    )
    expired_entry = entry.model_copy(update={"ttl_seconds": 0.01})
    expired_cache.put(expired_entry, resp)
    time.sleep(0.02)
    assert expired_cache.get(k_hybrid) is None


@pytest.mark.anyio
async def test_multi_query_retriever_cache_filters_and_embedding_mismatch(tmp_path: Path) -> None:
    settings = Settings(storage=Storage(database_path=tmp_path / "mq.sqlite3"))

    def transport_fn(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        body = json.loads(request.content)
        if path == "/api/embed":
            return httpx.Response(
                200,
                json={
                    "model": settings.runtime.embedding_model,
                    "embeddings": [[1.0, 0.0] for _ in body["input"]],
                },
            )
        return httpx.Response(404, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport_fn)) as client:
        resources = ResourceManager(settings)
        analyzer = TextAnalyzer.from_config(settings.sparse)
        ingestion = IngestionService(settings, resources)
        sparse_index = SparseIndex(ingestion.repository, analyzer)
        ingestion.sparse_index = sparse_index
        provider = Ollama(client, settings)
        coordinator = Coordinator(settings, provider)
        emb = EmbeddingEngine(settings, provider, coordinator, ingestion.repository)
        dense = DenseRetriever(settings, emb)
        sparse = SparseRetriever(settings, sparse_index, analyzer)
        hybrid = HybridRetriever(settings, dense, sparse, analyzer)
        cache = RetrievalCache(settings.retrieval_cache)
        mq = MultiQueryRetriever(settings, hybrid, cache)

        planner = AdaptiveQueryPlanner(settings, analyzer)
        req = RAGRequest(
            question="Compare vector storage with SQLite WAL and explain limitations.",
            strategy="adaptive",
            filters=MetadataFilter(document_id="doc-a"),
        )
        plan = await planner.plan(req)

        c_a = _make_chunk("c_a", "doc-a", "Vector storage uses float32 files.")
        c_b = _make_chunk("c_b", "doc-a", "SQLite WAL supports concurrent reads.", start=100)

        calls: list[MetadataFilter] = []

        async def fake_search(r):  # type: ignore[no-untyped-def]
            calls.append(r.resolved_filters())
            return _make_resp(r.query_id or "0", [c_a, c_b])

        hybrid.search = AsyncMock(side_effect=fake_search)  # type: ignore[method-assign]
        res1 = await mq.retrieve(plan, filters=req.filters)
        assert res1.retrieval_executions == len(plan.variants)
        assert all(f.document_id == "doc-a" for f in calls)

        # Second call hits cache with 0 new executions.
        res2 = await mq.retrieve(plan, filters=req.filters)
        assert res2.retrieval_executions == 0
        assert res2.cache_hits == len(plan.variants)

        # Embedding space mismatch raises 500 IngestionError.
        cache.invalidate()
        responses = iter(
            [
                _make_resp("0", [c_a], model="m1", digest="d1"),
                _make_resp("1", [c_b], model="m2", digest="d2"),
                _make_resp("2", [c_a], model="m1", digest="d1"),
            ]
        )

        async def mismatch_search(r):  # type: ignore[no-untyped-def]
            return next(responses)

        hybrid.search = AsyncMock(side_effect=mismatch_search)  # type: ignore[method-assign]
        with pytest.raises(IngestionError) as exc_info:
            await mq.retrieve(plan)
        assert exc_info.value.code == "embedding_space_mismatch"
        assert exc_info.value.status == 500

        # Sparse mode + execution cap (max_retrieval_executions=1) + duplicate variant
        sparse_plan = plan.model_copy(
            update={
                "retrieval_mode": "sparse",
                "budget": plan.budget.model_copy(update={"max_retrieval_executions": 1}),
            }
        )
        hybrid.search = AsyncMock(side_effect=fake_search)  # type: ignore[method-assign]
        res_sparse = await mq.retrieve(sparse_plan)
        assert res_sparse.retrieval_executions == 1


@pytest.mark.anyio
async def test_evidence_consolidation_preserves_adjacent_and_reports_coverage() -> None:
    settings = Settings()
    planner = AdaptiveQueryPlanner(settings)
    plan = await planner.plan(
        RAGRequest(
            question="Compare vector storage with SQLite WAL and explain limitations.",
            strategy="adaptive",
        )
    )
    # Two adjacent non-overlapping chunks from the same doc, plus an exact duplicate.
    c1 = _make_chunk("c1", "doc1", "Vector storage uses float32 files.", start=0)
    c2 = _make_chunk("c2", "doc1", "SQLite WAL provides transaction recovery.", start=100)
    c_dup = _make_chunk("c_dup", "doc2", "Vector storage uses float32 files.", start=0)

    v_orig = plan.variants[0]
    v_sq1 = plan.variants[1]
    cands, details = multi_query_rrf(
        [
            (v_orig, _make_resp("0", [c1, c_dup])),
            (v_sq1, _make_resp("1", [c2])),
        ],
        mq_settings=settings.multi_query,
        max_candidates=10,
    )
    from evorag.query_types import MultiQueryResult

    mq_res = MultiQueryResult(
        plan_id=plan.plan_id,
        effective_mode="hybrid",
        retrieval_executions=2,
        candidates=cands,
        fused_details=details,
        subquery_candidate_counts={"sq_1": 1, "sq_2": 0},
    )
    selected, cov = consolidate_evidence(mq_res, plan, max_items=5)
    assert len(selected) == 2  # c1 and c2 kept; c_dup deduplicated!
    assert cov.duplicate_evidence_ratio > 0.0
    assert "sq_1" in cov.subqueries_in_selected_evidence
    assert "sq_2" in cov.missing_subqueries
    assert any(w.startswith("missing_subquery_evidence:") for w in cov.warnings)
    await asyncio.sleep(0)
