"""Offline retrieval quality fixture evaluation and performance benchmark.

Note:
    Because live Ollama inference is not used in offline benchmarks, dense vectors
    are deterministic synthetic fixtures. Quality numbers represent algorithmic
    fixture evaluations rather than real semantic retrieval quality.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import statistics
import tempfile
import time
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import psutil
from evorag.analyzer import TextAnalyzer
from evorag.config import Chunking, Embeddings, Reranking, Retrieval, Settings, Storage
from evorag.embedding_engine import QUERY_PREFIX, EmbeddingEngine
from evorag.evaluation import AggregateEvaluation, aggregate_evaluations, evaluate_query
from evorag.hybrid_retrieval import HybridRetriever, RetrieveRequest
from evorag.inference_types import ModelInfo
from evorag.ingestion import IngestionService
from evorag.resources import GIB, ResourceManager
from evorag.retrieval import DenseRetriever
from evorag.sparse_index import SparseIndex
from evorag.sparse_retrieval import SparseRetriever

DOCUMENTS: list[tuple[str, bytes]] = [
    (
        "http_errors.md",
        b"HTTP 404 Not Found indicates the requested endpoint /api/v1/retrieve or resource "
        b"does not exist. Return status 404 without exposing internal filesystem paths.",
    ),
    (
        "gpu_budget.md",
        b"Target laptop hardware uses NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM and "
        b"16 GB system RAM. Keep minimum_free_ram_gb at 3 GiB for admission.",
    ),
    (
        "embeddings.md",
        b"Qwen3-Embedding 0.6B produces 1024-dimensional float32 vectors. "
        b"Set EVORAG_RUNTIME__BASE_URL to http://127.0.0.1:11550 for local Ollama.",
    ),
    (
        "vector_recovery.md",
        b"When vector_store_corrupted occurs, truncated or missing float32 vector files "
        b"fail closed with HTTP 500 rather than silently degrading to sparse search.",
    ),
    (
        "python_env.md",
        b"EvoRag targets Python 3.12 on Windows 11 with SQLite WAL mode enabled and "
        b"custom NumPy memmap storage.",
    ),
    (
        "bm25_notes.md",
        b"Okapi BM25 scores sparse candidates using Robertson-Sparck Jones IDF with "
        b"k1=1.5 and b=0.75 over an inverted index.",
    ),
    (
        "rrf_fusion.md",
        b"Reciprocal Rank Fusion combines dense and sparse rankings using "
        b"weight / (k + rank) with k=60 and deterministic chunk_id tie-breaking.",
    ),
    (
        "reranking.md",
        b"CPU-only heuristic reranking boosts exact phrase matches, technical identifiers, "
        b"and term coverage while penalizing near-duplicate chunks.",
    ),
    (
        "security.md",
        b"All SQL queries use parameterized placeholders. Never log private document text "
        b"or expose database file paths in API responses.",
    ),
    (
        "revisions.md",
        b"Legacy revision v1 deprecated endpoint /api/v0/search.",
    ),
]
# Revision v2 supersedes v1 for revisions.md
REVISION_V2 = (
    "revisions.md",
    b"Current revision v2 uses /api/v1/retrieve with hybrid RRF and metadata filters.",
)

# 54 manually specified queries: 27 dev, 27 held_out, mapping to expected source filenames.
QUALITY_CASES: list[tuple[str, str, str, frozenset[str]]] = [
    # dev split (27)
    ("dev_01", "dev", "HTTP 404", frozenset({"http_errors.md"})),
    ("dev_02", "dev", "RTX 3050", frozenset({"gpu_budget.md"})),
    ("dev_03", "dev", "Qwen3-Embedding", frozenset({"embeddings.md"})),
    ("dev_04", "dev", "EVORAG_RUNTIME__BASE_URL", frozenset({"embeddings.md"})),
    ("dev_05", "dev", "vector_store_corrupted", frozenset({"vector_recovery.md"})),
    ("dev_06", "dev", "Python 3.12", frozenset({"python_env.md"})),
    ("dev_07", "dev", "/api/v1/retrieve", frozenset({"http_errors.md", "revisions.md"})),
    (
        "dev_08",
        "dev",
        "What GPU hardware and VRAM does the laptop have?",
        frozenset({"gpu_budget.md"}),
    ),
    ("dev_09", "dev", "How does BM25 calculate document scores?", frozenset({"bm25_notes.md"})),
    ("dev_10", "dev", "How are dense and sparse ranks fused?", frozenset({"rrf_fusion.md"})),
    (
        "dev_11",
        "dev",
        "What happens when vector files are missing?",
        frozenset({"vector_recovery.md"}),
    ),
    ("dev_12", "dev", "SQLite WAL mode storage", frozenset({"python_env.md"})),
    ("dev_13", "dev", "parameterized placeholders SQL", frozenset({"security.md"})),
    ("dev_14", "dev", "exact phrase matches duplicate chunks", frozenset({"reranking.md"})),
    ("dev_15", "dev", "minimum_free_ram_gb admission", frozenset({"gpu_budget.md"})),
    ("dev_16", "dev", "1024-dimensional float32 vectors", frozenset({"embeddings.md"})),
    ("dev_17", "dev", "k1=1.5 b=0.75 inverted index", frozenset({"bm25_notes.md"})),
    ("dev_18", "dev", "k=60 tie-breaking", frozenset({"rrf_fusion.md"})),
    ("dev_19", "dev", "Current revision v2 hybrid RRF", frozenset({"revisions.md"})),
    ("dev_20", "dev", "deprecated endpoint /api/v0/search", frozenset()),  # superseded v1!
    ("dev_21", "dev", "quantum teleportation protocol", frozenset()),  # no evidence
    ("dev_22", "dev", "blockchain consensus mining", frozenset()),  # no evidence
    (
        "dev_23",
        "dev",
        "internal filesystem paths error 404",
        frozenset({"http_errors.md", "security.md"}),
    ),
    ("dev_24", "dev", "NumPy memmap Windows 11", frozenset({"python_env.md"})),
    ("dev_25", "dev", "Ollama local 127.0.0.1:11550", frozenset({"embeddings.md"})),
    ("dev_26", "dev", "CPU-only heuristic reranking", frozenset({"reranking.md"})),
    ("dev_27", "dev", "fail closed HTTP 500", frozenset({"vector_recovery.md"})),
    # held_out split (27)
    ("ho_01", "held_out", "404 Not Found endpoint", frozenset({"http_errors.md"})),
    ("ho_02", "held_out", "NVIDIA 4 GB VRAM system RAM", frozenset({"gpu_budget.md"})),
    ("ho_03", "held_out", "0.6B embedding model dimensions", frozenset({"embeddings.md"})),
    ("ho_04", "held_out", "truncated float32 vector corruption", frozenset({"vector_recovery.md"})),
    ("ho_05", "held_out", "Windows 11 Python version", frozenset({"python_env.md"})),
    ("ho_06", "held_out", "Robertson-Sparck Jones IDF", frozenset({"bm25_notes.md"})),
    ("ho_07", "held_out", "Reciprocal Rank Fusion formula", frozenset({"rrf_fusion.md"})),
    ("ho_08", "held_out", "penalizing near-duplicate chunks", frozenset({"reranking.md"})),
    ("ho_09", "held_out", "never log private document text", frozenset({"security.md"})),
    ("ho_10", "held_out", "metadata filters revision v2", frozenset({"revisions.md"})),
    ("ho_11", "held_out", "sparse search degradation policy", frozenset({"vector_recovery.md"})),
    ("ho_12", "held_out", "3 GiB RAM floor", frozenset({"gpu_budget.md"})),
    (
        "ho_13",
        "held_out",
        "database file paths API responses",
        frozenset({"security.md", "http_errors.md"}),
    ),
    ("ho_14", "held_out", "technical identifiers term coverage", frozenset({"reranking.md"})),
    ("ho_15", "held_out", "deterministic chunk_id tie-breaking", frozenset({"rrf_fusion.md"})),
    ("ho_16", "held_out", "Okapi sparse candidates", frozenset({"bm25_notes.md"})),
    ("ho_17", "held_out", "custom NumPy memmap", frozenset({"python_env.md"})),
    ("ho_18", "held_out", "EVORAG_RUNTIME__BASE_URL 11550", frozenset({"embeddings.md"})),
    (
        "ho_19",
        "held_out",
        "HTTP 404 /api/v1/retrieve",
        frozenset({"http_errors.md", "revisions.md"}),
    ),
    ("ho_20", "held_out", "v1 legacy search", frozenset()),  # superseded v1!
    ("ho_21", "held_out", "astrophysics dark matter", frozenset()),  # no evidence
    ("ho_22", "held_out", "culinary sourdough recipe", frozenset()),  # no evidence
    ("ho_23", "held_out", "16 GB system RAM RTX", frozenset({"gpu_budget.md"})),
    ("ho_24", "held_out", "SQLite WAL Python 3.12", frozenset({"python_env.md"})),
    ("ho_25", "held_out", "weight / (k + rank)", frozenset({"rrf_fusion.md"})),
    ("ho_26", "held_out", "vector_store_corrupted HTTP 500", frozenset({"vector_recovery.md"})),
    ("ho_27", "held_out", "parameterized SQL security", frozenset({"security.md"})),
]


def _synthetic_vector(text: str, dim: int = 32) -> tuple[float, ...]:
    """Deterministic bag-of-trigrams hashed vector for offline algorithmic fixtures."""
    vec = np.zeros(dim, dtype=np.float64)
    normed = text.lower()
    for i in range(max(1, len(normed) - 2)):
        gram = normed[i : i + 3]
        h = int(hashlib.blake2b(gram.encode(), digest_size=4).hexdigest(), 16)
        vec[h % dim] += 1.0
    n = float(np.linalg.norm(vec))
    if n > 0.0:
        vec /= n
    else:
        vec[0] = 1.0
    return tuple(float(x) for x in vec)


class _FixtureRuntime:
    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self.model = ModelInfo(name="qwen3-embedding:0.6b", digest="fixture-d1", size_bytes=64)

    async def models(self) -> tuple[ModelInfo, ...]:
        return (self.model,)

    @asynccontextmanager
    async def reserve(self, model_name: str, capability: Literal["generation", "embedding"]):  # type: ignore[no-untyped-def]
        yield self.model

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(_synthetic_vector(t.removeprefix(QUERY_PREFIX), self.dim) for t in texts)


async def run_quality_evaluation() -> dict[str, dict[str, AggregateEvaluation]]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        settings = Settings(
            storage=Storage(database_path=root / "eval.sqlite3"),
            embeddings=Embeddings(dimensions=32, batch_size=8, normalize=True),
            chunking=Chunking(chunk_size=512, overlap=0, strategy="fixed"),
            retrieval=Retrieval(dense_top_k=10, sparse_top_k=10, final_top_k=5),
            reranking=Reranking(enabled=True, max_candidates=20),
        )
        ingestion = IngestionService(settings, ResourceManager(settings, lambda: 8 * GIB))
        analyzer = TextAnalyzer.from_config(settings.sparse)
        sparse_index = SparseIndex(ingestion.repository, analyzer)
        ingestion.sparse_index = sparse_index

        source_by_doc_id: dict[str, str] = {}
        runtime = _FixtureRuntime(dim=32)
        engine = EmbeddingEngine(settings, runtime, runtime, ingestion.repository)

        for name, payload in [*DOCUMENTS, REVISION_V2]:
            res = ingestion.ingest(name, payload)
            source_by_doc_id[res.document_id] = name
            await engine.run_job(res.document_id, res.version)

        dense = DenseRetriever(settings, engine)
        sparse = SparseRetriever(settings, sparse_index, analyzer)
        hybrid = HybridRetriever(settings, dense, sparse, analyzer)

        configs: list[tuple[str, Literal["dense", "sparse", "hybrid"], bool]] = [
            ("dense_only", "dense", False),
            ("bm25_only", "sparse", False),
            ("hybrid_rrf", "hybrid", False),
            ("hybrid_rerank", "hybrid", True),
        ]
        summary: dict[str, dict[str, AggregateEvaluation]] = {}
        for split in ("dev", "held_out"):
            summary[split] = {}
            cases = [c for c in QUALITY_CASES if c[1] == split]
            for label, mode, rerank in configs:
                evals = []
                for qid, _, qtext, expected_sources in cases:
                    resp = await hybrid.search(
                        RetrieveRequest(
                            query=qtext,
                            mode=mode,
                            top_k=5,
                            reranking_enabled=rerank,
                        )
                    )
                    retrieved_sources = [
                        source_by_doc_id[r.chunk.document_id]
                        for r in resp.results
                        if mode != "dense" or r.score > 0.0
                    ]
                    # For sparse/hybrid, keep all returned hits.
                    if mode in ("sparse", "hybrid"):
                        retrieved_sources = [
                            source_by_doc_id[r.chunk.document_id] for r in resp.results
                        ]
                    evals.append(evaluate_query(qid, retrieved_sources, expected_sources, k=5))
                summary[split][label] = aggregate_evaluations(evals, k=5)
        return summary


def _percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(pct * len(ordered)) - 1))
    return ordered[idx]


async def run_performance_benchmark(num_chunks: int) -> dict[str, object]:
    proc = psutil.Process()
    baseline_rss = proc.memory_info().rss
    peak_rss = baseline_rss

    def sample_rss() -> None:
        nonlocal peak_rss
        rss = proc.memory_info().rss
        if rss > peak_rss:
            peak_rss = rss

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dim = 64
        settings = Settings(
            storage=Storage(database_path=root / f"perf_{num_chunks}.sqlite3"),
            embeddings=Embeddings(dimensions=dim, batch_size=64, normalize=True),
            chunking=Chunking(chunk_size=128, overlap=0, strategy="fixed"),
            retrieval=Retrieval(dense_top_k=20, sparse_top_k=20, final_top_k=5),
            reranking=Reranking(enabled=True, max_candidates=40),
        )
        ingestion = IngestionService(settings, ResourceManager(settings, lambda: 8 * GIB))
        analyzer = TextAnalyzer.from_config(settings.sparse)
        sparse_index = SparseIndex(ingestion.repository, analyzer)

        # Ingest docs (50 chunks each) without auto-indexing to time index build separately.
        chunks_per_doc = 50
        num_docs = max(1, num_chunks // chunks_per_doc)
        doc_ids: list[str] = []
        for i in range(num_docs):
            lines = [
                f"Document {i} chunk {j} HTTP 404 RTX 3050 Python 3.12 token_{j % 200} "
                for j in range(chunks_per_doc)
            ]
            # Pad each line to ~100 chars so fixed chunker produces ~1 chunk per line.
            payload = "".join(line.ljust(128) for line in lines).encode("utf-8")
            res = ingestion.ingest(f"doc_{i}.txt", payload)
            doc_ids.append(res.document_id)

        t0 = time.perf_counter()
        for doc_id in doc_ids:
            sparse_index.index_version(doc_id, 1)
        index_build_ms = (time.perf_counter() - t0) * 1000.0
        sample_rss()

        # Incremental indexing of 1 additional document.
        inc_res = ingestion.ingest(
            "incremental.txt", ("HTTP 404 RTX 3050 " * 8).ljust(128).encode()
        )
        t_inc = time.perf_counter()
        sparse_index.index_version(inc_res.document_id, 1)
        incremental_ms = (time.perf_counter() - t_inc) * 1000.0

        # Populate dense vector store directly for offline retrieval timing.
        runtime = _FixtureRuntime(dim=dim)
        engine = EmbeddingEngine(settings, runtime, runtime, ingestion.repository)
        space = await engine._space()
        rng = np.random.default_rng(42)
        for doc_id in [*doc_ids, inc_res.document_id]:
            job = engine.store.prepare(doc_id, 1, space)
            engine.store.claim(job.job_id, "bench", 600.0)
            chunks = ingestion.repository.chunks(doc_id, 1, 100, 0)
            vecs = rng.normal(size=(len(chunks), dim)).astype("<f4")
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-8)
            rows = [(c.chunk_id, vecs[idx].tobytes()) for idx, c in enumerate(chunks)]
            engine.store.commit_batch(job.job_id, "bench", 0, 600.0, rows)
        sample_rss()

        dense = DenseRetriever(settings, engine)
        sparse = SparseRetriever(settings, sparse_index, analyzer)
        hybrid = HybridRetriever(settings, dense, sparse, analyzer)

        queries = [
            "HTTP 404 RTX 3050",
            "Python 3.12 token_42",
            "Document 5 chunk 10",
            "token_99 RTX 3050",
            "HTTP 404 token_1",
        ]

        async def measure_mode(
            mode: Literal["dense", "sparse", "hybrid"], rerank: bool
        ) -> dict[str, object]:
            cold_t0 = time.perf_counter()
            await hybrid.search(
                RetrieveRequest(query=queries[0], mode=mode, reranking_enabled=rerank)
            )
            cold_ms = (time.perf_counter() - cold_t0) * 1000.0
            warm_latencies: list[float] = []
            for i in range(20):
                q = queries[i % len(queries)]
                t_q = time.perf_counter()
                await hybrid.search(RetrieveRequest(query=q, mode=mode, reranking_enabled=rerank))
                warm_latencies.append((time.perf_counter() - t_q) * 1000.0)
                sample_rss()
            total_s = sum(warm_latencies) / 1000.0
            return {
                "samples": len(warm_latencies),
                "cold_ms": round(cold_ms, 2),
                "median_ms": round(statistics.median(warm_latencies), 2),
                "p95_ms": round(_percentile(warm_latencies, 0.95), 2),
                "min_ms": round(min(warm_latencies), 2),
                "max_ms": round(max(warm_latencies), 2),
                "qps": round(len(warm_latencies) / total_s, 1) if total_s > 0 else 0.0,
            }

        dense_m = await measure_mode("dense", False)
        sparse_m = await measure_mode("sparse", False)
        hybrid_m = await measure_mode("hybrid", False)
        rerank_m = await measure_mode("hybrid", True)

        storage_bytes = await asyncio.to_thread(
            lambda: sum(p.stat().st_size for p in root.iterdir() if p.is_file())
        )
        return {
            "chunks": num_chunks,
            "index_build_ms": round(index_build_ms, 2),
            "incremental_ms": round(incremental_ms, 2),
            "peak_additional_mib": round((peak_rss - baseline_rss) / (1024 * 1024), 2),
            "storage_mib": round(storage_bytes / (1024 * 1024), 2),
            "dense": dense_m,
            "sparse": sparse_m,
            "hybrid": hybrid_m,
            "hybrid_rerank": rerank_m,
        }


async def main() -> None:
    print("=== Retrieval Quality (Algorithmic Fixture Evaluation, Not Live Semantics) ===")
    quality = await run_quality_evaluation()
    for split, methods in quality.items():
        print(f"\nSplit: {split}")
        for name, agg in methods.items():
            print(
                f"  {name:14s} | P@5={agg.mean_precision:.3f} R@5={agg.mean_recall:.3f} "
                f"Hit@5={agg.mean_hit_rate:.3f} MRR={agg.mean_reciprocal_rank:.3f} "
                f"nDCG@5={agg.mean_ndcg:.3f} Abstain={agg.no_evidence_abstention_rate:.3f}"
            )

    print("\n=== Performance Benchmarks ===")
    for size in (1000, 10000):
        res = await run_performance_benchmark(size)
        print(
            f"\nCorpus: {res['chunks']} chunks | Index={res['index_build_ms']} ms | "
            f"Inc={res['incremental_ms']} ms | PeakAddRAM={res['peak_additional_mib']} MiB | "
            f"Storage={res['storage_mib']} MiB"
        )
        for key in ("dense", "sparse", "hybrid", "hybrid_rerank"):
            m = res[key]
            print(f"  {key:14s}: {m}")


if __name__ == "__main__":
    asyncio.run(main())
