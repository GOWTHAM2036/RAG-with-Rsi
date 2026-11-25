# Changelog

## Phase 04 · Sparse retrieval, hybrid search and reranking

Added deterministic Unicode-aware lexical analysis (`analyzer.py`), SQLite-backed inverted index with revision-aware active corpus tracking (`sparse_index.py`), from-scratch Okapi BM25 (`bm25.py`, `sparse_retrieval.py`), block-wise exact dense scoring (`similarity.py`, `retrieval.py`), typed metadata filtering (`filtering.py`), weighted Reciprocal Rank Fusion (`fusion.py`), optional CPU-only heuristic reranking (`reranking.py`), hybrid coordination with explicit sparse-only fallback (`hybrid_retrieval.py`), index management endpoints (`/api/v1/index/*`), retrieval evaluation metrics (`evaluation.py`), and offline quality/performance benchmarks (`benchmarks/retrieval_benchmark.py`).

## Phase 03 · Embedding and dense retrieval

Added Ollama HTTP client, inference coordinator, embedding engine with SQLite cache and resumable jobs, append-only float32 vector store with memmap and crash recovery, exact dense retrieval, and `/api/v1/retrieve`.

## Phase 02 · Document ingestion

Added local raw-byte document upload, bounded batch ingestion, UTF-8 parsing and source spans, three token-budgeted chunking strategies, atomic SQLite version publication, job inspection and a synthetic ingestion benchmark.

## 0.1.0 · Phase 01

Added the local Python service, health and readiness endpoints, TOML and environment settings, typed contracts and provider ports, resource reporting and admission checks, offline tests and foundation CI. See [project state](docs/PROJECT_STATE.md) for verification and integration status.
