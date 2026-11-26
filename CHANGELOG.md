# Changelog

## Frontend UI Phase 11.3 · Functional AI Workspace

Implemented **RAG Playground**, **Document Management**, and **Retrieval Lab** on top of the Phase 11.2 Neural Intelligence Console, with route-level `React.lazy` code splitting and strict `LIVE` vs `DEMO` separation:
- **Typed API & NDJSON Streaming (`frontend/src/api.ts`)**: Consolidated verified OpenAPI contracts (documents, versions, chunks, embedding jobs/cancel, index status/rebuild, retrieve, query/plan/stream, traces, models) with UTF-8 partial-frame NDJSON parsing, `AbortController` cancellation, deadline enforcement, malformed-line handling, and `503` resource admission errors (`ApiError`).
- **RAG Playground (`frontend/src/components/RAGPlayground.tsx`)**: 3-panel conversational research workspace with bounded local threads, retrieval/strategy/reflection controls, XSS-safe Markdown (`frontend/src/utils/markdown.tsx`), stop/retry, interactive `[1]` citations, and cross-view links to Document Management, Knowledge Explorer, and Pipeline Explorer.
- **Document Management (`frontend/src/components/DocumentManagement.tsx`)**: Drag-and-drop & file picker ingestion validated to `.txt`, `.md`, `.csv`, `.json`, duplicate revision handling, revision/chunk/span inspection, embedding job triggers/cancellation, and cross-view jump to Knowledge Explorer.
- **Retrieval Lab (`frontend/src/components/RetrievalLab.tsx`)**: `POST /api/v1/retrieve` ranking comparison (`dense`, `sparse`, `hybrid`, `rerank`) plus an explicitly labeled **Educational Retrieval Math Simulator** (cosine, dot, Euclidean, BM25, RRF).
- **Verification**: `tsc --noEmit` passed; Vitest **6/6 tests** passed; Vite build passed (`249.50 kB` main JS / `73.75 kB` gzipped + lazy workspace chunks); Playwright screenshots verified at 1440px & 390px mobile.

## Frontend UI Phase 11.2 · Premium Neural Intelligence Console

Upgraded `frontend/` into the EvoRAG Neural Intelligence Console:
- **Application Shell & Design System**: Collapsible grouped sidebar (Signature Views + Planned Phase 11.3+ views), EvoRAG network SVG logo & favicon, breadcrumbs, global `Ctrl+K` command palette, notifications/inspector/appearance drawers, and `Primitives.tsx` (`SurfaceCard`, `ProvenanceBadge`, `MetricReadout`, `SkeletonBlock`).
- **Command Center & Living RAG Core**: Made Intelligence Command Center the default landing bento grid with a 3D-projected HTML5 Canvas `LivingRagCore` neural sphere (2D fallback, reduced-motion, `IntersectionObserver` offscreen pause, pointer orbit, fullscreen, and event pulses).
- **Knowledge Explorer, Pipeline Explorer & RSI Evolution Tree**: Upgraded Knowledge Explorer with a deterministic 240+ node demo corpus (`demoCorpus.ts`), cluster isolation, progressive zoom labels, 1-hop/2-hop neighbor expansion, and accessible table view; upgraded Pipeline Explorer to an interactive 10-stage execution DAG with timeline scrubber; upgraded RSI Evolution Tree with side-by-side statistical candidate comparison and owner-CLI guardrails.
- **Verification**: `tsc --noEmit`, `vitest run` (3/3 tests passed), `vite build` (`238.23 kB` JS / `70.30 kB` gzipped), and Playwright browser screenshots at 1440px, 1280px, 768px, and 390px.

## Frontend UI & Knowledge Explorer · UI-00 / UI-01

Added React + TypeScript + Vite + Tailwind CSS console under `frontend/` featuring a near-black (`#05070B`) canvas, cyan (`#22D3EE`) provenance links, purple (`#A855F7`) reflection/evolution nodes, four views (`Knowledge Explorer`, `Command Center`, `Pipeline Explorer`, `RSI Evolution Tree`), reduced-motion and low-performance modes, accessible table fallback, and bounded `GET /api/v1/graph/provenance` (`backend/evorag/graph_api.py`) with CORS middleware for local Vite dev origins.

## Phase 10 · System hardening, live verification and performance optimization

Closed Phase 08/09 branch coverage gaps (`rsi_search_space.py`, `rsi_analysis.py`, `rsi_optimizer.py`, `rsi_engine.py`), verified 2-cycle recursive RSI refinement and experiment isolation, batched `SparseIndex.rebuild_all` job progress WAL transactions, and verified live `Coordinator.admission('qwen3:4b')` against `127.0.0.1:11550` (properly declined at ~1.76 GiB available RAM vs 6.33 GiB required).

## Phase 09 · Recursive self-improvement engine

Completed Phase 08 reflection hardening (`correction_planner.py` >= 90%, `reflection_engine.py` >= 90%). Added bounded RSI contracts (`rsi_types.py`), 21-state `RSIStateMachine` (`rsi_state.py`), trusted parameter search space & validator (`rsi_search_space.py`), failure pattern analysis & hypothesis engine (`rsi_analysis.py`), coordinate/random/local/Pareto optimizer (`rsi_optimizer.py`), SQLite WAL configuration registry, improvement memory, approvals, and rollback (`rsi_store.py`), `RSIEngine` (`rsi_engine.py`), owner CLI (`rsi_cli.py`), read-only/bounded `/api/v1/rsi/*` endpoints, and 30-scenario RSI benchmark (`benchmarks/rsi_benchmark.py`).

## Phase 08 · Self-reflective RAG and bounded correction engine

Added bounded reflection contracts (`reflection_types.py`), legal-transition state machine (`reflection_state.py`), deterministic verification and sanitized semantic critique parsing (`verification.py`), deterministic correction planner, targeted retrieval, evidence reconciliation, and candidate comparison (`correction_planner.py`), `ReflectionEngine` (`reflection_engine.py`), read-only Phase 09 RSI diagnostic interface (`rsi_interfaces.py`), `GET /api/v1/rag/reflection/status`, and 80-case reflection benchmark (`benchmarks/reflection_benchmark.py`).

## Phase 07 · Comprehensive evaluation and experimentation engine

Migrated `ModelConfiguration.generation_model` default to `qwen3:4b`. Added typed evaluation/experiment contracts (`eval_types.py`), versioned dataset registry with 108-case `evorag_comprehensive_v1` and contamination checks (`dataset_registry.py`), retrieval/answer/citation/faithfulness/abstention metrics (`eval_metrics.py`), 20-category failure taxonomy (`failure_analysis.py`), seeded bootstrap & Pareto comparison engine (`eval_stats.py`), strict quality gates (`quality_gates.py`), SQLite WAL experiment store (`experiment_store.py`), sequential `ExperimentRunner` with JSON/Markdown reports (`experiment_runner.py`), CLI (`eval_cli.py`), `/api/v1/evaluation/*` endpoints, and `benchmarks/evaluation_benchmark.py`.

## Phase 06 · Advanced adaptive RAG and query intelligence

Added typed query-intelligence contracts (`query_types.py`), CPU-only deterministic query analysis and rule-based classification (`query_analysis.py`), bounded query rewriting with injection/preservation validation (`query_rewriting.py`), bounded non-recursive query decomposition (`query_decomposition.py`), adaptive strategy planner (`query_planner.py`), bounded in-memory LRU+TTL retrieval cache (`retrieval_cache.py`), multi-query weighted RRF retriever (`multi_query_retrieval.py`), evidence consolidation and coverage diagnostics (`evidence_consolidation.py`), `RAGService` adaptive mode and `POST /api/v1/rag/plan`, `configs/example.toml` `qwen3:4b` update, and a 64-case adaptive RAG benchmark (`benchmarks/adaptive_rag_benchmark.py`).

## Phase 05 · Grounded generation and end-to-end RAG

Added typed RAG pipeline contracts (`rag_types.py`), deterministic evidence deduplication and 4096-token budget enforcement (`context_engine.py`), versioned grounded prompt templates with delimiter escaping (`prompt_engine.py`), strict bounded JSON output parsing and citation resolution (`citations.py`), deterministic answer admission and abstention (`admission.py`), bounded in-memory conversation context (`conversation.py`), SQLite WAL execution tracing (`tracing.py`), end-to-end `RAGService` with buffered NDJSON streaming (`rag_service.py`), `/api/v1/rag/*` endpoints, and a 42-case offline grounded-QA benchmark (`benchmarks/rag_benchmark.py`).

## Phase 04 · Sparse retrieval, hybrid search and reranking

Added deterministic Unicode-aware lexical analysis (`analyzer.py`), SQLite-backed inverted index with revision-aware active corpus tracking (`sparse_index.py`), from-scratch Okapi BM25 (`bm25.py`, `sparse_retrieval.py`), block-wise exact dense scoring (`similarity.py`, `retrieval.py`), typed metadata filtering (`filtering.py`), weighted Reciprocal Rank Fusion (`fusion.py`), optional CPU-only heuristic reranking (`reranking.py`), hybrid coordination with explicit sparse-only fallback (`hybrid_retrieval.py`), index management endpoints (`/api/v1/index/*`), retrieval evaluation metrics (`evaluation.py`), and offline quality/performance benchmarks (`benchmarks/retrieval_benchmark.py`).

## Phase 03 · Embedding and dense retrieval

Added Ollama HTTP client, inference coordinator, embedding engine with SQLite cache and resumable jobs, append-only float32 vector store with memmap and crash recovery, exact dense retrieval, and `/api/v1/retrieve`.

## Phase 02 · Document ingestion

Added local raw-byte document upload, bounded batch ingestion, UTF-8 parsing and source spans, three token-budgeted chunking strategies, atomic SQLite version publication, job inspection and a synthetic ingestion benchmark.

## 0.1.0 · Phase 01

Added the local Python service, health and readiness endpoints, TOML and environment settings, typed contracts and provider ports, resource reporting and admission checks, offline tests and foundation CI. See [project state](docs/PROJECT_STATE.md) for verification and integration status.
