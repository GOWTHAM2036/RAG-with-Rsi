# Resource budget

Phase 02 admits one ingestion per service instance, before reading the upload. It enforces the existing 3 GiB available-RAM floor and checks disk budget before publishing SQLite changes. One ASGI worker gives a single process-wide upload slot. Batch paths are read one at a time. Default caps are 10 MiB raw, 20 MiB extracted, 10 documents per batch and 10,000 chunks per document. Size settings ending in ingestion `_mb` use MiB. Output growth from overlapping chunk text and repeated provenance is capped separately.

Run `python benchmarks/ingestion_benchmark.py` with the project interpreter. Each synthetic input size runs in a fresh process with a temporary database and checks real RAM admission. Timings cover first ingestion, duplicate ingestion, isolated chunking and database publication from prepared chunks. Peak RSS is sampled every 2 ms, so very short spikes may be missed. See PROJECT_STATE.md for measurements or a recorded admission blocker.

The target machine has an i7-12650H, 10 physical cores, 16 logical processors, 16 GB of RAM and approximately 4 GB of dedicated VRAM. Keep Windows and development tools responsive. Resource settings ending in `_gb` use 1,073,741,824 bytes per unit.

| Default | Enforcement in Phase 01 |
| --- | --- |
| One generation and one concurrent request | A nonblocking process-local semaphore uses the smaller configured limit. Busy callers receive ResourceUnavailable. Permits release on failure or cancellation. Run one ASGI worker. |
| 3 GiB free RAM | Generation admission samples available physical RAM and declines work below the floor or when the measurement is unknown. |
| 20 GiB project storage | Writer preflight checks the proposed growth against project usage and free space on the destination volume. Future writers must serialize accounting and publication. |
| One loaded LLM | Phase 01 loads zero models. The future model adapter owns load/unload coordination with the runtime. |
| CPU embedding, batches of 8, width 1024 | Typed settings and returned-vector validation. Execution and device controls arrive with the adapter. |
| 4,096 context tokens, 768 output tokens | Configuration reserves input capacity. Generation will enforce the combined prompt and output limit. |

Run `python -m evorag.resources` from the project root using the installed environment for a JSON snapshot of available RAM, free bytes on the working volume and NVIDIA GPU names. Missing telemetry yields null. NVIDIA detection uses a two-second CLI timeout and runs only on request. The service performs no background sampling. The future generation adapter should prefer GPU, support CPU fallback, and serialize reflection and optimization with foreground generation.

Phase 04 retrieval runs entirely on CPU with bounded candidate sets (`dense_top_k=20`, `sparse_top_k=20`, `reranking.max_candidates=40`), block-wise `float64` similarity scoring (`block_size=2048`) over read-only `float32` memory maps, and bounded SQLite posting queries (`MAX_POSTING_CANDIDATES=5000`). See `docs/RETRIEVAL_BENCHMARKS.md` for measured 1,000- and 10,000-chunk RAM and latency results.

Phase 06 adaptive RAG enforces `RetrievalBudget` (`max_query_variants=3`, `max_subqueries=3`, `max_retrieval_executions=4`, `max_aggregate_candidates=80`, `max_final_evidence_items=5`, `max_planning_llm_calls=1`, `max_decomposition_depth=1`, `max_concurrent_generations=1`, `request_deadline_seconds=30.0`) and a bounded in-memory `RetrievalCache` (`max_entries=256`, `ttl_seconds=300.0`). Deterministic planning adds `< 0.5 ms` p95 latency and `< 3 MiB` additional peak RAM. See `docs/ADAPTIVE_RAG_BENCHMARKS.md`.
