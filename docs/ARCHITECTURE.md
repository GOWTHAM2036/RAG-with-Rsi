# Architecture

## Document ingestion

The application lifespan injects an IngestionService alongside the Phase 01 services. HTTP uploads and trusted local-file ingestion share resource admission, filename validation, raw SHA-256 identity, format parsing, normalization, chunk generation and a SQLite repository. The parser supplies normalized text and bounded structural regions; the chunker projects each window onto those regions. The repository streams the resulting typed chunks inside one version transaction. Batch ingestion invokes the same admission path per file. Chunk lookup, version history and job status read committed state. [Ingestion guide](INGESTION.md) · [Chunking](CHUNKING.md) · [Storage](STORAGE.md).

## Foundation runtime

`evorag.main:create_app` accepts validated settings and an optional HTTP transport. Its ASGI lifespan loads settings once, constructs a process-scoped resource manager and one reusable HTTPX client, then exposes them through FastAPI dependency injection. Shutdown clears the service reference and closes the client. Lifecycle events use a dedicated JSON logger.

`GET /health` checks liveness in process. `GET /ready` returns 503 when the lifespan has not initialized services. Once initialized, it returns 200 and independently probes the configured Ollama tag inventory. A successful listing yields booleans for the exact generation and embedding names. On timeout, HTTP failure, malformed inventory, a busy probe, or an unsupported provider, the response identifies the condition and reports unknown model availability as JSON null. Inventory confirms tag presence only. Generation capacity and successful inference are separate runtime concerns.

## Data boundaries

| Contract | Meaning |
| --- | --- |
| DocumentMetadata / Document | Source identity and media type with the document's text |
| Chunk | Document reference and a half-open span in normalized text measured in Unicode code points |
| Embedding | Chunk and model identity, configured width and actual finite provider coordinates |
| RetrievalResult / Evidence | Ranked, scored chunk and the evidence ID provided to generation |
| Citation / QueryResponse | Evidence reference and half-open span in the answer; response validates joins and bounds |
| QueryRequest / ExecutionTrace | Query identity, question, stage names, timezone-aware start and elapsed milliseconds |
| ModelConfiguration | Validated provider identity, endpoint, model tags, token and request budgets |
| ExperimentConfiguration / EvaluationResult | Versioned config and dataset, held-out split, seed, trace and measured metric |

Five asynchronous protocols describe ingestion, embedding, retrieval, generation and evaluation. Provider adapters translate untrusted responses into these contracts. The Ollama payload decoder accepts a decoded embed response, requested chunk IDs and active settings. It checks the model, row count, numeric coordinates and configured width before returning immutable embeddings. The future HTTP adapter can call it before persistence. End-to-end provenance checks that need a corpus belong to the future persistence layer.

## Milestone boundaries

Ingestion will validate and parse bounded inputs, normalize text, build chunks, request embeddings in batches, and update the vector and sparse indexes. SQLite WAL will hold metadata; a disk-backed NumPy float32 store will hold vectors. A future commit protocol must publish matching versions of both stores after interrupted writes.

The query path embeds a question, searches dense (memmap float32 with block-wise exact scoring) and sparse (SQLite inverted index + BM25) indexes with consistent metadata and revision filters, fuses ranks with weighted RRF, optionally applies CPU-only heuristic reranking, and returns typed results with optional explanations. If the embedding provider is unavailable and `allow_sparse_fallback` is enabled, hybrid retrieval degrades explicitly to sparse-only; vector corruption and embedding-space mismatches fail closed. Context construction, generation and citation validation belong to Phase 05. Ollama is the initial runtime. LM Studio, OpenAI, Gemini and OpenAI-compatible endpoints use separate adapters behind the same application interfaces.

Offline improvement follows observe → evaluate → diagnose → propose → experiment → validate → approve → deploy → monitor → repeat. Use versioned data and settings, held-out evaluation, fixed seeds, resource ceilings and regression checks. A human approves promotion. Store the previous version for rollback. Production code, permissions and deployment settings remain under owner control.
