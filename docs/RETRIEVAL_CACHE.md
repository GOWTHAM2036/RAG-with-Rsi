# Retrieval Cache

`RetrievalCache` (`evorag.retrieval_cache`) is a bounded, thread-safe in-memory LRU+TTL cache for non-degraded `RetrieveResponse` objects.

## Cache Key Identity

`build_cache_key` computes a SHA-256 digest over:

- `normalized_query`
- `retrieval_mode` (`dense`, `sparse`, `hybrid`)
- `config_fingerprint` (SHA-256 of `retrieval`, `sparse`, `fusion`, `reranking` settings)
- `analyzer_version`
- `index_generation` (sparse index generation)
- `embedding_space_id` (`model:dimensions`)
- `filters` (`MetadataFilter`)
- `document_revision_token` (corpus document and version count + latest version key)
- `reranking_enabled`

## Invalidation & Safety Rules

- Corpus updates increment `index_generation` / `document_revision_token`, automatically invalidating stale entries.
- Degraded responses (`degraded=True`, e.g., sparse fallback) and integrity failures are **never** cached.
- Raw private queries are **never** persisted to disk by the cache.
- Bounded by `[retrieval_cache].max_entries` (default `256`) and `[retrieval_cache].ttl_seconds` (default `300.0`).
- Hit/miss statistics are exposed via `GET /api/v1/rag/status` and `QueryExecutionTrace`.
