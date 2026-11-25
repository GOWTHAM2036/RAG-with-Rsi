# Hybrid Retrieval and Fusion

`HybridRetriever` (`backend/evorag/hybrid_retrieval.py`) coordinates `DenseRetriever`, `SparseRetriever`, weighted Reciprocal Rank Fusion (`backend/evorag/fusion.py`), metadata filtering (`backend/evorag/filtering.py`), and optional CPU-only reranking.

## Modes and Flow

Supported modes: `dense`, `sparse`, `hybrid` (default for `query` requests; legacy `question`-only requests without `mode` preserve Phase 03 `dense` behavior).

1. Validate `RetrieveRequest` and resolve `MetadataFilter` (`document_id`, `version`, `latest_only`, `file_type`, `source`).
2. Execute dense retrieval (`dense_top_k = 20`) and sparse retrieval (`sparse_top_k = 20`) with identical revision and metadata filters applied during candidate selection.
3. Deduplicate by `chunk_id` and fuse with weighted RRF:
   $$\text{RRF}(d) = \sum_{r \in \{\text{dense}, \text{sparse}\}} \frac{w_r}{k + \text{rank}_r(d)}$$
   with `k = 60`, `dense_weight = 1.0`, `sparse_weight = 1.0`, 1-based ranks, and ascending `chunk_id` tie-breaking.
4. Optionally apply `rerank_candidates` over up to `max_candidates` (`40`) fused items.
5. Return `final_top_k` (`5`) results and optional `RetrievalExplanation` records.

## Degradation and Integrity Policy

- If `allow_sparse_fallback = true` and dense retrieval fails with a provider/availability error (`provider_unavailable`, `connect_timeout`, `read_timeout`, `model_not_installed`, `resource_exhaustion`, `inference_busy`, `inference_queue_full`, `inference_queue_timeout`, `insufficient_memory`), hybrid retrieval degrades to `effective_mode = "sparse"` with `degraded = true` and `degradation_reason` set.
- Data integrity failures (`vector_store_corrupted`, `embedding_model_changed`, `invalid_response`) **never** degrade and always fail closed.
