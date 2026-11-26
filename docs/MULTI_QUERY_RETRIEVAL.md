# Multi-Query Retrieval and Evidence Consolidation

## Multi-Query Retrieval (`evorag.multi_query_retrieval`)

`MultiQueryRetriever` executes planned `QueryVariant` queries through `HybridRetriever` within `RetrievalBudget`:

- Enforces `max_retrieval_executions` (default 4) and `request_deadline_seconds`.
- Deduplicates identical normalized variant queries within the same plan.
- Preserves trusted `MetadataFilter` on every retrieval call; model-generated queries can never override filters.
- Verifies embedding-space consistency `(model, digest)` across all non-degraded dense/hybrid responses in a plan and raises `embedding_space_mismatch` (HTTP 500) on conflict.
- Never silently degrades on vector corruption or SQLite integrity failures.

## Weighted RRF (`multi_query_rrf`)

- Default weights: original `1.0`, rewrite `0.8`, subquery `0.7`.
- Normalizes non-original variant weights when their sum exceeds `variant_weight_cap * original_weight` (default `1.5x`) so generated variants cannot overwhelm the original query.
- Retains `original_rank`, `variant_ranks`, `per_query_fusion_scores`, `dense_score`, `sparse_score`, `aggregated_score`, and `contributing_query_ids`.
- Deterministic tie-breaking by ascending `chunk_id`.

## Evidence Consolidation (`evorag.evidence_consolidation`)

`consolidate_evidence` reuses Phase 05 overlap detection (`_is_redundant_overlap`) to deduplicate exact and >=85% overlapping spans while preserving adjacent non-overlapping chunks from the same document. For decomposed queries, it tracks and reports `EvidenceCoverage` (`subqueries_with_candidates`, `subqueries_in_selected_evidence`, `missing_subqueries`, `duplicate_evidence_ratio`, `document_diversity`, `evidence_budget_utilization`) without fabricating evidence for missing subqueries.
