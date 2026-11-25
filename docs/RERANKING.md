# Lightweight Reranking

`backend/evorag/reranking.py` provides an optional, deterministic, CPU-only heuristic reranker (`reranking.enabled = false` by default).

## Features and Weights

Over a bounded candidate set (`max_candidates = 40`):

- **Exact phrase match** (`phrase_weight = 0.35`): `1.0` if normalized multi-token query appears as a substring in the normalized chunk.
- **Query-term coverage** (`coverage_weight = 0.30`): fraction of unique query tokens present in the chunk.
- **Technical identifier match** (`identifier_weight = 0.25`): fraction of identifier-like query tokens (containing digits or `._/-`) present in the chunk.
- **Fused rank signal** (`rrf_weight = 0.40`): `1 / fused_rank`.
- **Near-duplicate penalty** (`duplicate_penalty = 0.20`): subtracted when token Jaccard similarity with an already-selected chunk is `>= 0.85`.
- **Document diversity penalty** (`diversity_penalty = 0.10`): multiplied by the count of already-selected chunks from the same `document_id`.

Reranking scores are uncalibrated ranking heuristics for relative ordering, not probabilities. Original `dense_score`, `bm25_score`, and `rrf_score` remain preserved in `RetrievalExplanation`.
