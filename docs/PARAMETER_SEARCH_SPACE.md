# Parameter Search Space

`TRUSTED_SEARCH_SPACE` (`backend/evorag/rsi_search_space.py`) restricts tuning to an explicit allowlist (`bm25.k1`, `bm25.b`, `fusion.dense_weight`, `fusion.sparse_weight`, `fusion.k`, `retrieval.dense_top_k`, `retrieval.sparse_top_k`, `retrieval.final_top_k`, `adaptive_rag.default_strategy`, `query_rewriting.enabled`, `query_decomposition.enabled`, `reranking.enabled`, `reflection.*`, `retrieval_cache.*`). Security controls, RAM floors, holdout protection, and reindexing-dependent settings (`allow_reindexing = false`) are strictly excluded.
