# Adaptive RAG Architecture

Phase 06 integrates `AdaptiveQueryPlanner`, `MultiQueryRetriever`, `consolidate_evidence`, and `RetrievalCache` into `RAGService` while preserving the Phase 05 `baseline` path as the default.

## Pipeline Flow (`strategy = "adaptive"`)

```text
User question
  -> Query validation & analysis (analyze_query)
  -> Rule-based classification (classify_query)
  -> Adaptive strategy selection (AdaptiveQueryPlanner)
  -> Optional bounded rewriting / decomposition (depth <= 1)
  -> Bounded RetrievalPlan (variants <= 3, executions <= 4)
  -> Multi-query hybrid retrieval + cache + weighted RRF
  -> Evidence consolidation & coverage analysis
  -> Phase 05 context budgeting, grounded generation, citation validation & admission
  -> Metadata-only execution tracing
```

## Strategy Routing

- `simple_factual`: single hybrid retrieval with original query.
- `technical_exact`: preserves exact identifiers, avoids rewriting, emphasizes sparse weight.
- `conversational_resolve`: expands conservative conversation references while preserving original.
- `decomposed_multi_part`: decomposes comparative/multi-part questions into up to 3 subqueries.
- `exploratory_expanded`: bounded variant expansion.
- `baseline`: single-query Phase 05 path (default).

## API Endpoints

- `POST /api/v1/rag/query`: supports `strategy` (`baseline` | `adaptive`), `query_rewriting_enabled`, `query_decomposition_enabled`, `include_query_plan`, `include_retrieval_diagnostics`.
- `POST /api/v1/rag/plan`: returns a bounded `RetrievalPlan` without executing answer generation.
- `GET /api/v1/rag/status`: reports `adaptive_rag_enabled`, `default_strategy`, and `retrieval_cache_stats`.
