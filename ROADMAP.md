# EvoRag roadmap

| Phase | Depends on | Deliverable and acceptance gate |
| --- | --- | --- |
| 01 · Foundation | Workspace audit | Start a single-worker service, validate settings, exercise resource guards, pass offline tests, Ruff, strict mypy and 80% coverage. |
| 02 · Ingestion and provenance | 01 and owner approval | Start with UTF-8 plain text. Bound bytes before decoding, normalize deterministically, assign versioned document identities, and persist metadata in SQLite WAL transactions. Implement fixed-size overlapping chunks with traceable spans and repeatable IDs. Verify identical re-ingestion, changed input, Unicode, rollback and resource limits. |
| 03 · Embedding and dense retrieval | 02 | Ollama HTTP embedding adapter, CPU batches, actual dimension checks, atomic disk-backed float32 vectors, mathematically tested dot product, cosine, Euclidean distance and exact nearest neighbors. |
| 04 · Sparse and hybrid retrieval | 03 and a versioned public fixture | BM25, reciprocal rank fusion, mathematical reranking, retrieval traces and a held-out baseline. |
| 05 · Grounded generation | 04 | Token budget, context builder, direct generation adapter, citation checks, bounded single-model execution, GPU preference and CPU fallback. Add optional query rewriting and reflection after evaluating failures. |
| 06 · Evaluation and deployment | 05 | Versioned experiments, quality and latency metrics, reproducible benchmark splits, React/TypeScript/Vite/Tailwind UI, Docker Compose and deployment validation. |
| 07 · Controlled improvement | 06 | Isolated proposals, regression and resource comparisons, human approval, deployment monitoring and rollback to a versioned baseline. |

Phase 02 was authorized on 25 November 2026. Its scope includes TXT, Markdown, CSV and JSON plus fixed, recursive and structure-aware chunking. Phase 03 offline dense retrieval and Phase 04 offline sparse/hybrid retrieval and reranking are implemented; live Ollama model verification remains pending owner authorization. See the project state for the current verification gates.

After owner approval, begin Phase 03 by specifying the mapping from immutable chunk IDs to versioned embedding rows. Implement bounded CPU batches, validate actual vector width, commit disk-backed float32 vectors and their SQLite metadata atomically, then test exact search against hand-computed fixtures. Verify recovery from interrupted writes before benchmarking retrieval quality.
