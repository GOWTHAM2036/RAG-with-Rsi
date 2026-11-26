# Grounded RAG Pipeline (Phase 05)

`RAGService` (`backend/evorag/rag_service.py`) coordinates end-to-end grounded question answering:

1. **Request Validation**: Validates `RAGRequest` (`question`, `retrieval_mode`, `top_k`, `filters`, `reranking_enabled`, `conversation_id`).
2. **Conversation Context**: Loads bounded prior turns from `ConversationStore` (in-memory by default).
3. **Hybrid Retrieval**: Calls `HybridRetriever.search(...)` (`dense`, `sparse`, or `hybrid` with explicit sparse-only degradation when the embedding provider is unavailable; vector corruption fails closed).
4. **Evidence Selection & Context Budgeting**: Deduplicates and packs `SelectedEvidence` (`E1..En`) within the 4096-token `ContextSettings` budget.
5. **Early Abstention**: When no evidence is retrieved, returns `insufficient_evidence` without invoking the generation model.
6. **Resource-Aware Generation**: Acquires `Coordinator.generate(...)` after retrieval has released any embedding reservation, enforcing the 3 GiB RAM floor and single-request serialization.
7. **Output Parsing, Citation Validation & Admission**: Strictly parses JSON (`parse_structured_answer`), resolves citations against prompt evidence (`validate_citations`), and applies `admit_answer` (`accepted`, `partially_accepted`, `insufficient_evidence`, `invalid_citations`, `invalid_model_output`, `generation_failed`, `resource_unavailable`).
8. **Execution Tracing**: Persists `RAGExecutionTrace` metadata to SQLite WAL (`TraceStore`).

## Endpoints

- `POST /api/v1/rag/query` — Buffered grounded RAG query.
- `POST /api/v1/rag/query/stream` — NDJSON stage events (`retrieval_started`, `retrieval_completed`, `evidence_selected`, `generation_started`, `validation_started`, `completed` / `failed`), validating the full response before emitting `completed`.
- `GET /api/v1/rag/traces/{trace_id}` — Retrieve stored execution trace metadata.
- `GET /api/v1/rag/status` — Report RAG configuration and trace/conversation counts.
