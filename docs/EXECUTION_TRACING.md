# Execution Tracing

`backend/evorag/tracing.py` persists `RAGExecutionTrace` records in SQLite WAL (`rag_traces` table) with bounded retention (`[tracing].max_records = 10000`).

## Recorded Metadata

- `trace_id`, `request_id`, `conversation_id`, `pipeline_version`, `prompt_template_version`
- `retrieval_mode`, `effective_retrieval_mode`, `degraded`, `degradation_reason`
- `embedding_model`, `generation_model`, `model_digest`, `selected_evidence_ids`
- `stage_metrics` (`retrieval_ms`, `context_ms`, `generation_ms`, `first_token_ms`, `validation_ms`, `total_ms`)
- `estimated_input_tokens`, `provider_usage`, `citation_validation`, `admission_outcome`, `resource_rejection_reason`

By default (`store_raw_prompts = false`, `store_raw_questions = false`), raw questions, prompts, and document contents are **not** persisted in traces. Trace persistence errors are logged and never alter the validity of a generated response.
