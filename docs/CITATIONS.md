# Citation Validation and Answer Admission

`backend/evorag/citations.py` and `backend/evorag/admission.py` enforce strict post-generation grounding checks:

1. **Strict JSON Parsing (`parse_structured_answer`)**: Enforces `max_output_bytes` (16 KiB) and `max_claims` (16), rejects malformed JSON without repairing it, validates `^E[1-9]\d*$` identifier syntax, and deduplicates `evidence_ids` per claim.
2. **Citation Resolution (`validate_citations`)**: Maps each `E1..En` strictly to the `SelectedEvidence` items presented in the prompt. Rejects invented or excluded IDs, detects uncited claims, strips directory prefixes from `source_id` so absolute local paths are never exposed, and assigns stable 1-based `citation_number` values.
3. **Answer Admission (`admit_answer`)**: Returns `accepted`, `partially_accepted`, `insufficient_evidence`, `invalid_citations`, `invalid_model_output`, `generation_failed`, or `resource_unavailable`.

> **Important:** Validating that a citation identifier exists in the prompt does **not** prove semantic entailment or factual accuracy. Semantic verification belongs to the future self-reflection phase.
