# Query Intelligence

Phase 06 adds deterministic CPU-only query analysis, rule-based classification, bounded rewriting, and bounded decomposition without external orchestration frameworks.

## Query Analysis (`evorag.query_analysis`)

`analyze_query` extracts structural and lexical signals while preserving `original_query` verbatim:

- Length (`char_length`, `estimated_tokens`)
- Question structure (`question_words`, `has_question_mark`)
- Technical identifiers & error codes (`technical_identifiers`, `error_codes`, `quoted_phrases`)
- Constraints (`numeric_references`, `negation_terms`, `has_negation`)
- Multi-part & comparative cues (`comparison_terms`, `conjunctions`, `temporal_references`)
- Conversational cues (`ambiguous_pronouns`, `conversation_references`)
- Validation (`is_valid`, `validation_errors`)

Normalization (`normalize_query_text`) applies Unicode NFKC and whitespace normalization without stripping negation, numbers, or technical identifiers.

### Heuristic Limitations

1. Signal extraction uses lexical patterns and closed-set vocabularies, not syntactic parsing.
2. Pronoun detection is lexical and may trigger on intra-sentence pronouns.
3. Classification labels are deterministic rule outputs, not calibrated probabilities.
4. Non-whitespace-delimited languages (e.g., CJK, Thai) are not segmented by `TextAnalyzer`.

## Query Classification

`classify_query` assigns multi-label `QueryIntent` tags (`factual`, `technical_lookup`, `comparative`, `multi_part`, `conversational`, `exploratory`, `unsupported`) and `QueryComplexity` (`simple`, `moderate`, `complex`, `invalid`), with explicit `triggered_signals` and `explanation`.

## Query Rewriting (`evorag.query_rewriting`)

- Defaults to deterministic normalization and conservative conversation-reference expansion.
- Optional LLM-assisted rewriting (`[query_rewriting].llm_enabled = false` by default) uses `REWRITE_PROMPT_V1` via `Coordinator.generate`.
- `validate_rewritten_query` rejects empty, oversized, malformed, prompt-injection-like, or identifier/number/negation-dropping outputs and falls back deterministically.

## Query Decomposition (`evorag.query_decomposition`)

- Splits comparative and multi-part queries up to `max_subqueries = 3` and `max_depth = 1` (never recursive).
- Deduplicates equivalent subqueries and falls back to the original query on failure.
