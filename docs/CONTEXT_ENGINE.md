# Context Construction and Token Budgeting

`backend/evorag/context_engine.py` implements deterministic evidence selection and token budget enforcement for a 4096-token model context window.

## Token Allocation (`[context]`)

| Bucket | Default Tokens |
| --- | ---: |
| `system_token_budget` | 384 |
| `question_token_budget` | 256 |
| `history_token_budget` | 384 |
| `evidence_token_budget` | 2048 |
| `reserved_output_tokens` | 768 |
| `safety_margin_tokens` | 256 |
| **Total (`max_context_tokens`)** | **4096** |

- Token counts use `estimate_tokens(text)` (`ceil(len(text) / 4)`), which is a conservative approximation and **not** exact provider tokenization.
- Oversized questions are rejected (`422 question_exceeds_token_budget`) by default, or truncated when `oversized_question_policy = "truncate"`.
- Older conversation history turns are trimmed first before sacrificing retrieved evidence.
- Evidence items (`E1..En`) are deduplicated (exact normalized text and >=85% same-document span overlap) while preserving distinct adjacent chunks. Truncated chunks record exact `included_start` and `included_end` offsets.
