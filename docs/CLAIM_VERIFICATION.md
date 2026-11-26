# Claim Verification

`backend/evorag/verification.py` separates structural/provenance checks from semantic support:

- **Deterministic Verification (`verify_deterministic`)**: Checks JSON structure, citation ID validity, provenance (`version >= 1`, valid spans), duplicate claims, and unchecked free-form content (`unchecked_freeform_content`). Runtime claims without semantic verification are explicitly marked `not_assessed`.
- **Optional Semantic Critique (`parse_reflection_critique`)**: Strictly parses bounded JSON from `qwen3:4b`, rejecting prompt-injection patterns, unauthorized actions, and invalid evidence IDs.
- **Limitation**: Citation existence and deterministic checks do not prove semantic entailment.
