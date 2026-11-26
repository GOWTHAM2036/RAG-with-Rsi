# Dataset Registry & Split Isolation

`DatasetRegistry` (`backend/evorag/dataset_registry.py`) manages versioned datasets with SHA-256 content fingerprints and contamination detection (`validate_dataset`).

- **Expanded Suite**: `evorag_comprehensive_v1` contains **108 cases** (`54 dev`, `54 held_out`) across 15 categories (direct factual, technical identifiers, semantic retrieval, comparative, multi-document, multi-part, ambiguous, conflicting revisions, insufficient evidence, unanswerable, adversarial injection, numeric constraints, negation, citation integrity, resource failure).
- **Split Protection**: `validate_dataset` rejects duplicate `case_id`s, fingerprint mismatches, and `dev`/`held_out` question overlap.
