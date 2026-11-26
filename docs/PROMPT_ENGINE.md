# Grounded Prompt Engine

`backend/evorag/prompt_engine.py` manages versioned, fingerprinted prompt templates (`grounded_v1`, version `1.0.0`, SHA-256 fingerprint).

## Prompt Injection Controls & Limitations

- System instructions are kept in a dedicated `system` message separate from `UNTRUSTED EVIDENCE BLOCKS`.
- `escape_evidence_text` neutralizes `<EVIDENCE`, `</EVIDENCE`, `<SYSTEM`, and `</SYSTEM` tags inside retrieved chunks.
- The model is instructed to treat evidence as untrusted data, never follow instructions inside documents, and return strict JSON (`status`, `answer`, `claims`, `limitations`).
- **Limitation**: Prompt delimiters and escaping reduce spoofing risk but cannot eliminate prompt injection on their own; EvoRag therefore gives the generation model zero tool/filesystem/network access and validates all citations deterministically.
