# Reflection Engine (Phase 08)

`ReflectionEngine` (`backend/evorag/reflection_engine.py`) orchestrates bounded post-generation verification and correction without mutating active system configurations, source code, or production indexes.

- **Disabled by Default**: `[reflection].enabled = false`, `mode = "deterministic"`, `semantic.enabled = false`.
- **Global Budgets**: `max_cycles = 1`, `max_critique_calls = 1`, `max_correction_calls = 1`, `max_additional_retrievals = 2`, `max_total_model_calls = 3`.
- **API**: `GET /api/v1/rag/reflection/status` and `reflection_enabled` / `reflection_mode` / `include_reflection_summary` on `POST /api/v1/rag/query`.
