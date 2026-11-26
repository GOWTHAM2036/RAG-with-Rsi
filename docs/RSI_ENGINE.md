# RSI Engine (Phase 09)

`RSIEngine` (`backend/evorag/rsi_engine.py`) orchestrates bounded, evidence-driven Recursive Self-Improvement over configuration parameters without modifying source code, neural weights, or active runtime state automatically.

- **Disabled by Default**: `[rsi].enabled = false`, `automatic_promotion = false`, `automatic_rollback = false`, `automatic_scheduling = false`.
- **Global Budgets**: `max_cycles = 2`, `max_candidates_per_cycle = 3`, `max_experiments = 6`, `max_execution_seconds = 300.0`.
- **API & CLI**: `GET /api/v1/rsi/{status,recommendations,configurations}`, `POST /api/v1/rsi/cycles`, `GET /api/v1/rsi/cycles/{id}`, `POST /api/v1/rsi/cycles/{id}/cancel`, `GET /api/v1/rsi/cycles/{id}/report`, and `python -m evorag.rsi_cli`. Promotion and rollback are owner-operated CLI/internal workflows only, never exposed on the unauthenticated API.
