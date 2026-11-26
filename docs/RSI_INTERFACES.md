# RSI Interfaces (Read-Only Diagnostic Snapshot)

`RSIFeedbackInterface` (`backend/evorag/rsi_interfaces.py`) exposes `RSIFeedbackSnapshot` for Phase 09:

- `config_fingerprint`, `total_traces`, `failure_counts`, `recent_run_ids`, `gate_results`, and `comparisons`.
- Strictly read-only with respect to active configuration, source code, and production indexes.
- Never exposes held-out evaluation labels.
