# RSI Security

- All feedback, traces, and candidate proposals are treated as untrusted.
- `validate_candidate` enforces strict parameter allowlists, types, bounds, and cross-field constraints, blocking reward-hacking attempts against `resources.*`, `citations.*`, or `eval_*`.
- Promotion and rollback are excluded from unauthenticated HTTP endpoints.
- Held-out evaluation labels are never exposed to `RSIEngine` candidate generation.
