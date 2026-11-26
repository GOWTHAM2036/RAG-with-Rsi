# Quality Gates

`backend/evorag/quality_gates.py` evaluates configurable `QualityGate` thresholds returning `pass`, `fail`, `inconclusive`, or `blocked`:

- Missing required metrics return `inconclusive` (never auto-pass).
- Gates with `require_live_mode=True` return `blocked` when `execution_mode != "live"`, preventing mocked scores from passing live semantic gates.
- `experiments.automatic_promotion` is locked to `False`.
