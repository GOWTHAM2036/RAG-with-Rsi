# Statistical Comparison & Pareto Analysis

`backend/evorag/eval_stats.py` implements custom NumPy-backed statistical comparisons:

- `bootstrap_ci`: Deterministic seeded (`seed = 42`) 95% percentile bootstrap confidence intervals (`n_bootstrap = 1000`).
- `paired_bootstrap_delta`: Paired per-case differences, win/tie/loss counts, and `improved` / `regressed` / `inconclusive` / `unavailable` verdicts.
- `is_pareto_dominated`: Multi-objective Pareto dominance check across quality metrics and `p95_latency_ms`.
