# Optimization Algorithms

`backend/evorag/rsi_optimizer.py` implements from-scratch:
- **Coordinate Search**: One-hypothesis-at-a-time parameter changes.
- **Seeded Random Search**: Deterministic NumPy `default_rng(seed)` sampling over `TRUSTED_SEARCH_SPACE`.
- **Local Refinement**: Bounded neighborhood scaling around the current experimental incumbent.
- **Pareto Selection (`rank_candidates_pareto`)**: Non-dominated ranking across quality utility, `p95_latency_ms`, and `additional_ram_mib`.
