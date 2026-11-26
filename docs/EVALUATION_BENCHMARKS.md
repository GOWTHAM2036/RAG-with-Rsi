# Evaluation Benchmarks (Offline Controlled Comparisons)

Run `.\.venv-phase01\Scripts\python.exe benchmarks\evaluation_benchmark.py` from the repository root.

> **Important:** Offline comparisons use deterministic synthetic embeddings and mocked `qwen3:4b` responses. They verify comparison mechanics, statistical aggregation, and quality gates, **not** live semantic intelligence.

## Offline Results (`evorag_comprehensive_v1` `dev` split, 54 cases per run)

| Metric | Measured Result |
| --- | ---: |
| Cases per run | 54 |
| Comparison A (Baseline vs Adaptive) | `improved` (mocked fixture) |
| Comparison B (Baseline vs Reranking) | `improved` (mocked fixture) |
| Baseline p95 latency | 137.88 ms |
| Adaptive p95 latency | 42.16 ms |
| Reranking p95 latency | 35.66 ms |
| Statistical comparison (2x paired bootstrap) | 28.24 ms |
| Additional process RAM | 12.79 MiB |

- Statistical comparison (`28.24 ms`) is well within the `< 100 ms` provisional target.
- Additional process RAM (`12.79 MiB`) is well within the `< 64 MiB` provisional target.
