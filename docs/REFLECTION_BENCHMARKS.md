# Reflection Benchmarks (Offline Fixtures)

Run `.\.venv-phase01\Scripts\python.exe benchmarks\reflection_benchmark.py` from the repository root.

> **Important:** Offline benchmarks use mocked `qwen3:4b` responses and synthetic embeddings; they measure orchestration overhead and deterministic repair mechanics, **not** live semantic intelligence.

| Metric | Measured Result (`40 dev` cases) |
| --- | ---: |
| Trigger Precision | 0.625 |
| Trigger Recall | 1.000 |
| Correction Success Rate | 1.000 |
| Regression Introduction Rate | 0.000 |
| Baseline p95 Latency | 66.07 ms |
| Reflective Median Latency | 37.08 ms |
| Reflective p95 Latency | 113.27 ms |
| Additional Process RAM | 5.06 MiB |
