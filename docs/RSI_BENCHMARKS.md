# RSI Benchmarks (Offline Fixtures)

Run `.\.venv-phase01\Scripts\python.exe benchmarks\rsi_benchmark.py` from the repository root.

> **Important:** Offline benchmarks use mocked `qwen3:4b` responses and synthetic embeddings; they measure optimizer overhead and cycle mechanics, **not** live real-world intelligence improvement.

| Metric | Measured Result (`30` scenarios + 3 bounded experiments) |
| --- | ---: |
| Candidate Generation Median | 0.127 ms |
| Candidate Generation p95 | 0.235 ms (target `< 50 ms`) |
| Candidate Validation p95 | 0.010 ms (target `< 20 ms`) |
| Additional Process RAM | 7.89 MiB (target `< 64 MiB`) |
| Exp A / Exp B / Exp C | `inconclusive` / `inconclusive` / `1` cycle |
