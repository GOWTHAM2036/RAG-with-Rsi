# Adaptive RAG Benchmarks

Run `python benchmarks/adaptive_rag_benchmark.py` for reproducible offline evaluation across 64 cases (`32 dev`, `32 held_out`) covering factual, technical-identifier, comparative, multi-part, multi-document, negation/numeric, ambiguous, adversarial, and insufficient-evidence queries.

> **Disclaimer**: This benchmark uses synthetic 4-dimensional embeddings and mocked Ollama responses. Mocked metrics verify deterministic routing, rank fusion, caching, and budget enforcement, **not** real LLM semantic intelligence.

## Query Analysis Microbenchmark

- Samples: `64`
- Median: `0.057 ms`
- p95: `0.081 ms`
- Range: `0.041 - 0.141 ms`

## Mode Comparison (64 Cases, k=5)

| Mode | P@5 | R@5 | MRR | nDCG@5 | Abstain | Mean Execs | Planning p95 ms | E2E Median ms | E2E p95 ms | Cache Hit Median ms | Add. Peak RAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_hybrid` | 1.0000 | 0.8611 | 1.0000 | 0.8925 | 1.0000 | 1.00 | 0.000 | 51.244 | 106.279 | — | 7.36 |
| `adaptive_deterministic` | 1.0000 | 0.8611 | 1.0000 | 0.8925 | 1.0000 | 1.00 | 0.380 | 45.834 | 95.271 | 17.034 | 2.73 |
| `adaptive_rewriting` | 1.0000 | 0.8611 | 1.0000 | 0.8925 | 1.0000 | 1.00 | 0.301 | 45.610 | 121.332 | 17.473 | 0.63 |
| `adaptive_decomposition` | 1.0000 | 0.8611 | 1.0000 | 0.8925 | 1.0000 | 1.28 | 0.393 | 84.134 | 181.077 | 18.870 | 0.51 |
| `adaptive_multi_query` | 1.0000 | 0.8611 | 1.0000 | 0.8925 | 1.0000 | 1.28 | 0.360 | 99.294 | 222.947 | 18.749 | 0.85 |

## Observations

- Deterministic planning p95 (`0.301 - 0.393 ms`) is well under the `20 ms` target.
- Additional peak RAM (`0.51 - 2.73 MiB`) is well under the `64 MiB` target.
- Decomposition/multi-query executes `1.28` retrievals on average vs `1.00` for baseline, trading higher E2E latency (`99.294 ms` vs `51.244 ms` median) for multi-subquery coverage.
- Because mocked evaluation does not measure live semantic gains, `default_strategy` remains `"baseline"`.
