# Grounded RAG Benchmarks (Offline Algorithmic Fixtures)

Run `.\.venv-phase01\Scripts\python.exe benchmarks\rag_benchmark.py` from the repository root.

> **Important:** Dense embeddings and chat responses in this benchmark use deterministic offline `httpx.MockTransport` fixtures. These measurements verify orchestration overhead, citation validation, and abstention mechanics, **not** live LLM intelligence or semantic faithfulness.

## Offline Results (42 QA Cases: 21 `dev`, 21 `held_out`)

| Metric | Measured Result |
| --- | ---: |
| QA Cases | 42 |
| Citation Validity Rate | 1.0000 |
| Expected Evidence Retrieval Recall | 0.8824 |
| Abstention Accuracy | 1.0000 |
| Cold Orchestration Latency | 44.77 ms |
| Median Orchestration Latency | 22.41 ms |
| p95 Orchestration Latency | 74.59 ms |
| Peak Additional Process RAM | 4.17 MiB |

- Orchestration p95 latency (`74.59 ms`) is within the `< 100 ms` provisional target (excluding live model inference).
- Additional process RAM (`4.17 MiB`) is well within the `< 64 MiB` provisional target.
- Live Ollama inference benchmarks (`qwen3-embedding:0.6b` + `qwen3:4b-instruct`) remain pending owner authorization and >3 GiB available RAM.
