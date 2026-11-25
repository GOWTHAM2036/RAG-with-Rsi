# Retrieval Benchmarks (Offline Algorithmic Fixtures)

Run `.\.venv-phase01\Scripts\python.exe benchmarks\retrieval_benchmark.py` from the repository root.

> **Important:** Because live Ollama inference is pending model authorization, dense embeddings in this benchmark are deterministic 32-dim/64-dim synthetic bag-of-trigrams/random fixtures. Quality numbers measure **algorithmic fixture behavior**, not real semantic retrieval quality.

## Quality Evaluation (54 Queries: 27 `dev`, 27 `held_out`)

Dataset includes exact technical identifiers (`HTTP 404`, `RTX 3050`, `Qwen3-Embedding`, `EVORAG_RUNTIME__BASE_URL`, `vector_store_corrupted`, `Python 3.12`, `/api/v1/retrieve`), semantic questions, document revision replacement (`revisions.md` v1 superseded by v2), and queries with no relevant evidence (scored separately via abstention rate).

| Split | Method | P@5 | R@5 | Hit@5 | MRR | nDCG@5 | No-Evidence Abstain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev` | `dense_only` | 0.217 | 1.000 | 1.000 | 0.809 | 0.846 | 0.000 |
| `dev` | `bm25_only` | 0.732 | 1.000 | 1.000 | 1.000 | 1.000 | 0.667 |
| `dev` | `hybrid_rrf` | 0.217 | 1.000 | 1.000 | 0.979 | 0.981 | 0.000 |
| `dev` | `hybrid_rerank` | 0.217 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| `held_out` | `dense_only` | 0.192 | 0.896 | 0.917 | 0.844 | 0.846 | 0.000 |
| `held_out` | `bm25_only` | 0.774 | 1.000 | 1.000 | 1.000 | 1.000 | 0.667 |
| `held_out` | `hybrid_rrf` | 0.217 | 1.000 | 1.000 | 0.979 | 0.981 | 0.000 |
| `held_out` | `hybrid_rerank` | 0.217 | 1.000 | 1.000 | 0.979 | 0.981 | 0.000 |

## Performance Benchmarks (20 Warm Samples + 1 Cold Sample per Mode)

| Corpus | Index Build (ms) | Incremental 1-doc (ms) | Peak Add RAM (MiB) | Storage (MiB) |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 chunks | 5985.82 | 101.89 | 5.16 | 2.02 |
| 10,000 chunks | 241071.73 | 142.09 | 9.98 | 19.11 |

### Latency & Throughput

| Corpus | Mode | Cold (ms) | Median (ms) | p95 (ms) | Range (ms) | QPS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | `dense` | 133.08 | 20.18 | 104.75 | 16.89–108.16 | 27.0 |
| 1,000 | `sparse` | 29.64 | 28.93 | 39.75 | 19.89–42.24 | 34.3 |
| 1,000 | `hybrid` | 87.55 | 54.48 | 94.42 | 22.97–110.87 | 17.0 |
| 1,000 | `hybrid_rerank` | 51.02 | 63.50 | 76.37 | 45.82–92.53 | 16.1 |
| 10,000 | `dense` | 209.36 | 101.95 | 160.18 | 76.77–165.26 | 9.3 |
| 10,000 | `sparse` | 242.32 | 229.69 | 393.22 | 153.04–481.37 | 4.0 |
| 10,000 | `hybrid` | 304.15 | 313.46 | 498.56 | 249.12–529.91 | 2.9 |
| 10,000 | `hybrid_rerank` | 334.59 | 310.65 | 490.94 | 248.16–524.80 | 3.0 |

### Observations & Bottlenecks

- Peak additional RAM at 10,000 chunks was **9.98 MiB** (well below the 256 MiB target).
- Warm p95 hybrid latency at 10,000 chunks was **498.56 ms** (`490.94 ms` with reranking), meeting the `< 500 ms` provisional target.
- Index build at 10,000 chunks (200 documents) took `241.07 s` because each `index_version` commits a separate SQLite WAL transaction; batching multiple versions per transaction in `rebuild_all` is a clear optimization for corpora >= 50,000 chunks.
- 50,000- and 100,000-chunk benchmarks remain pending.
