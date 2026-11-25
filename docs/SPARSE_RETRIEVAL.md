# Sparse Retrieval

EvoRag implements a from-scratch lexical analysis engine, SQLite-backed inverted index, and Okapi BM25 scorer under `backend/evorag/analyzer.py`, `backend/evorag/sparse_index.py`, `backend/evorag/bm25.py`, and `backend/evorag/sparse_retrieval.py`.

## Lexical Analyzer (`v1`)

1. Bounds input to `1,000,000` characters, applies Unicode NFKC normalization and `.casefold()`.
2. Preserves compound technical identifiers (`Qwen3-Embedding`, `EVORAG_RUNTIME__BASE_URL`, `vector_store_corrupted`, `Python 3.12`, `/api/v1/retrieve`) alongside their sub-tokens when `preserve_compound_identifiers = true`.
3. Optionally filters stop words (`remove_stopwords = false` by default so identifiers and error codes like `HTTP 404` retain full signal).
4. Enforces `max_query_tokens` (default `128`) and `max_chunk_tokens` (`8192`).
5. **Known limitation:** Uses Unicode word-boundary and identifier patterns; languages without whitespace word boundaries (e.g., Chinese, Japanese, Thai) are not segmented into dictionary words.

## Inverted Index Schema

Stored in the shared SQLite WAL database (`sparse_schema` version `1`):

- `sparse_meta`: `index_generation`, `analyzer_version`, active `doc_count`, active `total_length`.
- `sparse_chunks`: `chunk_id`, `version_key`, `document_id`, `version`, `chunk_length`, `analyzer_version`, `is_latest`.
- `sparse_terms`: `term`, active `df`.
- `sparse_postings`: `(term, chunk_id)` -> `tf`.
- `sparse_index_jobs`: explicit rebuild job status (`running`, `complete`, `failed`).

When a new revision of an existing `document_id` is indexed, prior active chunks are marked `is_latest = 0` and their `df`, `doc_count`, and `total_length` contributions are removed atomically so superseded revisions do not pollute active BM25 statistics.

## BM25 Formula

$$\text{IDF}(t) = \ln\left(1 + \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5}\right)$$

$$\text{BM25}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{\text{tf}(t, D) \cdot (k_1 + 1)}{\text{tf}(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

- Defaults: `k1 = 1.5`, `b = 0.75`, `sparse_top_k = 20`.
- Repeated query terms are summed in order (bounded by `max_query_tokens`).
- Ties break deterministically by ascending `chunk_id`.
