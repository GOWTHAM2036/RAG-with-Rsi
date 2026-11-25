# EvoRag

EvoRag is a Windows-first framework for building and evaluating a resource-bounded RAG pipeline. Phase 01 provides a FastAPI service, validated configuration, typed data contracts, hardware reporting and generation admission control. Retrieval and inference start in later milestones.

## Run locally

From the repository root, use Python 3.11 or newer. This checkout has Python 3.12 available as `python3.12`.

```powershell
python3.12 -m venv .venv-phase01
.\.venv-phase01\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv-phase01\Scripts\python.exe -m uvicorn evorag.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Open [health](http://127.0.0.1:8000/health), [readiness](http://127.0.0.1:8000/ready) or the [interactive API documentation](http://127.0.0.1:8000/docs). Health returns `{"status":"ok"}` without accessing Ollama. Readiness returns HTTP 200 once the application has initialized and reports model inventory separately. Use Ctrl+C to stop the server.

## Ingest a document

With the single-worker service running, upload a local UTF-8 file as a raw request body. The filename identifies the logical source. Both the body and the running service stay on the local machine.

```powershell
$api = 'http://127.0.0.1:8000/api/v1'
$result = Invoke-RestMethod -Method Post -Uri "$api/documents" -InFile '.\README.md' -ContentType 'application/octet-stream' -Headers @{'X-Filename'='README.md'}
Invoke-RestMethod "$api/documents/$($result.document_id)/versions"
Invoke-RestMethod "$api/documents/$($result.document_id)/chunks?version=$($result.version)&limit=5"
```

TXT, Markdown, CSV and JSON use the same ingestion pipeline. A new revision returns 201 and an identical revision returns 200. A 503 resource response means admission could not confirm capacity. See the [ingestion guide](docs/INGESTION.md), [chunking algorithms](docs/CHUNKING.md), [SQLite schema](docs/STORAGE.md), [sparse retrieval](docs/SPARSE_RETRIEVAL.md), [hybrid retrieval](docs/HYBRID_RETRIEVAL.md), [reranking](docs/RERANKING.md) and [retrieval benchmarks](docs/RETRIEVAL_BENCHMARKS.md).

## Retrieve chunks

```powershell
Invoke-RestMethod -Method Post -Uri "$api/retrieve" -ContentType 'application/json' -Body '{"query":"HTTP 404","mode":"hybrid","top_k":5,"include_explanations":true}'
Invoke-RestMethod "$api/index/status"
```

## Repository guide

| Path | Contents |
| --- | --- |
| `backend/evorag/` | Application, provider inventory, contracts and resource controls |
| `backend/tests/` | Deterministic unit and ASGI integration tests |
| `configs/example.toml`, `.env.example` | Optional file and process environment overrides |
| `docs/` | Architecture, decisions, development, security, hardware budget and verified project state |
| `.github/workflows/foundation.yml` | Windows and Linux verification |

Start with the [development guide](docs/DEVELOPMENT.md) for configuration and test commands. The [roadmap](ROADMAP.md) gives milestone dependencies and the [project state](docs/PROJECT_STATE.md) records measured results and open issues.
