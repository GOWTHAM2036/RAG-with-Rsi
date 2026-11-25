# Development

## Environment and checks

Install from the repository root using the [README commands](../README.md). The empty legacy requirements.txt is present in the owner's baseline. pyproject.toml supplies the runtime dependencies and the dev extra.

```powershell
$python = '.\.venv-phase01\Scripts\python.exe'
& $python -m ruff format --check backend
& $python -m ruff check backend
& $python -m mypy
& $python -m pytest -q
& $python -m evorag.resources
```

CI installs the package and runs the same format, lint, typing and test gates on Windows/Python 3.12 and Linux/Python 3.11. Coverage includes branches, with an 80% minimum. Tests inject HTTP transports and hardware readings. The Windows command to run the app is in the README. For a Unix environment use `python3.11 -m venv .venv-phase01` then `.venv-phase01/bin/python` in place of the Windows executable.

## Configuration

Load order is validated defaults, optional TOML, then process environment. Point `EVORAG_CONFIG` at a TOML file. Python callers may pass a path directly to `load_settings`. Environment keys use `EVORAG_SECTION__FIELD`; typos and unknown fields fail validation. Set variables in the shell or service manager. The example environment file is a reference. Invalid values fail startup with field locations and error codes.

```powershell
$env:EVORAG_CONFIG = 'configs/example.toml'
$env:EVORAG_RUNTIME__BASE_URL = 'http://localhost:11434'
$env:EVORAG_CHUNKING__CHUNK_SIZE = '512'
```

| Section | Defaults |
| --- | --- |
| runtime | ollama, http://localhost:11434, qwen3:4b-instruct, qwen3-embedding:0.6b, context 4096, output 768, temperature 0.1, concurrency 1 |
| embeddings | cpu, batch 8, dimensions 1024 |
| chunking | size 384, overlap 64 |
| retrieval | dense 20, sparse 20, final 5, RRF constant 60 |
| reflection | disabled, at most one retry |
| rsi | disabled, automatic promotion disabled, human approval required |
| storage | sqlite, custom_memmap, float32 |
| resources | one generation, one loaded LLM, 3 GiB RAM floor, 20 GiB project budget |

Run `ollama list` against the intended daemon and set the runtime base URL to its address. Use `GET /ready` to see local installation status. An untagged override follows Ollama's `:latest` convention. Named providers other than Ollama report unsupported inventory in this milestone. OpenAI and Gemini configurations require an explicit HTTPS URL. Credentials and inference adapters belong to their provider milestones.

## Extending the foundation

Use the ports in `evorag.interfaces` and frozen payloads in `evorag.contracts`. Fake transports belong in tests. Wrap future inference in `ResourceManager.generation_slot()` and call `check_storage` with the existing target volume directory, accounted project size and maximum expected growth before a serialized write. The current telemetry command works with a missing GPU or restricted sensors. See [resource units and operating limits](RESOURCE_BUDGET.md).
