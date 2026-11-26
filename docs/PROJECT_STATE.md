# Phase 01 handoff

## Workspace audit · 25 November 2026

The working directory was `C:\Users\B Gowtham Reddy\.vscode\EvoRag`. Local main and cached origin/main started at `4a05d78`. The index was clean. The working tree already contained 85 unstaged deletions and three edited files. The README contained a title, .gitignore contained basic Python/environment/IDE rules, and requirements.txt was empty. The tracked predecessor had ingestion, retrieval, optimization, frontend, Docker and benchmark files. Their worktree deletions were preserved.

The configured fetch and push URL is `https://github.com/GOWTHAM2036/RAG-With-RSI.git`. The live remote query for HEAD and all branch refs returned successfully with no advertised refs. The cached origin/main ref is historical. Review repository ownership, remote state and the pre-existing deletion set before publishing.

| Item | Observation |
| --- | --- |
| Python | System executable and existing .venv both 3.10.11. Separate Python 3.12.11 available. Phase 01 uses .venv-phase01. |
| Node and package tools | Node 24.14.1, npm 11.12.1, pip and pnpm present. No frontend install. |
| CPU / memory | i7-12650H, 16 logical processors. Node OS API reported 16,802,746,368 bytes total and 1,695,272,960 free during audit. CIM access was restricted. |
| GPU / storage | nvidia-smi reported RTX 3050 A Laptop GPU, 4,094 MiB total and 3,892 MiB free. Initial C: free space 108,497,555,456 bytes. These are point-in-time measurements. |
| Ollama | CLI 0.40.16. Host configuration uses 127.0.0.1:11550. The application default is localhost:11434. Set EVORAG_RUNTIME__BASE_URL for this host. |
| Installed tags from live daemon | qwen3:8b, qwen2.5:7b, qwen3:1.7b-q4_K_M, gemma3:4b, qwen3:4b, qwen2.5:1.5b-instruct-q4_K_M. |

The requested generation and embedding tags exist in the public registry. Neither exact name appeared in the live daemon inventory. Local readiness reports presence by name; it never assumes that similar names or different quantizations can substitute. No models were downloaded or loaded. Registry references are in [DECISIONS.md](DECISIONS.md).

## Delegation and review

Antigravity's help advertised model selection, session permissions, sandboxed plan mode and noninteractive prompts. Authenticated `agy models` listed Hydrogen Low, Medium, High and XHigh; `agy agents` returned no named profiles. Three distinct model-selected CLI sessions completed, with read-only ownership. The orchestrator made and reviewed all workspace edits. The existing dirty checkout stayed on its original branch.

| Session | Ownership and outcome |
| --- | --- |
| Hydrogen High · core | Proposed frozen typed contracts, provenance, coordinate validation and async ports. Integrated references and span checks appropriate to the foundation. |
| Hydrogen Medium · quality/security | Set acceptance cases for offline/lifecycle isolation, malformed metadata, contention, unavailable telemetry and sensitive data. |
| Hydrogen High · infrastructure | Independently inspected application source and packaging. Integrated cloud endpoint validation, default-tag normalization, package version metadata, public typing marker and full experiment settings snapshots. |

Application readiness stays independent of model inventory by design. Concurrent probes report busy. A redirect must produce a bounded failed inventory check, as covered by an integration test. Credentials and model residency move to the inference milestone. The storage guard takes an existing volume directory. These decisions resolve the other review suggestions within Phase 01 scope.

## Delivered files

Root additions include AGENTS.md, ROADMAP.md, CHANGELOG.md, pyproject.toml and .env.example. README.md and .gitignore were extended. The Python package contains config, contracts, errors, interfaces, inventory, provider_payloads, main, observability, resources, the package initializer and py.typed. Tests cover configuration, contracts, resources and status endpoints. configs/example.toml demonstrates partial overrides. docs contains architecture, development, decisions, resource budget, security and this handoff. The GitHub workflow is .github/workflows/foundation.yml.

## Reproduce the audit

| Purpose | Commands |
| --- | --- |
| Workspace and Git | `rg --files --hidden` with Git/venv excluded, `git status --short --branch`, `git remote -v`, `git branch -vv`, `git show HEAD:backend/app/main.py`, `git diff --cached --stat`, `git ls-remote --symref origin HEAD 'refs/heads/*'` |
| Tooling | `Get-Command`, `python --version`, `python3.12 --version`, `node --version`, `npm.cmd --version`, `agy --help`, `agy models`, `agy agents`, `ollama --version`, `ollama list` |
| Resources | Node os.totalmem/freemem/cpus, `Get-PSDrive C`, `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader` |
| Agent sessions | `agy --model <verified-id> --sandbox --mode plan --print-timeout <limit> --print <assignment>`. CLI authentication and Ollama socket access required running with approved host access. |
| Setup | `python3.12 -m venv .venv-phase01`, then `python -m pip install -e ".[dev]"` with the new environment's interpreter. The initial sandboxed install encountered DNS failure while fetching setuptools; the retry used approved index access. |

See [DEVELOPMENT.md](DEVELOPMENT.md) for verification commands and [ROADMAP.md](../ROADMAP.md) for the proposed ingestion milestone. Project owner approval is required before starting Phase 02.

## Verified outcome

| Check | Local Windows result |
| --- | --- |
| Offline unit and integration suite | 34 passed in 1.76 s. Branch-inclusive foundation coverage **97.14%** against an 80% gate. Test HTTP calls used in-process transports. |
| Format, lint, static types | Ruff format and lint passed. Strict mypy passed on all 14 source and test files. |
| Dependency integrity | pip check reported no broken requirements. Runtime packages installed directly were FastAPI 0.165.0, HTTPX 0.28.1, Pydantic 2.14.4, Uvicorn 0.54.0 and psutil 7.2.2. Development tools included pytest 9.2.0, pytest-cov 7.1.0, Ruff 0.17.6, mypy 2.4.1 and types-psutil 7.2.2.20261122. |
| Live HTTP smoke test | Started Uvicorn on an ephemeral loopback port, set the upstream to 127.0.0.1:11550, made real HTTP requests, and shut the server down. GET /health returned 200 and ok. GET /ready returned 200, application ready, inventory available and both installed flags false. |
| Resource reporter | python -m evorag.resources measured 423,489,536 available RAM bytes and 108,329,398,272 available disk bytes and detected the NVIDIA GPU. At that sample the RAM floor would decline a generation. |
| Security and Git | Sample private input, credential, database, vector, virtual environment and benchmark-result paths were ignored. A filename-only scan of source and config found no matching common private-key or access-token patterns. No prohibited RAG dependency was added. git diff --check passed. |

The initial sandboxed asyncio run hung creating Windows' event-loop socket pair. A traceback located the wait in socket.accept during loop initialization. The final offline suite and live socket smoke test ran with approved native loopback access. The final suite uses HTTPX ASGITransport directly and emitted no warnings.

Final status on local main: 85 pre-existing deleted paths, three modified tracked paths and 28 new untracked files. The requirements.txt edit was pre-existing. The index stayed empty; no commit, push, branch deletion or history change occurred. Hosted CI is configured and has not run. Missing local target tags and installation-time dependency downloads do not block foundation use; live vector width and inference remain unmeasured until the provider milestone. Phase 01 is complete locally.

## Phase 02 handoff · 26 November 2026

Phase 02 passed local Windows verification on Python 3.12.11. Preflight reproduced 34 Phase 01 tests and 97.14% branch-inclusive coverage, Ruff, mypy and pip check. Ollama inventory was reachable at 127.0.0.1:11550; both configured tags were absent. Ingestion used CPU and SQLite without inference or new dependencies.

The final offline suite passed **126 tests in 7.46 seconds** with **95.55% overall branch-inclusive coverage**. Module results rounded by coverage.py were parsing 93%, chunking 94%, orchestration 99%, repository 99% and document API 97%. Ruff format and lint passed, strict mypy passed on 27 files including the benchmark, and pip check found no broken requirements. Hosted CI was not run.

| Synthetic input | Chunks | First ingestion ms | Duplicate ms | SQLite publication ms | Chunk MiB/s | Sampled peak RSS MiB | Additional MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 KiB | 9 | 159.97 | 60.58 | 58.52 | 4.64 | 37.93 | 1.48 |
| 100 KiB | 84 | 207.19 | 62.22 | 92.50 | 10.12 | 41.02 | 4.50 |
| 1 MiB | 857 | 457.32 | 54.05 | 351.21 | 14.74 | 67.91 | 31.56 |

The initial preflight measured 1.50 GiB available RAM. A benchmark attempt below the 3 GiB floor correctly declined admission. On retry, the resource reporter measured 3,593,748,480 available RAM bytes and 108,362,756,096 free disk bytes. The benchmark then completed all three sizes with production admission enabled. These single-run timings use temporary databases and 2 ms RSS sampling.

Hydrogen High through agy produced parsing.py and parser tests before its CLI print deadline expired. The orchestrator corrected and verified that work. A sequential Hydrogen Medium session refined repository.py and added transaction, migration, concurrency and resource tests. Automatic approval review rejected supplying project files to external native child agents for independent review because source might be sensitive. A Hydrogen Medium parent used native invoke_subagent twice in an empty temporary workspace with synthetic questions. Both finished: Unicode reviewer 0eacfd72-c336-41be-9737-a5c80be73be7 and SQLite reviewer 85ad1363-e9d8-4a53-befe-9f721ff29698. The orchestrator reviewed the actual implementation locally. Checks covered normalized-to-original section spans, bounded streams, local file identity, malformed structured inputs, transaction rollback and duplicate races.

Changes comprise six application modules (parsing, chunking, documents, repository, ingestion and document_api), six new test files, the synthetic benchmark, three guides (INGESTION, CHUNKING and STORAGE), and updates to main, config, resources, errors, example settings, pytest coverage configuration, README, roadmap, changelog and architecture/security/resource/decision documentation. Git remains on main at 4a05d78 with 85 pre-existing deletions, three modified tracked paths and 44 untracked files in total. The index is empty. No commit, push or branch change occurred.

Use the README upload example, the development guide verification commands and the resource budget benchmark command. The API is loopback-oriented, unauthenticated and single-namespace. SQLite stores plaintext. Token counts are estimates and structured provenance is row- or pointer-based. Owner approval is required before Phase 03 CPU embedding and exact retrieval.

## Phase 03 audit & Phase 04 handoff · 26 November 2026

### Phase 03 completeness audit (verified)

Preflight reproduced **187 passing offline tests in 21.56 s** with **96.09%** branch-inclusive coverage (`embedding_store.py` 100%, `retrieval.py` 100%, `similarity.py` 100%, `vector_store.py` 97%), Ruff clean, strict mypy clean on 41 files, and `pip check` clean. All Phase 03 offline provider requirements were present: streaming NDJSON generation (`/api/v1/generate/stream`), SQLite embedding cache with SHA-256 verification and FIFO eviction, resumable leased embedding jobs, client-disconnect and job cancellation, and `Coordinator` resource/queue admission. Pending from Phase 03: live Ollama verification (`qwen3-embedding:0.6b` and `qwen3:4b-instruct` not downloaded/verified live) and benchmarks above 10,000 vectors.

### Phase 04 verified outcome (offline)

| Check | Local Windows result |
| --- | --- |
| Offline unit and integration suite | **203 passed in 36.41 s**. Branch-inclusive coverage **95.22%** (gate 90%). New modules: `analyzer.py` 98%, `bm25.py` 97%, `evaluation.py` 91%, `filtering.py` 100%, `fusion.py` 91%, `hybrid_retrieval.py` 94%, `reranking.py` 95%, `retrieval.py` 98%, `similarity.py` 92%, `sparse_index.py` 91%, `sparse_retrieval.py` 97%. |
| Format, lint, static types | `ruff check backend`, `ruff format --check backend`, and strict `mypy` passed on 56 source files. `pip check` reported no broken requirements. |
| Resource snapshot | `python -m evorag.resources`: 1,760,403,456 available RAM bytes, 107,818,053,632 free disk bytes, `NVIDIA GeForce RTX 3050 A Laptop GPU`. |
| Benchmarks | 54-query offline fixture evaluation + 1,000/10,000-chunk latency/RAM benchmarks recorded in [RETRIEVAL_BENCHMARKS.md](RETRIEVAL_BENCHMARKS.md). |

Live Ollama model verification and 50k/100k-chunk benchmarks remain pending owner authorization and sufficient free RAM. Owner approval is required before Phase 05 grounded generation.

## Phase 05 handoff · 26 November 2026

Phase 05 offline grounded generation and end-to-end RAG passed local Windows verification on Python 3.12.11. Preflight reproduced 203 Phase 04 tests (95.22% coverage), Ruff, strict mypy (56 files), and `pip check`. Available RAM was ~2.25 GiB (below the 3 GiB floor) and `qwen3:4b-instruct` / `qwen3-embedding:0.6b` were not installed, so live model verification remains pending owner authorization.

| Check | Local Windows result |
| --- | --- |
| Offline unit and integration suite | **224 passed**, branch-inclusive coverage **>= 95%** (gate 90%), all new Phase 05 modules >= 90%. |
| Format, lint, static types | `ruff check backend benchmarks`, `ruff format --check backend benchmarks`, strict `mypy` (68 source files), and `pip check` all passed. |
| Offline RAG QA & orchestration benchmark | 42 QA cases (`21 dev`, `21 held_out`): citation validity `1.0000`, expected evidence recall `0.8824`, abstention accuracy `1.0000`, cold `44.77 ms`, median `22.41 ms`, p95 `74.59 ms`, additional RAM `4.17 MiB`. |

New Phase 05 modules: `rag_types.py`, `context_engine.py`, `prompt_engine.py`, `conversation.py`, `citations.py`, `admission.py`, `tracing.py`, `rag_service.py`, plus `/api/v1/rag/query`, `/api/v1/rag/query/stream`, `/api/v1/rag/traces/{trace_id}`, `/api/v1/rag/status`, and `benchmarks/rag_benchmark.py`. Owner approval is required before Phase 06 evaluation/UI/deployment and before live Ollama verification.

## Phase 06 handoff · 26 November 2026

Phase 06 offline adaptive RAG and query intelligence passed local Windows verification on Python 3.12.11. Preflight reproduced all 224 Phase 05 tests (`95.28%` coverage), Ruff, strict mypy (68 files), and `pip check`. Live Ollama inspection confirmed `qwen3:4b` (`4.0B`, `Q4_K_M`, digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`, capabilities `completion`/`tools`/`thinking`), and `Ollama._chat_body` enforces `think: false`. `configs/example.toml` was updated with `generation_model = "qwen3:4b"` while keeping `"qwen3:4b-instruct"` as the `ModelConfiguration` schema default so Phase 03 unit tests remain untouched. `qwen3-embedding:0.6b` remains absent and pending owner-authorized download; available RAM was ~3.01 GiB (`3,230,195,712` bytes), so live end-to-end verification remains pending.

| Check | Local Windows result |
| --- | --- |
| Offline unit and integration suite | **242 passed**, branch-inclusive coverage **94.95%** (gate 90%), all new Phase 06 modules >= 90%. |
| Format, lint, static types | `ruff check backend benchmarks`, `ruff format --check backend benchmarks`, strict `mypy` (79 source files), and `pip check` all passed. |
| Adaptive RAG benchmark (`benchmarks/adaptive_rag_benchmark.py`) | 64 cases (`32 dev`, `32 held_out`): query analysis median `0.057 ms` (p95 `0.081 ms`), deterministic planning p95 `0.301 - 0.393 ms`, cache-hit median `17.03 - 18.87 ms`, additional peak RAM `0.51 - 2.73 MiB`, synthetic/mocked `P@5 = 1.0000`, `R@5 = 0.8611`, `MRR = 1.0000`, `nDCG@5 = 0.8925`. |

New Phase 06 modules: `query_types.py`, `query_analysis.py`, `query_rewriting.py`, `query_decomposition.py`, `query_planner.py`, `retrieval_cache.py`, `multi_query_retrieval.py`, `evidence_consolidation.py`, plus `POST /api/v1/rag/plan`, `strategy="adaptive"` on `POST /api/v1/rag/query`, and `benchmarks/adaptive_rag_benchmark.py`. `default_strategy` remains `"baseline"` until live model evaluation demonstrates semantic gains. Owner approval is required before live Ollama verification, Phase 07 evaluation engine, frontend development, or deployment.

## Phase 07 handoff · 26 November 2026

Phase 07 offline evaluation and experimentation engine passed local Windows verification on Python 3.12.11. Preflight reproduced all 244 Phase 06 tests (`94.97%` coverage), Ruff, strict mypy (83 files), and `pip check`. Resolved generation-model configuration debt by migrating `ModelConfiguration.generation_model` default from `qwen3:4b-instruct` to the installed `qwen3:4b` (`Q4_K_M`, digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`). `qwen3-embedding:0.6b` remains absent and pending owner-authorized download.

| Check | Local Windows result |
| --- | --- |
| Offline unit and integration suite | **249 passed**, branch-inclusive coverage **>= 94%** (gate 90%), all new Phase 07 evaluation modules >= 90%. |
| Format, lint, static types | `ruff check backend benchmarks`, `ruff format --check backend benchmarks`, strict `mypy` (95 source files), and `pip check` all passed. |
| Evaluation benchmark (`benchmarks/evaluation_benchmark.py`) | `evorag_comprehensive_v1` (108 cases: `54 dev`, `54 held_out`): 54 dev cases/run, statistical comparison `28.24 ms`, additional peak RAM `12.79 MiB`, `automatic_promotion = False`. |

New Phase 07 modules: `eval_types.py`, `dataset_registry.py`, `eval_metrics.py`, `failure_analysis.py`, `eval_stats.py`, `quality_gates.py`, `experiment_store.py`, `experiment_runner.py`, `eval_cli.py`, `/api/v1/evaluation/*` endpoints, and `benchmarks/evaluation_benchmark.py`. Owner approval is required before live Ollama verification or Phase 08 self-reflective RAG.

## Phase 08 handoff · 26 November 2026

Phase 08 offline self-reflective RAG and bounded correction engine passed local Windows verification on Python 3.12.11. Preflight reproduced all 249 Phase 07 tests (`94.95%` coverage), Ruff, strict mypy (95 files), and `pip check`. Reflection remains disabled by default (`reflection.enabled = false`, `reflection.semantic.enabled = false`), and `qwen3-embedding:0.6b` remains absent and pending owner-authorized download.

| Check | Local Windows result |
| --- | --- |
| Offline unit and integration suite | **256 passed**, branch-inclusive coverage **>= 94%** (gate 90%). |
| Format, lint, static types | `ruff check backend benchmarks`, `ruff format --check backend benchmarks`, strict `mypy` (104 source files), and `pip check` all passed. |
| Reflection benchmark (`benchmarks/reflection_benchmark.py`) | 80-case dataset (`40 dev`, `40 held_out`): trigger recall `1.0`, correction success `1.0`, regression rate `0.0`, reflective median `37.08 ms`, p95 `113.27 ms`, additional RAM `5.06 MiB`. |

New Phase 08 modules: `reflection_types.py`, `reflection_state.py`, `verification.py`, `correction_planner.py`, `reflection_engine.py`, `rsi_interfaces.py`, `GET /api/v1/rag/reflection/status`, and `benchmarks/reflection_benchmark.py`. Owner approval is required before live Ollama verification or Phase 09 RSI.

## Phase 09 handoff · 26 November 2026

Phase 09 offline Recursive Self-Improvement (RSI) engine passed local Windows verification on Python 3.12.11. Phase 08 hardening raised `correction_planner.py` and `reflection_engine.py` coverage above 90%. RSI remains disabled by default (`rsi.enabled = false`, `automatic_promotion = false`, `automatic_rollback = false`), and `qwen3-embedding:0.6b` remains absent and pending owner-authorized download.

| Check | Local Windows result |
| --- | --- |
| Offline unit and integration suite | **262 passed**, branch-inclusive coverage **>= 94%** (gate 90%). |
| Format, lint, static types | `ruff check backend benchmarks`, `ruff format --check backend benchmarks`, strict `mypy` (115 source files), and `pip check` all passed. |
| RSI benchmark (`benchmarks/rsi_benchmark.py`) | 30 scenarios + 3 bounded experiments: candidate gen p95 `0.235 ms`, validation p95 `0.010 ms`, additional RAM `7.89 MiB`. |

New Phase 09 modules: `rsi_types.py`, `rsi_state.py`, `rsi_search_space.py`, `rsi_analysis.py`, `rsi_optimizer.py`, `rsi_store.py`, `rsi_engine.py`, `rsi_cli.py`, `/api/v1/rsi/*` read-only/bounded endpoints, and `benchmarks/rsi_benchmark.py`. Owner approval is required before live Ollama verification or Phase 10 optimization/UI/deployment.

## Phase 10 handoff · 26 November 2026

Phase 10 (System Hardening, Live Verification and Performance Optimization) passed local Windows verification on Python 3.12.11:
- **Hardening & 2-Cycle RSI**: Closed remaining coverage gaps in `rsi_search_space.py`, `rsi_analysis.py`, `rsi_optimizer.py`, `rsi_engine.py`, and `reflection_engine.py`; verified 2-cycle recursive improvement (`cycles_completed == 2`) with strict experiment isolation (`active_version` and runtime `settings` unchanged).
- **Performance Optimization**: Reduced SQLite WAL transaction overhead in `SparseIndex.rebuild_all` by batching job progress updates every 25 document versions.
- **Live Admission Verification**: Checked `Coordinator.admission('qwen3:4b')` against live Ollama (`127.0.0.1:11550`): `admitted = False`, `reason = 'insufficient_memory'` (`available_ram_bytes = 1,887,117,312` < `required_ram_bytes = 6,792,261,227`). The 3 GiB floor was preserved and no models were downloaded.

Owner approval is required before frontend development or deployment.

## Frontend UI & Knowledge Explorer · 26 November 2026

Implemented `frontend/` (React 18, TypeScript, Vite, Tailwind CSS) and bounded `GET /api/v1/graph/provenance` (`backend/evorag/graph_api.py`, `backend/evorag/main.py` CORS for `http://127.0.0.1:5173` and `http://localhost:5173`, `backend/tests/integration/test_graph_api.py`):
- **Four Signature Views**: `Knowledge Explorer` (3-panel force-directed 2D graph with zoom/pan, search, cluster/type filters, density slider, low-perf toggle, camera focus/reset, fullscreen, accessible table view, query-driven pulse/particle animation), `Command Center` (live status + `POST /api/v1/rag/query/stream` console), `Pipeline Explorer`, and `RSI Evolution Tree` (`v0_baseline` lineage and owner CLI workflow; no unauthenticated promotion/rollback endpoints).
- **Backend Verification**: `ruff format`, `ruff check`, strict `mypy` (117 source files), and `pytest` (**263 passed**, **94.48%** branch-inclusive coverage).
- **Frontend Verification**: `npm.cmd run build` (`tsc -b && vite build`) succeeded cleanly (`dist/assets/index-vX4Ss4F4.js`, `178.23 kB`).

## Frontend UI Phase 11.2 · 26 November 2026

Upgraded `frontend/` to the **EvoRAG Neural Intelligence Console** (`Retrieval. Reflection. Evolution.`):
- **Shell & Design System**: Collapsible grouped sidebar, `EvoRagLogo`, `CommandPalette` (`Ctrl+K`), `ShellDrawer` (notifications, global inspector, appearance/performance toggles), `PlannedPageView` (honest `PLANNED` badges for Phase 11.3+ pages), and `Primitives.tsx`.
- **Signature Views**:
  - `CommandCenter.tsx` (default landing view, asymmetrical bento grid + `LivingRagCore.tsx` 3D-projected neural sphere with 2D fallback and offscreen `IntersectionObserver` suspension).
  - `KnowledgeExplorer.tsx` (240+ node deterministic demo corpus in `src/data/demoCorpus.ts`, cluster isolation, progressive labels, 1-hop/2-hop neighbor expansion, 2D/3D depth toggle, accessible table fallback, explicit `DEMO` vs `LIVE` separation).
  - `PipelineExplorer.tsx` (10-stage interactive execution DAG, timeline scrubber, static architecture vs recorded execution distinction).
  - `RSIEvolutionTree.tsx` (configuration lineage + side-by-side candidate comparison with statistical intervals and owner CLI workflow; no HTTP promotion/rollback controls).
- **Frontend Checks**: `npm.cmd run typecheck` (`tsc --noEmit`), `npm.cmd run test` (`vitest run`: 3 passed), `npm.cmd run build` (`238.23 kB` JS / `70.30 kB` gzipped), and Playwright browser verification across `1440x900`, `1280x800`, `768x1024`, and `390x844`.

## Frontend UI Phase 11.3 · 26 November 2026

Implemented **RAG Playground**, **Document Management**, and **Retrieval Lab** with strict `LIVE` vs `DEMO` separation (explicit user-consented switching; never silently mixing live and synthetic data):
- **New Components**: `RAGPlayground.tsx` (lazy), `DocumentManagement.tsx` (lazy), `RetrievalLab.tsx` (lazy), `utils/markdown.tsx` (XSS-safe Markdown renderer), expanded `api.ts` (`consumeNdjsonStream`, `ApiError`, document/embedding/retrieve/trace endpoints) and `demoCorpus.ts`.
- **Cross-View Integration**: Citations navigate to Document Management & Knowledge Explorer; traces link to Pipeline Explorer; Document Management & Retrieval Lab jump directly to Knowledge Explorer nodes.
- **Verification**: `npm.cmd run typecheck` passed; `npm.cmd run test` passed (**6/6 tests**); `npm.cmd run build` passed (`249.50 kB` main JS / `73.75 kB` gzipped + lazy chunks); Playwright verified RAG Playground, Citation Inspector, Document Management, Retrieval Lab, and mobile (`390x844`) navigation.

Owner approval is required before Evaluation Arena, Reflection Lab, and Experiment Lab (Phase 11.4).
