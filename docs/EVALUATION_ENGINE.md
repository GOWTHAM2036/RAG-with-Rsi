# Evaluation Engine (Phase 07)

`ExperimentRunner` (`backend/evorag/experiment_runner.py`) coordinates bounded, sequential evaluation runs over versioned datasets without mutating active production configuration or user document indexes.

- **Execution Modes**: `offline`, `synthetic`, `live`, `human_annotated` (clearly labeled in every report).
- **Concurrency & Bounds**: `max_concurrent_runs = 1`, `max_cases_per_run = 200`, cancellation via `POST /api/v1/evaluation/runs/{run_id}/cancel`.
- **CLI**: `python -m evorag.eval_cli {list-datasets,validate-datasets,show-run}`.
