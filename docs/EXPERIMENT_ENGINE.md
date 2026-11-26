# Experiment Engine & Persistence

`ExperimentConfiguration` (`backend/evorag/eval_types.py`) captures an immutable snapshot of dataset, retrieval, context, prompt, and model settings with a canonical SHA-256 `fingerprint()`.

`ExperimentStore` (`backend/evorag/experiment_store.py`) persists runs, case outcomes, comparisons, and quality gate results in SQLite WAL (`eval_runs`, `eval_case_results`, `eval_comparisons`, `eval_gate_results`) with `PRAGMA foreign_keys=ON` and bounded retention (`max_runs = 100`).
