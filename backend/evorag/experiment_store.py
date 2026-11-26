"""SQLite WAL persistence for evaluation runs, case outcomes, comparisons and gates."""

import sqlite3
from pathlib import Path

from .config import EvaluationStorageSettings
from .eval_types import (
    EvaluationRun,
    EvaluationRunResult,
    ExperimentComparison,
    QualityGateResult,
    RunStatus,
)

_DDL = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS eval_runs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    config_fingerprint TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    split TEXT NOT NULL,
    execution_mode TEXT NOT NULL,
    started_at TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_case_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    category TEXT NOT NULL,
    split TEXT NOT NULL,
    outcome_json TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES eval_runs(run_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS eval_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_run_id TEXT NOT NULL,
    candidate_run_id TEXT NOT NULL,
    comparison_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_gate_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    gate_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    gate_json TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES eval_runs(run_id) ON DELETE CASCADE
);
"""


class ExperimentStore:
    """Bounded SQLite WAL store for evaluation runs and comparisons."""

    def __init__(
        self,
        db_path: Path,
        storage: EvaluationStorageSettings,
        busy_timeout_ms: int = 5000,
    ) -> None:
        self.db_path = db_path
        self.storage = storage
        self.busy_timeout_ms = busy_timeout_ms
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=self.busy_timeout_ms / 1000.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms};")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)

    def save_run(self, result: EvaluationRunResult) -> None:
        run = result.run
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO eval_runs "
                "(run_id, status, config_fingerprint, dataset_id, dataset_version, "
                "split, execution_mode, started_at, result_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.run_id,
                    run.status,
                    run.config_fingerprint,
                    run.dataset_id,
                    run.dataset_version,
                    run.split,
                    run.execution_mode,
                    run.started_at.isoformat(),
                    result.model_dump_json(),
                ),
            )
            conn.execute("DELETE FROM eval_case_results WHERE run_id = ?", (run.run_id,))
            for case in result.case_outcomes:
                conn.execute(
                    "INSERT INTO eval_case_results "
                    "(run_id, case_id, category, split, outcome_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (run.run_id, case.case_id, case.category, case.split, case.model_dump_json()),
                )
            conn.execute(
                "DELETE FROM eval_runs WHERE seq NOT IN "
                "(SELECT seq FROM eval_runs ORDER BY seq DESC LIMIT ?)",
                (self.storage.max_runs,),
            )

    def get_run(self, run_id: str) -> EvaluationRunResult | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM eval_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return EvaluationRunResult.model_validate_json(str(row[0])) if row else None

    def update_status(self, run_id: str, status: RunStatus) -> EvaluationRun | None:
        existing = self.get_run(run_id)
        if existing is None:
            return None
        updated_run = existing.run.model_copy(update={"status": status})
        updated = existing.model_copy(update={"run": updated_run})
        self.save_run(updated)
        return updated_run

    def save_gates(self, run_id: str, gates: tuple[QualityGateResult, ...]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM eval_gate_results WHERE run_id = ?", (run_id,))
            for gate in gates:
                conn.execute(
                    "INSERT INTO eval_gate_results (run_id, gate_id, decision, gate_json) "
                    "VALUES (?, ?, ?, ?)",
                    (run_id, gate.gate_id, gate.decision, gate.model_dump_json()),
                )

    def save_comparison(self, comparison: ExperimentComparison) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO eval_comparisons (baseline_run_id, candidate_run_id, comparison_json) "
                "VALUES (?, ?, ?)",
                (
                    comparison.baseline_run_id,
                    comparison.candidate_run_id,
                    comparison.model_dump_json(),
                ),
            )
