"""SQLite WAL execution trace store with bounded retention and metadata-only defaults."""

import sqlite3
from pathlib import Path

from .config import TracingSettings
from .observability import application_logger
from .rag_types import RAGExecutionTrace

_DDL = """
CREATE TABLE IF NOT EXISTS rag_traces (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rag_traces_created ON rag_traces(seq);
"""


class TraceStore:
    """Stores structured RAGExecutionTrace records in SQLite WAL with bounded retention."""

    def __init__(self, db_path: Path, settings: TracingSettings, busy_timeout_ms: int = 5000):
        self.db_path = db_path
        self.settings = settings
        self.busy_timeout_ms = busy_timeout_ms
        self._logger = application_logger()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=self.busy_timeout_ms / 1000.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms};")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)

    def save(self, trace: RAGExecutionTrace) -> None:
        """Persist a trace without ever altering answer validity on storage failure."""
        if not self.settings.enabled:
            return
        try:
            payload = trace.model_dump_json()
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO rag_traces (trace_id, created_at, payload_json) "
                    "VALUES (?, ?, ?)",
                    (trace.trace_id, trace.created_at.isoformat(), payload),
                )
                conn.execute(
                    "DELETE FROM rag_traces WHERE seq NOT IN "
                    "(SELECT seq FROM rag_traces ORDER BY seq DESC LIMIT ?)",
                    (self.settings.max_records,),
                )
        except sqlite3.Error as exc:
            self._logger.warning("rag.trace_persist_failed:%s", type(exc).__name__)

    def get(self, trace_id: str) -> RAGExecutionTrace | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM rag_traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        if row is None:
            return None
        return RAGExecutionTrace.model_validate_json(str(row[0]))

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM rag_traces").fetchone()
        return int(row[0]) if row else 0
