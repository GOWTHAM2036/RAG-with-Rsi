"""SQLite WAL persistence for RSI cycles, configuration registry, approvals and memory."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import RSIStorageSettings
from .rsi_search_space import default_tunable_parameters
from .rsi_types import (
    CandidateConfiguration,
    ConfigurationVersion,
    ExperimentTrial,
    FailurePattern,
    ImprovementHypothesis,
    ImprovementMemoryEntry,
    PromotionApproval,
    PromotionRecommendation,
    PromotionRecord,
    RollbackRecord,
    RSICycle,
    RSIReport,
)

_DDL = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS rsi_cycles (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    outcome TEXT,
    report_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rsi_failure_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    pattern_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(cycle_id) REFERENCES rsi_cycles(cycle_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS rsi_hypotheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    hypothesis_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(cycle_id) REFERENCES rsi_cycles(cycle_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS rsi_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(cycle_id) REFERENCES rsi_cycles(cycle_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS rsi_trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT NOT NULL,
    trial_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(cycle_id) REFERENCES rsi_cycles(cycle_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS rsi_recommendations (
    recommendation_id TEXT PRIMARY KEY,
    cycle_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(cycle_id) REFERENCES rsi_cycles(cycle_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS configuration_versions (
    version_id TEXT PRIMARY KEY,
    parent_version_id TEXT,
    fingerprint TEXT NOT NULL,
    active INTEGER NOT NULL,
    approved INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS configuration_approvals (
    approval_id TEXT PRIMARY KEY,
    recommendation_id TEXT NOT NULL,
    candidate_fingerprint TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS configuration_promotions (
    promotion_id TEXT PRIMARY KEY,
    from_version_id TEXT NOT NULL,
    to_version_id TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS configuration_rollbacks (
    rollback_id TEXT PRIMARY KEY,
    from_version_id TEXT NOT NULL,
    to_version_id TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS improvement_memory (
    entry_id TEXT PRIMARY KEY,
    candidate_fingerprint TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    accepted INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);
"""


class RSIStore:
    """Atomic SQLite WAL store for RSI lineage, configuration versions and improvement memory."""

    def __init__(
        self, db_path: Path, storage: RSIStorageSettings, busy_timeout_ms: int = 5000
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
            row = conn.execute("SELECT COUNT(*) FROM configuration_versions").fetchone()
            if row and int(row[0]) == 0:
                params = default_tunable_parameters()
                v0 = ConfigurationVersion(
                    version_id="v0_baseline",
                    parent_version_id=None,
                    fingerprint=CandidateConfiguration.compute_fingerprint(params),
                    parameters=params,
                    active=True,
                    approved=True,
                    created_at=datetime.now(UTC),
                )
                conn.execute(
                    "INSERT INTO configuration_versions "
                    "(version_id, parent_version_id, fingerprint, active, approved, payload_json) "
                    "VALUES (?, ?, ?, 1, 1, ?)",
                    (v0.version_id, None, v0.fingerprint, v0.model_dump_json()),
                )

    def save_cycle_report(
        self,
        report: RSIReport,
        patterns: tuple[FailurePattern, ...] = (),
        hypotheses: tuple[ImprovementHypothesis, ...] = (),
        candidates: tuple[CandidateConfiguration, ...] = (),
        trials: tuple[ExperimentTrial, ...] = (),
    ) -> None:
        c = report.cycle
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO rsi_cycles (cycle_id, status, outcome, report_json) "
                "VALUES (?, ?, ?, ?)",
                (c.cycle_id, c.status, c.outcome, report.model_dump_json()),
            )
            for p in patterns:
                conn.execute(
                    "INSERT INTO rsi_failure_patterns (cycle_id, pattern_id, payload_json) "
                    "VALUES (?, ?, ?)",
                    (c.cycle_id, p.pattern_id, p.model_dump_json()),
                )
            for h in hypotheses:
                conn.execute(
                    "INSERT INTO rsi_hypotheses (cycle_id, hypothesis_id, payload_json) "
                    "VALUES (?, ?, ?)",
                    (c.cycle_id, h.hypothesis_id, h.model_dump_json()),
                )
            for cand in candidates:
                conn.execute(
                    "INSERT INTO rsi_candidates "
                    "(cycle_id, candidate_id, fingerprint, payload_json) "
                    "VALUES (?, ?, ?, ?)",
                    (c.cycle_id, cand.candidate_id, cand.fingerprint, cand.model_dump_json()),
                )
            for tr in trials:
                conn.execute(
                    "INSERT INTO rsi_trials (cycle_id, trial_id, payload_json) VALUES (?, ?, ?)",
                    (c.cycle_id, tr.trial_id, tr.model_dump_json()),
                )
            if report.recommendation is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO rsi_recommendations "
                    "(recommendation_id, cycle_id, payload_json) VALUES (?, ?, ?)",
                    (
                        report.recommendation.recommendation_id,
                        c.cycle_id,
                        report.recommendation.model_dump_json(),
                    ),
                )
            conn.execute(
                "DELETE FROM rsi_cycles WHERE seq NOT IN "
                "(SELECT seq FROM rsi_cycles ORDER BY seq DESC LIMIT ?)",
                (self.storage.max_cycles_retained,),
            )

    def get_report(self, cycle_id: str) -> RSIReport | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT report_json FROM rsi_cycles WHERE cycle_id = ?", (cycle_id,)
            ).fetchone()
        return RSIReport.model_validate_json(str(row[0])) if row else None

    def update_cycle_status(self, cycle_id: str, status: str) -> RSICycle | None:
        rep = self.get_report(cycle_id)
        if rep is None:
            return None
        upd = rep.cycle.model_copy(update={"status": status})
        self.save_cycle_report(rep.model_copy(update={"cycle": upd}))
        return upd

    def list_recommendations(self) -> tuple[PromotionRecommendation, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM rsi_recommendations ORDER BY rowid DESC"
            ).fetchall()
        return tuple(PromotionRecommendation.model_validate_json(str(r[0])) for r in rows)

    def get_recommendation(self, rec_id: str) -> PromotionRecommendation | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM rsi_recommendations WHERE recommendation_id = ?",
                (rec_id,),
            ).fetchone()
        return PromotionRecommendation.model_validate_json(str(row[0])) if row else None

    def list_versions(self) -> tuple[ConfigurationVersion, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM configuration_versions ORDER BY rowid ASC"
            ).fetchall()
        return tuple(ConfigurationVersion.model_validate_json(str(r[0])) for r in rows)

    def active_version(self) -> ConfigurationVersion:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM configuration_versions WHERE active = 1 LIMIT 1"
            ).fetchone()
        assert row is not None
        return ConfigurationVersion.model_validate_json(str(row[0]))

    def save_version(self, ver: ConfigurationVersion) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO configuration_versions "
                "(version_id, parent_version_id, fingerprint, active, approved, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ver.version_id,
                    ver.parent_version_id,
                    ver.fingerprint,
                    1 if ver.active else 0,
                    1 if ver.approved else 0,
                    ver.model_dump_json(),
                ),
            )

    def activate_version_atomic(
        self,
        from_ver: ConfigurationVersion,
        to_ver: ConfigurationVersion,
        record: PromotionRecord | RollbackRecord,
    ) -> None:
        with self._connect() as conn:
            deact = from_ver.model_copy(update={"active": False})
            act = to_ver.model_copy(update={"active": True})
            conn.execute(
                "UPDATE configuration_versions SET active = 0, payload_json = ? "
                "WHERE version_id = ?",
                (deact.model_dump_json(), from_ver.version_id),
            )
            conn.execute(
                "INSERT OR REPLACE INTO configuration_versions "
                "(version_id, parent_version_id, fingerprint, active, approved, payload_json) "
                "VALUES (?, ?, ?, 1, ?, ?)",
                (
                    act.version_id,
                    act.parent_version_id,
                    act.fingerprint,
                    1 if act.approved else 0,
                    act.model_dump_json(),
                ),
            )
            if isinstance(record, PromotionRecord):
                conn.execute(
                    "INSERT INTO configuration_promotions "
                    "(promotion_id, from_version_id, to_version_id, payload_json) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        record.promotion_id,
                        record.from_version_id,
                        record.to_version_id,
                        record.model_dump_json(),
                    ),
                )
            else:
                conn.execute(
                    "INSERT INTO configuration_rollbacks "
                    "(rollback_id, from_version_id, to_version_id, payload_json) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        record.rollback_id,
                        record.from_version_id,
                        record.to_version_id,
                        record.model_dump_json(),
                    ),
                )

    def save_approval(self, approval: PromotionApproval) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO configuration_approvals "
                "(approval_id, recommendation_id, candidate_fingerprint, payload_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    approval.approval_id,
                    approval.recommendation_id,
                    approval.candidate_fingerprint,
                    approval.model_dump_json(),
                ),
            )

    def get_approval(self, approval_id: str) -> PromotionApproval | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM configuration_approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        return PromotionApproval.model_validate_json(str(row[0])) if row else None

    def save_memory(self, entry: ImprovementMemoryEntry) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO improvement_memory "
                "(entry_id, candidate_fingerprint, dataset_version, accepted, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    entry.entry_id,
                    entry.candidate_fingerprint,
                    entry.dataset_version,
                    1 if entry.accepted else 0,
                    entry.model_dump_json(),
                ),
            )

    def has_rejected_memory(self, fingerprint: str, dataset_version: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM improvement_memory "
                "WHERE candidate_fingerprint = ? AND dataset_version = ? AND accepted = 0",
                (fingerprint, dataset_version),
            ).fetchone()
        return bool(row and int(row[0]) > 0)

    def count_cycles(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM rsi_cycles").fetchone()
        return int(row[0]) if row else 0

    def get_candidate_params(self, fingerprint: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM rsi_candidates WHERE fingerprint = ? LIMIT 1",
                (fingerprint,),
            ).fetchone()
        if row is None:
            return None
        data = json.loads(str(row[0]))
        params = data.get("parameters")
        return dict(params) if isinstance(params, dict) else None
