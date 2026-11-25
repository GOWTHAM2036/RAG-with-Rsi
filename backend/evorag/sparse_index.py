"""Persistent SQLite inverted index with atomic revision updates and bounded queries."""

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

from .analyzer import TextAnalyzer
from .bm25 import BM25Candidate, bm25_idf
from .contracts import Contract
from .documents import StoredChunk
from .errors import IngestionError
from .filtering import MetadataFilter
from .repository import Repository

SPARSE_DDL = (
    "CREATE TABLE sparse_schema(version INTEGER PRIMARY KEY)",
    "INSERT INTO sparse_schema VALUES(1)",
    """CREATE TABLE sparse_meta(
        id INTEGER PRIMARY KEY CHECK(id = 1),
        index_generation INTEGER NOT NULL DEFAULT 0,
        analyzer_version TEXT NOT NULL,
        doc_count INTEGER NOT NULL DEFAULT 0,
        total_length INTEGER NOT NULL DEFAULT 0)""",
    """CREATE TABLE sparse_chunks(
        chunk_id TEXT PRIMARY KEY REFERENCES chunks,
        version_key TEXT NOT NULL REFERENCES versions,
        document_id TEXT NOT NULL,
        version INTEGER NOT NULL,
        chunk_length INTEGER NOT NULL,
        analyzer_version TEXT NOT NULL,
        is_latest INTEGER NOT NULL DEFAULT 1,
        indexed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))""",
    "CREATE INDEX idx_sparse_chunks_doc_latest ON sparse_chunks(document_id, is_latest)",
    "CREATE INDEX idx_sparse_chunks_version_key ON sparse_chunks(version_key)",
    """CREATE TABLE sparse_terms(
        term TEXT PRIMARY KEY,
        df INTEGER NOT NULL CHECK(df >= 0))""",
    """CREATE TABLE sparse_postings(
        term TEXT NOT NULL,
        chunk_id TEXT NOT NULL REFERENCES sparse_chunks ON DELETE CASCADE,
        tf INTEGER NOT NULL CHECK(tf >= 1),
        PRIMARY KEY(term, chunk_id))""",
    "CREATE INDEX idx_sparse_postings_chunk ON sparse_postings(chunk_id)",
    """CREATE TABLE sparse_index_jobs(
        job_id TEXT PRIMARY KEY,
        status TEXT NOT NULL CHECK(status IN ('running','complete','failed')),
        total_chunks INTEGER NOT NULL DEFAULT 0,
        indexed_chunks INTEGER NOT NULL DEFAULT 0,
        error TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""",
)

MAX_POSTING_CANDIDATES = 5000


class SparseIndexStatus(Contract):
    doc_count: int
    chunk_count: int
    term_count: int
    total_length: int
    avg_chunk_length: float
    index_generation: int
    analyzer_version: str


class SparseIndexJob(Contract):
    job_id: str
    status: Literal["running", "complete", "failed"]
    total_chunks: int
    indexed_chunks: int
    error: str | None = None


class SparseIndex:
    """SQLite-backed inverted index sharing the repository's WAL database."""

    def __init__(self, repository: Repository, analyzer: TextAnalyzer) -> None:
        self.repository = repository
        self.analyzer = analyzer
        self._ready = False

    def _budget(self, growth: int) -> None:
        used = 0
        for suffix in ("", "-wal", "-shm", "-journal"):
            try:
                used += Path(str(self.repository.path) + suffix).stat().st_size
            except FileNotFoundError:
                continue
        self.repository.resources.check_storage(
            used_bytes=used, additional_bytes=growth, path=self.repository.path.parent
        )

    @contextmanager
    def transaction(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        if not self._ready:
            with self.repository._connect(write=True) as db:
                exists = db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sparse_schema'"
                ).fetchone()
                if not exists:
                    for statement in SPARSE_DDL:
                        db.execute(statement)
                    db.execute(
                        "INSERT INTO sparse_meta(id, index_generation, analyzer_version) "
                        "VALUES(1, 0, ?)",
                        (self.analyzer.version,),
                    )
                else:
                    ver = db.execute("SELECT version FROM sparse_schema").fetchone()[0]
                    if ver != 1:
                        raise IngestionError("unsupported_sparse_schema", 503)
            self._ready = True
        with self.repository._connect(write=write) as db:
            yield db

    def status(self) -> SparseIndexStatus:
        with self.transaction() as db:
            meta = db.execute("SELECT * FROM sparse_meta WHERE id=1").fetchone()
            doc_count = int(meta["doc_count"])
            total_length = int(meta["total_length"])
            term_count = int(
                db.execute("SELECT count(*) FROM sparse_terms WHERE df > 0").fetchone()[0]
            )
            total_chunks = int(db.execute("SELECT count(*) FROM sparse_chunks").fetchone()[0])
            avg = total_length / doc_count if doc_count > 0 else 0.0
            return SparseIndexStatus(
                doc_count=doc_count,
                chunk_count=total_chunks,
                term_count=term_count,
                total_length=total_length,
                avg_chunk_length=avg,
                index_generation=int(meta["index_generation"]),
                analyzer_version=str(meta["analyzer_version"]),
            )

    def job_status(self, job_id: str) -> SparseIndexJob:
        with self.transaction() as db:
            row = db.execute("SELECT * FROM sparse_index_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise IngestionError("index_job_not_found", 404)
            return SparseIndexJob(
                job_id=str(row["job_id"]),
                status=row["status"],
                total_chunks=int(row["total_chunks"]),
                indexed_chunks=int(row["indexed_chunks"]),
                error=row["error"],
            )

    def _deactivate_prior_active(self, db: sqlite3.Connection, document_id: str) -> None:
        """Mark existing active chunks for document_id as superseded and update term DFs."""
        prior_rows = db.execute(
            "SELECT chunk_id, chunk_length FROM sparse_chunks WHERE document_id=? AND is_latest=1",
            (document_id,),
        ).fetchall()
        if not prior_rows:
            return
        prior_ids = [str(r["chunk_id"]) for r in prior_rows]
        removed_length = sum(int(r["chunk_length"]) for r in prior_rows)
        placeholders = ",".join("?" for _ in prior_ids)
        term_counts = db.execute(
            f"SELECT term, count(*) AS cnt FROM sparse_postings "
            f"WHERE chunk_id IN ({placeholders}) GROUP BY term",
            prior_ids,
        ).fetchall()
        for row in term_counts:
            db.execute(
                "UPDATE sparse_terms SET df = max(0, df - ?) WHERE term=?",
                (int(row["cnt"]), str(row["term"])),
            )
        db.execute("DELETE FROM sparse_terms WHERE df <= 0")
        db.execute(
            f"UPDATE sparse_chunks SET is_latest=0 WHERE chunk_id IN ({placeholders})",
            prior_ids,
        )
        db.execute(
            "UPDATE sparse_meta SET doc_count = max(0, doc_count - ?), "
            "total_length = max(0, total_length - ?) WHERE id=1",
            (len(prior_ids), removed_length),
        )

    def _delete_version_chunks(self, db: sqlite3.Connection, version_key: str) -> None:
        """Remove stale postings/chunks for a specific version_key before re-indexing."""
        existing = db.execute(
            "SELECT chunk_id, chunk_length, is_latest FROM sparse_chunks WHERE version_key=?",
            (version_key,),
        ).fetchall()
        if not existing:
            return
        active_ids = [str(r["chunk_id"]) for r in existing if int(r["is_latest"]) == 1]
        if active_ids:
            removed_length = sum(
                int(r["chunk_length"]) for r in existing if int(r["is_latest"]) == 1
            )
            placeholders = ",".join("?" for _ in active_ids)
            term_counts = db.execute(
                f"SELECT term, count(*) AS cnt FROM sparse_postings "
                f"WHERE chunk_id IN ({placeholders}) GROUP BY term",
                active_ids,
            ).fetchall()
            for row in term_counts:
                db.execute(
                    "UPDATE sparse_terms SET df = max(0, df - ?) WHERE term=?",
                    (int(row["cnt"]), str(row["term"])),
                )
            db.execute("DELETE FROM sparse_terms WHERE df <= 0")
            db.execute(
                "UPDATE sparse_meta SET doc_count = max(0, doc_count - ?), "
                "total_length = max(0, total_length - ?) WHERE id=1",
                (len(active_ids), removed_length),
            )
        all_ids = [str(r["chunk_id"]) for r in existing]
        all_placeholders = ",".join("?" for _ in all_ids)
        db.execute(
            f"DELETE FROM sparse_postings WHERE chunk_id IN ({all_placeholders})",
            all_ids,
        )
        db.execute(
            f"DELETE FROM sparse_chunks WHERE chunk_id IN ({all_placeholders})",
            all_ids,
        )

    def index_version(
        self, document_id: str, version: int | None = None, *, force: bool = False
    ) -> int:
        """Atomically and idempotently index one document revision. Returns chunks indexed."""
        with self.transaction(write=True) as db:
            ver_row = db.execute(
                "SELECT version_key, version, chunk_count FROM versions "
                "WHERE document_id=? AND (? IS NULL OR version=?) "
                "ORDER BY version DESC LIMIT 1",
                (document_id, version, version),
            ).fetchone()
            if ver_row is None:
                raise IngestionError("document_revision_not_found", 404)
            version_key = str(ver_row["version_key"])
            resolved_version = int(ver_row["version"])
            max_ver = int(
                db.execute(
                    "SELECT max(version) FROM versions WHERE document_id=?", (document_id,)
                ).fetchone()[0]
            )
            is_latest = 1 if resolved_version == max_ver else 0

            if not force:
                already = db.execute(
                    "SELECT count(*), min(analyzer_version) FROM sparse_chunks WHERE version_key=?",
                    (version_key,),
                ).fetchone()
                if (
                    int(already[0]) == int(ver_row["chunk_count"])
                    and already[1] == self.analyzer.version
                ):
                    return 0

            chunk_rows = db.execute(
                "SELECT chunk_id, payload FROM chunks WHERE version_key=? ORDER BY ordinal ASC",
                (version_key,),
            ).fetchall()
            if not chunk_rows:
                return 0

            self._budget(len(chunk_rows) * 2048)
            self._delete_version_chunks(db, version_key)
            if is_latest:
                self._deactivate_prior_active(db, document_id)

            df_deltas: dict[str, int] = {}
            total_added_length = 0
            posting_rows: list[tuple[str, str, int]] = []
            chunk_inserts: list[tuple[str, str, str, int, int, str, int]] = []

            for row in chunk_rows:
                stored = StoredChunk.model_validate_json(row["payload"])
                analyzed = self.analyzer.analyze(stored.text, for_query=False)
                chunk_inserts.append(
                    (
                        stored.chunk_id,
                        version_key,
                        document_id,
                        resolved_version,
                        analyzed.length,
                        analyzed.analyzer_version,
                        is_latest,
                    )
                )
                total_added_length += analyzed.length
                for term, tf in analyzed.term_frequencies.items():
                    posting_rows.append((term, stored.chunk_id, tf))
                    if is_latest:
                        df_deltas[term] = df_deltas.get(term, 0) + 1

            db.executemany(
                "INSERT INTO sparse_chunks("
                "chunk_id, version_key, document_id, version, chunk_length, "
                "analyzer_version, is_latest) VALUES(?,?,?,?,?,?,?)",
                chunk_inserts,
            )
            db.executemany(
                "INSERT INTO sparse_postings(term, chunk_id, tf) VALUES(?,?,?)",
                posting_rows,
            )
            if is_latest:
                db.executemany(
                    "INSERT INTO sparse_terms(term, df) VALUES(?, ?) "
                    "ON CONFLICT(term) DO UPDATE SET df = sparse_terms.df + excluded.df",
                    list(df_deltas.items()),
                )
                db.execute(
                    "UPDATE sparse_meta SET "
                    "index_generation = index_generation + 1, "
                    "analyzer_version = ?, "
                    "doc_count = doc_count + ?, "
                    "total_length = total_length + ? WHERE id=1",
                    (self.analyzer.version, len(chunk_inserts), total_added_length),
                )
            else:
                db.execute(
                    "UPDATE sparse_meta SET index_generation = index_generation + 1, "
                    "analyzer_version = ? WHERE id=1",
                    (self.analyzer.version,),
                )
            return len(chunk_inserts)

    def rebuild_all(self, *, force: bool = True) -> SparseIndexJob:
        """Explicitly rebuild or synchronize the sparse index across all document versions."""
        with self.repository.resources.ingestion_slot():
            job_id = uuid4().hex
            with self.transaction(write=True) as db:
                versions = db.execute(
                    "SELECT document_id, version, chunk_count FROM versions "
                    "ORDER BY document_id ASC, version ASC"
                ).fetchall()
                total_chunks = sum(int(v["chunk_count"]) for v in versions)
                db.execute(
                    "INSERT INTO sparse_index_jobs(job_id, status, total_chunks, indexed_chunks) "
                    "VALUES(?, 'running', ?, 0)",
                    (job_id, total_chunks),
                )
            indexed = 0
            try:
                for row in versions:
                    doc_id = str(row["document_id"])
                    ver = int(row["version"])
                    self.index_version(doc_id, ver, force=force)
                    indexed += int(row["chunk_count"])
                    with self.transaction(write=True) as db:
                        db.execute(
                            "UPDATE sparse_index_jobs SET indexed_chunks=? WHERE job_id=?",
                            (indexed, job_id),
                        )
                with self.transaction(write=True) as db:
                    db.execute(
                        "UPDATE sparse_index_jobs SET status='complete', indexed_chunks=? "
                        "WHERE job_id=?",
                        (indexed, job_id),
                    )
            except Exception as exc:
                with self.transaction(write=True) as db:
                    db.execute(
                        "UPDATE sparse_index_jobs SET status='failed', error=? WHERE job_id=?",
                        (type(exc).__name__, job_id),
                    )
                raise
            return self.job_status(job_id)

    def query_candidates(
        self,
        query_terms: Sequence[str],
        metadata_filter: MetadataFilter,
        *,
        max_candidates: int = MAX_POSTING_CANDIDATES,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> tuple[int, float, dict[str, int], list[tuple[BM25Candidate, StoredChunk]]]:
        """Fetch top matching posting candidates by BM25 score and active corpus statistics."""
        query_tf: dict[str, int] = {}
        for term in query_terms:
            query_tf[term] = query_tf.get(term, 0) + 1
        unique_terms = list(query_tf)
        if not unique_terms:
            return 0, 0.0, {}, []

        with self.transaction() as db:
            use_fast_meta = (
                metadata_filter.latest_only
                and metadata_filter.version is None
                and metadata_filter.document_id is None
                and metadata_filter.file_type is None
                and metadata_filter.source is None
            )
            if use_fast_meta:
                meta = db.execute("SELECT doc_count, total_length FROM sparse_meta").fetchone()
                corpus_size = int(meta["doc_count"])
                total_length = int(meta["total_length"])
                if corpus_size == 0 or total_length <= 0:
                    return 0, 0.0, {}, []
                avg_dl = total_length / corpus_size

                term_ph = ",".join("?" for _ in unique_terms)
                df_rows = db.execute(
                    f"SELECT term, df FROM sparse_terms WHERE term IN ({term_ph}) AND df > 0",
                    unique_terms,
                ).fetchall()
                doc_freqs = {str(r["term"]): int(r["df"]) for r in df_rows}
                active_terms = [t for t in doc_freqs if bm25_idf(corpus_size, doc_freqs[t]) > 0.0]
                if not active_terms:
                    return corpus_size, avg_dl, doc_freqs, []

                active_ph = ",".join("?" for _ in active_terms)
                case_parts = " ".join("WHEN ? THEN ?" for _ in active_terms)
                case_params: list[object] = []
                for term in active_terms:
                    case_params.extend(
                        [term, bm25_idf(corpus_size, doc_freqs[term]) * query_tf[term]]
                    )
                top_rows = db.execute(
                    f"SELECT p.chunk_id, "
                    f"SUM((CASE p.term {case_parts} ELSE 0.0 END) "
                    f"* (p.tf * (? + 1.0)) "
                    f"/ (p.tf + ? * (1.0 - ? + ? * CAST(sc.chunk_length AS REAL) / ?))) "
                    f"AS bm25_score "
                    f"FROM sparse_postings p "
                    f"JOIN sparse_chunks sc ON p.chunk_id = sc.chunk_id "
                    f"WHERE p.term IN ({active_ph}) AND sc.is_latest = 1 "
                    f"AND sc.chunk_length > 0 "
                    f"GROUP BY p.chunk_id "
                    f"ORDER BY bm25_score DESC, p.chunk_id ASC LIMIT ?",
                    [*case_params, k1, k1, b, b, avg_dl, *active_terms, max_candidates],
                ).fetchall()
            else:
                clauses: list[str] = []
                params: list[object] = []
                if metadata_filter.document_id is not None:
                    clauses.append("sc.document_id = ?")
                    params.append(metadata_filter.document_id)
                if metadata_filter.version is not None:
                    clauses.append("sc.version = ?")
                    params.append(metadata_filter.version)
                elif metadata_filter.latest_only:
                    clauses.append("sc.is_latest = 1")
                if metadata_filter.source is not None:
                    clauses.append("d.source = ?")
                    params.append(metadata_filter.source)
                ext = metadata_filter.normalized_extension()
                if ext is not None:
                    clauses.append("substr(lower(d.source), -length(?)) = ?")
                    params.extend([ext, ext])

                where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                stats_row = db.execute(
                    f"SELECT count(*) AS cnt, coalesce(sum(sc.chunk_length), 0) AS total_len "
                    f"FROM sparse_chunks sc "
                    f"JOIN documents d ON sc.document_id = d.document_id {where_sql}",
                    params,
                ).fetchone()
                corpus_size = int(stats_row["cnt"])
                total_length = int(stats_row["total_len"])
                if corpus_size == 0 or total_length <= 0:
                    return 0, 0.0, {}, []
                avg_dl = total_length / corpus_size

                term_ph = ",".join("?" for _ in unique_terms)
                term_clause = f"p.term IN ({term_ph})"
                where_terms = (
                    f"{where_sql} AND {term_clause}" if where_sql else f"WHERE {term_clause}"
                )
                df_rows = db.execute(
                    f"SELECT p.term, count(*) AS df "
                    f"FROM sparse_postings p "
                    f"JOIN sparse_chunks sc ON p.chunk_id = sc.chunk_id "
                    f"JOIN documents d ON sc.document_id = d.document_id "
                    f"{where_terms} GROUP BY p.term",
                    [*params, *unique_terms],
                ).fetchall()
                doc_freqs = {str(r["term"]): int(r["df"]) for r in df_rows}
                active_terms = [t for t in doc_freqs if bm25_idf(corpus_size, doc_freqs[t]) > 0.0]
                if not active_terms:
                    return corpus_size, avg_dl, doc_freqs, []

                active_ph = ",".join("?" for _ in active_terms)
                active_clause = f"p.term IN ({active_ph}) AND sc.chunk_length > 0"
                where_postings = (
                    f"{where_sql} AND {active_clause}" if where_sql else f"WHERE {active_clause}"
                )
                case_parts = " ".join("WHEN ? THEN ?" for _ in active_terms)
                case_params = []
                for term in active_terms:
                    case_params.extend(
                        [term, bm25_idf(corpus_size, doc_freqs[term]) * query_tf[term]]
                    )
                top_rows = db.execute(
                    f"SELECT p.chunk_id, "
                    f"SUM((CASE p.term {case_parts} ELSE 0.0 END) "
                    f"* (p.tf * (? + 1.0)) "
                    f"/ (p.tf + ? * (1.0 - ? + ? * CAST(sc.chunk_length AS REAL) / ?))) "
                    f"AS bm25_score "
                    f"FROM sparse_postings p "
                    f"JOIN sparse_chunks sc ON p.chunk_id = sc.chunk_id "
                    f"JOIN documents d ON sc.document_id = d.document_id "
                    f"{where_postings} "
                    f"GROUP BY p.chunk_id "
                    f"ORDER BY bm25_score DESC, p.chunk_id ASC LIMIT ?",
                    [*case_params, k1, k1, b, b, avg_dl, *params, *active_terms, max_candidates],
                ).fetchall()

            top_ids = [str(r["chunk_id"]) for r in top_rows]
            if not top_ids:
                return corpus_size, avg_dl, doc_freqs, []

            cid_ph = ",".join("?" for _ in top_ids)
            chunk_rows = db.execute(
                f"SELECT sc.chunk_id, sc.chunk_length, c.payload "
                f"FROM sparse_chunks sc "
                f"JOIN chunks c ON sc.chunk_id = c.chunk_id "
                f"WHERE sc.chunk_id IN ({cid_ph})",
                top_ids,
            ).fetchall()
            chunk_info: dict[str, tuple[int, StoredChunk]] = {
                str(r["chunk_id"]): (
                    int(r["chunk_length"]),
                    StoredChunk.model_validate_json(r["payload"]),
                )
                for r in chunk_rows
            }

            active_ph = ",".join("?" for _ in active_terms)
            posting_rows = db.execute(
                f"SELECT chunk_id, term, tf FROM sparse_postings "
                f"WHERE chunk_id IN ({cid_ph}) AND term IN ({active_ph})",
                [*top_ids, *active_terms],
            ).fetchall()
            tfs_by_chunk: dict[str, dict[str, int]] = {cid: {} for cid in top_ids}
            for row in posting_rows:
                tfs_by_chunk[str(row["chunk_id"])][str(row["term"])] = int(row["tf"])

        candidates: list[tuple[BM25Candidate, StoredChunk]] = []
        for cid in top_ids:
            length, stored = chunk_info[cid]
            candidates.append(
                (
                    BM25Candidate(
                        chunk_id=cid,
                        chunk_length=length,
                        term_frequencies=tfs_by_chunk[cid],
                    ),
                    stored,
                )
            )
        return corpus_size, avg_dl, doc_freqs, candidates
