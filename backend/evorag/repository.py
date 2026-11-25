"""SQLite metadata store. Every connection and writer transaction has a bounded lifetime."""

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from threading import Lock

from .config import Storage
from .documents import DocumentInfo, IngestResult, JobInfo, StoredChunk, VersionInfo
from .errors import IngestionError
from .observability import application_logger
from .resources import ResourceManager

DDL = (
    "CREATE TABLE schema_versions("
    "version INTEGER PRIMARY KEY, applied_at TEXT DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE documents(document_id TEXT PRIMARY KEY, source TEXT NOT NULL UNIQUE)",
    """CREATE TABLE versions(
       version_key TEXT PRIMARY KEY,
       document_id TEXT NOT NULL REFERENCES documents,
       version INTEGER NOT NULL,
       content_hash TEXT NOT NULL,
       parser TEXT NOT NULL,
       normalization TEXT NOT NULL,
       fingerprint TEXT NOT NULL,
       chunk_count INTEGER NOT NULL DEFAULT 0,
       created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
       UNIQUE(document_id, version))""",
    """CREATE TABLE chunks(
       chunk_id TEXT PRIMARY KEY,
       version_key TEXT NOT NULL REFERENCES versions,
       ordinal INTEGER NOT NULL,
       payload TEXT NOT NULL,
       UNIQUE(version_key, ordinal))""",
    """CREATE TABLE ingestion_jobs(
       job_id TEXT PRIMARY KEY,
       status TEXT NOT NULL CHECK(status IN ('created','duplicate','failed')),
       error TEXT,
       version_key TEXT REFERENCES versions,
       created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""",
    "INSERT INTO schema_versions(version) VALUES (1)",
    "PRAGMA user_version=1",
)

VERSION_FIELDS = (
    "document_id, version, content_hash, fingerprint, "
    "parser, normalization, chunk_count, created_at"
)
DOC_QUERY = """SELECT d.document_id, d.source, max(v.version) AS latest_version
               FROM documents d JOIN versions v USING(document_id)"""
MAX_LIMIT = 100
MAX_OFFSET = 2_147_483_647
PREFLIGHT_STEP_BYTES = 256 * 1024


def _validate_pagination(limit: int, offset: int) -> None:
    if isinstance(limit, bool) or not 1 <= limit <= MAX_LIMIT:
        raise IngestionError("invalid_pagination", 422)
    if isinstance(offset, bool) or not 0 <= offset <= MAX_OFFSET:
        raise IngestionError("invalid_pagination", 422)


class Repository:
    def __init__(self, config: Storage, resources: ResourceManager):
        self.path = config.database_path.absolute()
        self.config = config
        self.resources = resources
        self._initialized = False
        self._migration_lock = Lock()

    def _budget(self, growth: int) -> None:
        used = 0
        for suffix in ("", "-wal", "-shm", "-journal"):
            try:
                used += Path(str(self.path) + suffix).stat().st_size
            except FileNotFoundError:
                continue
        self.resources.check_storage(
            used_bytes=used, additional_bytes=growth, path=self.path.parent
        )

    @contextmanager
    def _connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(
                sqlite3.connect(
                    self.path, timeout=self.config.busy_timeout_ms / 1000, isolation_level=None
                )
            ) as db:
                db.row_factory = sqlite3.Row
                db.execute("PRAGMA foreign_keys=ON")
                with self._migration_lock:
                    if not self._initialized:
                        if self.config.sqlite_wal:
                            try:
                                db.execute("PRAGMA journal_mode=WAL")
                            except sqlite3.OperationalError:
                                application_logger().warning("storage.wal_unavailable")
                        db.execute("BEGIN IMMEDIATE")
                        schema = db.execute("PRAGMA user_version").fetchone()[0]
                        if schema == 0:
                            self._budget(PREFLIGHT_STEP_BYTES)
                            for statement in DDL:
                                db.execute(statement)
                        elif schema != 1:
                            raise IngestionError("unsupported_schema", 503)
                        db.commit()
                        self._initialized = True
                db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                if write:
                    self._budget(64 * 1024)
                try:
                    yield db
                    db.commit()
                except BaseException:
                    if db.in_transaction:
                        db.rollback()
                    raise
        except (sqlite3.Error, OSError):
            raise IngestionError("storage_unavailable", 503) from None

    @staticmethod
    def _duplicate(db: sqlite3.Connection, job_id: str, key: str) -> IngestResult | None:
        row = db.execute("SELECT * FROM versions WHERE version_key=?", (key,)).fetchone()
        if row is None:
            return None
        db.execute(
            "INSERT INTO ingestion_jobs(job_id,status,version_key) VALUES (?,'duplicate',?)",
            (job_id, key),
        )
        return IngestResult(
            job_id=job_id,
            document_id=row["document_id"],
            version=row["version"],
            chunk_count=row["chunk_count"],
            status="duplicate",
        )

    def duplicate(self, job_id: str, key: str) -> IngestResult | None:
        with self._connect(write=True) as db:
            return self._duplicate(db, job_id, key)

    def publish(
        self,
        job_id: str,
        doc_id: str,
        source: str,
        key: str,
        content_hash: str,
        parser: str,
        normalization: str,
        fingerprint: str,
        chunks: Iterable[StoredChunk],
    ) -> IngestResult:
        with self._connect(write=True) as db:
            duplicate = self._duplicate(db, job_id, key)
            if duplicate is not None:
                return duplicate
            db.execute(
                "INSERT INTO documents VALUES (?,?) ON CONFLICT(document_id) DO NOTHING",
                (doc_id, source),
            )
            version = db.execute(
                "SELECT coalesce(max(version),0)+1 FROM versions WHERE document_id=?", (doc_id,)
            ).fetchone()[0]
            db.execute(
                """INSERT INTO versions(version_key,document_id,version,content_hash,parser,
                       normalization,fingerprint) VALUES (?,?,?,?,?,?,?)""",
                (key, doc_id, version, content_hash, parser, normalization, fingerprint),
            )
            count = growth = next_check = 0
            for count, chunk in enumerate(chunks, 1):
                payload = chunk.model_dump_json()
                # Allow table/index pages plus WAL frames. Recheck every 256 KiB of growth.
                estimated = len(payload.encode()) * 3 + 2048
                growth += estimated
                if growth >= next_check:
                    self._budget(growth + PREFLIGHT_STEP_BYTES)
                    next_check = growth + PREFLIGHT_STEP_BYTES
                db.execute(
                    "INSERT INTO chunks VALUES (?,?,?,?)", (chunk.chunk_id, key, count, payload)
                )
            if not count:
                raise IngestionError("empty_content")
            db.execute("UPDATE versions SET chunk_count=? WHERE version_key=?", (count, key))
            db.execute(
                "INSERT INTO ingestion_jobs(job_id,status,version_key) VALUES (?,'created',?)",
                (job_id, key),
            )
            return IngestResult(
                job_id=job_id,
                document_id=doc_id,
                version=version,
                chunk_count=count,
                status="created",
            )

    def fail(self, job_id: str, code: str) -> None:
        with self._connect(write=True) as db:
            db.execute(
                "INSERT INTO ingestion_jobs(job_id,status,error) VALUES (?,'failed',?)",
                (job_id, code),
            )

    def documents(self, limit: int, offset: int) -> list[DocumentInfo]:
        _validate_pagination(limit, offset)
        with self._connect() as db:
            rows = db.execute(
                DOC_QUERY + " GROUP BY d.document_id ORDER BY d.document_id LIMIT ? OFFSET ?",
                (limit, offset),
            )
            return [DocumentInfo.model_validate(dict(r)) for r in rows]

    def document(self, doc_id: str) -> DocumentInfo:
        with self._connect() as db:
            row = db.execute(
                DOC_QUERY + " WHERE d.document_id=? GROUP BY d.document_id", (doc_id,)
            ).fetchone()
            if row is None:
                raise IngestionError("document_not_found", 404)
            return DocumentInfo.model_validate(dict(row))

    def versions(self, doc_id: str, limit: int, offset: int) -> list[VersionInfo]:
        _validate_pagination(limit, offset)
        self.document(doc_id)
        with self._connect() as db:
            rows = db.execute(
                f"SELECT {VERSION_FIELDS} FROM versions "
                "WHERE document_id=? ORDER BY version DESC LIMIT ? OFFSET ?",
                (doc_id, limit, offset),
            )
            return [VersionInfo.model_validate(dict(r)) for r in rows]

    def chunks(
        self, doc_id: str, version: int | None, limit: int, offset: int
    ) -> list[StoredChunk]:
        _validate_pagination(limit, offset)
        selected = version if version is not None else self.document(doc_id).latest_version
        with self._connect() as db:
            row = db.execute(
                "SELECT version_key FROM versions WHERE document_id=? AND version=?",
                (doc_id, selected),
            ).fetchone()
            if row is None:
                raise IngestionError("version_not_found", 404)
            rows = db.execute(
                "SELECT payload FROM chunks WHERE version_key=? ORDER BY ordinal LIMIT ? OFFSET ?",
                (row[0], limit, offset),
            )
            return [StoredChunk.model_validate_json(r[0]) for r in rows]

    def job(self, job_id: str) -> JobInfo:
        with self._connect() as db:
            row = db.execute(
                """SELECT job_id,status,error,document_id,version FROM ingestion_jobs
                             LEFT JOIN versions USING(version_key) WHERE job_id=?""",
                (job_id,),
            ).fetchone()
            if row is None:
                raise IngestionError("job_not_found", 404)
            return JobInfo.model_validate(dict(row))
