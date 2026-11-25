"""Durable embedding jobs and a separately evictable cache in the ingestion database."""

import hashlib
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from .contracts import Contract
from .documents import StoredChunk, digest
from .inference_types import ProviderError
from .repository import Repository
from .vector_store import VectorStore


class EmbeddingSpace(Contract):
    endpoint: str
    provider: str
    model: str
    digest: str
    dimensions: int
    normalized: bool
    formatting_version: str


class EmbeddingJob(Contract):
    job_id: str
    document_id: str
    version: int
    total: int
    completed: int
    status: Literal["pending", "running", "failed", "cancelled", "complete"]
    space: EmbeddingSpace
    error: str | None = None


DDL = (
    "CREATE TABLE embedding_schema(version INTEGER PRIMARY KEY)",
    "INSERT INTO embedding_schema VALUES(1)",
    """CREATE TABLE embedding_cache(
        key TEXT PRIMARY KEY, vector BLOB NOT NULL, checksum TEXT NOT NULL)""",
    """CREATE TABLE embedding_jobs(
        job_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, version INTEGER NOT NULL,
        total INTEGER NOT NULL, completed INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL,
        space TEXT NOT NULL, error TEXT, owner TEXT, expires REAL,
        FOREIGN KEY(document_id,version) REFERENCES versions(document_id,version))""",
    """CREATE TABLE vector_spaces(
        space_id TEXT PRIMARY KEY, space_json TEXT NOT NULL, dimensions INTEGER NOT NULL,
        committed_rows INTEGER NOT NULL DEFAULT 0, file_name TEXT NOT NULL)""",
    """CREATE TABLE chunk_embeddings(
        job_id TEXT NOT NULL REFERENCES embedding_jobs, chunk_id TEXT NOT NULL REFERENCES chunks,
        space_id TEXT NOT NULL REFERENCES vector_spaces, row_index INTEGER NOT NULL,
        checksum TEXT NOT NULL, PRIMARY KEY(job_id,chunk_id), UNIQUE(space_id,row_index))""",
)


def space_id_for(space: EmbeddingSpace) -> str:
    return digest("space-v1", space.model_dump_json())


class EmbeddingStore:
    def __init__(self, repository: Repository, max_cache_bytes: int) -> None:
        self.repository = repository
        self.max_cache_bytes = max_cache_bytes
        self.vectors = VectorStore(repository.path.parent, repository.path.name)
        self._ready = False

    def _budget(self, growth: int) -> None:
        used = self.vectors.total_bytes()
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
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='embedding_schema'"
                ).fetchone()
                if not exists:
                    for statement in DDL:
                        db.execute(statement)
                elif db.execute("SELECT version FROM embedding_schema").fetchone()[0] != 1:
                    raise ProviderError("unsupported_embedding_schema", 503)
            self._ready = True
        with self.repository._connect(write=write) as db:
            yield db

    def cached(self, keys: Sequence[str], width: int) -> dict[str, bytes]:
        with self.transaction() as db:
            hits = {}
            for key in keys:
                row = db.execute(
                    "SELECT vector,checksum FROM embedding_cache WHERE key=? AND length(vector)=?",
                    (key, width * 4),
                ).fetchone()
                if row and hashlib.sha256(row[0]).hexdigest() == row[1]:
                    hits[key] = bytes(row[0])
            return hits

    def save_cache(self, entries: Mapping[str, bytes]) -> None:
        with self.transaction(write=True) as db:
            self._budget(sum(len(blob) + 256 for blob in entries.values()) * 3)
            for key, blob in entries.items():
                if len(blob) + 256 <= self.max_cache_bytes:
                    db.execute(
                        "INSERT OR REPLACE INTO embedding_cache VALUES(?,?,?)",
                        (key, blob, hashlib.sha256(blob).hexdigest()),
                    )
            size = db.execute(
                "SELECT coalesce(sum(length(vector)+256),0) FROM embedding_cache"
            ).fetchone()[0]
            # FIFO insertion order. SQLite reuses freed pages on subsequent writes.
            for rowid, cost in db.execute(
                "SELECT rowid,length(vector)+256 FROM embedding_cache ORDER BY rowid"
            ):
                if size <= self.max_cache_bytes:
                    break
                db.execute("DELETE FROM embedding_cache WHERE rowid=?", (rowid,))
                size -= cost

    def stats(self) -> dict[str, int]:
        with self.transaction() as db:
            row = db.execute(
                "SELECT count(*),coalesce(sum(length(vector)+256),0) FROM embedding_cache"
            ).fetchone()
            return {"entries": row[0], "charged_bytes": row[1]}

    @staticmethod
    def _job(db: sqlite3.Connection, job_id: str) -> EmbeddingJob:
        row = db.execute("SELECT * FROM embedding_jobs WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            raise ProviderError("embedding_job_not_found", 404)
        return EmbeddingJob(
            job_id=job_id,
            document_id=row["document_id"],
            version=row["version"],
            total=row["total"],
            completed=row["completed"],
            status=row["status"],
            space=EmbeddingSpace.model_validate_json(row["space"]),
            error=row["error"],
        )

    def status(self, job_id: str) -> EmbeddingJob:
        with self.transaction() as db:
            return self._job(db, job_id)

    def prepare(self, doc_id: str, version: int | None, space: EmbeddingSpace) -> EmbeddingJob:
        with self.transaction(write=True) as db:
            row = db.execute(
                """SELECT version,version_key,chunk_count FROM versions WHERE document_id=?
                   AND (? IS NULL OR version=?) ORDER BY version DESC LIMIT 1""",
                (doc_id, version, version),
            ).fetchone()
            if not row:
                raise ProviderError("document_revision_not_found", 404)
            job_id = digest("embed-job-v1", row["version_key"], space.model_dump_json())
            db.execute(
                """INSERT INTO embedding_jobs(job_id,document_id,version,total,status,space)
                       VALUES(?,?,?,?,'pending',?) ON CONFLICT(job_id) DO NOTHING""",
                (job_id, doc_id, row["version"], row["chunk_count"], space.model_dump_json()),
            )
            return self._job(db, job_id)

    def claim(self, job_id: str, owner: str, lease: float) -> bool:
        with self.transaction(write=True) as db:
            cursor = db.execute(
                """UPDATE embedding_jobs SET status='running',owner=?,expires=?,error=NULL
                   WHERE job_id=? AND status != 'complete' AND (owner IS NULL OR expires < ?)""",
                (owner, time.time() + lease, job_id, time.time()),
            )
            return cursor.rowcount == 1

    def commit_batch(
        self, job_id: str, owner: str, offset: int, lease: float, rows: Sequence[tuple[str, bytes]]
    ) -> EmbeddingJob:
        with self.transaction(write=True) as db:
            job = self._job(db, job_id)
            if not db.execute(
                "SELECT 1 FROM embedding_jobs WHERE job_id=? AND owner=? AND completed=?",
                (job_id, owner, offset),
            ).fetchone():
                return job
            sid = space_id_for(job.space)
            vector_bytes = len(rows) * job.space.dimensions * 4
            self._budget(sum(len(blob) + 256 for _, blob in rows) * 3 + vector_bytes)
            db.execute(
                """INSERT INTO vector_spaces(
                       space_id,space_json,dimensions,committed_rows,file_name
                   ) VALUES(?,?,?,0,?) ON CONFLICT(space_id) DO NOTHING""",
                (
                    sid,
                    job.space.model_dump_json(),
                    job.space.dimensions,
                    self.vectors.path_for(sid).name,
                ),
            )
            committed_rows = int(
                db.execute(
                    "SELECT committed_rows FROM vector_spaces WHERE space_id=?", (sid,)
                ).fetchone()[0]
            )
            new_committed = self.vectors.append_batch(
                sid, committed_rows, job.space.dimensions, [blob for _, blob in rows]
            )
            db.executemany(
                "INSERT INTO chunk_embeddings VALUES(?,?,?,?,?)",
                (
                    (
                        job_id,
                        chunk_id,
                        sid,
                        committed_rows + index,
                        hashlib.sha256(blob).hexdigest(),
                    )
                    for index, (chunk_id, blob) in enumerate(rows)
                ),
            )
            db.execute(
                "UPDATE vector_spaces SET committed_rows=? WHERE space_id=?",
                (new_committed, sid),
            )
            completed = offset + len(rows)
            done = completed == job.total
            db.execute(
                "UPDATE embedding_jobs SET completed=?, status=?,owner=?,expires=? WHERE job_id=?",
                (
                    completed,
                    "complete" if done else "running",
                    None if done else owner,
                    None if done else time.time() + lease,
                    job_id,
                ),
            )
            return self._job(db, job_id)

    def recover_space(self, space: EmbeddingSpace) -> int:
        sid = space_id_for(space)
        with self.transaction() as db:
            row = db.execute(
                "SELECT committed_rows FROM vector_spaces WHERE space_id=?", (sid,)
            ).fetchone()
            committed_rows = int(row[0]) if row else 0
        self.vectors.recover(sid, committed_rows, space.dimensions)
        return committed_rows

    def space_rows(
        self,
        space: EmbeddingSpace,
        *,
        document_id: str | None = None,
        version: int | None = None,
        latest_only: bool = True,
        file_type: str | None = None,
        source: str | None = None,
    ) -> tuple[str, int, list[tuple[int, StoredChunk]]]:
        sid = space_id_for(space)
        with self.transaction() as db:
            meta = db.execute(
                "SELECT committed_rows FROM vector_spaces WHERE space_id=?", (sid,)
            ).fetchone()
            if meta is None or int(meta[0]) == 0:
                return sid, 0, []
            committed_rows = int(meta[0])
            clauses = ["ce.space_id = ?"]
            params: list[object] = [sid]
            if document_id is not None:
                clauses.append("v.document_id = ?")
                params.append(document_id)
            if version is not None:
                clauses.append("v.version = ?")
                params.append(version)
            elif latest_only:
                clauses.append(
                    "v.version = ("
                    "SELECT max(v2.version) FROM versions v2 WHERE v2.document_id = v.document_id"
                    ")"
                )
            if source is not None:
                clauses.append("d.source = ?")
                params.append(source)
            if file_type is not None:
                ext = file_type.lower()
                ext = ext if ext.startswith(".") else f".{ext}"
                clauses.append("substr(lower(d.source), -length(?)) = ?")
                params.extend([ext, ext])
            sql = (
                "SELECT ce.row_index, c.payload "
                "FROM chunk_embeddings ce "
                "JOIN chunks c ON ce.chunk_id = c.chunk_id "
                "JOIN versions v ON c.version_key = v.version_key "
                "JOIN documents d ON v.document_id = d.document_id "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY ce.row_index ASC"
            )
            rows = db.execute(sql, params).fetchall()
            return (
                sid,
                committed_rows,
                [
                    (int(r["row_index"]), StoredChunk.model_validate_json(r["payload"]))
                    for r in rows
                ],
            )

    def stop(
        self, job_id: str, owner: str | None, status: Literal["failed", "cancelled"], code: str
    ) -> EmbeddingJob:
        with self.transaction(write=True) as db:
            db.execute(
                """UPDATE embedding_jobs SET status=?,error=?,owner=NULL,expires=NULL
                WHERE job_id=? AND status!='complete' AND (? IS NULL OR owner=?)""",
                (status, code, job_id, owner, owner),
            )
            return self._job(db, job_id)
