import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any, cast

import pytest
from evorag.config import Resources, Settings, Storage
from evorag.documents import IngestResult, SourceSpan, StoredChunk, digest
from evorag.errors import IngestionError, ResourceUnavailable
from evorag.repository import Repository
from evorag.resources import ResourceManager


def _repo(
    db_path: Path,
    *,
    wal: bool = True,
    budget_gb: float = 20.0,
) -> tuple[Repository, ResourceManager]:
    settings = Settings(
        storage=Storage(database_path=db_path, sqlite_wal=wal),
        resources=Resources(project_storage_budget_gb=budget_gb),
    )
    resources = ResourceManager(settings, lambda: 4 * 1024**3)
    return Repository(settings.storage, resources), resources


def _chunk(
    doc_id: str,
    key: str,
    ordinal: int,
    text: str = "hello",
) -> StoredChunk:
    start = ordinal * 10
    end = start + len(text)
    return StoredChunk(
        chunk_id=digest(key, str(ordinal)),
        document_id=doc_id,
        start=start,
        end=end,
        text=text,
        token_count=max(1, len(text) // 4),
        estimated=True,
        sources=(
            SourceSpan(label="body", start=start, end=end, parsed_start=start, parsed_end=end),
        ),
    )


def test_transactional_rollback_on_iterator_exception(tmp_path: Path) -> None:
    db_path = tmp_path / "repo.sqlite3"
    repo, _ = _repo(db_path)

    def failing_chunks() -> Iterator[StoredChunk]:
        yield _chunk("doc1", "key1", 0)
        raise RuntimeError("boom mid-stream")

    with pytest.raises(RuntimeError, match="boom mid-stream"):
        repo.publish(
            "job1", "doc1", "a.txt", "key1", "h1", "txt-v1", "nfc-lf-v1", "fp1", failing_chunks()
        )

    assert repo.documents(10, 0) == []
    with pytest.raises(IngestionError, match="job_not_found"):
        repo.job("job1")

    with pytest.raises(IngestionError, match="empty_content"):
        repo.publish("job2", "doc1", "a.txt", "key2", "h1", "txt-v1", "nfc-lf-v1", "fp1", iter(()))
    assert repo.documents(10, 0) == []


def test_foreign_keys_enforced(tmp_path: Path) -> None:
    db_path = tmp_path / "fk.sqlite3"
    repo, _ = _repo(db_path)
    repo.documents(10, 0)

    with repo._connect(write=True) as db:
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO versions(version_key,document_id,version,content_hash,parser,"
                "normalization,fingerprint) VALUES ('k','missing',1,'h','p','n','f')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO chunks VALUES ('c','missing_key',1,'{}')")
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO ingestion_jobs(job_id,status,version_key) "
                "VALUES ('j','created','missing_key')"
            )


def test_failure_job_isolation(tmp_path: Path) -> None:
    db_path = tmp_path / "fail.sqlite3"
    repo, _ = _repo(db_path)

    with pytest.raises(IngestionError, match="empty_content"):
        repo.publish("job-fail", "doc1", "a.txt", "k1", "h1", "txt-v1", "nfc", "fp", iter(()))
    repo.fail("job-fail", "empty_content")

    job = repo.job("job-fail")
    assert job.status == "failed"
    assert job.error == "empty_content"
    assert job.document_id is None
    assert job.version is None
    assert repo.documents(10, 0) == []

    res = repo.publish(
        "job-ok", "doc1", "a.txt", "k2", "h2", "txt-v1", "nfc", "fp", [_chunk("doc1", "k2", 0)]
    )
    assert res.status == "created"
    assert repo.job("job-fail").status == "failed"


def test_concurrent_same_key_duplicate_from_two_connections(tmp_path: Path) -> None:
    db_path = tmp_path / "concurrent.sqlite3"
    repo_a, _ = _repo(db_path)
    repo_b, _ = _repo(db_path)
    barrier = Barrier(2)

    def run(repo: Repository, job_id: str) -> IngestResult:
        barrier.wait(timeout=5)
        return repo.publish(
            job_id,
            "doc1",
            "a.txt",
            "same-key",
            "h1",
            "txt-v1",
            "nfc",
            "fp",
            [_chunk("doc1", "same-key", 0), _chunk("doc1", "same-key", 1)],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(run, repo_a, "job-a")
        fut_b = pool.submit(run, repo_b, "job-b")
        results = [fut_a.result(), fut_b.result()]

    assert sorted(r.status for r in results) == ["created", "duplicate"]
    assert all(r.document_id == "doc1" and r.version == 1 and r.chunk_count == 2 for r in results)
    assert len(repo_a.chunks("doc1", 1, 10, 0)) == 2
    assert {repo_a.job("job-a").status, repo_a.job("job-b").status} == {"created", "duplicate"}


def test_migrations_and_future_schema_rejection(tmp_path: Path) -> None:
    db_path = tmp_path / "schema.sqlite3"
    repo, _ = _repo(db_path)
    assert repo.documents(10, 0) == []

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert conn.execute("SELECT version FROM schema_versions").fetchall() == [(1,)]

    reopened, _ = _repo(db_path)
    assert reopened.documents(10, 0) == []

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA user_version=2")

    future_repo, _ = _repo(db_path)
    with pytest.raises(IngestionError, match="unsupported_schema") as caught:
        future_repo.documents(10, 0)
    assert caught.value.status == 503


def test_persisted_wal_mode_and_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wal_db = tmp_path / "wal.sqlite3"
    repo_wal, _ = _repo(wal_db, wal=True)
    repo_wal.documents(10, 0)
    with sqlite3.connect(wal_db) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

    no_wal_db = tmp_path / "nowal.sqlite3"
    repo_no_wal, _ = _repo(no_wal_db, wal=False)
    repo_no_wal.documents(10, 0)
    with sqlite3.connect(no_wal_db) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal"

    fallback_db = tmp_path / "fallback.sqlite3"
    real_connect = sqlite3.connect

    class WalRejectingConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
            if sql.strip().upper() == "PRAGMA JOURNAL_MODE=WAL":
                raise sqlite3.OperationalError("wal unsupported")
            return super().execute(sql, parameters)

    def fake_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        kwargs["factory"] = WalRejectingConnection
        return cast(sqlite3.Connection, real_connect(*args, **kwargs))

    monkeypatch.setattr(sqlite3, "connect", fake_connect)
    repo_fb, _ = _repo(fallback_db, wal=True)
    assert repo_fb.documents(10, 0) == []


def test_storage_admission_and_coarse_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny_db = tmp_path / "tiny.sqlite3"
    tiny_repo, _ = _repo(tiny_db, budget_gb=1e-9)
    with pytest.raises(ResourceUnavailable):
        tiny_repo.documents(10, 0)

    coarse_db = tmp_path / "coarse.sqlite3"
    repo, resources = _repo(coarse_db)
    # Initialize schema first so we measure only publish's chunk preflight calls.
    repo.documents(10, 0)

    calls: list[int] = []
    real_check = resources.check_storage

    def spy_check(*, used_bytes: int, additional_bytes: int, path: Path) -> None:
        calls.append(additional_bytes)
        real_check(used_bytes=used_bytes, additional_bytes=additional_bytes, path=path)

    monkeypatch.setattr(resources, "check_storage", spy_check)

    # 120 chunks * (~2500 estimated bytes each) > 256 KiB cumulative growth.
    many_chunks = [_chunk("doc1", "big-key", i, "x" * 100) for i in range(120)]
    res = repo.publish(
        "job-big", "doc1", "big.txt", "big-key", "h", "txt-v1", "nfc", "fp", many_chunks
    )
    assert res.chunk_count == 120
    # 1 initial write preflight + 2 coarse 256 KiB preflights (not 120 per-chunk stats).
    assert len(calls) <= 4


def test_pagination_bounds(tmp_path: Path) -> None:
    db_path = tmp_path / "page.sqlite3"
    repo, _ = _repo(db_path)
    repo.publish(
        "job1",
        "doc1",
        "a.txt",
        "k1",
        "h1",
        "txt-v1",
        "nfc",
        "fp",
        [_chunk("doc1", "k1", 0), _chunk("doc1", "k1", 1)],
    )

    assert len(repo.chunks("doc1", 1, limit=1, offset=1)) == 1
    for bad_limit, bad_offset in [
        (0, 0),
        (-1, 0),
        (101, 0),
        (True, 0),
        (1, -1),
        (1, 2_147_483_648),
        (1, False),
    ]:
        with pytest.raises(IngestionError, match="invalid_pagination") as exc_docs:
            repo.documents(bad_limit, bad_offset)
        assert exc_docs.value.status == 422
        with pytest.raises(IngestionError, match="invalid_pagination"):
            repo.versions("doc1", bad_limit, bad_offset)
        with pytest.raises(IngestionError, match="invalid_pagination"):
            repo.chunks("doc1", 1, bad_limit, bad_offset)
