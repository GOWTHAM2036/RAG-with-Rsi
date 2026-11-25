import hashlib
from pathlib import Path

import pytest
from evorag.config import Chunking, Ingestion, Settings, Storage
from evorag.documents import BatchFailure, IngestResult
from evorag.errors import IngestionError
from evorag.ingestion import IngestionService
from evorag.resources import GIB, ResourceManager


def engine(folder: Path, *, overlap: int = 0, ram: int = 4 * GIB) -> IngestionService:
    settings = Settings(
        storage=Storage(database_path=folder / "store.db"),
        chunking=Chunking(chunk_size=4, overlap=overlap),
        ingestion=Ingestion(max_batch_documents=3, max_file_size_mb=1),
    )
    return IngestionService(settings, ResourceManager(settings, lambda: ram))


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("x.txt", b"\xef\xbb\xbfcafe\xcc\x81\r\nline 2"),
        ("x.md", b"# Header\n\n## Section\ntext"),
        ("x.csv", b'n,v\r\n1,"a,b"\r\n'),
        ("x.json", b'{"x":["a",2]}'),
    ],
)
def test_ingest_formats(name: str, payload: bytes, tmp_path: Path) -> None:
    service = engine(tmp_path)
    created = service.ingest(name, payload)
    assert created.status == "created" and created.version == 1
    versions = service.repository.versions(created.document_id, 20, 0)
    assert versions[0].content_hash == hashlib.sha256(payload).hexdigest()
    chunks = service.repository.chunks(created.document_id, 1, 100, 0)
    assert len(chunks) == created.chunk_count
    assert all(c.sources and c.start < c.end for c in chunks)
    assert service.repository.job(created.job_id).document_id == created.document_id
    repeat = engine(tmp_path).ingest(name, payload)
    assert repeat.status == "duplicate" and repeat.document_id == created.document_id
    other = engine(tmp_path / "other").ingest(name, payload)
    assert other.document_id == created.document_id
    assert engine(tmp_path / "other").repository.chunks(other.document_id, 1, 100, 0) == chunks


def test_revisions_and_logical_identity(tmp_path: Path) -> None:
    service = engine(tmp_path)
    a = service.ingest("a.txt", b"one")
    b = service.ingest("b.txt", b"one")
    assert a.document_id != b.document_id
    assert service.ingest("a.txt", b"two").version == 2
    assert service.ingest("a.txt", b"one").version == 1
    changed = engine(tmp_path, overlap=1).ingest("a.txt", b"one")
    assert changed.version == 3
    old = service.repository.chunks(a.document_id, 1, 1, 0)[0]
    new = service.repository.chunks(a.document_id, 3, 1, 0)[0]
    assert new.chunk_id != old.chunk_id


def test_bounded_batch(tmp_path: Path) -> None:
    service = engine(tmp_path)
    (tmp_path / "a.txt").write_bytes(b"ok")
    batch = service.ingest_batch(tmp_path, ["a.txt", "absent.txt", "a.txt"])
    assert [r.status for r in batch] == ["created", "failed", "duplicate"]
    assert isinstance(batch[0], IngestResult) and isinstance(batch[1], BatchFailure)
    assert batch[1].job_id and service.repository.job(batch[1].job_id).status == "failed"
    with pytest.raises(IngestionError, match="batch_too_large"):
        service.ingest_batch(tmp_path, ["a.txt"] * 4)
    assert engine(tmp_path, ram=1).ingest_batch(tmp_path, ["a.txt"])[0].status == "failed"
    with pytest.raises(IngestionError) as caught:
        service.ingest("a.txt", b"x" * (1024**2 + 1))
    assert caught.value.status == 413
