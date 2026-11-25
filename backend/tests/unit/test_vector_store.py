"""Tests for atomic disk-backed float32 vectors and crash recovery."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from evorag.config import Chunking, Embeddings, Settings, Storage
from evorag.embedding_engine import EmbeddingEngine, encode_vector
from evorag.inference_types import ModelInfo, ProviderError
from evorag.ingestion import IngestionService
from evorag.resources import GIB, ResourceManager
from evorag.vector_store import VectorStore

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_vector_store_append_map_and_recovery(tmp_path: Path) -> None:
    missing_dir_store = VectorStore(tmp_path / "missing_dir", "meta.sqlite3")
    assert missing_dir_store.total_bytes() == 0

    store = VectorStore(tmp_path, "meta.sqlite3")
    sid = "a" * 64
    assert store.total_bytes() == 0

    with store.map_readonly(sid, 0, 2) as empty:
        assert empty.shape == (0, 2)

    b0 = encode_vector((1.0, 0.0), 2, False)
    b1 = encode_vector((0.0, 1.0), 2, False)
    assert store.append_batch(sid, 0, 2, []) == 0
    assert store.append_batch(sid, 0, 2, [b0, b1]) == 2
    assert store.total_bytes() == 16

    with store.map_readonly(sid, 2, 2) as mat:
        assert mat.dtype == np.dtype("<f4")
        assert mat.shape == (2, 2)
        assert mat[0].tolist() == pytest.approx([1.0, 0.0])
        assert mat[1].tolist() == pytest.approx([0.0, 1.0])

    # Simulate an interrupted partial write appended to the .f32 file without SQLite commit.
    vec_path = store.path_for(sid)
    with vec_path.open("ab") as handle:
        handle.write(b"\x01\x02\x03")
    assert vec_path.stat().st_size == 19

    # Read-only memmap still safely maps only the 2 committed rows.
    with store.map_readonly(sid, 2, 2) as mat:
        assert mat.shape == (2, 2)
        assert mat[1].tolist() == pytest.approx([0.0, 1.0])

    # Recovery truncates the uncommitted tail back to 16 bytes, and Windows releases the handle.
    store.recover(sid, 2, 2)
    assert vec_path.stat().st_size == 16

    # Subsequent append writes cleanly at row 2.
    b2 = encode_vector((0.6, 0.8), 2, False)
    assert store.append_batch(sid, 2, 2, [b2]) == 3
    with store.map_readonly(sid, 3, 2) as mat:
        assert mat[2].tolist() == pytest.approx([0.6, 0.8])


def test_vector_store_corruption_and_validation(tmp_path: Path) -> None:
    store = VectorStore(tmp_path, "meta.sqlite3")
    sid = "b" * 64

    # Missing file when committed_rows > 0 fails closed.
    with pytest.raises(ProviderError, match="vector_store_corrupted"):
        store.recover(sid, 1, 2)
    with pytest.raises(ProviderError, match="vector_store_corrupted"):
        with store.map_readonly(sid, 1, 2):
            pass

    # Truncated file when committed_rows > 0 fails closed.
    vec_path = store.path_for(sid)
    vec_path.write_bytes(b"\x00" * 4)
    with pytest.raises(ProviderError, match="vector_store_corrupted"):
        store.recover(sid, 1, 2)
    with pytest.raises(ProviderError, match="vector_store_corrupted"):
        with store.map_readonly(sid, 1, 2):
            pass

    # Invalid arguments and blob sizes.
    with pytest.raises(ProviderError, match="invalid_response"):
        store.append_batch(sid, 0, 2, [b"\x00" * 4])
    with pytest.raises(ValueError):
        store.recover(sid, -1, 2)
    with pytest.raises(ValueError):
        with store.map_readonly(sid, 0, 0):
            pass


class _StubRuntime:
    def __init__(self) -> None:
        self.model = ModelInfo(name="qwen3-embedding:0.6b", digest="d1", size_bytes=64)

    async def models(self) -> tuple[ModelInfo, ...]:
        return (self.model,)

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple((float(len(t)), 1.0) for t in texts)


class _StubAdmission:
    def __init__(self, model: ModelInfo) -> None:
        self.model = model

    def reserve(self, model_name: str, capability: str):  # type: ignore[no-untyped-def]
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _cm():  # type: ignore[no-untyped-def]
            yield self.model

        return _cm()


async def test_interrupted_sqlite_commit_recovers_on_resume(tmp_path: Path) -> None:
    settings = Settings(
        storage=Storage(database_path=tmp_path / "db.sqlite3"),
        embeddings=Embeddings(dimensions=2, batch_size=2),
        chunking=Chunking(chunk_size=1, overlap=0, strategy="fixed"),
    )
    ingestion = IngestionService(settings, ResourceManager(settings, lambda: 8 * GIB))
    doc = ingestion.ingest("notes.txt", b"abcdefghijklmnop")
    runtime = _StubRuntime()
    engine = EmbeddingEngine(settings, runtime, _StubAdmission(runtime.model), ingestion.repository)

    space = await engine._space()
    original_append = engine.store.vectors.append_batch
    calls = 0

    def fail_after_vector_write(
        space_id: str, committed_rows: int, dimensions: int, blobs: list[bytes]
    ) -> int:
        nonlocal calls
        calls += 1
        new_count = original_append(space_id, committed_rows, dimensions, blobs)
        if calls == 2:
            # Vector file now has 4 rows written, but SQLite transaction aborts before committing!
            raise ProviderError("storage_unavailable", 503)
        return new_count

    with patch.object(engine.store.vectors, "append_batch", side_effect=fail_after_vector_write):
        with pytest.raises(ProviderError, match="storage_unavailable"):
            await engine.run_job(doc.document_id, 1)

    # SQLite committed 2 rows, while .f32 has 4 rows (16 uncommitted extra bytes).
    sid, committed, rows = engine.store.space_rows(space)
    assert committed == 2
    assert len(rows) == 2
    vec_path = engine.store.vectors.path_for(sid)
    assert vec_path.stat().st_size == 32

    # Explicit recovery heals the .f32 file back to 16 bytes.
    assert engine.store.recover_space(space) == 2
    assert vec_path.stat().st_size == 16

    # Resuming the job completes all 4 chunks cleanly.
    finished = await engine.run_job(doc.document_id, 1)
    assert finished.status == "complete"
    assert finished.completed == 4
    sid, committed, rows = engine.store.space_rows(space)
    assert committed == 4
    assert len(rows) == 4
    assert vec_path.stat().st_size == 32
