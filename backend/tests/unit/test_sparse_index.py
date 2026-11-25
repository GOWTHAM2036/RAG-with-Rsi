"""Unit tests for persistent SQLite inverted index, revisions and recovery."""

from pathlib import Path
from unittest.mock import patch

import pytest
from evorag.analyzer import TextAnalyzer
from evorag.config import Chunking, Settings, Storage
from evorag.errors import IngestionError, ResourceUnavailable
from evorag.filtering import MetadataFilter
from evorag.ingestion import IngestionService
from evorag.resources import GIB, ResourceManager
from evorag.sparse_index import SparseIndex


def _setup(tmp_path: Path) -> tuple[IngestionService, SparseIndex]:
    settings = Settings(
        storage=Storage(database_path=tmp_path / "idx.sqlite3"),
        chunking=Chunking(chunk_size=64, overlap=0, strategy="fixed"),
    )
    ingestion = IngestionService(settings, ResourceManager(settings, lambda: 8 * GIB))
    index = SparseIndex(ingestion.repository, TextAnalyzer.from_config(settings.sparse))
    return ingestion, index


def test_incremental_idempotent_and_revision_replacement(tmp_path: Path) -> None:
    ingestion, index = _setup(tmp_path)
    assert index.status().doc_count == 0

    d1_v1 = ingestion.ingest("guide.md", b"HTTP 404 legacy error")
    assert index.index_version(d1_v1.document_id, 1) == 1
    # Idempotent re-index is a no-op.
    assert index.index_version(d1_v1.document_id, 1) == 0
    assert index.status().doc_count == 1

    # Revision 2 supersedes revision 1 in active corpus.
    d1_v2 = ingestion.ingest("guide.md", b"RTX 3050 GPU guide")
    assert d1_v2.version == 2
    assert index.index_version(d1_v2.document_id, 2) == 1
    status = index.status()
    assert status.doc_count == 1
    assert status.chunk_count == 2

    # Default latest_only=True excludes "legacy" from v1.
    _, _, dfs_legacy, cands_legacy = index.query_candidates(["legacy"], MetadataFilter())
    assert dfs_legacy == {}
    assert cands_legacy == []

    # Explicit version=1 or latest_only=False still finds v1.
    _, _, dfs_v1, cands_v1 = index.query_candidates(
        ["legacy"], MetadataFilter(version=1, latest_only=False)
    )
    assert dfs_v1 == {"legacy": 1}
    assert len(cands_v1) == 1

    # Forced re-index replaces postings cleanly.
    assert index.index_version(d1_v2.document_id, 2, force=True) == 1
    assert index.status().doc_count == 1


def test_atomic_rollback_and_rebuild_jobs(tmp_path: Path) -> None:
    ingestion, index = _setup(tmp_path)
    d1 = ingestion.ingest("a.txt", b"Python 3.12 async")
    index.index_version(d1.document_id, 1)
    before = index.status()

    d2 = ingestion.ingest("b.txt", b"vector_store_corrupted recovery")
    with patch.object(index, "_budget", side_effect=IngestionError("disk_full", 507)):
        with pytest.raises(IngestionError, match="disk_full"):
            index.index_version(d2.document_id, 1)

    # Interrupted index leaves prior committed state intact.
    assert index.status().doc_count == before.doc_count

    job = index.rebuild_all(force=True)
    assert job.status == "complete"
    assert job.indexed_chunks == 2
    assert index.job_status(job.job_id).status == "complete"
    assert index.status().doc_count == 2

    with pytest.raises(IngestionError, match="index_job_not_found"):
        index.job_status("missing")
    with pytest.raises(IngestionError, match="document_revision_not_found"):
        index.index_version("missing-doc", 1)

    # Metadata filters: source and file_type.
    _, _, dfs_src, cands_src = index.query_candidates(
        ["python"], MetadataFilter(source="a.txt", file_type="txt")
    )
    assert dfs_src == {"python": 1} and len(cands_src) == 1
    _, _, dfs_miss, _ = index.query_candidates(["python"], MetadataFilter(file_type=".md"))
    assert dfs_miss == {}


def test_candidate_selection_preserves_rare_terms_and_rebuild_admission(tmp_path: Path) -> None:
    ingestion, index = _setup(tmp_path)
    for i in range(60):
        res = ingestion.ingest(f"doc_{i:03d}.txt", f"system check item {i}".encode())
        index.index_version(res.document_id, 1)
    rare = ingestion.ingest("zzz_rare_doc.txt", b"system vector_store_corrupted fatal")
    index.index_version(rare.document_id, 1)

    _, _, dfs, cands = index.query_candidates(
        ["system", "vector_store_corrupted"], MetadataFilter(), max_candidates=20
    )
    assert dfs["vector_store_corrupted"] == 1
    assert len(cands) == 20
    assert any("vector_store_corrupted" in cand.term_frequencies for cand, _ in cands)

    # Rebuild must enforce the 3 GiB RAM admission floor.
    low_ram_settings = Settings(storage=Storage(database_path=tmp_path / "idx.sqlite3"))
    low_ram_ing = IngestionService(low_ram_settings, ResourceManager(low_ram_settings, lambda: GIB))
    low_ram_index = SparseIndex(low_ram_ing.repository, TextAnalyzer())
    with pytest.raises(ResourceUnavailable):
        low_ram_index.rebuild_all(force=True)
