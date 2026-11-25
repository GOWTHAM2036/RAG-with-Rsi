"""Public ingestion and repository contracts."""

import hashlib
import json
from typing import Literal

from pydantic import AwareDatetime

from .contracts import Chunk, Contract


def digest(*parts: str) -> str:
    return hashlib.sha256(
        json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


class SourceSpan(Contract):
    label: str
    start: int
    end: int
    parsed_start: int | None
    parsed_end: int | None


class StoredChunk(Chunk):
    token_count: int
    estimated: bool
    sources: tuple[SourceSpan, ...]


class DocumentInfo(Contract):
    document_id: str
    source: str
    latest_version: int


class VersionInfo(Contract):
    document_id: str
    version: int
    content_hash: str
    fingerprint: str
    parser: str
    normalization: str
    chunk_count: int
    created_at: AwareDatetime


class IngestResult(Contract):
    job_id: str
    document_id: str
    version: int
    chunk_count: int
    status: Literal["created", "duplicate"]


class JobInfo(Contract):
    job_id: str
    status: Literal["created", "duplicate", "failed"]
    error: str | None = None
    document_id: str | None = None
    version: int | None = None


class BatchFailure(Contract):
    status: Literal["failed"] = "failed"
    error: str
    job_id: str | None = None
