"""CPU-only orchestration; each admitted operation owns one bounded input."""

import hashlib
from asyncio import to_thread
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from .chunking import EstimatedTokens, iter_chunks
from .config import Settings
from .contracts import Document, DocumentMetadata
from .documents import BatchFailure, IngestResult, digest
from .errors import IngestionError, ResourceUnavailable
from .parsing import parse_document, read_local, validate_filename
from .repository import Repository
from .resources import ResourceManager
from .sparse_index import SparseIndex


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        resources: ResourceManager,
        sparse_index: SparseIndex | None = None,
    ):
        self.settings = settings
        self.resources = resources
        self.repository = Repository(settings.storage, resources)
        self.tokenizer = EstimatedTokens()
        self.sparse_index = sparse_index

    async def parse(self, payload: bytes, metadata: DocumentMetadata) -> Document:
        """Implement the foundation Ingestor port for callers that only need parsed text."""

        def parse_in_slot() -> Document:
            with self.resources.ingestion_slot():
                name = validate_filename(metadata.source, self.settings.ingestion)
                parsed = parse_document(name, payload, self.settings.ingestion)
                return Document(
                    document_id=digest("document-v1", name), metadata=metadata, text=parsed.text
                )

        return await to_thread(parse_in_slot)

    def ingest(self, name: str, payload: bytes) -> IngestResult:
        with self.resources.ingestion_slot():
            return self.ingest_admitted(name, payload)

    def ingest_admitted(self, name: str, payload: bytes | bytearray) -> IngestResult:
        """Caller holds the admission permit from before it buffered the input."""
        job_id = uuid4().hex
        limits = self.settings.ingestion
        try:
            name = validate_filename(name, limits)
            if len(payload) > limits.max_file_size_mb * 1024**2:
                raise IngestionError("file_too_large", 413)
            content_hash = hashlib.sha256(payload).hexdigest()
            document_id = digest("document-v1", name)
            parser = name.rsplit(".", 1)[1].lower() + "-v1"
            normalization = "nfc-lf-v1/" + limits.whitespace
            fingerprint = digest(
                "chunks-v1", self.settings.chunking.model_dump_json(), self.tokenizer.identifier
            )
            version_key = digest(document_id, content_hash, parser, normalization, fingerprint)
            duplicate = self.repository.duplicate(job_id, version_key)
            if duplicate is not None:
                return duplicate
            parsed = parse_document(name, bytes(payload), limits)
            chunks = iter_chunks(parsed, document_id, version_key, self.settings.chunking, limits)
            result = self.repository.publish(
                job_id,
                document_id,
                name,
                version_key,
                content_hash,
                parser,
                normalization,
                fingerprint,
                chunks,
            )
            if result.status == "created" and self.sparse_index is not None:
                self.sparse_index.index_version(result.document_id, result.version)
            return result
        except IngestionError as exc:
            # Storage failures cannot reliably write their own job record.
            if exc.status < 500:
                self.repository.fail(job_id, exc.code)
                exc.job_id = job_id
            raise

    def ingest_file(self, root: Path, name: str) -> IngestResult:
        with self.resources.ingestion_slot():
            try:
                payload = read_local(root, name, self.settings.ingestion)
            except IngestionError as exc:
                job_id = uuid4().hex
                self.repository.fail(job_id, exc.code)
                exc.job_id = job_id
                raise
            return self.ingest_admitted(name, payload)

    def ingest_batch(self, root: Path, names: Sequence[str]) -> list[IngestResult | BatchFailure]:
        if len(names) > self.settings.ingestion.max_batch_documents:
            raise IngestionError("batch_too_large", 413)
        results: list[IngestResult | BatchFailure] = []
        for name in names:
            try:
                results.append(self.ingest_file(root, name))
            except IngestionError as exc:
                results.append(BatchFailure(error=exc.code, job_id=exc.job_id))
            except ResourceUnavailable:
                results.append(BatchFailure(error="resource_unavailable"))
        return results
