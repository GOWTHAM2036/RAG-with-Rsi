"""Local document API. Uploads are raw byte streams; filenames are logical labels."""

from asyncio import timeout
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from .documents import DocumentInfo, IngestResult, JobInfo, StoredChunk, VersionInfo
from .errors import IngestionError
from .ingestion import IngestionService
from .parsing import validate_filename

router = APIRouter(prefix="/api/v1")


async def service(request: Request) -> IngestionService:
    active = getattr(request.app.state, "ingestion", None)
    if not isinstance(active, IngestionService):
        raise IngestionError("application_unavailable", 503)
    return active


Engine = Annotated[IngestionService, Depends(service)]
Limit = Annotated[int, Query(ge=1, le=100)]
Offset = Annotated[int, Query(ge=0, le=2_147_483_647)]


@router.post(
    "/documents",
    status_code=201,
    responses={200: {"model": IngestResult}},
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
            },
        }
    },
)
async def upload(
    request: Request,
    response: Response,
    engine: Engine,
    filename: Annotated[str, Header(alias="X-Filename", max_length=255)],
) -> IngestResult:
    limits = engine.settings.ingestion
    validate_filename(filename, limits)
    if request.headers.get("content-encoding", "identity").lower() != "identity":
        raise IngestionError("unsupported_content_encoding", 415)
    maximum = limits.max_file_size_mb * 1024**2
    declared = request.headers.get("content-length")
    if declared is not None:
        if not declared.isascii() or not declared.isdecimal() or len(declared) > 16:
            raise IngestionError("invalid_content_length", 400)
        if int(declared) > maximum:
            raise IngestionError("file_too_large", 413)
    with engine.resources.ingestion_slot():
        data = bytearray()
        try:
            async with timeout(30):
                async for block in request.stream():
                    if len(block) > maximum - len(data):
                        raise IngestionError("file_too_large", 413)
                    data.extend(block)
        except TimeoutError:
            raise IngestionError("upload_timeout", 408) from None
        except ClientDisconnect:
            raise IngestionError("upload_interrupted", 400) from None
        if declared is not None and len(data) != int(declared):
            raise IngestionError("content_length_mismatch", 400)
        result = await run_in_threadpool(engine.ingest_admitted, filename, data)
    response.status_code = 200 if result.status == "duplicate" else 201
    return result


@router.get("/documents")
def documents(engine: Engine, limit: Limit = 20, offset: Offset = 0) -> list[DocumentInfo]:
    return engine.repository.documents(limit, offset)


@router.get("/documents/{document_id}")
def document(document_id: str, engine: Engine) -> DocumentInfo:
    return engine.repository.document(document_id)


@router.get("/documents/{document_id}/versions")
def versions(
    document_id: str, engine: Engine, limit: Limit = 20, offset: Offset = 0
) -> list[VersionInfo]:
    return engine.repository.versions(document_id, limit, offset)


@router.get("/documents/{document_id}/chunks")
def chunks(
    document_id: str,
    engine: Engine,
    version: Annotated[int | None, Query(ge=1, le=2147483647)] = None,
    limit: Limit = 20,
    offset: Offset = 0,
) -> list[StoredChunk]:
    return engine.repository.chunks(document_id, version, limit, offset)


@router.get("/ingestion-jobs/{job_id}")
def job(job_id: str, engine: Engine) -> JobInfo:
    return engine.repository.job(job_id)
