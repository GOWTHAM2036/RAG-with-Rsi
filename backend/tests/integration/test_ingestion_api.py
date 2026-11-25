from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from evorag.config import Ingestion, Settings, Storage
from evorag.contracts import DocumentMetadata
from evorag.document_api import service
from evorag.ingestion import IngestionService
from evorag.interfaces import Ingestor
from evorag.main import create_app
from evorag.resources import GIB, ResourceManager

pytestmark = pytest.mark.anyio


async def test_foundation_ingestor_port(tmp_path: Path) -> None:
    settings = Settings(storage=Storage(database_path=tmp_path / "unused"))
    parser: Ingestor = IngestionService(settings, ResourceManager(settings, lambda: 4 * GIB))
    document = await parser.parse(
        b"one\r\ntwo", DocumentMetadata(source="a.txt", media_type="text/plain")
    )
    assert document.text == "one\ntwo"
    assert not settings.storage.database_path.exists()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@asynccontextmanager
async def api(folder: Path, ram: int = 4 * GIB) -> AsyncIterator[httpx.AsyncClient]:
    settings = Settings(
        ingestion=Ingestion(max_file_size_mb=1),
        storage=Storage(database_path=folder / "db.sqlite3"),
    )
    app = create_app(settings)
    engine = IngestionService(settings, ResourceManager(settings, lambda: ram))
    app.dependency_overrides[service] = lambda: engine
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        yield client


async def test_upload_query_and_jobs(tmp_path: Path) -> None:
    url = "/api/v1/documents"
    async with api(tmp_path) as client:
        assert (await client.get(url)).json() == []
        head = {"X-Filename": "notes.txt"}
        created = await client.post(url, headers=head, content="A 🐍".encode())
        assert created.status_code == 201
        doc = created.json()["document_id"]
        repeat = await client.post(url, headers=head, content="A 🐍".encode())
        assert repeat.status_code == 200 and repeat.json()["status"] == "duplicate"
        second = await client.post(url, headers=head, content=b"revision")
        assert second.json()["version"] == 2
        assert (await client.get(f"{url}/{doc}")).json()["latest_version"] == 2
        assert (await client.get(url, params={"offset": 1})).json() == []
        versions = (
            await client.get(f"{url}/{doc}/versions", params={"limit": 1, "offset": 1})
        ).json()
        assert [v["version"] for v in versions] == [1]
        assert (await client.get(f"{url}/{doc}/chunks")).json()[0]["text"] == "revision"
        assert (await client.get(f"{url}/{doc}/chunks?version=1")).json()[0]["text"] == "A 🐍"
        assert (await client.get(f"{url}/{doc}/chunks?version=3")).status_code == 404
        assert (await client.get(f"{url}/unknown")).status_code == 404
        assert (await client.get("/api/v1/ingestion-jobs/unknown")).status_code == 404
        failed = await client.post(url, headers=head, content=b"\xff")
        assert failed.status_code == 422
        job = (await client.get("/api/v1/ingestion-jobs/" + failed.json()["job_id"])).json()
        assert job["status"] == "failed" and job["error"] == failed.json()["error"]


async def test_admission_and_request_limits(tmp_path: Path) -> None:
    url = "/api/v1/documents"
    async with api(tmp_path, ram=GIB) as client:
        denied = await client.post(url, headers={"X-Filename": "a.txt"}, content=b"private data")
        assert denied.status_code == 503 and "private data" not in denied.text
    head = {"X-Filename": "a.txt"}
    async with api(tmp_path) as client:
        for headers, code in [
            ({}, 422),
            ({"X-Filename": "../a.txt"}, 422),
            ({**head, "Content-Encoding": "gzip"}, 415),
            ({**head, "Content-Length": "x"}, 400),
            ({**head, "Content-Length": "1048577"}, 413),
            ({**head, "Content-Length": "3"}, 400),
        ]:
            result = await client.post(url, headers=headers, content=b"z")
            assert result.status_code == code
            assert "error" in result.json()

        async def stream() -> AsyncIterator[bytes]:
            yield b"a" * (1024**2)
            yield b"b"

        assert (await client.post(url, headers=head, content=stream())).status_code == 413
        assert (await client.get(url + "?limit=101")).status_code == 422
        assert (await client.get(url + "?offset=-1")).status_code == 422


async def test_real_lifespan_dependency(tmp_path: Path) -> None:
    app = create_app(Settings(storage=Storage(database_path=tmp_path / "db")))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/api/v1/documents")).status_code == 503
        async with app.router.lifespan_context(app):
            assert (await client.get("/api/v1/documents")).json() == []
