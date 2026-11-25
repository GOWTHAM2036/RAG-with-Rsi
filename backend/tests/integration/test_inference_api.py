"""Offline public-API checks with a synthetic Ollama transport and synthetic RAM readings."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from evorag.config import Embeddings, Settings, Storage
from evorag.inference_api import _while_connected
from evorag.inference_types import ProviderError
from evorag.main import create_app
from starlette.requests import Request
from starlette.types import Message

pytestmark = pytest.mark.anyio
CHAT = {"messages": [{"role": "user", "content": "synthetic prompt"}]}
API = "/api/v1"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def fixture_transport(settings: Settings, request: httpx.Request) -> httpx.Response:
    generation = settings.runtime.generation_model
    embedding = settings.runtime.embedding_model
    path = request.url.path
    if path == "/api/tags":
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": m, "digest": "digest-" + m, "size": 16}
                    for m in (generation, embedding)
                ]
            },
        )
    if path == "/api/ps":
        return httpx.Response(200, json={"models": []})
    body = json.loads(request.content)
    if path == "/api/show":
        capabilities = ["completion"] if body["model"] == generation else ["embedding"]
        return httpx.Response(200, json={"capabilities": capabilities})
    if path == "/api/embed":
        return httpx.Response(
            200, json={"model": embedding, "embeddings": [[1.0, 0.0]] * len(body["input"])}
        )
    assert path == "/api/chat"
    event = {
        "model": generation,
        "message": {"role": "assistant", "content": "fixture", "thinking": "PRIVATE"},
        "done": True,
        "done_reason": "stop",
    }
    if body["stream"]:
        first = {**event, "done": False}
        return httpx.Response(200, content=(json.dumps(first) + "\n" + json.dumps(event)).encode())
    return httpx.Response(200, json=event)


@asynccontextmanager
async def client_for(
    directory: Path,
    handler: Callable[[Settings, httpx.Request], httpx.Response] = fixture_transport,
) -> AsyncIterator[tuple[httpx.AsyncClient, Any]]:
    settings = Settings(
        storage=Storage(database_path=directory / "meta.sqlite3"),
        embeddings=Embeddings(dimensions=2),
    )
    app = create_app(settings, transport=httpx.MockTransport(lambda r: handler(settings, r)))
    async with app.router.lifespan_context(app):
        with patch.object(app.state.inference.coordinator, "ram_probe", return_value=32 * 1024**3):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                yield client, app


async def test_metadata_and_generation(tmp_path: Path) -> None:
    async with client_for(tmp_path) as (client, app):
        assert (await client.get(API + "/models")).json()[0]["size_bytes"] == 16
        model_status = (await client.get(API + "/models/status")).json()
        assert model_status["reachable"] and model_status["resident"] == []
        assert model_status["generation_installed"] and model_status["embedding_installed"]
        ready = (await client.get("/ready?inference=true")).json()
        assert ready["application"] == "ready" and ready["inference"]["reachable"]
        assert "/api/v1/generate/stream" in app.openapi()["paths"]
        result = await client.post(API + "/generate", json=CHAT)
        assert result.status_code == 200, result.text
        assert result.json()["content"] == "fixture" and "PRIVATE" not in result.text
        streamed = await client.post(API + "/generate/stream", json=CHAT)
        assert streamed.headers["content-type"].startswith("application/x-ndjson")
        events = [json.loads(line) for line in streamed.text.splitlines()]
        assert [item["done"] for item in events] == [False, True]
        assert "PRIVATE" not in streamed.text


async def test_embed_revision_and_manage_job(tmp_path: Path) -> None:
    async with client_for(tmp_path) as (client, app):
        # Empty corpus returns zero results.
        empty_ret = await client.post(API + "/retrieve", json={"question": "sample"})
        assert empty_ret.status_code == 200
        assert empty_ret.json()["results"] == []

        # The only replaced hardware sensor is available RAM; every inference route still admits.
        with patch.object(app.state.ingestion.resources, "_ram_probe", return_value=32 * 1024**3):
            made = await client.post(
                API + "/documents", headers={"X-Filename": "note.txt"}, content=b"sample"
            )
        assert made.status_code == 201, made.text
        doc_id = made.json()["document_id"]
        url = f"{API}/documents/{doc_id}/embed"
        prepared = await client.post(url + "?prepare_only=true&version=1")
        assert prepared.status_code == 200, prepared.text
        job_id = prepared.json()["job_id"]
        path = f"{API}/embedding-jobs/{job_id}"
        assert (await client.get(path)).json()["status"] == "pending"
        assert (await client.post(path + "/cancel")).json()["status"] == "cancelled"
        finished = await client.post(url + "?version=1")
        assert finished.json()["status"] == "complete", finished.text
        response = await client.post(API + "/embeddings", json={"texts": ["sample"]})
        assert response.json()["embeddings"] == [[1.0, 0.0]], response.text
        assert response.json()["cache_hits"] == 1
        assert (await client.get(API + "/embeddings/cache/stats")).json()["hits"] >= 1
        retrieved = await client.post(
            API + "/retrieve", json={"question": "sample", "top_k": 3, "metric": "cosine"}
        )
        assert retrieved.status_code == 200, retrieved.text
        payload = retrieved.json()
        assert payload["metric"] == "cosine"
        assert len(payload["results"]) == 1
        assert payload["results"][0]["rank"] == 1
        assert payload["results"][0]["chunk"]["document_id"] == doc_id
        assert payload["results"][0]["chunk"]["text"] == "sample"
        assert payload["results"][0]["score"] == pytest.approx(1.0)

        # Validation errors on /retrieve.
        for bad_payload in (
            {"question": "   "},
            {"question": "ok", "top_k": 0},
            {"question": "ok", "top_k": 101},
            {"question": "ok", "metric": "manhattan"},
            {"question": "ok", "version": 0},
        ):
            assert (await client.post(API + "/retrieve", json=bad_payload)).status_code == 422

        # Corrupted/missing .f32 file after commit fails closed with 500 vector_store_corrupted.
        def _unlink_vectors() -> None:
            for vec_file in tmp_path.glob("*.vectors.f32"):
                vec_file.unlink()

        await asyncio.to_thread(_unlink_vectors)
        corrupted = await client.post(API + "/retrieve", json={"question": "sample"})
        assert corrupted.status_code == 500
        assert corrupted.json() == {"error": "vector_store_corrupted"}


async def test_provider_failures_are_redacted(tmp_path: Path) -> None:
    def unavailable(settings: Settings, request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "PRIVATE source"})

    async with client_for(tmp_path, unavailable) as (client, _):
        assert not (await client.get(API + "/models/status")).json()["reachable"]
        for path in ("generate", "generate/stream"):
            response = await client.post(f"{API}/{path}", json=CHAT)
            assert response.status_code >= 400 and "PRIVATE" not in response.text


async def test_raw_upload_limits_and_validation(tmp_path: Path) -> None:
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.post(API + "/generate", json=CHAT)).status_code == 503
    async with client_for(tmp_path) as (client, app):
        for headers, expected in [
            ({"content-length": "-1"}, 400),
            ({"content-encoding": "gzip"}, 415),
            ({"content-length": "99999999"}, 413),
            ({"content-length": "4"}, 400),
        ]:
            assert (
                await client.post(API + "/generate", headers=headers, content=b"xx")
            ).status_code == expected

        async def body() -> AsyncIterator[bytes]:
            yield b"a" * 262144
            yield b"b"

        assert (await client.post(API + "/generate", content=body())).status_code == 413
        invalid = await client.post(API + "/generate", json={**CHAT, "base_url": "http://private"})
        assert invalid.status_code == 422 and "private" not in invalid.text
        slots = app.state.inference.readers
        while not slots.locked():
            await slots.acquire()
        try:
            assert (await client.post(API + "/generate", json=CHAT)).status_code == 503
        finally:
            slots.release()


async def test_connection_loss_cancels_pending_work() -> None:
    released = asyncio.Event()

    async def disconnect() -> Message:
        return {"type": "http.disconnect"}

    async def waiting() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            released.set()

    with pytest.raises(ProviderError, match="request_cancelled"):
        await _while_connected(Request({"type": "http"}, disconnect), waiting())
    assert released.is_set()


async def test_ndjson_error_after_first_output(tmp_path: Path) -> None:
    def malformed(settings: Settings, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            first = {
                "model": settings.runtime.generation_model,
                "message": {"role": "assistant", "content": "ok"},
                "done": False,
            }
            return httpx.Response(200, content=json.dumps(first).encode() + b"\nnot-json PRIVATE\n")
        return fixture_transport(settings, request)

    async with client_for(tmp_path, malformed) as (client, _):
        response = await client.post(API + "/generate/stream", json=CHAT)
        assert response.status_code == 200
        assert json.loads(response.text.splitlines()[-1]) == {"error": "invalid_response"}
        assert "PRIVATE" not in response.text
