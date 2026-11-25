"""Integration tests for Phase 04 hybrid retrieval and sparse index endpoints."""

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from evorag.config import Embeddings, Settings, Storage
from evorag.main import create_app

pytestmark = pytest.mark.anyio
API = "/api/v1"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _transport(settings: Settings, request: httpx.Request) -> httpx.Response:
    gen = settings.runtime.generation_model
    emb = settings.runtime.embedding_model
    if request.url.path == "/api/tags":
        return httpx.Response(
            200,
            json={"models": [{"name": m, "digest": "d-" + m, "size": 16} for m in (gen, emb)]},
        )
    if request.url.path == "/api/ps":
        return httpx.Response(200, json={"models": []})
    body = json.loads(request.content)
    if request.url.path == "/api/show":
        return httpx.Response(
            200, json={"capabilities": ["embedding" if body["model"] == emb else "completion"]}
        )
    if request.url.path == "/api/embed":
        return httpx.Response(
            200, json={"model": emb, "embeddings": [[1.0, 0.0]] * len(body["input"])}
        )
    raise AssertionError(f"Unexpected path: {request.url.path}")


@asynccontextmanager
async def _client(
    directory: Path,
    handler: Callable[[Settings, httpx.Request], httpx.Response] = _transport,
) -> AsyncIterator[tuple[httpx.AsyncClient, Any]]:
    settings = Settings(
        storage=Storage(database_path=directory / "api.sqlite3"),
        embeddings=Embeddings(dimensions=2),
    )
    app = create_app(settings, transport=httpx.MockTransport(lambda r: handler(settings, r)))
    async with app.router.lifespan_context(app):
        with (
            patch.object(app.state.inference.coordinator, "ram_probe", return_value=32 * 1024**3),
            patch.object(app.state.ingestion.resources, "_ram_probe", return_value=32 * 1024**3),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                yield client, app


async def test_hybrid_retrieve_and_index_endpoints(tmp_path: Path) -> None:
    async with _client(tmp_path) as (client, _):
        r1 = await client.post(
            API + "/documents",
            headers={"X-Filename": "errors.md"},
            content=b"HTTP 404 vector_store_corrupted guide",
        )
        assert r1.status_code == 201
        doc_id = r1.json()["document_id"]
        assert (await client.post(f"{API}/documents/{doc_id}/embed")).status_code == 200

        status = await client.get(API + "/index/status")
        assert status.status_code == 200
        assert status.json()["doc_count"] == 1
        assert status.json()["analyzer_version"] == "v1"

        rebuilt = await client.post(API + "/index/rebuild")
        assert rebuilt.status_code == 200
        job_id = rebuilt.json()["job_id"]
        assert rebuilt.json()["status"] == "complete"
        assert (await client.get(f"{API}/index/jobs/{job_id}")).json()["status"] == "complete"

        hybrid = await client.post(
            API + "/retrieve",
            json={
                "query": "HTTP 404",
                "mode": "hybrid",
                "top_k": 3,
                "reranking_enabled": True,
                "include_explanations": True,
                "filters": {"file_type": ".md"},
            },
        )
        assert hybrid.status_code == 200, hybrid.text
        payload = hybrid.json()
        assert payload["mode"] == "hybrid"
        assert payload["effective_mode"] == "hybrid"
        assert payload["reranking_applied"] is True
        assert len(payload["results"]) == 1
        assert payload["explanations"][0]["bm25_score"] > 0

        sparse = await client.post(
            API + "/retrieve",
            json={
                "query": "vector_store_corrupted",
                "mode": "sparse",
                "include_explanations": True,
            },
        )
        assert sparse.status_code == 200
        assert sparse.json()["effective_mode"] == "sparse"
        assert len(sparse.json()["results"]) == 1


async def test_http_sparse_fallback_when_ollama_unavailable(tmp_path: Path) -> None:
    def _down(settings: Settings, request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    async with _client(tmp_path, _down) as (client, _):
        r1 = await client.post(
            API + "/documents",
            headers={"X-Filename": "gpu.txt"},
            content=b"RTX 3050 Laptop GPU 4GB VRAM",
        )
        assert r1.status_code == 201
        resp = await client.post(
            API + "/retrieve",
            json={"query": "RTX 3050", "mode": "hybrid", "include_explanations": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["degraded"] is True
        assert data["effective_mode"] == "sparse"
        assert data["degradation_reason"] == "provider_unavailable"
        assert len(data["results"]) == 1
