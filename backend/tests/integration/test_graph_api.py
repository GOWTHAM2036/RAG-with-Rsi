"""Offline integration tests for bounded provenance graph API."""

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from evorag.config import Embeddings, Settings, Storage
from evorag.main import create_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_provenance_graph_endpoint_and_bounds(tmp_path: Path) -> None:
    settings = Settings(
        storage=Storage(database_path=tmp_path / "graph.sqlite3"),
        embeddings=Embeddings(dimensions=2),
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        with patch.object(app.state.ingestion.resources, "_ram_probe", return_value=32 * 1024**3):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                empty = await client.get("/api/v1/graph/provenance")
                assert empty.status_code == 200
                assert empty.json()["is_demo"] is False
                assert any(n["kind"] == "config_version" for n in empty.json()["nodes"])

                up = await client.post(
                    "/api/v1/documents",
                    headers={"X-Filename": "guide.md"},
                    content=b"EvoRag provenance graph tracks documents, revisions and chunks.",
                )
                assert up.status_code == 201

                full = await client.get("/api/v1/graph/provenance?max_nodes=50")
                assert full.status_code == 200
                data = full.json()
                kinds = {n["kind"] for n in data["nodes"]}
                assert {"document", "revision", "chunk", "config_version"} <= kinds
                edge_kinds = {e["kind"] for e in data["edges"]}
                assert {"document_has_revision", "document_contains_chunk"} <= edge_kinds
