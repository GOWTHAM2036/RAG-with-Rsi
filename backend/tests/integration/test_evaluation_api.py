"""Offline integration tests for Phase 07 evaluation API, persistence and comparisons."""

import json
from collections.abc import AsyncIterator
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


def _eval_transport(settings: Settings, request: httpx.Request) -> httpx.Response:
    gen = settings.runtime.generation_model
    emb = settings.runtime.embedding_model
    path = request.url.path
    if path == "/api/tags":
        return httpx.Response(
            200,
            json={"models": [{"name": m, "digest": f"d-{m}", "size": 16} for m in (gen, emb)]},
        )
    if path == "/api/ps":
        return httpx.Response(200, json={"models": []})
    body = json.loads(request.content)
    if path == "/api/show":
        caps = ["completion"] if body["model"] == gen else ["embedding"]
        return httpx.Response(200, json={"capabilities": caps})
    if path == "/api/embed":
        return httpx.Response(
            200, json={"model": emb, "embeddings": [[1.0, 0.0] for _ in body["input"]]}
        )
    payload = {
        "status": "answered",
        "answer": "EvoRag uses SQLite WAL mode.",
        "claims": [{"text": "EvoRag uses SQLite WAL mode.", "evidence_ids": ["E1"]}],
        "limitations": [],
    }
    return httpx.Response(
        200,
        json={
            "model": gen,
            "message": {"role": "assistant", "content": json.dumps(payload)},
            "done": True,
            "done_reason": "stop",
        },
    )


@asynccontextmanager
async def eval_client(directory: Path) -> AsyncIterator[tuple[httpx.AsyncClient, Any]]:
    settings = Settings(
        storage=Storage(database_path=directory / "meta.sqlite3"),
        embeddings=Embeddings(dimensions=2),
    )
    app = create_app(
        settings, transport=httpx.MockTransport(lambda r: _eval_transport(settings, r))
    )
    async with app.router.lifespan_context(app):
        with (
            patch.object(app.state.ingestion.resources, "_ram_probe", return_value=32 * 1024**3),
            patch.object(app.state.inference.coordinator, "ram_probe", return_value=32 * 1024**3),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                yield client, app


async def test_evaluation_runs_report_cancel_and_compare(tmp_path: Path) -> None:
    async with eval_client(tmp_path) as (client, app):
        doc = await client.post(
            API + "/documents",
            headers={"X-Filename": "arch.md"},
            content=b"EvoRag uses SQLite WAL mode and float32 vectors.",
        )
        assert doc.status_code == 201
        await client.post(f"{API}/documents/{doc.json()['document_id']}/embed?version=1")

        datasets = await client.get(API + "/evaluation/datasets")
        assert datasets.status_code == 200
        assert datasets.json()[0]["dataset_id"] == "evorag_comprehensive_v1"

        app.state.inference.settings.evaluation.model_copy(update={"max_cases_per_run": 5})
        cfg_base = {
            "dataset_id": "evorag_comprehensive_v1",
            "dataset_version": "1.0.0",
            "split": "dev",
            "retrieval_mode": "sparse",
            "strategy": "baseline",
        }
        r1 = await client.post(API + "/evaluation/runs", json=cfg_base)
        assert r1.status_code == 200, r1.text
        run1_id = r1.json()["run"]["run_id"]

        r2 = await client.post(API + "/evaluation/runs", json={**cfg_base, "strategy": "adaptive"})
        assert r2.status_code == 200
        run2_id = r2.json()["run"]["run_id"]

        fetched = await client.get(f"{API}/evaluation/runs/{run1_id}")
        assert fetched.status_code == 200

        report = await client.get(f"{API}/evaluation/runs/{run1_id}/report")
        assert report.status_code == 200
        assert report.json()["provenance"]["automatic_promotion_enabled"] is False
        assert "Dataset:" in app.state.inference.experiment_runner.render_markdown_report(run1_id)

        comp = await client.post(
            f"{API}/evaluation/compare?baseline_run_id={run1_id}&candidate_run_id={run2_id}"
        )
        assert comp.status_code == 200
        assert comp.json()["baseline_run_id"] == run1_id

        cancelled = await client.post(f"{API}/evaluation/runs/{run1_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        assert (await client.get(f"{API}/evaluation/runs/missing")).status_code == 404
