"""Offline integration tests for Phase 09 RSI cycles, approval, promotion and rollback."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from evorag.config import RSI, Embeddings, EvaluationSettings, Settings, Storage
from evorag.main import create_app

pytestmark = pytest.mark.anyio
API = "/api/v1"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _rsi_transport(settings: Settings, request: httpx.Request) -> httpx.Response:
    gen = settings.runtime.generation_model
    emb = settings.runtime.embedding_model
    if request.url.path == "/api/tags":
        return httpx.Response(
            200, json={"models": [{"name": m, "digest": f"d-{m}", "size": 16} for m in (gen, emb)]}
        )
    if request.url.path == "/api/ps":
        return httpx.Response(200, json={"models": []})
    body = json.loads(request.content)
    if request.url.path == "/api/show":
        caps = ["completion"] if body["model"] == gen else ["embedding"]
        return httpx.Response(200, json={"capabilities": caps})
    if request.url.path == "/api/embed":
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
async def rsi_client(
    directory: Path, *, enabled: bool = True
) -> AsyncIterator[tuple[httpx.AsyncClient, Any]]:
    settings = Settings(
        storage=Storage(database_path=directory / "meta.sqlite3"),
        embeddings=Embeddings(dimensions=2),
        evaluation=EvaluationSettings(max_cases_per_run=5),
        rsi=RSI(enabled=enabled, max_cycles=2, max_candidates_per_cycle=2, max_experiments=4),
    )
    app = create_app(settings, transport=httpx.MockTransport(lambda r: _rsi_transport(settings, r)))
    async with app.router.lifespan_context(app):
        with (
            patch.object(app.state.ingestion.resources, "_ram_probe", return_value=32 * 1024**3),
            patch.object(app.state.inference.coordinator, "ram_probe", return_value=32 * 1024**3),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                yield client, app


async def test_rsi_cycle_approval_promotion_and_rollback(tmp_path: Path) -> None:
    async with rsi_client(tmp_path, enabled=True) as (client, app):
        doc = await client.post(
            API + "/documents",
            headers={"X-Filename": "arch.md"},
            content=b"EvoRag uses SQLite WAL mode and float32 vectors.",
        )
        assert doc.status_code == 201
        await client.post(f"{API}/documents/{doc.json()['document_id']}/embed?version=1")

        st = await client.get(API + "/rsi/status")
        assert st.status_code == 200 and st.json()["automatic_promotion"] is False

        report = await app.state.inference.rsi_engine.run_cycle(
            failure_counts={"missing_relevant_documents": 12},
            max_dev_cases=1,
            doc_id_to_name={doc.json()["document_id"]: "arch.md"},
        )
        assert report.cycle.outcome == "improvement_recommended"
        assert report.recommendation is not None
        rec_id = report.recommendation.recommendation_id

        # Offline result cannot authorize live_runtime promotion.
        with pytest.raises(ValueError, match="offline_result_cannot_authorize_live_promotion"):
            app.state.inference.rsi_engine.approve_recommendation(
                rec_id, approved_by="owner", activation_scope="live_runtime"
            )

        approval = app.state.inference.rsi_engine.approve_recommendation(
            rec_id, approved_by="owner", activation_scope="offline_sandbox"
        )
        promoted = app.state.inference.rsi_engine.promote_approved(approval.approval_id)
        assert promoted.active is True and promoted.parent_version_id == "v0_baseline"

        # Stale approval cannot be re-applied after baseline version changed.
        with pytest.raises(ValueError, match="stale_approval"):
            app.state.inference.rsi_engine.promote_approved(approval.approval_id)

        rolled_back = app.state.inference.rsi_engine.rollback_to(
            "v0_baseline", reason="regression_check"
        )
        assert rolled_back.version_id == "v0_baseline"

        assert (await client.get(f"{API}/rsi/cycles/{report.cycle.cycle_id}")).status_code == 200
        assert (
            await client.get(f"{API}/rsi/cycles/{report.cycle.cycle_id}/report")
        ).status_code == 200
        assert len((await client.get(API + "/rsi/recommendations")).json()) >= 1
        assert len((await client.get(API + "/rsi/configurations")).json()) >= 2
        assert (await client.get(API + "/rsi/cycles/missing")).status_code == 404

    async with rsi_client(tmp_path / "off", enabled=False) as (client, _):
        assert (await client.post(API + "/rsi/cycles")).status_code == 503
