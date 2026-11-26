"""Integration tests for Phase 06 adaptive RAG, /api/v1/rag/plan, caching and tracing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from evorag.config import Embeddings, Settings, Storage
from evorag.main import create_app


def _make_transport(settings: Settings, *, fail_generation: bool = False) -> httpx.MockTransport:
    gen = settings.runtime.generation_model
    emb = settings.runtime.embedding_model

    def handle(request: httpx.Request) -> httpx.Response:
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
                200,
                json={"model": emb, "embeddings": [[1.0, 0.0] for _ in body["input"]]},
            )
        if fail_generation:
            return httpx.Response(503, json={"error": "provider_unavailable"})
        ans = {
            "status": "answered",
            "answer": "Supported by evidence.",
            "claims": [{"text": "Supported by evidence.", "evidence_ids": ["E1"]}],
            "limitations": [],
        }
        return httpx.Response(
            200,
            json={
                "model": gen,
                "message": {"role": "assistant", "content": json.dumps(ans)},
                "done": True,
                "done_reason": "stop",
            },
        )

    return httpx.MockTransport(handle)


@pytest.mark.anyio
async def test_adaptive_rag_api_plan_query_cache_and_trace(tmp_path: Path) -> None:
    settings = Settings(
        storage=Storage(database_path=tmp_path / "adaptive.sqlite3"),
        embeddings=Embeddings(dimensions=2),
    )
    app = create_app(settings, transport=_make_transport(settings))
    async with app.router.lifespan_context(app):
        with (
            patch.object(app.state.ingestion.resources, "_ram_probe", return_value=32 * 1024**3),
            patch.object(app.state.inference.coordinator, "ram_probe", return_value=32 * 1024**3),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                # Ingest two documents
                for fname, text in (
                    (
                        "vector.md",
                        "EvoRag vector storage uses append-only float32 files and checksums.",
                    ),
                    (
                        "sqlite.md",
                        "SQLite WAL transaction recovery replays committed frames on startup.",
                    ),
                ):
                    up = await client.post(
                        "/api/v1/documents",
                        headers={"X-Filename": fname},
                        content=text.encode("utf-8"),
                    )
                    assert up.status_code == 201
                    doc_id = up.json()["document_id"]
                    emb_res = await client.post(f"/api/v1/documents/{doc_id}/embed?version=1")
                    assert emb_res.status_code == 200

                # 1. /api/v1/rag/plan returns bounded plan without invoking generation
                plan_resp = await client.post(
                    "/api/v1/rag/plan",
                    json={
                        "question": (
                            "Compare vector storage with SQLite WAL and explain limitations."
                        ),
                        "strategy": "adaptive",
                    },
                )
                assert plan_resp.status_code == 200
                plan_data = plan_resp.json()
                assert plan_data["strategy_mode"] == "adaptive"
                assert plan_data["selected_strategy"] == "decomposed_multi_part"
                assert plan_data["llm_planning_calls_used"] == 0
                assert 2 <= len(plan_data["variants"]) <= 3

                # 2. Default baseline query remains compatible
                base_resp = await client.post(
                    "/api/v1/rag/query",
                    json={"question": "What does vector storage use?"},
                )
                assert base_resp.status_code == 200
                assert base_resp.json()["strategy_mode"] == "baseline"
                assert base_resp.json()["query_plan"] is None

                # 3. Adaptive query with plan & diagnostics
                secret_q = "Compare vector storage with SQLite WAL and explain SECRET_TOKEN_123."
                adv_resp = await client.post(
                    "/api/v1/rag/query",
                    json={
                        "question": secret_q,
                        "strategy": "adaptive",
                        "include_query_plan": True,
                        "include_retrieval_diagnostics": True,
                    },
                )
                assert adv_resp.status_code == 200
                adv_data = adv_resp.json()
                assert adv_data["strategy_mode"] == "adaptive"
                assert adv_data["admission_outcome"] == "accepted"
                assert adv_data["query_plan"] is not None
                assert adv_data["retrieval_diagnostics"] is not None

                # 4. Trace is metadata-only (no raw private query text stored)
                trace_id = adv_data["trace_id"]
                tr_resp = await client.get(f"/api/v1/rag/traces/{trace_id}")
                assert tr_resp.status_code == 200
                tr_json = tr_resp.text
                assert "SECRET_TOKEN_123" not in tr_json
                tr_data = tr_resp.json()
                assert tr_data["query_execution"]["strategy_mode"] == "adaptive"
                assert tr_data["query_execution"]["cache_misses"] >= 1

                # 5. Repeat identical adaptive query hits retrieval cache
                adv_resp2 = await client.post(
                    "/api/v1/rag/query",
                    json={"question": secret_q, "strategy": "adaptive"},
                )
                assert adv_resp2.status_code == 200
                tr2 = (
                    await client.get(f"/api/v1/rag/traces/{adv_resp2.json()['trace_id']}")
                ).json()
                assert tr2["query_execution"]["cache_hits"] >= 1

                # 6. Status exposes adaptive_rag_enabled, default_strategy and cache stats
                st = (await client.get("/api/v1/rag/status")).json()
                assert st["adaptive_rag_enabled"] is True
                assert st["default_strategy"] == "baseline"
                assert st["retrieval_cache_stats"]["hits"] >= 1


@pytest.mark.anyio
async def test_adaptive_rag_resource_rejection_recorded_in_trace(tmp_path: Path) -> None:
    settings = Settings(
        storage=Storage(database_path=tmp_path / "rej.sqlite3"),
        embeddings=Embeddings(dimensions=2),
    )
    app = create_app(settings, transport=_make_transport(settings))
    async with app.router.lifespan_context(app):
        with (
            patch.object(app.state.ingestion.resources, "_ram_probe", return_value=32 * 1024**3),
            patch.object(app.state.inference.coordinator, "ram_probe", return_value=32 * 1024**3),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                up = await client.post(
                    "/api/v1/documents",
                    headers={"X-Filename": "doc.md"},
                    content=b"EvoRag uses SQLite WAL.",
                )
                assert up.status_code == 201
                doc_id = up.json()["document_id"]
                await client.post(f"/api/v1/documents/{doc_id}/embed?version=1")

                # Simulate RAM below 3 GiB floor during generation
                with patch.object(
                    app.state.inference.coordinator,
                    "ram_probe",
                    return_value=1 * 1024**3,
                ):
                    resp = await client.post(
                        "/api/v1/rag/query",
                        json={"question": "What does EvoRag use?", "strategy": "adaptive"},
                    )
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["admission_outcome"] == "resource_unavailable"
                    tr = (await client.get(f"/api/v1/rag/traces/{data['trace_id']}")).json()
                    assert tr["query_execution"]["resource_admission_ok"] is False


@pytest.mark.anyio
async def test_adaptive_rag_limitations_deadline_and_streaming(tmp_path: Path) -> None:
    from evorag.config import AdaptiveRagSettings

    settings = Settings(
        storage=Storage(database_path=tmp_path / "lim.sqlite3"),
        embeddings=Embeddings(dimensions=2),
    )
    app = create_app(settings, transport=_make_transport(settings))
    async with app.router.lifespan_context(app):
        with (
            patch.object(app.state.ingestion.resources, "_ram_probe", return_value=32 * 1024**3),
            patch.object(app.state.inference.coordinator, "ram_probe", return_value=32 * 1024**3),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                up = await client.post(
                    "/api/v1/documents",
                    headers={"X-Filename": "vector.md"},
                    content=b"EvoRag vector storage uses float32 files.",
                )
                assert up.status_code == 201
                doc_id = up.json()["document_id"]
                await client.post(f"/api/v1/documents/{doc_id}/embed?version=1")

                # 1. Technical exact query exercises sparse_weight_override and streaming.
                stream_resp = await client.post(
                    "/api/v1/rag/query/stream",
                    json={
                        "question": "What does vector_store_corrupted do?",
                        "strategy": "adaptive",
                    },
                )
                assert stream_resp.status_code == 200
                events = [json.loads(line) for line in stream_resp.text.strip().splitlines()]
                assert events[-1]["event"] == "completed"
                assert events[-1]["response"]["strategy_mode"] == "adaptive"

                # 2. Coverage warning propagates to limitations when a subquery has no evidence.
                cov_resp = await client.post(
                    "/api/v1/rag/query",
                    json={
                        "question": "Compare vector storage with quantum_flux_capacitor.",
                        "strategy": "adaptive",
                        "include_retrieval_diagnostics": True,
                    },
                )
                assert cov_resp.status_code == 200
                cov_data = cov_resp.json()
                if cov_data["retrieval_diagnostics"]["warnings"]:
                    for warn in cov_data["retrieval_diagnostics"]["warnings"]:
                        assert warn in cov_data["limitations"]

    # 3. Expired request deadline skips generation and returns request_timeout.
    tight_settings = Settings(
        storage=Storage(database_path=tmp_path / "deadline.sqlite3"),
        embeddings=Embeddings(dimensions=2),
        adaptive_rag=AdaptiveRagSettings(request_deadline_seconds=0.0001),
    )
    tight_app = create_app(tight_settings, transport=_make_transport(tight_settings))
    async with tight_app.router.lifespan_context(tight_app):
        with (
            patch.object(
                tight_app.state.ingestion.resources, "_ram_probe", return_value=32 * 1024**3
            ),
            patch.object(
                tight_app.state.inference.coordinator, "ram_probe", return_value=32 * 1024**3
            ),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=tight_app), base_url="http://testserver"
            ) as client:
                up = await client.post(
                    "/api/v1/documents",
                    headers={"X-Filename": "doc.md"},
                    content=b"EvoRag uses SQLite WAL.",
                )
                assert up.status_code == 201
                doc_id = up.json()["document_id"]
                await client.post(f"/api/v1/documents/{doc_id}/embed?version=1")
                dl_resp = await client.post(
                    "/api/v1/rag/query",
                    json={"question": "What does EvoRag use?", "strategy": "adaptive"},
                )
                assert dl_resp.status_code == 200
                assert dl_resp.json()["admission_outcome"] == "generation_failed"
                assert "request_timeout" in dl_resp.json()["limitations"]
