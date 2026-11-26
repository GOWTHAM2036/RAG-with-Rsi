"""Reproducible Phase 07 controlled configuration comparisons and aggregation benchmark."""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import psutil
from evorag.config import Embeddings, EvaluationSettings, Settings, Storage
from evorag.eval_types import ExperimentConfiguration
from evorag.main import create_app


def _transport(settings: Settings, request: httpx.Request) -> httpx.Response:
    gen = settings.runtime.generation_model
    emb = settings.runtime.embedding_model
    path = request.url.path
    if path == "/api/tags":
        return httpx.Response(
            200, json={"models": [{"name": m, "digest": f"d-{m}", "size": 16} for m in (gen, emb)]}
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
    user_msg = body["messages"][-1]["content"]
    ans: dict[str, Any]
    if "World Cup" in user_msg or "price" in user_msg:
        ans = {
            "status": "insufficient_evidence",
            "answer": "Insufficient evidence.",
            "claims": [],
            "limitations": ["missing_evidence"],
        }
    else:
        ans = {
            "status": "answered",
            "answer": "EvoRag uses SQLite WAL mode, float32 vectors, and RTX 3050 A GPU.",
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


async def run_benchmark() -> dict[str, Any]:
    proc = psutil.Process()
    base_rss = proc.memory_info().rss
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(
            storage=Storage(database_path=Path(tmp) / "eval_bench.sqlite3"),
            embeddings=Embeddings(dimensions=2),
            evaluation=EvaluationSettings(max_cases_per_run=54),
        )
        app = create_app(settings, transport=httpx.MockTransport(lambda r: _transport(settings, r)))
        async with app.router.lifespan_context(app):
            with (
                patch.object(
                    app.state.ingestion.resources, "_ram_probe", return_value=32 * 1024**3
                ),
                patch.object(
                    app.state.inference.coordinator, "ram_probe", return_value=32 * 1024**3
                ),
            ):
                docs = {
                    "arch.md": b"EvoRag uses SQLite WAL mode and append-only float32 vectors.",
                    "errors.md": b"vector_store_corrupted fails closed with HTTP 500.",
                    "hardware.md": (
                        b"RTX 3050 A Laptop GPU, Python 3.12, minimum free RAM floor is 3 GiB."
                    ),
                    "adversarial.md": b"Retrieval endpoint is POST /api/v1/retrieve.",
                    "revisions.md": b"Latest active revision applies by default.",
                }
                doc_map: dict[str, str] = {}
                for name, body in docs.items():
                    ing = app.state.ingestion.ingest(name, body)
                    doc_map[ing.document_id] = name
                    await app.state.inference.embeddings.run_job(ing.document_id, 1)

                runner = app.state.inference.experiment_runner
                cfg_base = ExperimentConfiguration(
                    dataset_id="evorag_comprehensive_v1",
                    dataset_version="1.0.0",
                    split="dev",
                    retrieval_mode="hybrid",
                    strategy="baseline",
                    reranking_enabled=False,
                )
                cfg_adapt = cfg_base.model_copy(update={"strategy": "adaptive"})
                cfg_rerank = cfg_base.model_copy(update={"reranking_enabled": True})

                res_base = await runner.run(cfg_base, doc_id_to_name=doc_map)
                res_adapt = await runner.run(cfg_adapt, doc_id_to_name=doc_map)
                res_rerank = await runner.run(cfg_rerank, doc_id_to_name=doc_map)

                t_comp = time.perf_counter()
                comp_a = runner.compare(res_base.run.run_id, res_adapt.run.run_id)
                comp_b = runner.compare(res_base.run.run_id, res_rerank.run.run_id)
                comp_ms = (time.perf_counter() - t_comp) * 1000.0
                peak_rss = proc.memory_info().rss

                return {
                    "cases_per_run": res_base.run.completed_cases,
                    "comparison_a_baseline_vs_adaptive": comp_a.overall_verdict,
                    "comparison_b_baseline_vs_rerank": comp_b.overall_verdict,
                    "baseline_p95_ms": round(res_base.p95_latency_ms, 2),
                    "adaptive_p95_ms": round(res_adapt.p95_latency_ms, 2),
                    "rerank_p95_ms": round(res_rerank.p95_latency_ms, 2),
                    "statistical_comparison_ms": round(comp_ms, 2),
                    "additional_ram_mib": round((peak_rss - base_rss) / (1024**2), 2),
                }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_benchmark()), indent=2))
