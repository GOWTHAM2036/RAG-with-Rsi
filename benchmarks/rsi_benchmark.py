"""Reproducible Phase 09 RSI offline benchmark (30+ scenarios & 3 bounded experiments)."""

import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import psutil
from evorag.config import RSI, Embeddings, EvaluationSettings, Settings, Storage
from evorag.main import create_app
from evorag.rsi_analysis import (
    analyze_failure_patterns,
    generate_hypotheses,
    prioritize_opportunities,
)
from evorag.rsi_optimizer import generate_candidates
from evorag.rsi_search_space import default_tunable_parameters, validate_candidate


def _transport(settings: Settings, request: httpx.Request) -> httpx.Response:
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
    ans = {
        "status": "answered",
        "answer": "EvoRag uses SQLite WAL mode.",
        "claims": [{"text": "EvoRag uses SQLite WAL mode.", "evidence_ids": ["E1"]}],
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
    base_params = default_tunable_parameters()
    patterns = analyze_failure_patterns({"missing_relevant_documents": 12}, minimum_samples=1)
    hyps = generate_hypotheses(prioritize_opportunities(patterns))

    cand_lat: list[float] = []
    val_lat: list[float] = []
    for i in range(30):
        t0 = time.perf_counter()
        cands = generate_candidates(base_params, hyps, strategy="seeded_random_search", seed=i)
        cand_lat.append((time.perf_counter() - t0) * 1000.0)
        t1 = time.perf_counter()
        for c in cands:
            validate_candidate(c)
        val_lat.append((time.perf_counter() - t1) * 1000.0)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(
            storage=Storage(database_path=Path(tmp) / "rsi_bench.sqlite3"),
            embeddings=Embeddings(dimensions=2),
            evaluation=EvaluationSettings(max_cases_per_run=5),
            rsi=RSI(enabled=True, max_cycles=2, max_candidates_per_cycle=2, max_experiments=4),
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
                ing = app.state.ingestion.ingest(
                    "arch.md", b"EvoRag uses SQLite WAL mode and float32 vectors."
                )
                await app.state.inference.embeddings.run_job(ing.document_id, 1)
                rsi = app.state.inference.rsi_engine
                exp_a = await rsi.run_cycle(
                    failure_counts={"missing_relevant_documents": 12}, max_dev_cases=3
                )
                exp_b = await rsi.run_cycle(
                    failure_counts={"incorrect_retrieval_ranking": 10}, max_dev_cases=3
                )
                exp_c = await rsi.run_cycle(
                    failure_counts={"incomplete_evidence_coverage": 8}, max_dev_cases=3
                )
                peak_rss = proc.memory_info().rss
                return {
                    "scenarios_benchmarked": 30,
                    "candidate_gen_p95_ms": round(
                        sorted(cand_lat)[int(0.95 * (len(cand_lat) - 1))], 3
                    ),
                    "candidate_val_p95_ms": round(
                        sorted(val_lat)[int(0.95 * (len(val_lat) - 1))], 3
                    ),
                    "candidate_gen_median_ms": round(statistics.median(cand_lat), 3),
                    "exp_a_outcome": exp_a.cycle.outcome,
                    "exp_b_outcome": exp_b.cycle.outcome,
                    "exp_c_cycles": exp_c.cycle.cycles_completed,
                    "additional_ram_mib": round((peak_rss - base_rss) / (1024**2), 2),
                }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_benchmark()), indent=2))
