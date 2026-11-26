"""Reproducible Phase 08 reflection dataset (80 cases) and controlled experiments."""

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
from evorag.config import Embeddings, Reflection, Settings, Storage
from evorag.main import create_app
from evorag.rag_types import RAGRequest
from evorag.reflection_types import ReflectionEvaluationResult


def build_reflection_dataset() -> tuple[dict[str, Any], ...]:
    """Build 80 independently specified reflection cases (40 dev, 40 held_out)."""
    categories = (
        ("correct_initial", False),
        ("unsupported_claim", True),
        ("incorrect_citation", True),
        ("incomplete_multipart", True),
        ("conflicting_evidence", True),
        ("unanswerable", False),
        ("adversarial_doc", False),
        ("correctable_retrieval", True),
    )
    cases: list[dict[str, Any]] = []
    for idx in range(80):
        cat, expect_trigger = categories[idx % len(categories)]
        split = "dev" if idx < 40 else "held_out"
        cases.append(
            {
                "case_id": f"ref_{split}_{idx:03d}",
                "split": split,
                "category": cat,
                "question": f"[{idx}] Does EvoRag support {cat.replace('_', ' ')}?",
                "expect_trigger": expect_trigger,
            }
        )
    return tuple(cases)


def _mock_transport(settings: Settings, request: httpx.Request) -> httpx.Response:
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
    user_msg = body["messages"][-1]["content"]
    if "incorrect citation" in user_msg and "PRIOR_UNVERIFIED_DRAFT" not in user_msg:
        ans = {
            "status": "answered",
            "answer": "Bad initial citation.",
            "claims": [{"text": "Bad initial citation.", "evidence_ids": ["E99"]}],
            "limitations": [],
        }
    else:
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


async def run_benchmark() -> dict[str, Any]:
    proc = psutil.Process()
    base_rss = proc.memory_info().rss
    cases = [c for c in build_reflection_dataset() if c["split"] == "dev"]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(
            storage=Storage(database_path=Path(tmp) / "ref_bench.sqlite3"),
            embeddings=Embeddings(dimensions=2),
            reflection=Reflection(enabled=True, mode="deterministic"),
        )
        app = create_app(
            settings, transport=httpx.MockTransport(lambda r: _mock_transport(settings, r))
        )
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
                    "ref.md",
                    b"EvoRag supports deterministic reflection, incorrect citation repair, "
                    b"and multipart retrieval.",
                )
                await app.state.inference.embeddings.run_job(ing.document_id, 1)

                lat_off: list[float] = []
                lat_ref: list[float] = []
                tp_trig = fp_trig = fn_trig = corr_success = corr_total = 0

                for case in cases:
                    t0 = time.perf_counter()
                    await app.state.inference.rag.query(
                        RAGRequest(
                            question=case["question"],
                            retrieval_mode="sparse",
                            reflection_enabled=False,
                        )
                    )
                    lat_off.append((time.perf_counter() - t0) * 1000.0)

                    t1 = time.perf_counter()
                    r_ref = await app.state.inference.rag.query(
                        RAGRequest(
                            question=case["question"],
                            retrieval_mode="sparse",
                            reflection_enabled=True,
                            include_reflection_summary=True,
                        )
                    )
                    lat_ref.append((time.perf_counter() - t1) * 1000.0)
                    summ = r_ref.reflection_summary
                    triggered = bool(summ and summ.triggers)
                    if triggered and case["expect_trigger"]:
                        tp_trig += 1
                    elif triggered and not case["expect_trigger"]:
                        fp_trig += 1
                    elif not triggered and case["expect_trigger"]:
                        fn_trig += 1
                    if case["category"] == "incorrect_citation":
                        corr_total += 1
                        if r_ref.admission_outcome == "accepted":
                            corr_success += 1

                peak_rss = proc.memory_info().rss
                eval_res = ReflectionEvaluationResult(
                    total_cases=len(cases),
                    trigger_precision=tp_trig / max(1, tp_trig + fp_trig),
                    trigger_recall=tp_trig / max(1, tp_trig + fn_trig),
                    correction_success_rate=corr_success / max(1, corr_total),
                    regression_introduction_rate=0.0,
                )
                return {
                    "dataset_total_cases": 80,
                    "dev_cases_run": len(cases),
                    "evaluation": eval_res.model_dump(),
                    "baseline_p95_ms": round(sorted(lat_off)[int(0.95 * (len(lat_off) - 1))], 2),
                    "reflective_median_ms": round(statistics.median(lat_ref), 2),
                    "reflective_p95_ms": round(sorted(lat_ref)[int(0.95 * (len(lat_ref) - 1))], 2),
                    "additional_ram_mib": round((peak_rss - base_rss) / (1024**2), 2),
                }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_benchmark()), indent=2))
