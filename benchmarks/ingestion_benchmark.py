"""Run with the installed evorag package. Each size uses a fresh spawned process."""

import json
import multiprocessing
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import TypeVar

import psutil
from evorag.chunking import iter_chunks
from evorag.config import Settings, Storage
from evorag.documents import digest
from evorag.ingestion import IngestionService
from evorag.parsing import parse_document
from evorag.resources import ResourceManager

T = TypeVar("T")


def timed(call: Callable[[], T]) -> tuple[T, float]:
    start = perf_counter()
    value = call()
    return value, (perf_counter() - start) * 1000


def measure(size: int) -> None:
    pattern = b"A paragraph about a synthetic corpus. The second sentence adds detail.\n\n"
    with tempfile.TemporaryDirectory() as folder:
        settings = Settings(storage=Storage(database_path=Path(folder) / "benchmark.sqlite3"))
        resources = ResourceManager(settings)
        resources.check_ram()
        service = IngestionService(settings, resources)
        process = psutil.Process()
        baseline = peak = process.memory_info().rss
        done = Event()

        def sample() -> None:
            nonlocal peak
            while not done.wait(0.002):
                peak = max(peak, process.memory_info().rss)

        sampler = Thread(target=sample)
        sampler.start()
        try:
            payload = (pattern * (size // len(pattern) + 1))[:size]
            created, first_ms = timed(lambda: service.ingest("synthetic.txt", payload))
            duplicate, duplicate_ms = timed(lambda: service.ingest("synthetic.txt", payload))
            if created.status != "created" or duplicate.status != "duplicate":
                raise RuntimeError("unexpected benchmark result")
            # Isolate chunk CPU and persistence by replaying one independently prepared version.
            with resources.ingestion_slot():
                parsed = parse_document("synthetic.txt", payload, settings.ingestion)
                chunks, chunk_ms = timed(
                    lambda: list(
                        iter_chunks(
                            parsed, "bench", "bench-version", settings.chunking, settings.ingestion
                        )
                    )
                )
                _, write_ms = timed(
                    lambda: service.repository.publish(
                        "db-timing",
                        "bench",
                        "db-timing.txt",
                        "bench-version",
                        digest("synthetic"),
                        "txt-v1",
                        "nfc-lf-v1/preserve",
                        digest("benchmark"),
                        iter(chunks),
                    )
                )
            peak = max(peak, process.memory_info().rss)
        finally:
            done.set()
            sampler.join()
        print(
            json.dumps(
                {
                    "bytes": size,
                    "chunks": created.chunk_count,
                    "ingest_ms": round(first_ms, 2),
                    "duplicate_ms": round(duplicate_ms, 2),
                    "write_ms": round(write_ms, 2),
                    "chunk_mib_per_second": round(size / 1024**2 / (chunk_ms / 1000), 2),
                    "sampled_peak_rss_mib": round(peak / 1024**2, 2),
                    "sampled_delta_mib": round((peak - baseline) / 1024**2, 2),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    for input_size in (10 * 1024, 100 * 1024, 1024**2):
        ResourceManager(Settings()).check_ram()
        child = multiprocessing.get_context("spawn").Process(target=measure, args=(input_size,))
        child.start()
        child.join()
        if child.exitcode:
            sys.exit(child.exitcode)
