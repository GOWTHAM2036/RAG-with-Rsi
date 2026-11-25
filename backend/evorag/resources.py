"""Local resource measurements and process-scoped admission control."""

import shutil
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import BoundedSemaphore

import psutil

from .config import Settings
from .errors import ResourceUnavailable

GIB = 1024**3


def available_ram() -> int | None:
    try:
        return psutil.virtual_memory().available
    except (OSError, psutil.Error, NotImplementedError):
        return None


def available_disk(path: Path) -> int | None:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def nvidia_gpus() -> tuple[str, ...] | None:
    """Best-effort NVIDIA detection. None means the probe is unavailable."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [binary, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            check=True,
            text=True,
            errors="replace",
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


@dataclass(frozen=True)
class ResourceReport:
    available_ram_bytes: int | None
    available_disk_bytes: int | None
    nvidia_gpus: tuple[str, ...] | None


def resource_report(path: Path) -> ResourceReport:
    return ResourceReport(available_ram(), available_disk(path), nvidia_gpus())


class ResourceManager:
    def __init__(self, settings: Settings, ram_probe: Callable[[], int | None] = available_ram):
        self._limits = settings.resources
        self._ram_probe = ram_probe
        self._ingestion_slot = BoundedSemaphore(1)
        self._slots = BoundedSemaphore(
            min(settings.runtime.max_concurrent_requests, self._limits.max_parallel_generations)
        )

    @contextmanager
    def generation_slot(self) -> Iterator[None]:
        """Fail immediately when busy. Exceptions and cancellation release the permit."""
        if not self._slots.acquire(blocking=False):
            raise ResourceUnavailable("Generation capacity is occupied")
        try:
            self.check_ram()
            yield
        finally:
            self._slots.release()

    def check_ram(self) -> None:
        ram = self._ram_probe()
        if ram is None or ram < self._limits.minimum_free_ram_gb * GIB:
            raise ResourceUnavailable("Cannot confirm sufficient available RAM")

    @contextmanager
    def ingestion_slot(self) -> Iterator[None]:
        if not self._ingestion_slot.acquire(blocking=False):
            raise ResourceUnavailable("Ingestion capacity is occupied")
        try:
            self.check_ram()
            yield
        finally:
            self._ingestion_slot.release()

    def check_storage(self, *, used_bytes: int, additional_bytes: int, path: Path) -> None:
        """Preflight for a future serialized writer that owns its usage accounting."""
        if min(used_bytes, additional_bytes) < 0:
            raise ValueError("storage byte counts must be nonnegative")
        if used_bytes + additional_bytes > self._limits.project_storage_budget_gb * GIB:
            raise ResourceUnavailable("Project storage budget exceeded")
        free = available_disk(path)
        if free is None or additional_bytes > free:
            raise ResourceUnavailable("Cannot confirm sufficient available disk space")


if __name__ == "__main__":
    import json

    print(json.dumps(asdict(resource_report(Path.cwd()))))
