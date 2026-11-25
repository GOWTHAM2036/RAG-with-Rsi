"""Atomic disk-backed float32 vector storage with crash recovery."""

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .inference_types import ProviderError


class VectorStore:
    """Append-only little-endian float32 vector files, one uniform-width file per space."""

    def __init__(self, base_dir: Path, prefix: str) -> None:
        self.base_dir = base_dir
        self.prefix = prefix

    def path_for(self, space_id: str) -> Path:
        return self.base_dir / f"{self.prefix}.{space_id[:16]}.vectors.f32"

    def total_bytes(self) -> int:
        if not self.base_dir.exists():
            return 0
        total = 0
        for item in self.base_dir.glob(f"{self.prefix}.*.vectors.f32"):
            try:
                total += item.stat().st_size
            except FileNotFoundError:
                continue
        return total

    def recover(self, space_id: str, committed_rows: int, dimensions: int) -> None:
        if committed_rows < 0 or dimensions < 1:
            raise ValueError("committed_rows must be >= 0 and dimensions must be >= 1")
        path = self.path_for(space_id)
        expected = committed_rows * dimensions * 4
        if not path.exists():
            if committed_rows > 0:
                raise ProviderError("vector_store_corrupted", 500)
            return
        actual = path.stat().st_size
        if actual < expected:
            raise ProviderError("vector_store_corrupted", 500)
        if actual > expected:
            with path.open("r+b") as handle:
                handle.truncate(expected)
                handle.flush()
                os.fsync(handle.fileno())

    def append_batch(
        self, space_id: str, committed_rows: int, dimensions: int, blobs: Sequence[bytes]
    ) -> int:
        row_bytes = dimensions * 4
        if any(len(blob) != row_bytes for blob in blobs):
            raise ProviderError("invalid_response")
        self.recover(space_id, committed_rows, dimensions)
        if not blobs:
            return committed_rows
        self.base_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(space_id)
        mode = "r+b" if path.exists() else "wb"
        with path.open(mode) as handle:
            handle.seek(committed_rows * row_bytes)
            for blob in blobs:
                handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        return committed_rows + len(blobs)

    @contextmanager
    def map_readonly(
        self, space_id: str, committed_rows: int, dimensions: int
    ) -> Iterator[np.ndarray]:
        if committed_rows < 0 or dimensions < 1:
            raise ValueError("committed_rows must be >= 0 and dimensions must be >= 1")
        if committed_rows == 0:
            yield np.empty((0, dimensions), dtype="<f4")
            return
        path = self.path_for(space_id)
        expected = committed_rows * dimensions * 4
        if not path.exists() or path.stat().st_size < expected:
            raise ProviderError("vector_store_corrupted", 500)
        matrix = np.memmap(path, dtype="<f4", mode="r", shape=(committed_rows, dimensions))
        try:
            yield matrix
        finally:
            mmap_obj = getattr(matrix, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
