"""Mathematical similarity, distance metrics and exact nearest-neighbor search."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

Metric = Literal["cosine", "dot", "euclidean"]


def _validate_inputs(
    query: NDArray[np.floating] | NDArray[np.integer],
    matrix: NDArray[np.floating] | NDArray[np.integer],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    q = np.asarray(query, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    if q.ndim != 1 or q.shape[0] == 0:
        raise ValueError("query must be a non-empty 1D vector")
    if m.ndim != 2 or m.shape[1] != q.shape[0]:
        raise ValueError("matrix must be 2D with width matching query dimension")
    if not np.isfinite(q).all() or not np.isfinite(m).all():
        raise ValueError("query and matrix must contain only finite values")
    return q, m


def dot_product(
    query: NDArray[np.floating] | NDArray[np.integer],
    matrix: NDArray[np.floating] | NDArray[np.integer],
) -> NDArray[np.float64]:
    q, m = _validate_inputs(query, matrix)
    if m.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    result: NDArray[np.float64] = m @ q
    return result


def cosine_similarity(
    query: NDArray[np.floating] | NDArray[np.integer],
    matrix: NDArray[np.floating] | NDArray[np.integer],
) -> NDArray[np.float64]:
    q, m = _validate_inputs(query, matrix)
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        raise ValueError("query vector must have non-zero norm for cosine similarity")
    if m.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    row_norms = np.linalg.norm(m, axis=1)
    dots = m @ q
    denom = row_norms * q_norm
    out = np.zeros(m.shape[0], dtype=np.float64)
    np.divide(dots, denom, out=out, where=denom > 0.0)
    return out


def euclidean_distance(
    query: NDArray[np.floating] | NDArray[np.integer],
    matrix: NDArray[np.floating] | NDArray[np.integer],
) -> NDArray[np.float64]:
    q, m = _validate_inputs(query, matrix)
    if m.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    result: NDArray[np.float64] = np.linalg.norm(m - q, axis=1)
    return result


DEFAULT_BLOCK_SIZE = 2048


def exact_nearest_neighbors(
    query: NDArray[np.floating] | NDArray[np.integer],
    matrix: NDArray[np.floating] | NDArray[np.integer],
    top_k: int,
    metric: Metric = "cosine",
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> list[tuple[int, float]]:
    if isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if isinstance(block_size, bool) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if metric not in ("cosine", "dot", "euclidean"):
        raise ValueError(f"unsupported metric: {metric}")

    q = np.asarray(query, dtype=np.float64)
    if q.ndim != 1 or q.shape[0] == 0:
        raise ValueError("query must be a non-empty 1D vector")
    if not np.isfinite(q).all():
        raise ValueError("query and matrix must contain only finite values")
    if matrix.ndim != 2 or matrix.shape[1] != q.shape[0]:
        raise ValueError("matrix must be 2D with width matching query dimension")

    if metric == "cosine":
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            raise ValueError("query vector must have non-zero norm for cosine similarity")

    total_rows = int(matrix.shape[0])
    if total_rows == 0:
        return []

    descending = metric in ("cosine", "dot")
    best: list[tuple[int, float]] = []

    for start in range(0, total_rows, block_size):
        end = min(start + block_size, total_rows)
        block = matrix[start:end]
        if metric == "cosine":
            scores = cosine_similarity(q, block)
        elif metric == "dot":
            scores = dot_product(q, block)
        else:
            scores = euclidean_distance(q, block)

        block_items = [(start + offset, float(scores[offset])) for offset in range(len(scores))]
        best.extend(block_items)
        if len(best) > top_k:
            if descending:
                best.sort(key=lambda item: (-item[1], item[0]))
            else:
                best.sort(key=lambda item: (item[1], item[0]))
            best = best[:top_k]

    if descending:
        best.sort(key=lambda item: (-item[1], item[0]))
    else:
        best.sort(key=lambda item: (item[1], item[0]))
    return best[:top_k]
