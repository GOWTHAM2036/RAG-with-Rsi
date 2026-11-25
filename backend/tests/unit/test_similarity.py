"""Hand-computed mathematical fixtures and edge-case tests for dense similarity and search."""

import math

import numpy as np
import pytest
from evorag.similarity import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    exact_nearest_neighbors,
)


def test_hand_computed_dot_cosine_and_euclidean() -> None:
    query = np.array([3.0, 4.0], dtype=np.float32)  # ||q|| = 5
    matrix = np.array(
        [
            [3.0, 4.0],  # identical: dot=25, cos=1.0, l2=0.0
            [4.0, -3.0],  # orthogonal: dot=0, cos=0.0, l2=sqrt(1 + 49)=sqrt(50)
            [-3.0, -4.0],  # opposite: dot=-25, cos=-1.0, l2=10.0
            [0.0, 0.0],  # zero row: dot=0, cos=0.0, l2=5.0
            [6.0, 8.0],  # scaled 2x: dot=50, cos=1.0, l2=5.0
        ],
        dtype=np.float32,
    )

    dots = dot_product(query, matrix)
    assert dots == pytest.approx([25.0, 0.0, -25.0, 0.0, 50.0])

    cosines = cosine_similarity(query, matrix)
    assert cosines == pytest.approx([1.0, 0.0, -1.0, 0.0, 1.0])

    distances = euclidean_distance(query, matrix)
    assert distances == pytest.approx([0.0, math.sqrt(50.0), 10.0, 5.0, 5.0])


def test_exact_nearest_neighbors_ranking_and_tie_breaking() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array(
        [
            [0.0, 1.0],  # idx 0: dot=0, cos=0, l2=sqrt(2)
            [2.0, 0.0],  # idx 1: dot=2, cos=1, l2=1
            [1.0, 0.0],  # idx 2: dot=1, cos=1, l2=0
            [-1.0, 0.0],  # idx 3: dot=-1, cos=-1, l2=2
        ],
        dtype=np.float32,
    )

    # Cosine: idx 1 and idx 2 both have cos=1.0 -> tie-break by smaller index (1 before 2)
    cos_top = exact_nearest_neighbors(query, matrix, top_k=3, metric="cosine")
    assert [idx for idx, _ in cos_top] == [1, 2, 0]
    assert cos_top[0][1] == pytest.approx(1.0)
    assert cos_top[1][1] == pytest.approx(1.0)
    assert cos_top[2][1] == pytest.approx(0.0)

    # Dot product: idx 1 (2.0) > idx 2 (1.0) > idx 0 (0.0)
    dot_top = exact_nearest_neighbors(query, matrix, top_k=3, metric="dot")
    assert [idx for idx, _ in dot_top] == [1, 2, 0]
    assert dot_top[0][1] == pytest.approx(2.0)

    # Euclidean: idx 2 (0.0) < idx 1 (1.0) < idx 0 (sqrt(2))
    l2_top = exact_nearest_neighbors(query, matrix, top_k=10, metric="euclidean")
    assert [idx for idx, _ in l2_top] == [2, 1, 0, 3]
    assert l2_top[0][1] == pytest.approx(0.0)
    assert l2_top[1][1] == pytest.approx(1.0)
    assert l2_top[2][1] == pytest.approx(math.sqrt(2.0))


def test_empty_matrix_and_validation_errors() -> None:
    query = np.array([1.0, 2.0], dtype=np.float32)
    empty = np.empty((0, 2), dtype=np.float32)

    assert dot_product(query, empty).shape == (0,)
    assert cosine_similarity(query, empty).shape == (0,)
    assert euclidean_distance(query, empty).shape == (0,)
    assert exact_nearest_neighbors(query, empty, top_k=5, metric="cosine") == []

    # Zero-norm query is valid for dot and euclidean, invalid for cosine
    zero_q = np.array([0.0, 0.0], dtype=np.float32)
    one_row = np.array([[3.0, 4.0]], dtype=np.float32)
    assert dot_product(zero_q, one_row) == pytest.approx([0.0])
    assert euclidean_distance(zero_q, one_row) == pytest.approx([5.0])
    with pytest.raises(ValueError, match="non-zero norm"):
        cosine_similarity(zero_q, one_row)

    for bad_k in (0, -1, True):
        with pytest.raises(ValueError, match="top_k"):
            exact_nearest_neighbors(query, one_row, top_k=bad_k)

    with pytest.raises(ValueError, match="unsupported metric"):
        exact_nearest_neighbors(query, one_row, top_k=1, metric="manhattan")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="1D"):
        dot_product(np.array([[1.0, 2.0]]), one_row)

    with pytest.raises(ValueError, match="2D"):
        dot_product(query, np.array([[1.0, 2.0, 3.0]]))

    with pytest.raises(ValueError, match="finite"):
        dot_product(np.array([float("nan"), 1.0]), one_row)

    with pytest.raises(ValueError, match="finite"):
        dot_product(query, np.array([[float("inf"), 1.0]]))
