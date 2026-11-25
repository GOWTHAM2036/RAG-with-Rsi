"""Weighted Reciprocal Rank Fusion (RRF) for deterministic rank combination."""

from collections.abc import Sequence
from dataclasses import dataclass

from .contracts import Chunk


@dataclass(frozen=True)
class RankedCandidate:
    chunk: Chunk
    rank: int
    score: float


@dataclass(frozen=True)
class FusedCandidate:
    chunk: Chunk
    rrf_score: float
    fused_rank: int
    dense_rank: int | None
    dense_score: float | None
    sparse_rank: int | None
    sparse_score: float | None
    matched_terms: tuple[str, ...]


def reciprocal_rank_fusion(
    dense_results: Sequence[RankedCandidate],
    sparse_results: Sequence[RankedCandidate],
    *,
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    sparse_matched_terms: dict[str, tuple[str, ...]] | None = None,
    top_k: int | None = None,
) -> list[FusedCandidate]:
    """Fuse dense and sparse ranked lists using weighted Reciprocal Rank Fusion.

    RRF(d) = sum over lists r of weight(r) / (k + rank_r(d))
    Ranks are 1-based. Missing candidates contribute 0.0 for that list.
    Duplicate chunk_ids within a single list keep their best (lowest) rank.
    Ties are broken deterministically by ascending chunk_id.
    """
    if isinstance(k, bool) or k < 1:
        raise ValueError("rrf k must be a positive integer")
    if dense_weight < 0.0 or sparse_weight < 0.0 or (dense_weight + sparse_weight) <= 0.0:
        raise ValueError("fusion weights must be non-negative with positive sum")
    if top_k is not None and (isinstance(top_k, bool) or top_k < 1):
        raise ValueError("top_k must be a positive integer")

    chunks: dict[str, Chunk] = {}
    dense_info: dict[str, tuple[int, float]] = {}
    sparse_info: dict[str, tuple[int, float]] = {}

    for item in dense_results:
        if item.rank < 1:
            raise ValueError("ranks must be 1-based positive integers")
        cid = item.chunk.chunk_id
        chunks.setdefault(cid, item.chunk)
        if cid not in dense_info or item.rank < dense_info[cid][0]:
            dense_info[cid] = (item.rank, item.score)

    for item in sparse_results:
        if item.rank < 1:
            raise ValueError("ranks must be 1-based positive integers")
        cid = item.chunk.chunk_id
        chunks.setdefault(cid, item.chunk)
        if cid not in sparse_info or item.rank < sparse_info[cid][0]:
            sparse_info[cid] = (item.rank, item.score)

    terms_map = sparse_matched_terms or {}
    fused: list[tuple[float, str, Chunk, int | None, float | None, int | None, float | None]] = []
    for cid, chunk in chunks.items():
        d_entry = dense_info.get(cid)
        s_entry = sparse_info.get(cid)
        score = 0.0
        if d_entry is not None and dense_weight > 0.0:
            score += dense_weight / (k + d_entry[0])
        if s_entry is not None and sparse_weight > 0.0:
            score += sparse_weight / (k + s_entry[0])
        fused.append(
            (
                score,
                cid,
                chunk,
                d_entry[0] if d_entry else None,
                d_entry[1] if d_entry else None,
                s_entry[0] if s_entry else None,
                s_entry[1] if s_entry else None,
            )
        )

    fused.sort(key=lambda row: (-row[0], row[1]))
    limited = fused[:top_k] if top_k is not None else fused
    return [
        FusedCandidate(
            chunk=row[2],
            rrf_score=row[0],
            fused_rank=idx,
            dense_rank=row[3],
            dense_score=row[4],
            sparse_rank=row[5],
            sparse_score=row[6],
            matched_terms=terms_map.get(row[1], ()),
        )
        for idx, row in enumerate(limited, start=1)
    ]
