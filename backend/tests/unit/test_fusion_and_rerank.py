"""Unit tests for weighted RRF, CPU-only reranking, and evaluation metrics."""

import pytest
from evorag.analyzer import TextAnalyzer
from evorag.config import Reranking
from evorag.contracts import Chunk
from evorag.evaluation import (
    aggregate_evaluations,
    evaluate_query,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evorag.fusion import RankedCandidate, reciprocal_rank_fusion
from evorag.reranking import rerank_candidates


def _chunk(cid: str, doc_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, document_id=doc_id, start=0, end=len(text), text=text)


def test_rrf_mathematical_correctness_weights_and_dedup() -> None:
    c1 = _chunk("c1", "d1", "HTTP 404 error")
    c2 = _chunk("c2", "d2", "RTX 3050 GPU")
    c3 = _chunk("c3", "d3", "Python 3.12")

    dense = [
        RankedCandidate(c1, rank=1, score=0.95),
        RankedCandidate(c2, rank=2, score=0.80),
        RankedCandidate(c1, rank=3, score=0.50),  # duplicate within list -> keeps rank 1
    ]
    sparse = [
        RankedCandidate(c2, rank=1, score=3.2),
        RankedCandidate(c3, rank=2, score=2.1),
    ]
    fused = reciprocal_rank_fusion(
        dense,
        sparse,
        k=60,
        dense_weight=1.0,
        sparse_weight=2.0,
        sparse_matched_terms={"c2": ("rtx",), "c3": ("python",)},
    )
    # c2: 1/(60+2) + 2/(60+1) = 1/62 + 2/61
    # c3: 0 + 2/(60+2) = 2/62
    # c1: 1/(60+1) + 0 = 1/61
    assert [f.chunk.chunk_id for f in fused] == ["c2", "c3", "c1"]
    assert fused[0].rrf_score == pytest.approx(1.0 / 62.0 + 2.0 / 61.0)
    assert fused[0].dense_rank == 2
    assert fused[0].sparse_rank == 1
    assert fused[0].matched_terms == ("rtx",)
    assert fused[1].dense_rank is None
    assert fused[2].sparse_rank is None

    # Tie-breaking by ascending chunk_id.
    tied = reciprocal_rank_fusion(
        [RankedCandidate(_chunk("b", "d1", "x"), 1, 1.0)],
        [RankedCandidate(_chunk("a", "d2", "y"), 1, 1.0)],
        k=60,
    )
    assert [t.chunk.chunk_id for t in tied] == ["a", "b"]

    with pytest.raises(ValueError, match="rrf k"):
        reciprocal_rank_fusion([], [], k=0)
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([], [], dense_weight=0.0, sparse_weight=0.0)


def test_reranker_determinism_and_penalties() -> None:
    analyzer = TextAnalyzer()
    cfg = Reranking(
        enabled=True,
        phrase_weight=0.5,
        coverage_weight=0.3,
        identifier_weight=0.4,
        rrf_weight=0.2,
        duplicate_penalty=0.5,
        diversity_penalty=0.1,
    )
    c_exact = _chunk("c1", "d1", "Fix HTTP 404 on /api/v1/retrieve endpoint")
    c_dup = _chunk("c2", "d1", "Fix HTTP 404 on /api/v1/retrieve endpoint")
    c_partial = _chunk("c3", "d2", "HTTP status codes overview")

    fused = reciprocal_rank_fusion(
        [
            RankedCandidate(c_partial, 1, 0.9),
            RankedCandidate(c_exact, 2, 0.8),
            RankedCandidate(c_dup, 3, 0.7),
        ],
        [],
        k=60,
    )
    reranked = rerank_candidates("HTTP 404 /api/v1/retrieve", fused, analyzer, cfg, top_k=3)
    # Exact phrase + identifier match boosts c1 to rank 1; c2 is penalized as near-duplicate.
    assert reranked[0].fused.chunk.chunk_id == "c1"
    assert reranked[-1].fused.chunk.chunk_id == "c2"
    assert rerank_candidates("HTTP 404", [], analyzer, cfg) == []


def test_evaluation_metrics_and_no_evidence_handling() -> None:
    rel = frozenset({"c1", "c3"})
    ret = ["c2", "c1", "c4", "c3"]
    assert precision_at_k(ret, rel, 4) == pytest.approx(0.5)
    assert recall_at_k(ret, rel, 2) == pytest.approx(0.5)
    assert hit_rate_at_k(ret, rel, 2) == 1.0
    assert reciprocal_rank(ret, rel, 4) == pytest.approx(0.5)
    ndcg = ndcg_at_k(ret, rel, 4)
    assert ndcg is not None and 0.0 < ndcg < 1.0

    q1 = evaluate_query("q1", ret, rel, k=4)
    q_empty = evaluate_query("q2", [], frozenset(), k=4)
    q_fp = evaluate_query("q3", ["c9"], frozenset(), k=4)
    agg = aggregate_evaluations([q1, q_empty, q_fp], k=4)
    assert agg.evidence_queries == 1
    assert agg.no_evidence_queries == 2
    assert agg.no_evidence_abstention_rate == pytest.approx(0.5)
    assert aggregate_evaluations([], k=4).total_queries == 0
    assert precision_at_k([], rel, 4) == 0.0
    assert recall_at_k(ret, frozenset(), 4) is None
    assert hit_rate_at_k(ret, frozenset(), 4) is None
    assert reciprocal_rank(ret, frozenset(), 4) is None
    assert ndcg_at_k(ret, frozenset(), 4) is None
    with pytest.raises(ValueError, match="k must be >= 1"):
        precision_at_k(ret, rel, 0)
