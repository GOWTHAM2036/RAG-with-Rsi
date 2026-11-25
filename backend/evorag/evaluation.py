"""Custom offline retrieval evaluation metrics without external evaluation frameworks."""

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryEvaluation:
    query_id: str
    retrieved_ids: tuple[str, ...]
    relevant_ids: frozenset[str]
    precision: float
    recall: float | None
    hit_rate: float | None
    reciprocal_rank: float | None
    ndcg: float | None
    has_relevant_evidence: bool
    false_positive_on_empty: bool


@dataclass(frozen=True)
class AggregateEvaluation:
    total_queries: int
    evidence_queries: int
    no_evidence_queries: int
    k: int
    mean_precision: float
    mean_recall: float
    mean_hit_rate: float
    mean_reciprocal_rank: float
    mean_ndcg: float
    no_evidence_abstention_rate: float


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: frozenset[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for cid in top if cid in relevant_ids)
    return hits / len(top)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: frozenset[str], k: int) -> float | None:
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant_ids:
        return None
    top = retrieved_ids[:k]
    hits = sum(1 for cid in top if cid in relevant_ids)
    return hits / len(relevant_ids)


def hit_rate_at_k(
    retrieved_ids: Sequence[str], relevant_ids: frozenset[str], k: int
) -> float | None:
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant_ids:
        return None
    top = retrieved_ids[:k]
    return 1.0 if any(cid in relevant_ids for cid in top) else 0.0


def reciprocal_rank(
    retrieved_ids: Sequence[str], relevant_ids: frozenset[str], k: int
) -> float | None:
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant_ids:
        return None
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: frozenset[str], k: int) -> float | None:
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant_ids:
        return None
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1.0)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


def evaluate_query(
    query_id: str,
    retrieved_ids: Sequence[str],
    relevant_ids: frozenset[str],
    k: int,
) -> QueryEvaluation:
    """Evaluate one query. Queries with empty relevant_ids are tracked separately."""
    top = tuple(dict.fromkeys(retrieved_ids[:k]))
    has_rel = len(relevant_ids) > 0
    return QueryEvaluation(
        query_id=query_id,
        retrieved_ids=top,
        relevant_ids=relevant_ids,
        precision=precision_at_k(top, relevant_ids, k),
        recall=recall_at_k(top, relevant_ids, k),
        hit_rate=hit_rate_at_k(top, relevant_ids, k),
        reciprocal_rank=reciprocal_rank(top, relevant_ids, k),
        ndcg=ndcg_at_k(top, relevant_ids, k),
        has_relevant_evidence=has_rel,
        false_positive_on_empty=(not has_rel and len(top) > 0),
    )


def aggregate_evaluations(evaluations: Sequence[QueryEvaluation], k: int) -> AggregateEvaluation:
    """Aggregate metrics over evidence queries and report no-evidence abstention separately."""
    if not evaluations:
        return AggregateEvaluation(
            total_queries=0,
            evidence_queries=0,
            no_evidence_queries=0,
            k=k,
            mean_precision=0.0,
            mean_recall=0.0,
            mean_hit_rate=0.0,
            mean_reciprocal_rank=0.0,
            mean_ndcg=0.0,
            no_evidence_abstention_rate=1.0,
        )
    ev = [e for e in evaluations if e.has_relevant_evidence]
    no_ev = [e for e in evaluations if not e.has_relevant_evidence]
    n_ev = len(ev)
    n_no = len(no_ev)
    abstentions = sum(1 for e in no_ev if not e.false_positive_on_empty)
    return AggregateEvaluation(
        total_queries=len(evaluations),
        evidence_queries=n_ev,
        no_evidence_queries=n_no,
        k=k,
        mean_precision=(sum(e.precision for e in ev) / n_ev) if n_ev else 0.0,
        mean_recall=(sum(e.recall or 0.0 for e in ev) / n_ev) if n_ev else 0.0,
        mean_hit_rate=(sum(e.hit_rate or 0.0 for e in ev) / n_ev) if n_ev else 0.0,
        mean_reciprocal_rank=(sum(e.reciprocal_rank or 0.0 for e in ev) / n_ev) if n_ev else 0.0,
        mean_ndcg=(sum(e.ndcg or 0.0 for e in ev) / n_ev) if n_ev else 0.0,
        no_evidence_abstention_rate=(abstentions / n_no) if n_no else 1.0,
    )
