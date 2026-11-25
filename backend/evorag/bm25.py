"""From-scratch Okapi BM25 scoring with Robertson-Sparck Jones IDF."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BM25Candidate:
    chunk_id: str
    chunk_length: int
    term_frequencies: Mapping[str, int]


@dataclass(frozen=True)
class BM25TermContribution:
    term: str
    tf: int
    df: int
    idf: float
    score: float


@dataclass(frozen=True)
class BM25ScoredChunk:
    chunk_id: str
    score: float
    matched_terms: tuple[str, ...]
    contributions: tuple[BM25TermContribution, ...]


def bm25_idf(corpus_size: int, document_frequency: int) -> float:
    """Compute BM25 IDF: ln(1 + (N - df + 0.5) / (df + 0.5))."""
    if corpus_size <= 0 or document_frequency <= 0:
        return 0.0
    effective_df = min(document_frequency, corpus_size)
    return math.log(1.0 + (corpus_size - effective_df + 0.5) / (effective_df + 0.5))


def validate_bm25_params(k1: float, b: float, top_k: int) -> None:
    if isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not math.isfinite(k1) or k1 < 0.0:
        raise ValueError("k1 must be a finite non-negative float")
    if not math.isfinite(b) or not 0.0 <= b <= 1.0:
        raise ValueError("b must be a finite float in [0.0, 1.0]")


def score_bm25(
    query_tokens: Sequence[str],
    candidates: Sequence[BM25Candidate],
    *,
    corpus_size: int,
    avg_doc_length: float,
    document_frequencies: Mapping[str, int],
    k1: float = 1.5,
    b: float = 0.75,
    top_k: int = 20,
) -> list[BM25ScoredChunk]:
    """Score candidate chunks against query_tokens using BM25.

    Repeated query terms are summed in order (equivalent to weighting by query term
    frequency), bounded by the analyzer's max_query_tokens limit. Ties are broken
    deterministically by ascending chunk_id.
    """
    validate_bm25_params(k1, b, top_k)
    if corpus_size <= 0 or avg_doc_length <= 0.0 or not query_tokens or not candidates:
        return []

    term_idfs: dict[str, float] = {}
    for term in query_tokens:
        if term not in term_idfs:
            df = document_frequencies.get(term, 0)
            term_idfs[term] = bm25_idf(corpus_size, df)

    if not any(val > 0.0 for val in term_idfs.values()):
        return []

    scored: list[BM25ScoredChunk] = []
    for candidate in candidates:
        if candidate.chunk_length <= 0:
            continue
        length_norm = 1.0 - b + b * (candidate.chunk_length / avg_doc_length)
        total_score = 0.0
        matched: list[str] = []
        contributions: list[BM25TermContribution] = []
        for term in query_tokens:
            idf = term_idfs[term]
            if idf <= 0.0:
                continue
            tf = candidate.term_frequencies.get(term, 0)
            if tf <= 0:
                continue
            denom = tf + k1 * length_norm
            if denom <= 0.0:
                continue
            term_score = idf * (tf * (k1 + 1.0)) / denom
            total_score += term_score
            if term not in matched:
                matched.append(term)
            contributions.append(
                BM25TermContribution(
                    term=term,
                    tf=tf,
                    df=document_frequencies.get(term, 0),
                    idf=idf,
                    score=term_score,
                )
            )
        if total_score > 0.0:
            scored.append(
                BM25ScoredChunk(
                    chunk_id=candidate.chunk_id,
                    score=total_score,
                    matched_terms=tuple(matched),
                    contributions=tuple(contributions),
                )
            )

    scored.sort(key=lambda item: (-item.score, item.chunk_id))
    return scored[:top_k]
