"""Deterministic CPU-only heuristic reranker over bounded candidate sets.

Note:
    Heuristic reranking scores are uncalibrated ranking signals for relative
    ordering and must not be interpreted as calibrated probabilities.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .analyzer import TextAnalyzer, is_technical_identifier
from .config import Reranking
from .fusion import FusedCandidate


@dataclass(frozen=True)
class RerankedCandidate:
    fused: FusedCandidate
    rerank_score: float
    final_rank: int


def _jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def rerank_candidates(
    query: str,
    candidates: Sequence[FusedCandidate],
    analyzer: TextAnalyzer,
    config: Reranking,
    *,
    top_k: int | None = None,
) -> list[RerankedCandidate]:
    """Deterministically rerank fused candidates using lexical and diversity signals."""
    if not candidates:
        return []
    bounded = list(candidates[: config.max_candidates])
    query_norm = analyzer.normalize(query).strip()
    query_tokens = analyzer.tokenize(query, for_query=True)
    unique_query_tokens = tuple(dict.fromkeys(query_tokens))
    identifier_tokens = tuple(t for t in unique_query_tokens if is_technical_identifier(t))

    chunk_token_sets: list[frozenset[str]] = []
    base_scores: list[float] = []

    for cand in bounded:
        chunk_norm = analyzer.normalize(cand.chunk.text)
        chunk_tokens = frozenset(analyzer.tokenize(cand.chunk.text, for_query=False))
        chunk_token_sets.append(chunk_tokens)

        phrase_match = (
            1.0
            if len(unique_query_tokens) >= 2 and query_norm and query_norm in chunk_norm
            else 0.0
        )
        coverage = (
            sum(1 for t in unique_query_tokens if t in chunk_tokens) / len(unique_query_tokens)
            if unique_query_tokens
            else 0.0
        )
        id_match = (
            sum(1 for t in identifier_tokens if t in chunk_tokens) / len(identifier_tokens)
            if identifier_tokens
            else 0.0
        )
        rank_signal = 1.0 / float(cand.fused_rank)

        score = (
            config.phrase_weight * phrase_match
            + config.coverage_weight * coverage
            + config.identifier_weight * id_match
            + config.rrf_weight * rank_signal
        )
        base_scores.append(score)

    # Greedy selection applying duplicate and document-diversity penalties deterministically.
    remaining = list(range(len(bounded)))
    selected_indices: list[int] = []
    selected_scores: list[float] = []
    doc_counts: dict[str, int] = {}
    limit = len(bounded) if top_k is None else min(top_k, len(bounded))

    while remaining and len(selected_indices) < limit:
        best_idx = -1
        best_score = float("-inf")
        best_chunk_id = ""
        for idx in remaining:
            cand = bounded[idx]
            tokens = chunk_token_sets[idx]
            dup_pen = 0.0
            for prev_idx in selected_indices:
                if _jaccard_similarity(tokens, chunk_token_sets[prev_idx]) >= 0.85:
                    dup_pen = config.duplicate_penalty
                    break
            doc_pen = doc_counts.get(cand.chunk.document_id, 0) * config.diversity_penalty
            adjusted = base_scores[idx] - dup_pen - doc_pen
            cid = cand.chunk.chunk_id
            if (
                best_idx == -1
                or adjusted > best_score
                or (adjusted == best_score and cid < best_chunk_id)
            ):
                best_idx = idx
                best_score = adjusted
                best_chunk_id = cid
        remaining.remove(best_idx)
        selected_indices.append(best_idx)
        selected_scores.append(best_score)
        doc_id = bounded[best_idx].chunk.document_id
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

    return [
        RerankedCandidate(
            fused=bounded[idx],
            rerank_score=score,
            final_rank=rank,
        )
        for rank, (idx, score) in enumerate(
            zip(selected_indices, selected_scores, strict=True), start=1
        )
    ]
