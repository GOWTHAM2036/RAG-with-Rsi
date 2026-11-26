"""Bounded evidence consolidation and deterministic coverage diagnostics."""

from __future__ import annotations

from .context_engine import _is_redundant_overlap, estimate_tokens
from .query_types import (
    EvidenceCandidate,
    EvidenceCoverage,
    FusedMultiQueryCandidate,
    MultiQueryResult,
    RetrievalPlan,
)
from .rag_types import SelectedEvidence


def consolidate_evidence(
    multi_result: MultiQueryResult,
    plan: RetrievalPlan,
    *,
    max_items: int,
    max_evidence_tokens: int = 2048,
) -> tuple[tuple[SelectedEvidence, ...], EvidenceCoverage]:
    """Consolidate deduplicated evidence with subquery coverage and diversity diagnostics.

    Preserves source provenance and adjacent non-overlapping chunks from the same document.
    Never fabricates evidence for missing subqueries.
    """
    candidates = multi_result.candidates
    details_by_chunk: dict[str, FusedMultiQueryCandidate] = {
        d.chunk_id: d for d in multi_result.fused_details
    }

    selected: list[SelectedEvidence] = []
    accepted_spans: list[tuple[str, int, int, str]] = []
    selected_chunk_ids: set[str] = set()
    duplicate_count = 0

    def _try_accept(cand: EvidenceCandidate) -> bool:
        nonlocal duplicate_count
        chunk = cand.result.chunk
        if chunk.chunk_id in selected_chunk_ids:
            return False
        if _is_redundant_overlap(
            chunk.text, chunk.document_id, chunk.start, chunk.end, accepted_spans
        ):
            duplicate_count += 1
            return False
        eid = f"E{len(selected) + 1}"
        accepted_spans.append((chunk.document_id, chunk.start, chunk.end, chunk.text))
        selected_chunk_ids.add(chunk.chunk_id)
        selected.append(
            SelectedEvidence(
                evidence_id=eid,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                version=cand.version,
                source_id=cand.source_id,
                sources=cand.sources,
                text=chunk.text,
                original_rank=cand.result.rank,
                retrieval_mode=cand.retrieval_mode,
                score=cand.result.score,
                truncated=False,
                included_start=chunk.start,
                included_end=chunk.end,
            )
        )
        return True

    # If decomposed into multiple subqueries, first ensure at least one top candidate per subquery
    # (when available) before filling remaining slots by aggregated score.
    subquery_ids = [sq.subquery_id for sq in plan.decomposition.subqueries]
    if plan.decomposition.applied and subquery_ids and max_items > 0:
        for sq_id in subquery_ids:
            if len(selected) >= max_items:
                break
            for cand in candidates:
                detail = details_by_chunk.get(cand.result.chunk.chunk_id)
                if detail is not None and sq_id in detail.contributing_query_ids:
                    if _try_accept(cand):
                        break

    for cand in candidates:
        if len(selected) >= max_items:
            break
        _try_accept(cand)

    subqueries_with_candidates = tuple(
        sq_id for sq_id, count in multi_result.subquery_candidate_counts.items() if count > 0
    )
    selected_subqueries_set: set[str] = set()
    for item in selected:
        detail = details_by_chunk.get(item.chunk_id)
        if detail is not None:
            for qid in detail.contributing_query_ids:
                if qid in subquery_ids:
                    selected_subqueries_set.add(qid)
    subqueries_in_selected = tuple(sq for sq in subquery_ids if sq in selected_subqueries_set)
    missing_subqueries = tuple(sq for sq in subquery_ids if sq not in selected_subqueries_set)

    total_considered = len(selected) + duplicate_count
    dup_ratio = (duplicate_count / total_considered) if total_considered > 0 else 0.0
    unique_docs = len({item.document_id for item in selected})
    diversity = (unique_docs / len(selected)) if selected else 0.0
    est_ev_tokens = sum(estimate_tokens(item.text) + 24 for item in selected)
    utilization = min(1.0, est_ev_tokens / max(1, max_evidence_tokens))

    warnings: list[str] = []
    if missing_subqueries:
        warnings.append("missing_subquery_evidence:" + ",".join(missing_subqueries))
    if multi_result.degraded:
        warnings.append(f"retrieval_degraded:{multi_result.degradation_reason or 'unknown'}")

    coverage = EvidenceCoverage(
        subqueries_with_candidates=subqueries_with_candidates,
        subqueries_in_selected_evidence=subqueries_in_selected,
        missing_subqueries=missing_subqueries,
        duplicate_evidence_ratio=round(dup_ratio, 6),
        unique_document_count=unique_docs,
        document_diversity=round(diversity, 6),
        evidence_budget_utilization=round(utilization, 6),
        retrieval_degraded=multi_result.degraded,
        degradation_reason=multi_result.degradation_reason,
        warnings=tuple(warnings),
    )
    return tuple(selected), coverage
