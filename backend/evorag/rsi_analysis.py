"""Deterministic failure pattern analysis, opportunity prioritization and hypotheses."""

from collections.abc import Sequence

from .rsi_types import FailurePattern, ImprovementHypothesis, ImprovementOpportunity

_SEVERITY: dict[str, float] = {
    "missing_relevant_documents": 0.9,
    "incorrect_retrieval_ranking": 0.85,
    "incomplete_evidence_coverage": 0.75,
    "invalid_citation": 0.95,
    "incorrect_abstention": 0.80,
    "context_budget_overflow": 0.70,
}


def analyze_failure_patterns(
    failure_counts: dict[str, int], *, minimum_samples: int = 1
) -> tuple[FailurePattern, ...]:
    """Aggregate recurring failure counts into bounded FailurePattern records."""
    patterns: list[FailurePattern] = []
    for category, count in sorted(failure_counts.items()):
        if count < minimum_samples:
            continue
        sev = _SEVERITY.get(category, 0.5)
        patterns.append(
            FailurePattern(
                pattern_id=f"pat_{category}",
                category=category,
                sample_count=count,
                severity=sev,
                description=f"Observed {count} occurrences of {category}",
            )
        )
    return tuple(patterns)


def prioritize_opportunities(
    patterns: Sequence[FailurePattern],
) -> tuple[ImprovementOpportunity, ...]:
    """Compute priority: normalized_freq * severity * benefit * reliability / cost."""
    if not patterns:
        return ()
    max_count = max(p.sample_count for p in patterns) or 1
    opportunities: list[ImprovementOpportunity] = []
    for pat in patterns:
        norm_freq = pat.sample_count / max_count
        cost = 1.0
        score = (norm_freq * pat.severity * 0.8 * 0.9) / cost
        opportunities.append(
            ImprovementOpportunity(
                opportunity_id=f"opp_{pat.category}",
                pattern_ids=(pat.pattern_id,),
                priority_score=round(score, 4),
                estimated_cost=cost,
                reversible=True,
                description=f"Address {pat.category} ({pat.sample_count} samples)",
            )
        )
    opportunities.sort(key=lambda o: (-o.priority_score, o.opportunity_id))
    return tuple(opportunities)


def generate_hypotheses(
    opportunities: Sequence[ImprovementOpportunity], *, max_hypotheses: int = 3
) -> tuple[ImprovementHypothesis, ...]:
    """Generate deterministic, reversible improvement hypotheses from top opportunities."""
    hypotheses: list[ImprovementHypothesis] = []
    for opp in opportunities[:max_hypotheses]:
        if "missing_relevant_documents" in opp.opportunity_id:
            hypotheses.append(
                ImprovementHypothesis(
                    hypothesis_id=f"hyp_{opp.opportunity_id}",
                    pattern_ids=opp.pattern_ids,
                    proposed_changes={"fusion.sparse_weight": 1.25, "fusion.dense_weight": 0.8},
                    expected_benefit=(
                        "Increase lexical recall on technical identifiers and rare terms"
                    ),
                    possible_regression="Slightly lower semantic recall on paraphrased queries",
                )
            )
        elif "incorrect_retrieval_ranking" in opp.opportunity_id:
            hypotheses.append(
                ImprovementHypothesis(
                    hypothesis_id=f"hyp_{opp.opportunity_id}",
                    pattern_ids=opp.pattern_ids,
                    proposed_changes={"reranking.enabled": True},
                    expected_benefit=(
                        "Improve top-k ranking via heuristic phrase and identifier overlap"
                    ),
                    possible_regression="Minor additional CPU latency per query",
                )
            )
        elif "incomplete_evidence_coverage" in opp.opportunity_id:
            hypotheses.append(
                ImprovementHypothesis(
                    hypothesis_id=f"hyp_{opp.opportunity_id}",
                    pattern_ids=opp.pattern_ids,
                    proposed_changes={"adaptive_rag.default_strategy": "adaptive"},
                    expected_benefit="Improve multi-part coverage via query decomposition",
                    possible_regression="Higher retrieval latency from multi-query execution",
                )
            )
        else:
            hypotheses.append(
                ImprovementHypothesis(
                    hypothesis_id=f"hyp_{opp.opportunity_id}",
                    pattern_ids=opp.pattern_ids,
                    proposed_changes={"bm25.k1": 1.2},
                    expected_benefit="Tune term-frequency saturation for balanced lexical scoring",
                    possible_regression="Minor shift in BM25 score distribution",
                )
            )
    return tuple(hypotheses)
