"""Structured failure taxonomy distinguishing observed outcomes from inferred causes."""

from collections import Counter
from collections.abc import Sequence

from .eval_types import EvaluationCase, EvaluationCaseOutcome, FailureClassification
from .rag_types import RAGResponse


def classify_case_failures(
    case: EvaluationCase,
    response: RAGResponse,
    recall_at_k: float | None,
    citation_validity: float | None,
    answer_f1: float | None,
) -> tuple[FailureClassification, ...]:
    """Classify observed and inferred failures for a single evaluation case."""
    items: list[FailureClassification] = []
    cid = case.case_id

    if response.admission_outcome == "resource_unavailable":
        items.append(
            FailureClassification(
                case_id=cid,
                category="resource_admission_failure",
                observed=True,
                detail="Resource admission rejected generation",
            )
        )
        return tuple(items)

    if response.admission_outcome == "invalid_model_output":
        items.append(
            FailureClassification(
                case_id=cid,
                category="invalid_model_output",
                observed=True,
                detail="Structured output parser rejected model output",
            )
        )
    elif response.admission_outcome == "invalid_citations":
        items.append(
            FailureClassification(
                case_id=cid,
                category="invalid_citation",
                observed=True,
                detail="Citation validation failed",
            )
        )
    elif response.admission_outcome == "generation_failed":
        items.append(
            FailureClassification(
                case_id=cid,
                category="provider_timeout",
                observed=True,
                detail="Generation provider failed or timed out",
            )
        )

    abstained = response.admission_outcome == "insufficient_evidence"
    if case.ground_truth.expect_abstain != abstained:
        items.append(
            FailureClassification(
                case_id=cid,
                category="incorrect_abstention",
                observed=True,
                detail=f"expected_abstain={case.ground_truth.expect_abstain}, actual={abstained}",
            )
        )

    if recall_at_k is not None and recall_at_k == 0.0 and not case.ground_truth.expect_abstain:
        items.append(
            FailureClassification(
                case_id=cid,
                category="missing_relevant_documents",
                observed=True,
                detail="Expected documents were absent from retrieved top-k",
            )
        )

    if citation_validity is not None and citation_validity < 1.0:
        items.append(
            FailureClassification(
                case_id=cid,
                category="invalid_citation",
                observed=True,
                detail=f"citation_validity={citation_validity:.3f}",
            )
        )

    if answer_f1 is not None and answer_f1 < 0.2 and not case.ground_truth.expect_abstain:
        items.append(
            FailureClassification(
                case_id=cid,
                category="incorrect_answer",
                observed=False,
                detail=f"Low lexical proxy F1={answer_f1:.3f}",
            )
        )
    return tuple(items)


def summarize_failures(outcomes: Sequence[EvaluationCaseOutcome]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for outcome in outcomes:
        for failure in outcome.failures:
            counts[failure.category] += 1
    return dict(sorted(counts.items()))
