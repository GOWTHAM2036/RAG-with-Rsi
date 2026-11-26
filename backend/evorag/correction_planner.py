"""Deterministic reflection triggers, correction planning, reconciliation and comparison."""

from collections.abc import Sequence

from .config import Reflection
from .context_engine import select_evidence
from .query_types import EvidenceCandidate, EvidenceCoverage
from .rag_types import (
    AnswerAdmissionResult,
    CitationValidationResult,
    SelectedEvidence,
    StructuredAnswer,
)
from .reflection_types import (
    AnswerComparison,
    CorrectionAction,
    CorrectionPlan,
    ReflectionCritique,
    ReflectionTrigger,
    VerificationReport,
)


def evaluate_triggers(
    report: VerificationReport,
    admission: AnswerAdmissionResult,
    coverage: EvidenceCoverage | None,
    critique: ReflectionCritique | None,
    *,
    explicit_request: bool,
    is_complex_multipart: bool,
    settings: Reflection,
) -> tuple[ReflectionTrigger, ...]:
    """Identify deterministic and critique-based reflection triggers."""
    triggers: list[ReflectionTrigger] = []
    if explicit_request:
        triggers.append(
            ReflectionTrigger(
                kind="explicit_request", reason="Caller explicitly requested reflection"
            )
        )
    if not report.structurally_valid:
        triggers.append(
            ReflectionTrigger(
                kind="invalid_structured_response",
                reason="Initial model output failed structured validation",
            )
        )
    if not report.citations_valid:
        triggers.append(
            ReflectionTrigger(
                kind="invalid_citations", reason="Initial answer had invalid or missing citations"
            )
        )
    if admission.outcome == "insufficient_evidence" and admission.status == "insufficient_evidence":
        triggers.append(
            ReflectionTrigger(
                kind="missing_evidence", reason="Insufficient evidence for initial answer"
            )
        )
    if report.missing_components or (critique and critique.missing_components):
        triggers.append(
            ReflectionTrigger(
                kind="missing_components", reason="Question components appear unanswered"
            )
        )
    if any(c.category == "contradicted" for c in report.claim_assessments) or (
        critique and any(c.assessment == "contradicted" for c in critique.claim_assessments)
    ):
        triggers.append(
            ReflectionTrigger(kind="contradictory_evidence", reason="Contradicted claim detected")
        )
    if admission.status == "partially_answered":
        triggers.append(
            ReflectionTrigger(kind="partial_answer", reason="Initial answer is partial")
        )
    if coverage and coverage.missing_subqueries:
        triggers.append(
            ReflectionTrigger(
                kind="low_evidence_coverage",
                reason=f"Missing subqueries: {','.join(coverage.missing_subqueries)}",
            )
        )
    if settings.trigger_policy == "always_complex" and is_complex_multipart:
        triggers.append(
            ReflectionTrigger(
                kind="complex_multipart", reason="Complex multi-part query trigger policy"
            )
        )
    return tuple(triggers)


def plan_correction(
    question: str,
    triggers: Sequence[ReflectionTrigger],
    report: VerificationReport,
    critique: ReflectionCritique | None,
    settings: Reflection,
) -> CorrectionPlan:
    """Select the least expensive allowed corrective action for verified findings."""
    if not triggers:
        return CorrectionPlan(
            actions=(
                CorrectionAction(
                    action="accept_answer",
                    triggers=(),
                    justification="No reflection triggers fired",
                ),
            )
        )

    kinds = tuple(t.kind for t in triggers)
    kind_set = set(kinds)

    # Prefer deterministic removal of unsupported claims when no retrieval is needed.
    if critique and any(
        c.assessment == "insufficient_evidence" for c in critique.claim_assessments
    ):
        if settings.correction.allow_targeted_retrieval and (
            "missing_components" in kind_set or "low_evidence_coverage" in kind_set
        ):
            queries = _build_corrective_queries(
                question, critique.missing_components, settings.correction.max_query_variants
            )
            return CorrectionPlan(
                actions=(
                    CorrectionAction(
                        action="retrieve_more",
                        triggers=kinds,
                        justification="Targeted retrieval for missing components",
                        estimated_model_calls=1,
                        estimated_retrievals=len(queries),
                    ),
                ),
                corrective_queries=queries,
            )
        return CorrectionPlan(
            actions=(
                CorrectionAction(
                    action="remove_unsupported_claims",
                    triggers=kinds,
                    justification=(
                        "Deterministic removal of unsupported claims without regeneration"
                    ),
                ),
            )
        )

    if "contradictory_evidence" in kind_set:
        if settings.correction.allow_regeneration:
            return CorrectionPlan(
                actions=(
                    CorrectionAction(
                        action="reconcile_conflicting_evidence",
                        triggers=kinds,
                        justification="Reconcile conflicting evidence via bounded regeneration",
                        estimated_model_calls=1,
                    ),
                )
            )
        return CorrectionPlan(
            actions=(
                CorrectionAction(
                    action="return_partial_answer",
                    triggers=kinds,
                    justification="Regeneration disabled; return partial answer",
                ),
            )
        )

    if (
        "missing_evidence" in kind_set or "low_evidence_coverage" in kind_set
    ) and settings.correction.allow_targeted_retrieval:
        missing = critique.missing_components if critique else report.missing_components
        queries = _build_corrective_queries(
            question, missing, settings.correction.max_query_variants
        )
        return CorrectionPlan(
            actions=(
                CorrectionAction(
                    action="retrieve_more",
                    triggers=kinds,
                    justification="Targeted retrieval for missing evidence",
                    estimated_model_calls=1 if settings.correction.allow_regeneration else 0,
                    estimated_retrievals=len(queries),
                ),
            ),
            corrective_queries=queries,
        )

    if (
        "invalid_structured_response" in kind_set or "invalid_citations" in kind_set
    ) and settings.correction.allow_regeneration:
        return CorrectionPlan(
            actions=(
                CorrectionAction(
                    action="revise_answer",
                    triggers=kinds,
                    justification="Regenerate to fix invalid structure or citations",
                    estimated_model_calls=1,
                ),
            )
        )

    if "partial_answer" in kind_set and settings.correction.allow_partial_answer:
        return CorrectionPlan(
            actions=(
                CorrectionAction(
                    action="return_partial_answer",
                    triggers=kinds,
                    justification="Accept partial answer with explicit limitations",
                ),
            )
        )

    return CorrectionPlan(
        actions=(
            CorrectionAction(
                action="accept_answer",
                triggers=kinds,
                justification="No corrective action required",
            ),
        )
    )


def _build_corrective_queries(
    question: str, missing_components: Sequence[str], max_variants: int
) -> tuple[str, ...]:
    queries: list[str] = []
    for comp in missing_components:
        cleaned = " ".join(comp.split())
        if cleaned and cleaned not in queries:
            queries.append(f"{question} {cleaned}".strip()[:512])
    if not queries:
        queries.append(question)
    return tuple(queries[:max_variants])


def reconcile_evidence(
    initial: Sequence[EvidenceCandidate],
    additional: Sequence[EvidenceCandidate],
    max_items: int,
) -> tuple[SelectedEvidence, ...]:
    """Combine initial and corrective evidence deterministically while deduplicating."""
    combined = [*initial, *additional]
    return select_evidence(combined, max_items)


def compare_candidates(
    original_parsed: StructuredAnswer | None,
    original_validation: CitationValidationResult | None,
    original_admission: AnswerAdmissionResult,
    corrected_parsed: StructuredAnswer | None,
    corrected_validation: CitationValidationResult | None,
    corrected_admission: AnswerAdmissionResult,
) -> AnswerComparison:
    """Compare original vs corrected candidate without auto-preferring the newest answer."""
    orig_valid = (
        original_parsed is not None
        and original_admission.accepted
        and (original_validation.valid if original_validation else True)
    )
    corr_valid = (
        corrected_parsed is not None
        and corrected_admission.accepted
        and (corrected_validation.valid if corrected_validation else True)
    )

    if corr_valid and not orig_valid:
        return AnswerComparison(
            selected="corrected",
            original_valid=orig_valid,
            corrected_valid=corr_valid,
            reason="Corrected answer resolved initial validation or admission failure",
        )
    if orig_valid and not corr_valid:
        return AnswerComparison(
            selected="original",
            original_valid=orig_valid,
            corrected_valid=corr_valid,
            reason="Corrected answer introduced regressions; retained valid original",
        )
    if corr_valid and orig_valid:
        orig_claims = len(original_parsed.claims) if original_parsed else 0
        corr_claims = len(corrected_parsed.claims) if corrected_parsed else 0
        if (
            original_admission.status == "partially_answered"
            and corrected_admission.status == "answered"
        ) or corr_claims > orig_claims:
            return AnswerComparison(
                selected="corrected",
                original_valid=orig_valid,
                corrected_valid=corr_valid,
                reason="Corrected answer improved completeness while preserving validity",
            )
        return AnswerComparison(
            selected="original",
            original_valid=orig_valid,
            corrected_valid=corr_valid,
            reason="Original answer already satisfied admission; avoided unnecessary change",
        )

    if corrected_admission.status == "partially_answered" and corrected_admission.accepted:
        return AnswerComparison(
            selected="partial",
            original_valid=orig_valid,
            corrected_valid=corr_valid,
            reason="Selected validated partial answer",
        )
    return AnswerComparison(
        selected="abstained",
        original_valid=orig_valid,
        corrected_valid=corr_valid,
        reason="Neither candidate satisfied final admission requirements",
    )
