"""Deterministic answer admission and abstention policy."""

from .rag_types import (
    AdmissionOutcome,
    AnswerAdmissionResult,
    CitationValidationResult,
    StructuredAnswer,
)

_INSUFFICIENT_MSG = "Insufficient evidence in the indexed documents to answer this question."


def admit_answer(
    parsed: StructuredAnswer | None,
    validation: CitationValidationResult | None,
    *,
    has_evidence: bool,
    allow_partial: bool,
    require_citations: bool,
    error_code: str | None = None,
) -> AnswerAdmissionResult:
    """Apply deterministic admission checks without fabricating replacement citations."""
    if error_code is not None:
        outcome: AdmissionOutcome = (
            "resource_unavailable"
            if error_code in {"insufficient_memory", "resource_unavailable", "inference_busy"}
            else "generation_failed"
        )
        return AnswerAdmissionResult(
            outcome=outcome,
            accepted=False,
            status="insufficient_evidence",
            answer=_INSUFFICIENT_MSG,
            citations=(),
            limitations=(error_code,),
            reasons=(error_code,),
        )

    if not has_evidence:
        return AnswerAdmissionResult(
            outcome="insufficient_evidence",
            accepted=True,
            status="insufficient_evidence",
            answer=_INSUFFICIENT_MSG,
            citations=(),
            limitations=("no_relevant_evidence_retrieved",),
            reasons=("no_relevant_evidence_retrieved",),
        )

    if parsed is None:
        return AnswerAdmissionResult(
            outcome="invalid_model_output",
            accepted=False,
            status="insufficient_evidence",
            answer=_INSUFFICIENT_MSG,
            citations=(),
            limitations=("invalid_model_output",),
            reasons=("invalid_model_output",),
        )

    if parsed.status == "insufficient_evidence":
        return AnswerAdmissionResult(
            outcome="insufficient_evidence",
            accepted=True,
            status="insufficient_evidence",
            answer=parsed.answer or _INSUFFICIENT_MSG,
            citations=validation.citations if validation else (),
            limitations=parsed.limitations or ("model_reported_insufficient_evidence",),
            reasons=("model_reported_insufficient_evidence",),
        )

    assert validation is not None
    if validation.invalid_evidence_ids:
        return AnswerAdmissionResult(
            outcome="invalid_citations",
            accepted=False,
            status="insufficient_evidence",
            answer=_INSUFFICIENT_MSG,
            citations=(),
            limitations=("invalid_evidence_identifiers",),
            reasons=validation.errors,
        )

    if require_citations and (not validation.citations or validation.uncited_claim_indices):
        return AnswerAdmissionResult(
            outcome="invalid_citations",
            accepted=False,
            status="insufficient_evidence",
            answer=_INSUFFICIENT_MSG,
            citations=(),
            limitations=("missing_required_citations",),
            reasons=validation.errors or ("missing_required_citations",),
        )

    if parsed.status == "partially_answered" or parsed.limitations:
        if not allow_partial:
            return AnswerAdmissionResult(
                outcome="insufficient_evidence",
                accepted=False,
                status="insufficient_evidence",
                answer=_INSUFFICIENT_MSG,
                citations=(),
                limitations=("partial_answers_disabled",),
                reasons=("partial_answers_disabled",),
            )
        return AnswerAdmissionResult(
            outcome="partially_accepted",
            accepted=True,
            status="partially_answered",
            answer=parsed.answer,
            citations=validation.citations,
            limitations=parsed.limitations,
            reasons=("partially_answered",),
        )

    return AnswerAdmissionResult(
        outcome="accepted",
        accepted=True,
        status="answered",
        answer=parsed.answer,
        citations=validation.citations,
        limitations=parsed.limitations,
        reasons=(),
    )
