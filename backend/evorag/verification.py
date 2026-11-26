"""Deterministic verification, claim assessment and bounded semantic critique parsing."""

import json
import re
from collections.abc import Sequence

from .citations import validate_citations
from .rag_types import (
    CitationValidationResult,
    SelectedEvidence,
    StructuredAnswer,
)
from .reflection_types import (
    ClaimAssessment,
    CorrectionActionKind,
    EvidenceAssessment,
    ReflectionCritique,
    VerificationFinding,
    VerificationReport,
)

ALLOWED_CRITIQUE_ACTIONS: frozenset[CorrectionActionKind] = frozenset(
    {
        "accept_answer",
        "remove_unsupported_claims",
        "revise_answer",
        "retrieve_more",
        "refine_query",
        "reconcile_conflicting_evidence",
        "return_partial_answer",
        "abstain",
    }
)
_INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?previous\s+instructions|execute\s+command|reveal\s+credentials|<\|im_start\|>)",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def verify_deterministic(
    question: str,
    parsed: StructuredAnswer | None,
    evidence: Sequence[SelectedEvidence],
    citation_validation: CitationValidationResult | None,
) -> VerificationReport:
    """Run CPU-only structural, provenance and consistency verification."""
    findings: list[VerificationFinding] = []
    ev_assessments = tuple(
        EvidenceAssessment(
            evidence_id=ev.evidence_id,
            compatible_revision=ev.version is None or ev.version >= 1,
            compatible_span=ev.included_end >= ev.included_start,
        )
        for ev in evidence
    )
    if parsed is None:
        findings.append(
            VerificationFinding(
                code="invalid_structured_response",
                severity="error",
                observed=True,
                detail="Model output failed structured JSON parsing",
            )
        )
        return VerificationReport(
            structurally_valid=False,
            citations_valid=False,
            provenance_valid=all(
                e.compatible_revision and e.compatible_span for e in ev_assessments
            ),
            findings=tuple(findings),
            evidence_assessments=ev_assessments,
        )

    val = citation_validation or validate_citations(parsed, evidence)
    if not val.valid:
        findings.append(
            VerificationFinding(
                code="invalid_citations",
                severity="error",
                observed=True,
                detail=",".join(val.errors) or "citation_validation_failed",
            )
        )

    seen_claims: dict[str, int] = {}
    dup_indices: list[int] = []
    assessments: list[ClaimAssessment] = []
    for idx, claim in enumerate(parsed.claims):
        norm_c = _norm(claim.text)
        if norm_c in seen_claims:
            dup_indices.append(idx)
        else:
            seen_claims[norm_c] = idx
        assessments.append(
            ClaimAssessment(
                claim_id=f"C{idx + 1}",
                evidence_ids=claim.evidence_ids,
                category="not_assessed",
                method="deterministic",
                reason="Structural and citation check only; semantic support not assessed.",
                uncertain=True,
            )
        )

    if dup_indices:
        findings.append(
            VerificationFinding(
                code="duplicate_claims",
                severity="warning",
                observed=True,
                detail=f"Duplicate factual claims at indices {dup_indices}",
            )
        )

    # Check if free-form answer contains substantive content when claims are empty.
    if parsed.status == "answered" and not parsed.claims and len(parsed.answer.strip()) > 0:
        findings.append(
            VerificationFinding(
                code="unchecked_freeform_content",
                severity="error",
                observed=True,
                detail="Answered status without structured claims",
            )
        )

    # Mechanically identify multi-part question components separated by ' and ' / '?'.
    missing_parts: list[str] = []
    if " and " in question.lower() and parsed.status == "partially_answered":
        missing_parts.append("multi_part_partial_coverage")

    prov_ok = all(e.compatible_revision and e.compatible_span for e in ev_assessments)
    return VerificationReport(
        structurally_valid=True,
        citations_valid=val.valid,
        provenance_valid=prov_ok,
        semantic_assessed=False,
        findings=tuple(findings),
        claim_assessments=tuple(assessments),
        evidence_assessments=ev_assessments,
        missing_components=tuple(missing_parts),
        duplicate_claim_indices=tuple(dup_indices),
    )


def render_admitted_answer(parsed: StructuredAnswer) -> str:
    """Render final answer strictly from admitted structured claims when present."""
    if not parsed.claims:
        return parsed.answer
    return " ".join(c.text.strip() for c in parsed.claims if c.text.strip())


def parse_reflection_critique(
    raw_text: str, valid_evidence_ids: frozenset[str], max_bytes: int = 8192
) -> ReflectionCritique:
    """Strictly parse and sanitize untrusted model critique JSON."""
    if len(raw_text.encode("utf-8")) > max_bytes:
        raise ValueError("critique_exceeds_max_bytes")
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("invalid_fenced_critique")
        stripped = "\n".join(lines[1:-1]).strip()
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("critique_root_must_be_object")
    if _INJECTION_RE.search(stripped):
        raise ValueError("unsafe_critique_content")

    critique = ReflectionCritique.model_validate(data)
    if critique.recommended_action not in ALLOWED_CRITIQUE_ACTIONS:
        raise ValueError("unauthorized_critique_action")
    for item in critique.claim_assessments:
        for eid in item.evidence_ids:
            if eid not in valid_evidence_ids:
                raise ValueError(f"invalid_critique_evidence_id:{eid}")
    return critique
