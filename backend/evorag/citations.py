"""Strict bounded structured-answer parsing and deterministic citation validation."""

import json
import re
from collections.abc import Sequence
from pathlib import PurePath

from pydantic import ValidationError

from .rag_types import (
    CitationReference,
    CitationValidationResult,
    SelectedEvidence,
    StructuredAnswer,
    StructuredClaim,
)

_EVIDENCE_ID_RE = re.compile(r"^E[1-9]\d*$")


class OutputParseError(ValueError):
    """Raised when raw model output violates the bounded structured-answer contract."""


def _extract_json_payload(raw_text: str, max_bytes: int) -> str:
    encoded = raw_text.encode("utf-8", errors="strict")
    if len(encoded) > max_bytes:
        raise OutputParseError("model_output_exceeds_max_bytes")
    stripped = raw_text.strip()
    if not stripped:
        raise OutputParseError("empty_model_output")
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if (
            len(lines) < 3
            or lines[0].strip() not in {"```", "```json"}
            or lines[-1].strip() != "```"
        ):
            raise OutputParseError("invalid_fenced_json")
        return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_structured_answer(
    raw_text: str, *, max_bytes: int = 16384, max_claims: int = 16
) -> StructuredAnswer:
    """Strictly parse and validate a structured JSON answer without repairing broken JSON."""
    payload = _extract_json_payload(raw_text, max_bytes)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise OutputParseError("malformed_json") from exc
    if not isinstance(data, dict):
        raise OutputParseError("json_root_must_be_object")
    if "status" not in data or "answer" not in data:
        raise OutputParseError("missing_required_fields")

    raw_claims = data.get("claims", [])
    if not isinstance(raw_claims, list) or len(raw_claims) > max_claims:
        raise OutputParseError("invalid_or_excessive_claims")

    normalized_claims: list[dict[str, object]] = []
    for item in raw_claims:
        if not isinstance(item, dict):
            raise OutputParseError("claim_must_be_object")
        raw_ids = item.get("evidence_ids", [])
        if not isinstance(raw_ids, list):
            raise OutputParseError("evidence_ids_must_be_list")
        deduped_ids: list[str] = []
        seen: set[str] = set()
        for eid in raw_ids:
            if not isinstance(eid, str) or not _EVIDENCE_ID_RE.match(eid):
                raise OutputParseError("invalid_evidence_id_format")
            if eid not in seen:
                seen.add(eid)
                deduped_ids.append(eid)
        normalized_claims.append({"text": item.get("text"), "evidence_ids": deduped_ids})

    try:
        parsed = StructuredAnswer.model_validate(
            {
                "status": data["status"],
                "answer": data["answer"],
                "claims": normalized_claims,
                "limitations": data.get("limitations", []),
            }
        )
    except ValidationError as exc:
        raise OutputParseError("schema_validation_failed") from exc

    if parsed.status != "insufficient_evidence" and not parsed.answer.strip():
        raise OutputParseError("empty_answer")
    return parsed


def _safe_source_id(source_id: str | None) -> str | None:
    if not source_id:
        return None
    # Never expose absolute filesystem paths; keep only the final basename.
    return PurePath(source_id).name or None


def validate_citations(
    parsed: StructuredAnswer,
    evidence: Sequence[SelectedEvidence],
    *,
    validate_identifiers: bool = True,
    reject_unknown_sources: bool = True,
) -> CitationValidationResult:
    """Resolve model evidence IDs strictly against the evidence items sent in the prompt."""
    by_id = {item.evidence_id: item for item in evidence}
    invalid_ids: list[str] = []
    uncited_claims: list[int] = []
    errors: list[str] = []
    claim_map: dict[str, list[int]] = {}
    ordered_eids: list[str] = []

    claims: tuple[StructuredClaim, ...] = parsed.claims
    for idx, claim in enumerate(claims):
        if not claim.evidence_ids:
            uncited_claims.append(idx)
            continue
        for eid in claim.evidence_ids:
            if validate_identifiers and eid not in by_id:
                if eid not in invalid_ids:
                    invalid_ids.append(eid)
                continue
            if eid in by_id:
                if eid not in claim_map:
                    claim_map[eid] = []
                    ordered_eids.append(eid)
                claim_map[eid].append(idx)

    if invalid_ids and reject_unknown_sources:
        errors.append("unknown_evidence_identifiers")
    if uncited_claims:
        errors.append("uncited_claims")

    citations: list[CitationReference] = []
    for num, eid in enumerate(ordered_eids, start=1):
        ev = by_id[eid]
        citations.append(
            CitationReference(
                citation_number=num,
                evidence_id=ev.evidence_id,
                document_id=ev.document_id,
                version=ev.version,
                chunk_id=ev.chunk_id,
                source_id=_safe_source_id(ev.source_id),
                sources=ev.sources,
                excerpt=ev.text[:240],
                claim_indices=tuple(claim_map[eid]),
            )
        )

    valid = not errors
    return CitationValidationResult(
        valid=valid,
        citations=tuple(citations),
        invalid_evidence_ids=tuple(invalid_ids),
        uncited_claim_indices=tuple(uncited_claims),
        errors=tuple(errors),
    )
