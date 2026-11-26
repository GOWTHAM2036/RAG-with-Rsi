"""Unit tests for strict structured-answer parsing, citations and admission."""

import pytest
from evorag.admission import admit_answer
from evorag.citations import OutputParseError, parse_structured_answer, validate_citations
from evorag.rag_types import SelectedEvidence


def _ev(eid: str = "E1", source_id: str | None = "C:/secret/path/doc.md") -> SelectedEvidence:
    return SelectedEvidence(
        evidence_id=eid,
        chunk_id="chunk-1",
        document_id="doc-1",
        version=2,
        source_id=source_id,
        text="EvoRag uses SQLite WAL for metadata persistence.",
        original_rank=1,
        retrieval_mode="hybrid",
        score=0.95,
        included_start=0,
        included_end=47,
    )


def test_parse_structured_answer_valid_and_fenced() -> None:
    raw = (
        '```json\n{"status":"answered","answer":"SQLite WAL is used.",'
        '"claims":[{"text":"SQLite WAL is used.","evidence_ids":["E1","E1"]}],'
        '"limitations":[]}\n```'
    )
    parsed = parse_structured_answer(raw)
    assert parsed.status == "answered"
    assert parsed.claims[0].evidence_ids == ("E1",)


@pytest.mark.parametrize(
    "bad_raw",
    [
        "",
        "not json at all",
        '{"status":"answered", broken',
        "[1, 2, 3]",
        '{"status":"answered"}',
        '{"status":"answered","answer":"   ","claims":[]}',
        '{"status":"answered","answer":"ok","claims":[{"text":"x","evidence_ids":["BAD"]}]}',
        '```json\n{"status":"answered","answer":"ok"}',
    ],
)
def test_parse_structured_answer_rejects_malformed_and_invalid(bad_raw: str) -> None:
    with pytest.raises(OutputParseError):
        parse_structured_answer(bad_raw)


def test_parse_structured_answer_enforces_size_and_claim_limits() -> None:
    with pytest.raises(OutputParseError, match="max_bytes"):
        parse_structured_answer('{"status":"answered","answer":"hello"}', max_bytes=10)
    with pytest.raises(OutputParseError, match="claims"):
        parse_structured_answer(
            '{"status":"answered","answer":"ok","claims":[{"text":"a","evidence_ids":["E1"]},{"text":"b","evidence_ids":["E1"]}]}',
            max_claims=1,
        )


def test_citation_validation_and_path_redaction() -> None:
    parsed = parse_structured_answer(
        '{"status":"answered","answer":"Fact A and Fact B.",'
        '"claims":[{"text":"Fact A","evidence_ids":["E1"]},'
        '{"text":"Fact B","evidence_ids":["E1"]}]}'
    )
    res = validate_citations(parsed, [_ev()])
    assert res.valid is True
    assert len(res.citations) == 1
    assert res.citations[0].citation_number == 1
    assert res.citations[0].source_id == "doc.md"
    assert res.citations[0].claim_indices == (0, 1)

    invented = parse_structured_answer(
        '{"status":"answered","answer":"Fake.","claims":[{"text":"Fake","evidence_ids":["E99"]}]}'
    )
    bad = validate_citations(invented, [_ev()])
    assert bad.valid is False
    assert bad.invalid_evidence_ids == ("E99",)


def test_admission_outcomes() -> None:
    no_ev = admit_answer(None, None, has_evidence=False, allow_partial=True, require_citations=True)
    assert no_ev.outcome == "insufficient_evidence" and no_ev.accepted is True

    res_unavail = admit_answer(
        None,
        None,
        has_evidence=True,
        allow_partial=True,
        require_citations=True,
        error_code="insufficient_memory",
    )
    assert res_unavail.outcome == "resource_unavailable" and res_unavail.accepted is False

    bad_out = admit_answer(
        None, None, has_evidence=True, allow_partial=True, require_citations=True
    )
    assert bad_out.outcome == "invalid_model_output" and bad_out.accepted is False

    parsed = parse_structured_answer(
        '{"status":"partially_answered","answer":"Partial.",'
        '"claims":[{"text":"Partial","evidence_ids":["E1"]}],"limitations":["missing detail"]}'
    )
    val = validate_citations(parsed, [_ev()])
    partial = admit_answer(
        parsed, val, has_evidence=True, allow_partial=True, require_citations=True
    )
    assert partial.outcome == "partially_accepted" and partial.accepted is True

    disallowed = admit_answer(
        parsed, val, has_evidence=True, allow_partial=False, require_citations=True
    )
    assert disallowed.accepted is False

    model_insuff = parse_structured_answer(
        '{"status":"insufficient_evidence","answer":"Not enough data.",'
        '"claims":[],"limitations":[]}'
    )
    insuff_res = admit_answer(
        model_insuff, None, has_evidence=True, allow_partial=True, require_citations=True
    )
    assert insuff_res.outcome == "insufficient_evidence" and insuff_res.accepted is True

    uncited = parse_structured_answer('{"status":"answered","answer":"Uncited.","claims":[]}')
    uncited_val = validate_citations(uncited, [_ev()])
    uncited_adm = admit_answer(
        uncited, uncited_val, has_evidence=True, allow_partial=True, require_citations=True
    )
    assert uncited_adm.outcome == "invalid_citations" and uncited_adm.accepted is False
