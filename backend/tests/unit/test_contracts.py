from datetime import UTC, datetime

import pytest
from evorag.config import Embeddings, Settings
from evorag.contracts import (
    Chunk,
    Citation,
    Document,
    DocumentMetadata,
    Embedding,
    EvaluationResult,
    Evidence,
    ExecutionTrace,
    ExperimentConfiguration,
    QueryRequest,
    QueryResponse,
    RetrievalResult,
)
from evorag.provider_payloads import ProviderResponseError, parse_ollama_embeddings
from pydantic import ValidationError


def test_provenance_round_trip() -> None:
    doc = Document(
        document_id="d",
        metadata=DocumentMetadata(source="a.txt", media_type="text/plain"),
        text=" a🐍",
    )
    chunk = Chunk(chunk_id="c", document_id=doc.document_id, start=0, end=3, text=doc.text)
    evidence = Evidence(evidence_id="e", result=RetrievalResult(chunk=chunk, rank=1, score=-0.25))
    citation = Citation(evidence_id="e", answer_start=0, answer_end=1)
    query = QueryRequest(query_id="q", question="Which symbol?")
    trace = ExecutionTrace(
        trace_id="t",
        query_id=query.query_id,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        elapsed_ms=0,
        stages=(),
    )
    response = QueryResponse(answer="🐍", evidence=(evidence,), citations=(citation,), trace=trace)
    assert QueryResponse.model_validate_json(response.model_dump_json()) == response
    for items, answer in [((), "a"), ((evidence, evidence), "a"), ((evidence,), "a")]:
        broken = citation.model_copy(update={"answer_end": 2}) if items == (evidence,) else citation
        with pytest.raises(ValidationError):
            QueryResponse(answer=answer, citations=(broken,), evidence=items, trace=trace)
    exp = ExperimentConfiguration(
        experiment_id="exp",
        configuration_version="v1",
        dataset_version="corpus@1",
        held_out_split="test",
        seed=0,
        settings=Settings(),
    )
    result = EvaluationResult(
        experiment_id=exp.experiment_id, trace_id=trace.trace_id, metric="recall@5", score=1
    )
    assert result.trace_id == response.trace.trace_id


def test_validate_provider_vector() -> None:
    embedding = Embedding(chunk_id="c", model="test", dimensions=2, values=(0.0, 0.5))
    assert embedding.dimensions == len(embedding.values)
    for values in [(0.1,), (float("nan"), 0.2), (float("inf"), 0.2)]:
        with pytest.raises(ValidationError):
            Embedding(chunk_id="c", model="test", dimensions=2, values=values)


def test_invalid_spans_and_blank_query() -> None:
    with pytest.raises(ValidationError):
        Chunk(chunk_id="c", document_id="d", start=0, end=3, text="ab")
    with pytest.raises(ValidationError):
        Citation(evidence_id="e", answer_start=2, answer_end=2)
    with pytest.raises(ValidationError):
        QueryRequest(query_id="q", question="   ")


def test_ollama_embed_response_boundary() -> None:
    settings = Settings(embeddings=Embeddings(dimensions=2))
    model = settings.runtime.embedding_model
    payload = {"model": model, "embeddings": [[1, 0], [0.25, 0.75]]}
    rows = parse_ollama_embeddings(payload, ["a", "b"], settings)
    assert [(row.chunk_id, row.values) for row in rows] == [("a", (1, 0)), ("b", (0.25, 0.75))]
    with pytest.raises(ProviderResponseError):
        parse_ollama_embeddings(payload, ["a"], settings)
    with pytest.raises(ProviderResponseError):
        parse_ollama_embeddings({**payload, "model": "other"}, ["a", "b"], settings)
    for bad in [[1], ["3", 1], [True, 1]]:
        with pytest.raises(ValidationError):
            parse_ollama_embeddings({"model": model, "embeddings": [bad]}, ["a"], settings)
