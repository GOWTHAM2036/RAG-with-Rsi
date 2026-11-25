"""Provider-independent payloads. Spans count Unicode code points in normalized text."""

from typing import Annotated, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

from .config import Name, Settings

Text = Annotated[str, Field(min_length=1)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DocumentMetadata(Contract):
    source: Name
    media_type: Name
    title: Name | None = None


class Document(Contract):
    document_id: Name
    metadata: DocumentMetadata
    text: Text


class Chunk(Contract):
    chunk_id: Name
    document_id: Name
    start: NonNegativeInt
    end: PositiveInt
    text: Text

    @model_validator(mode="after")
    def check_span(self) -> Self:
        if self.end - self.start != len(self.text):
            raise ValueError("chunk span must match the length of its text")
        return self


class Embedding(Contract):
    """Construct with the configured dimension and the actual returned coordinate sequence."""

    chunk_id: Name
    model: Name
    dimensions: PositiveInt
    values: tuple[Annotated[FiniteFloat, Field(strict=True)], ...]

    @model_validator(mode="after")
    def check_dimensions(self) -> Self:
        if len(self.values) != self.dimensions:
            raise ValueError("provider vector length differs from configured dimensions")
        return self


class RetrievalResult(Contract):
    chunk: Chunk
    rank: PositiveInt
    score: FiniteFloat


class Evidence(Contract):
    evidence_id: Name
    result: RetrievalResult


class Citation(Contract):
    evidence_id: Name
    answer_start: NonNegativeInt
    answer_end: PositiveInt

    @model_validator(mode="after")
    def check_span(self) -> Self:
        if self.answer_start >= self.answer_end:
            raise ValueError("citation requires a nonempty answer span")
        return self


class QueryRequest(Contract):
    query_id: Name
    question: Annotated[str, Field(min_length=1, max_length=16_384, pattern=r"\S")]


class ExecutionTrace(Contract):
    trace_id: Name
    query_id: Name
    started_at: AwareDatetime
    elapsed_ms: Annotated[FiniteFloat, Field(ge=0)]
    stages: tuple[Name, ...]


class QueryResponse(Contract):
    answer: Annotated[str, Field(min_length=1, pattern=r"\S")]
    evidence: tuple[Evidence, ...]
    citations: tuple[Citation, ...]
    trace: ExecutionTrace

    @model_validator(mode="after")
    def check_citations(self) -> Self:
        ids = {item.evidence_id for item in self.evidence}
        if len(ids) != len(self.evidence):
            raise ValueError("evidence IDs must be unique")
        for citation in self.citations:
            if citation.evidence_id not in ids or citation.answer_end > len(self.answer):
                raise ValueError("citation must identify evidence and a span within the answer")
        return self


class EvaluationResult(Contract):
    experiment_id: Name
    trace_id: Name
    metric: Name
    score: FiniteFloat


class ExperimentConfiguration(Contract):
    experiment_id: Name
    configuration_version: Name
    dataset_version: Name
    held_out_split: Name
    seed: NonNegativeInt
    settings: Settings
