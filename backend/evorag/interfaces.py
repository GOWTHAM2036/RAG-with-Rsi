"""Ports to implement in subsequent milestones."""

from collections.abc import Sequence
from typing import Protocol

from .contracts import (
    Chunk,
    Document,
    DocumentMetadata,
    Embedding,
    EvaluationResult,
    Evidence,
    ExperimentConfiguration,
    QueryRequest,
    QueryResponse,
    RetrievalResult,
)


class Ingestor(Protocol):
    async def parse(self, payload: bytes, metadata: DocumentMetadata) -> Document: ...


class EmbeddingProvider(Protocol):
    async def embed(self, chunks: Sequence[Chunk]) -> Sequence[Embedding]: ...


class Retriever(Protocol):
    async def retrieve(self, request: QueryRequest) -> Sequence[RetrievalResult]: ...


class Generator(Protocol):
    async def generate(
        self, request: QueryRequest, evidence: Sequence[Evidence]
    ) -> QueryResponse: ...


class Evaluator(Protocol):
    async def evaluate(
        self, response: QueryResponse, experiment: ExperimentConfiguration, reference: str
    ) -> Sequence[EvaluationResult]: ...
