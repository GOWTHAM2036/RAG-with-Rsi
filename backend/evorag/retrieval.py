"""Exact dense retrieval over disk-backed float32 vectors."""

import asyncio
from typing import Annotated
from uuid import uuid4

import numpy as np
from pydantic import Field

from .config import Name, Settings
from .contracts import Chunk, Contract, QueryRequest, RetrievalResult
from .embedding_engine import EmbeddingEngine, EmbeddingRequest
from .embedding_store import EmbeddingSpace
from .filtering import FileType
from .inference_types import ProviderError
from .similarity import DEFAULT_BLOCK_SIZE, Metric, exact_nearest_neighbors


class DenseRetrieveRequest(Contract):
    query_id: Name | None = None
    question: Annotated[str, Field(min_length=1, max_length=16_384, pattern=r"\S")]
    top_k: Annotated[int, Field(ge=1, le=100)] | None = None
    metric: Metric = "cosine"
    document_id: Name | None = None
    version: Annotated[int, Field(ge=1, le=2_147_483_647)] | None = None
    latest_only: bool = True
    file_type: FileType | None = None
    source: Name | None = None


class DenseRetrieveResponse(Contract):
    query_id: str
    model: str
    digest: str
    metric: Metric
    results: tuple[RetrievalResult, ...]


class DenseRetriever:
    """Exact nearest-neighbor dense retriever implementing the Retriever protocol."""

    def __init__(self, settings: Settings, embeddings: EmbeddingEngine) -> None:
        self.settings = settings
        self.embeddings = embeddings

    def _score_rows(
        self,
        space: EmbeddingSpace,
        query_vec: np.ndarray,
        top_k: int,
        metric: Metric,
        *,
        document_id: str | None,
        version: int | None,
        latest_only: bool,
        file_type: str | None = None,
        source: str | None = None,
    ) -> tuple[RetrievalResult, ...]:
        sid, committed_rows, rows = self.embeddings.store.space_rows(
            space,
            document_id=document_id,
            version=version,
            latest_only=latest_only,
            file_type=file_type,
            source=source,
        )
        if committed_rows == 0 or not rows:
            return ()
        row_indices = [item[0] for item in rows]
        chunks = [item[1] for item in rows]
        with self.embeddings.store.vectors.map_readonly(
            sid, committed_rows, space.dimensions
        ) as matrix:
            try:
                if len(row_indices) == committed_rows and row_indices == list(
                    range(committed_rows)
                ):
                    neighbors = exact_nearest_neighbors(query_vec, matrix, top_k, metric=metric)
                else:
                    # Score filtered subsets in bounded blocks to avoid full-matrix copies.
                    descending = metric in ("cosine", "dot")
                    best: list[tuple[int, float]] = []
                    for start in range(0, len(row_indices), DEFAULT_BLOCK_SIZE):
                        end = min(start + DEFAULT_BLOCK_SIZE, len(row_indices))
                        block_indices = row_indices[start:end]
                        block_matrix = matrix[block_indices]
                        block_top = exact_nearest_neighbors(
                            query_vec, block_matrix, top_k, metric=metric
                        )
                        for local_offset, score in block_top:
                            best.append((start + local_offset, score))
                    if descending:
                        best.sort(key=lambda item: (-item[1], item[0]))
                    else:
                        best.sort(key=lambda item: (item[1], item[0]))
                    neighbors = best[:top_k]
            except ValueError as exc:
                raise ProviderError("invalid_response") from exc
        results: list[RetrievalResult] = []
        for rank, (local_idx, score) in enumerate(neighbors, start=1):
            stored = chunks[local_idx]
            chunk = Chunk(
                chunk_id=stored.chunk_id,
                document_id=stored.document_id,
                start=stored.start,
                end=stored.end,
                text=stored.text,
            )
            results.append(RetrievalResult(chunk=chunk, rank=rank, score=score))
        return tuple(results)

    async def search(self, request: DenseRetrieveRequest) -> DenseRetrieveResponse:
        embedded, space = await self.embeddings.embed_with_space(
            EmbeddingRequest(texts=(request.question,), role="query")
        )
        query_vec = np.asarray(embedded.embeddings[0], dtype=np.float64)
        top_k = request.top_k if request.top_k is not None else self.settings.retrieval.dense_top_k
        results = await asyncio.to_thread(
            self._score_rows,
            space,
            query_vec,
            top_k,
            request.metric,
            document_id=request.document_id,
            version=request.version,
            latest_only=request.latest_only,
            file_type=request.file_type,
            source=request.source,
        )
        return DenseRetrieveResponse(
            query_id=request.query_id or uuid4().hex,
            model=space.model,
            digest=space.digest,
            metric=request.metric,
            results=results,
        )

    async def retrieve(self, request: QueryRequest) -> tuple[RetrievalResult, ...]:
        response = await self.search(
            DenseRetrieveRequest(
                query_id=request.query_id,
                question=request.question,
                top_k=self.settings.retrieval.dense_top_k,
            )
        )
        return response.results
