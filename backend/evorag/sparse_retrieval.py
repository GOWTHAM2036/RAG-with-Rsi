"""Sparse BM25 retriever implementing the Retriever protocol."""

import asyncio
from typing import Annotated
from uuid import uuid4

from pydantic import Field

from .analyzer import TextAnalyzer
from .bm25 import BM25ScoredChunk, score_bm25
from .config import Name, Settings
from .contracts import Chunk, Contract, QueryRequest, RetrievalResult
from .documents import StoredChunk
from .filtering import MetadataFilter
from .sparse_index import MAX_POSTING_CANDIDATES, SparseIndex


class SparseRetrieveRequest(Contract):
    query_id: Name | None = None
    question: Annotated[str, Field(min_length=1, max_length=16_384, pattern=r"\S")]
    top_k: Annotated[int, Field(ge=1, le=100)] | None = None
    filters: MetadataFilter = MetadataFilter()
    max_candidates: Annotated[int, Field(ge=1, le=MAX_POSTING_CANDIDATES)] = 500


class SparseHit(Contract):
    result: RetrievalResult
    stored: StoredChunk
    matched_terms: tuple[str, ...]


class SparseRetrieveResponse(Contract):
    query_id: str
    analyzer_version: str
    results: tuple[RetrievalResult, ...]


class SparseRetriever:
    """BM25 sparse retriever backed by the persistent SQLite inverted index."""

    def __init__(
        self,
        settings: Settings,
        index: SparseIndex,
        analyzer: TextAnalyzer | None = None,
    ) -> None:
        self.settings = settings
        self.index = index
        self.analyzer = analyzer or index.analyzer

    def _search_sync(
        self,
        question: str,
        top_k: int,
        filters: MetadataFilter,
        max_candidates: int,
    ) -> list[SparseHit]:
        query_tokens = self.analyzer.tokenize(question, for_query=True)
        if not query_tokens:
            return []

        corpus_size, avg_dl, doc_freqs, candidates = self.index.query_candidates(
            query_tokens,
            filters,
            max_candidates=max_candidates,
            k1=self.settings.sparse.k1,
            b=self.settings.sparse.b,
        )
        if not candidates or corpus_size <= 0 or avg_dl <= 0.0:
            return []

        chunk_map: dict[str, StoredChunk] = {stored.chunk_id: stored for _, stored in candidates}
        bm25_inputs = [cand for cand, _ in candidates]
        scored: list[BM25ScoredChunk] = score_bm25(
            query_tokens,
            bm25_inputs,
            corpus_size=corpus_size,
            avg_doc_length=avg_dl,
            document_frequencies=doc_freqs,
            k1=self.settings.sparse.k1,
            b=self.settings.sparse.b,
            top_k=top_k,
        )

        hits: list[SparseHit] = []
        for rank, item in enumerate(scored, start=1):
            stored = chunk_map[item.chunk_id]
            chunk = Chunk(
                chunk_id=stored.chunk_id,
                document_id=stored.document_id,
                start=stored.start,
                end=stored.end,
                text=stored.text,
            )
            hits.append(
                SparseHit(
                    result=RetrievalResult(chunk=chunk, rank=rank, score=item.score),
                    stored=stored,
                    matched_terms=item.matched_terms,
                )
            )
        return hits

    async def search_with_details(self, request: SparseRetrieveRequest) -> list[SparseHit]:
        top_k = request.top_k if request.top_k is not None else self.settings.retrieval.sparse_top_k
        return await asyncio.to_thread(
            self._search_sync,
            request.question,
            top_k,
            request.filters,
            request.max_candidates,
        )

    async def search(self, request: SparseRetrieveRequest) -> SparseRetrieveResponse:
        hits = await self.search_with_details(request)
        return SparseRetrieveResponse(
            query_id=request.query_id or uuid4().hex,
            analyzer_version=self.analyzer.version,
            results=tuple(hit.result for hit in hits),
        )

    async def retrieve(self, request: QueryRequest) -> tuple[RetrievalResult, ...]:
        response = await self.search(
            SparseRetrieveRequest(
                query_id=request.query_id,
                question=request.question,
                top_k=self.settings.retrieval.sparse_top_k,
            )
        )
        return response.results
