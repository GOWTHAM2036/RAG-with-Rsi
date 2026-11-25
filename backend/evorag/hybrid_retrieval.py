"""Hybrid retrieval coordinating dense, sparse, RRF fusion, reranking and degradation."""

import asyncio
from collections.abc import Sequence
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import Field, FiniteFloat, PositiveInt, model_validator

from .analyzer import TextAnalyzer
from .config import Name, Settings
from .contracts import Contract, QueryRequest, RetrievalResult
from .documents import SourceSpan
from .filtering import MetadataFilter
from .fusion import FusedCandidate, RankedCandidate, reciprocal_rank_fusion
from .inference_types import ProviderError
from .reranking import rerank_candidates
from .retrieval import DenseRetriever, DenseRetrieveRequest, DenseRetrieveResponse
from .similarity import Metric
from .sparse_retrieval import SparseHit, SparseRetriever, SparseRetrieveRequest

RetrievalMode = Literal["dense", "sparse", "hybrid"]
QueryText = Annotated[str, Field(min_length=1, max_length=16_384, pattern=r"\S")]

DEGRADABLE_DENSE_ERRORS: frozenset[str] = frozenset(
    {
        "provider_unavailable",
        "connect_timeout",
        "read_timeout",
        "model_not_installed",
        "resource_exhaustion",
        "inference_busy",
        "inference_queue_full",
        "inference_queue_timeout",
        "insufficient_memory",
    }
)


class RetrieveRequest(Contract):
    query_id: Name | None = None
    query: QueryText | None = None
    question: QueryText | None = None
    mode: RetrievalMode | None = None
    top_k: Annotated[int, Field(ge=1, le=100)] | None = None
    metric: Metric = "cosine"
    document_id: Name | None = None
    version: Annotated[int, Field(ge=1, le=2_147_483_647)] | None = None
    latest_only: bool = True
    filters: MetadataFilter | None = None
    reranking_enabled: bool | None = None
    include_explanations: bool = False

    @model_validator(mode="after")
    def check_query_text(self) -> Self:
        if self.query is None and self.question is None:
            raise ValueError("either query or question must be provided")
        if self.query is not None and self.question is not None and self.query != self.question:
            raise ValueError("query and question must match when both are provided")
        return self

    def resolved_query(self) -> str:
        text = self.query if self.query is not None else self.question
        assert text is not None
        return text

    def resolved_filters(self) -> MetadataFilter:
        if self.filters is not None:
            return self.filters
        return MetadataFilter(
            document_id=self.document_id,
            version=self.version,
            latest_only=self.latest_only,
        )


class RetrievalExplanation(Contract):
    chunk_id: str
    document_id: str
    version: int | None = None
    sources: tuple[SourceSpan, ...] = ()
    dense_score: FiniteFloat | None = None
    dense_rank: PositiveInt | None = None
    bm25_score: FiniteFloat | None = None
    sparse_rank: PositiveInt | None = None
    rrf_score: FiniteFloat | None = None
    rerank_score: FiniteFloat | None = None
    final_rank: PositiveInt
    matched_terms: tuple[str, ...] = ()
    retrieval_mode: RetrievalMode
    effective_mode: RetrievalMode
    degraded: bool = False
    degradation_reason: str | None = None


class RetrieveResponse(Contract):
    query_id: str
    mode: RetrievalMode
    effective_mode: RetrievalMode
    degraded: bool = False
    degradation_reason: str | None = None
    reranking_applied: bool = False
    model: str | None = None
    digest: str | None = None
    metric: Metric | None = None
    results: tuple[RetrievalResult, ...]
    explanations: tuple[RetrievalExplanation, ...] | None = None


class HybridRetriever:
    """Coordinates dense and sparse retrieval, RRF fusion, reranking and fallback."""

    def __init__(
        self,
        settings: Settings,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        analyzer: TextAnalyzer | None = None,
    ) -> None:
        self.settings = settings
        self.dense = dense
        self.sparse = sparse
        self.analyzer = analyzer or sparse.analyzer

    def _chunk_metadata(
        self, chunk_ids: Sequence[str]
    ) -> dict[str, tuple[int, tuple[SourceSpan, ...]]]:
        if not chunk_ids:
            return {}
        from .documents import StoredChunk

        with self.sparse.index.repository._connect() as db:
            ph = ",".join("?" for _ in chunk_ids)
            rows = db.execute(
                f"SELECT c.chunk_id, v.version, c.payload "
                f"FROM chunks c JOIN versions v ON c.version_key = v.version_key "
                f"WHERE c.chunk_id IN ({ph})",
                list(chunk_ids),
            ).fetchall()
        out: dict[str, tuple[int, tuple[SourceSpan, ...]]] = {}
        for row in rows:
            stored = StoredChunk.model_validate_json(row["payload"])
            out[str(row["chunk_id"])] = (int(row["version"]), stored.sources)
        return out

    async def search(self, request: RetrieveRequest) -> RetrieveResponse:
        query_text = request.resolved_query()
        query_id = request.query_id or uuid4().hex
        filters = request.resolved_filters()

        if request.mode is not None:
            requested_mode: RetrievalMode = request.mode
        elif request.query is not None:
            requested_mode = self.settings.retrieval.mode
        else:
            # Preserve Phase 03 default behavior for legacy {"question": ...} callers.
            requested_mode = "dense"

        rerank_enabled = (
            request.reranking_enabled
            if request.reranking_enabled is not None
            else self.settings.reranking.enabled
        )

        if requested_mode == "dense":
            top_k = (
                request.top_k if request.top_k is not None else self.settings.retrieval.dense_top_k
            )
            dense_resp = await self.dense.search(
                DenseRetrieveRequest(
                    query_id=query_id,
                    question=query_text,
                    top_k=top_k,
                    metric=request.metric,
                    document_id=filters.document_id,
                    version=filters.version,
                    latest_only=filters.latest_only,
                    file_type=filters.file_type,
                    source=filters.source,
                )
            )
            explanations = None
            if request.include_explanations:
                meta = await asyncio.to_thread(
                    self._chunk_metadata, [r.chunk.chunk_id for r in dense_resp.results]
                )
                explanations = tuple(
                    RetrievalExplanation(
                        chunk_id=r.chunk.chunk_id,
                        document_id=r.chunk.document_id,
                        version=meta.get(r.chunk.chunk_id, (None, ()))[0],
                        sources=meta.get(r.chunk.chunk_id, (None, ()))[1],
                        dense_score=r.score,
                        dense_rank=r.rank,
                        final_rank=r.rank,
                        retrieval_mode="dense",
                        effective_mode="dense",
                    )
                    for r in dense_resp.results
                )
            return RetrieveResponse(
                query_id=query_id,
                mode="dense",
                effective_mode="dense",
                model=dense_resp.model,
                digest=dense_resp.digest,
                metric=dense_resp.metric,
                results=dense_resp.results,
                explanations=explanations,
            )

        if requested_mode == "sparse":
            top_k = (
                request.top_k if request.top_k is not None else self.settings.retrieval.sparse_top_k
            )
            hits = await self.sparse.search_with_details(
                SparseRetrieveRequest(
                    query_id=query_id,
                    question=query_text,
                    top_k=top_k,
                    filters=filters,
                )
            )
            meta = (
                await asyncio.to_thread(
                    self._chunk_metadata, [h.result.chunk.chunk_id for h in hits]
                )
                if request.include_explanations
                else {}
            )
            return self._build_sparse_response(
                query_id=query_id,
                hits=hits,
                requested_mode="sparse",
                effective_mode="sparse",
                degraded=False,
                degradation_reason=None,
                include_explanations=request.include_explanations,
                chunk_meta=meta,
            )

        # Hybrid mode
        final_top_k = (
            request.top_k if request.top_k is not None else self.settings.retrieval.final_top_k
        )
        dense_k = min(100, max(self.settings.retrieval.dense_top_k, final_top_k))
        sparse_k = min(100, max(self.settings.retrieval.sparse_top_k, final_top_k))
        dense_resp_opt: DenseRetrieveResponse | None = None
        degraded = False
        degradation_reason: str | None = None

        try:
            dense_resp_opt = await self.dense.search(
                DenseRetrieveRequest(
                    query_id=query_id,
                    question=query_text,
                    top_k=dense_k,
                    metric=request.metric,
                    document_id=filters.document_id,
                    version=filters.version,
                    latest_only=filters.latest_only,
                    file_type=filters.file_type,
                    source=filters.source,
                )
            )
        except ProviderError as exc:
            if (
                self.settings.retrieval.allow_sparse_fallback
                and exc.code in DEGRADABLE_DENSE_ERRORS
            ):
                degraded = True
                degradation_reason = exc.code
            else:
                raise

        sparse_hits = await self.sparse.search_with_details(
            SparseRetrieveRequest(
                query_id=query_id,
                question=query_text,
                top_k=sparse_k,
                filters=filters,
            )
        )

        if degraded or dense_resp_opt is None:
            limited_hits = sparse_hits[:final_top_k]
            meta = (
                await asyncio.to_thread(
                    self._chunk_metadata, [h.result.chunk.chunk_id for h in limited_hits]
                )
                if request.include_explanations
                else {}
            )
            return self._build_sparse_response(
                query_id=query_id,
                hits=limited_hits,
                requested_mode="hybrid",
                effective_mode="sparse",
                degraded=True,
                degradation_reason=degradation_reason,
                include_explanations=request.include_explanations,
                chunk_meta=meta,
            )

        dense_ranked = [
            RankedCandidate(chunk=r.chunk, rank=r.rank, score=r.score)
            for r in dense_resp_opt.results
        ]
        sparse_ranked = [
            RankedCandidate(chunk=h.result.chunk, rank=h.result.rank, score=h.result.score)
            for h in sparse_hits
        ]
        terms_map = {h.result.chunk.chunk_id: h.matched_terms for h in sparse_hits}

        fused: list[FusedCandidate] = reciprocal_rank_fusion(
            dense_ranked,
            sparse_ranked,
            k=self.settings.fusion.k,
            dense_weight=self.settings.fusion.dense_weight,
            sparse_weight=self.settings.fusion.sparse_weight,
            sparse_matched_terms=terms_map,
            top_k=self.settings.reranking.max_candidates if rerank_enabled else final_top_k,
        )

        results_list: list[RetrievalResult] = []
        rerank_scores: dict[str, float] = {}
        if rerank_enabled and fused:
            reranked = rerank_candidates(
                query_text,
                fused,
                self.analyzer,
                self.settings.reranking,
                top_k=final_top_k,
            )
            for item in reranked:
                rerank_scores[item.fused.chunk.chunk_id] = item.rerank_score
                results_list.append(
                    RetrievalResult(
                        chunk=item.fused.chunk,
                        rank=item.final_rank,
                        score=item.rerank_score,
                    )
                )
            ordered_fused = [item.fused for item in reranked]
        else:
            limited_fused = fused[:final_top_k]
            for fused_item in limited_fused:
                results_list.append(
                    RetrievalResult(
                        chunk=fused_item.chunk,
                        rank=fused_item.fused_rank,
                        score=fused_item.rrf_score,
                    )
                )
            ordered_fused = limited_fused

        explanations = None
        if request.include_explanations:
            meta = await asyncio.to_thread(
                self._chunk_metadata, [item.chunk.chunk_id for item in ordered_fused]
            )
            explanations = tuple(
                RetrievalExplanation(
                    chunk_id=item.chunk.chunk_id,
                    document_id=item.chunk.document_id,
                    version=meta.get(item.chunk.chunk_id, (None, ()))[0],
                    sources=meta.get(item.chunk.chunk_id, (None, ()))[1],
                    dense_score=item.dense_score,
                    dense_rank=item.dense_rank,
                    bm25_score=item.sparse_score,
                    sparse_rank=item.sparse_rank,
                    rrf_score=item.rrf_score,
                    rerank_score=rerank_scores.get(item.chunk.chunk_id),
                    final_rank=rank,
                    matched_terms=item.matched_terms,
                    retrieval_mode="hybrid",
                    effective_mode="hybrid",
                )
                for rank, item in enumerate(ordered_fused, start=1)
            )

        return RetrieveResponse(
            query_id=query_id,
            mode="hybrid",
            effective_mode="hybrid",
            reranking_applied=rerank_enabled,
            model=dense_resp_opt.model,
            digest=dense_resp_opt.digest,
            metric=dense_resp_opt.metric,
            results=tuple(results_list),
            explanations=explanations,
        )

    def _build_sparse_response(
        self,
        *,
        query_id: str,
        hits: list[SparseHit],
        requested_mode: RetrievalMode,
        effective_mode: RetrievalMode,
        degraded: bool,
        degradation_reason: str | None,
        include_explanations: bool,
        chunk_meta: dict[str, tuple[int, tuple[SourceSpan, ...]]] | None = None,
    ) -> RetrieveResponse:
        results = tuple(
            RetrievalResult(chunk=hit.result.chunk, rank=rank, score=hit.result.score)
            for rank, hit in enumerate(hits, start=1)
        )
        explanations = None
        if include_explanations:
            meta = chunk_meta or {}
            explanations = tuple(
                RetrievalExplanation(
                    chunk_id=hit.result.chunk.chunk_id,
                    document_id=hit.result.chunk.document_id,
                    version=meta.get(hit.result.chunk.chunk_id, (None, ()))[0],
                    sources=hit.stored.sources,
                    bm25_score=hit.result.score,
                    sparse_rank=hit.result.rank,
                    final_rank=rank,
                    matched_terms=hit.matched_terms,
                    retrieval_mode=requested_mode,
                    effective_mode=effective_mode,
                    degraded=degraded,
                    degradation_reason=degradation_reason,
                )
                for rank, hit in enumerate(hits, start=1)
            )
        return RetrieveResponse(
            query_id=query_id,
            mode=requested_mode,
            effective_mode=effective_mode,
            degraded=degraded,
            degradation_reason=degradation_reason,
            results=results,
            explanations=explanations,
        )

    async def retrieve(self, request: QueryRequest) -> tuple[RetrievalResult, ...]:
        response = await self.search(
            RetrieveRequest(
                query_id=request.query_id,
                query=request.question,
                mode=self.settings.retrieval.mode,
            )
        )
        return response.results
