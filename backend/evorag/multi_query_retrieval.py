"""Multi-query retrieval and weighted Reciprocal Rank Fusion (RRF) aggregation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from .config import MultiQuerySettings, Settings
from .context_engine import _is_redundant_overlap
from .errors import IngestionError
from .filtering import MetadataFilter
from .hybrid_retrieval import (
    HybridRetriever,
    RetrievalExplanation,
    RetrievalMode,
    RetrieveRequest,
    RetrieveResponse,
)
from .query_analysis import normalize_query_text
from .query_types import (
    EvidenceCandidate,
    FusedMultiQueryCandidate,
    MultiQueryResult,
    QueryVariant,
    RetrievalCacheEntry,
    RetrievalPlan,
)
from .retrieval_cache import RetrievalCache, build_cache_key, build_config_fingerprint


def _normalize_variant_weights(
    variants: Sequence[QueryVariant],
    mq_settings: MultiQuerySettings,
) -> dict[str, float]:
    """Scale non-original variant weights if their sum exceeds the configured cap."""
    weights: dict[str, float] = {v.variant_id: float(v.weight) for v in variants}
    if not mq_settings.normalize_variant_weights:
        return weights
    orig_weight = max(
        (v.weight for v in variants if v.kind == "original"),
        default=mq_settings.original_weight,
    )
    non_orig = [v for v in variants if v.kind != "original"]
    total_non_orig = sum(v.weight for v in non_orig)
    cap = mq_settings.variant_weight_cap * orig_weight
    if total_non_orig > cap and total_non_orig > 0.0:
        scale = cap / total_non_orig
        for v in non_orig:
            weights[v.variant_id] = v.weight * scale
    return weights


def multi_query_rrf(
    variant_responses: Sequence[tuple[QueryVariant, RetrieveResponse]],
    *,
    mq_settings: MultiQuerySettings,
    max_candidates: int,
) -> tuple[tuple[EvidenceCandidate, ...], tuple[FusedMultiQueryCandidate, ...]]:
    """Aggregate ranked results across query variants using weighted RRF.

    Ties are broken deterministically by ascending chunk_id.
    """
    if not variant_responses:
        return (), ()

    k = mq_settings.rrf_k
    weights = _normalize_variant_weights([v for v, _ in variant_responses], mq_settings)

    best_candidate: dict[str, EvidenceCandidate] = {}
    agg_scores: dict[str, float] = {}
    orig_ranks: dict[str, int] = {}
    var_ranks: dict[str, dict[str, int]] = {}
    per_query_scores: dict[str, dict[str, float]] = {}
    dense_scores: dict[str, float] = {}
    sparse_scores: dict[str, float] = {}
    contributors: dict[str, list[str]] = {}

    for variant, resp in variant_responses:
        w = weights.get(variant.variant_id, variant.weight)
        exp_by_chunk: dict[str, RetrievalExplanation] = {
            exp.chunk_id: exp for exp in (resp.explanations or ())
        }
        seen_in_variant: set[str] = set()
        for res in resp.results:
            cid = res.chunk.chunk_id
            if cid in seen_in_variant:
                continue
            seen_in_variant.add(cid)
            exp = exp_by_chunk.get(cid)
            if cid not in best_candidate:
                best_candidate[cid] = EvidenceCandidate(
                    result=res,
                    version=exp.version if exp else None,
                    source_id=None,
                    sources=exp.sources if exp else (),
                    retrieval_mode=resp.effective_mode,
                )
            fusion_contrib = w / (k + res.rank)
            agg_scores[cid] = agg_scores.get(cid, 0.0) + fusion_contrib
            var_ranks.setdefault(cid, {})[variant.variant_id] = res.rank
            per_query_scores.setdefault(cid, {})[variant.variant_id] = fusion_contrib
            contributors.setdefault(cid, []).append(variant.variant_id)
            if variant.kind == "original" and cid not in orig_ranks:
                orig_ranks[cid] = res.rank
            if exp is not None:
                if exp.dense_score is not None and cid not in dense_scores:
                    dense_scores[cid] = exp.dense_score
                if exp.bm25_score is not None and cid not in sparse_scores:
                    sparse_scores[cid] = exp.bm25_score

    ranked_ids = sorted(agg_scores.keys(), key=lambda cid: (-agg_scores[cid], cid))[:max_candidates]

    candidates: list[EvidenceCandidate] = []
    fused_details: list[FusedMultiQueryCandidate] = []
    for final_rank, cid in enumerate(ranked_ids, start=1):
        cand = best_candidate[cid]
        score = agg_scores[cid]
        updated_result = cand.result.model_copy(update={"rank": final_rank, "score": score})
        candidates.append(cand.model_copy(update={"result": updated_result}))
        fused_details.append(
            FusedMultiQueryCandidate(
                chunk_id=cid,
                document_id=cand.result.chunk.document_id,
                aggregated_score=score,
                final_rank=final_rank,
                original_rank=orig_ranks.get(cid),
                variant_ranks=var_ranks[cid],
                per_query_fusion_scores=per_query_scores[cid],
                dense_score=dense_scores.get(cid),
                sparse_score=sparse_scores.get(cid),
                contributing_query_ids=tuple(contributors[cid]),
            )
        )
    return tuple(candidates), tuple(fused_details)


class MultiQueryRetriever:
    """Executes bounded multi-query retrieval with caching, consistency checks and RRF."""

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        cache: RetrievalCache | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.cache = cache
        self._config_fingerprint = build_config_fingerprint(settings)

    def _corpus_revision_state(self) -> tuple[int, str]:
        """Read current sparse index generation and corpus document count/revision token."""
        status = self.retriever.sparse.index.status()
        with self.retriever.sparse.index.repository._connect() as db:
            row = db.execute(
                "SELECT COUNT(*), COALESCE(MAX(version_key), '') FROM versions"
            ).fetchone()
        ver_count = int(row[0]) if row else 0
        latest_key = str(row[1]) if row else ""
        return status.index_generation, f"{status.doc_count}:{ver_count}:{latest_key}"

    def _embedding_space_id(self, mode: RetrievalMode) -> str | None:
        if mode == "sparse":
            return None
        return f"{self.settings.runtime.embedding_model}:{self.settings.embeddings.dimensions}"

    async def retrieve(
        self,
        plan: RetrievalPlan,
        *,
        filters: MetadataFilter | None = None,
        reranking_enabled: bool | None = None,
    ) -> MultiQueryResult:
        """Execute planned query variants within budget and deadline, then fuse results."""
        effective_filters = filters or MetadataFilter()
        rerank = (
            reranking_enabled if reranking_enabled is not None else self.settings.reranking.enabled
        )
        per_query_top_k = max(
            plan.budget.max_final_evidence_items,
            min(self.settings.retrieval.final_top_k * 2, 20),
        )
        index_gen, rev_token = await asyncio.to_thread(self._corpus_revision_state)
        emb_space_id = self._embedding_space_id(plan.retrieval_mode)

        variant_responses: list[tuple[QueryVariant, RetrieveResponse]] = []
        subquery_counts: dict[str, int] = {}
        cache_hits = 0
        cache_misses = 0
        executions = 0
        deadline = time.monotonic() + plan.budget.request_deadline_seconds
        seen_normalized: dict[str, RetrieveResponse] = {}

        for variant in plan.variants:
            if executions >= plan.budget.max_retrieval_executions:
                break
            if time.monotonic() >= deadline and variant_responses:
                break

            norm_text = normalize_query_text(variant.text)
            if norm_text in seen_normalized:
                resp = seen_normalized[norm_text]
                variant_responses.append((variant, resp))
                if variant.source_subquery_id:
                    subquery_counts[variant.source_subquery_id] = len(resp.results)
                continue

            cache_key = build_cache_key(
                normalized_query=norm_text,
                retrieval_mode=plan.retrieval_mode,
                config_fingerprint=self._config_fingerprint,
                analyzer_version=self.retriever.analyzer.version,
                index_generation=index_gen,
                embedding_space_id=emb_space_id,
                filters=effective_filters,
                document_revision_token=rev_token,
                reranking_enabled=rerank,
                dense_weight_override=plan.dense_weight_override,
                sparse_weight_override=plan.sparse_weight_override,
            )
            cached = self.cache.get(cache_key) if self.cache is not None else None
            if cached is not None:
                cache_hits += 1
                resp = cached
            else:
                if self.cache is not None:
                    cache_misses += 1
                resp = await self.retriever.search(
                    RetrieveRequest(
                        query_id=f"{plan.plan_id}:{variant.variant_id}",
                        question=variant.text,
                        mode=plan.retrieval_mode,
                        top_k=per_query_top_k,
                        filters=effective_filters,
                        reranking_enabled=rerank,
                        include_explanations=True,
                        dense_weight_override=plan.dense_weight_override,
                        sparse_weight_override=plan.sparse_weight_override,
                    )
                )
                executions += 1
                if self.cache is not None and not resp.degraded:
                    self.cache.put(
                        RetrievalCacheEntry(
                            cache_key=cache_key,
                            normalized_query=norm_text,
                            retrieval_mode=plan.retrieval_mode,
                            config_fingerprint=self._config_fingerprint,
                            analyzer_version=self.retriever.analyzer.version,
                            index_generation=index_gen,
                            embedding_space_id=emb_space_id,
                            document_revision_token=rev_token,
                            reranking_enabled=rerank,
                            created_at_monotonic=time.monotonic(),
                            ttl_seconds=self.settings.retrieval_cache.ttl_seconds,
                        ),
                        resp,
                    )

            seen_normalized[norm_text] = resp
            variant_responses.append((variant, resp))
            if variant.source_subquery_id:
                subquery_counts[variant.source_subquery_id] = len(resp.results)

        # Verify embedding-space consistency across all non-degraded dense/hybrid responses.
        spaces = {
            (r.model, r.digest)
            for _, r in variant_responses
            if not r.degraded and r.effective_mode in ("dense", "hybrid") and r.model is not None
        }
        if len(spaces) > 1:
            raise IngestionError("embedding_space_mismatch", 500)

        degraded = any(r.degraded for _, r in variant_responses)
        degradation_reasons = [
            r.degradation_reason for _, r in variant_responses if r.degradation_reason
        ]
        eff_mode: RetrievalMode = (
            "sparse"
            if degraded and all(r.effective_mode == "sparse" for _, r in variant_responses)
            else plan.retrieval_mode
        )
        model = next((r.model for _, r in variant_responses if r.model is not None), None)
        digest = next((r.digest for _, r in variant_responses if r.digest is not None), None)

        candidates, fused_details = multi_query_rrf(
            variant_responses,
            mq_settings=self.settings.multi_query,
            max_candidates=plan.budget.max_aggregate_candidates,
        )

        # Avoid discarding adjacent chunks; _is_redundant_overlap is applied in consolidation.
        _ = _is_redundant_overlap

        return MultiQueryResult(
            plan_id=plan.plan_id,
            effective_mode=eff_mode,
            degraded=degraded,
            degradation_reason=degradation_reasons[0] if degradation_reasons else None,
            retrieval_executions=executions,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            candidates=candidates,
            fused_details=fused_details,
            subquery_candidate_counts=subquery_counts,
            model=model,
            digest=digest,
        )
