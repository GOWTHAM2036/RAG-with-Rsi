"""Deterministic adaptive retrieval strategy planner with bounded resource budgets."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

from .analyzer import TextAnalyzer
from .config import Settings
from .coordinator import Coordinator
from .hybrid_retrieval import RetrievalMode
from .query_analysis import analyze_query, classify_query
from .query_decomposition import decompose_query
from .query_rewriting import rewrite_query
from .query_types import (
    QueryDecomposition,
    QueryRewriteResult,
    QueryVariant,
    RetrievalBudget,
    RetrievalPlan,
    RetrievalStrategy,
    StrategyMode,
)
from .rag_types import ConversationMessage, RAGRequest

PLANNING_VERSION = "0.6.0"


def _build_budget(settings: Settings, request: RAGRequest) -> RetrievalBudget:
    adv = settings.adaptive_rag
    top_k = request.top_k or settings.rag.max_evidence_items
    return RetrievalBudget(
        max_query_variants=adv.max_query_variants,
        max_subqueries=min(adv.max_subqueries, settings.query_decomposition.max_subqueries),
        max_retrieval_executions=adv.max_retrieval_executions,
        max_aggregate_candidates=adv.max_aggregate_candidates,
        max_final_evidence_items=min(top_k, settings.rag.max_evidence_items),
        max_planning_llm_calls=adv.max_planning_llm_calls,
        max_decomposition_depth=1,
        max_concurrent_generations=settings.resources.max_parallel_generations,
        request_deadline_seconds=adv.request_deadline_seconds,
    )


class AdaptiveQueryPlanner:
    """Selects bounded retrieval strategies and builds traceable RetrievalPlan contracts."""

    def __init__(self, settings: Settings, analyzer: TextAnalyzer | None = None) -> None:
        self.settings = settings
        self.analyzer = analyzer or TextAnalyzer.from_config(settings.sparse)

    async def plan(
        self,
        request: RAGRequest,
        *,
        history: Sequence[ConversationMessage] = (),
        coordinator: Coordinator | None = None,
        strategy_override: StrategyMode | None = None,
        rewriting_enabled: bool | None = None,
        decomposition_enabled: bool | None = None,
    ) -> RetrievalPlan:
        plan_id = uuid4().hex
        budget = _build_budget(self.settings, request)
        retrieval_mode: RetrievalMode = (
            request.retrieval_mode or self.settings.rag.default_retrieval_mode
        )
        requested_mode: StrategyMode = (
            strategy_override or request.strategy or self.settings.adaptive_rag.default_strategy
        )
        if not self.settings.adaptive_rag.enabled:
            requested_mode = "baseline"

        analysis = analyze_query(request.question, self.analyzer)
        classification = classify_query(analysis, has_conversation_history=bool(history))

        no_rewrite = QueryRewriteResult(
            original_query=request.question,
            rewritten_query=analysis.normalized_query or request.question,
            applied=False,
            method="none",
            preserved_identifiers=analysis.technical_identifiers,
        )
        no_decomp = QueryDecomposition(
            original_query=request.question,
            applied=False,
            method="none",
            depth=0,
        )
        original_variant = QueryVariant(
            variant_id="q_orig",
            kind="original",
            text=analysis.normalized_query or request.question,
            weight=self.settings.multi_query.original_weight,
        )

        if requested_mode == "baseline" or not analysis.is_valid:
            return RetrievalPlan(
                plan_id=plan_id,
                planning_version=PLANNING_VERSION,
                strategy_mode="baseline" if requested_mode == "baseline" else "adaptive",
                selected_strategy="baseline",
                retrieval_mode=retrieval_mode,
                analysis=analysis,
                classification=classification,
                rewrite=no_rewrite,
                decomposition=no_decomp,
                variants=(original_variant,),
                budget=budget,
                rationale=("Baseline single-query retrieval selected.",),
            )

        rw_enabled = (
            rewriting_enabled
            if rewriting_enabled is not None
            else self.settings.query_rewriting.enabled
        )
        dc_enabled = (
            decomposition_enabled
            if decomposition_enabled is not None
            else self.settings.query_decomposition.enabled
        )

        primary = classification.primary_intent
        rationale: list[str] = []
        fallback_reasons: list[str] = []
        llm_calls_used = 0
        dense_weight_override: float | None = None
        sparse_weight_override: float | None = None
        rewrite_res = no_rewrite
        decomp_res = no_decomp
        variants: list[QueryVariant] = []

        if primary in ("comparative", "multi_part"):
            selected_strategy: RetrievalStrategy = "decomposed_multi_part"
            rationale.append(f"Intent '{primary}' routed to bounded decomposition.")
            if dc_enabled:
                decomp_res, used = await decompose_query(
                    analysis,
                    self.settings,
                    self.settings.query_decomposition,
                    subquery_weight=self.settings.multi_query.subquery_weight,
                    coordinator=coordinator,
                    llm_budget_remaining=max(0, budget.max_planning_llm_calls - llm_calls_used),
                )
                llm_calls_used += used
                if decomp_res.fallback_reason:
                    fallback_reasons.append(decomp_res.fallback_reason)
            else:
                fallback_reasons.append("decomposition_disabled")

            if self.settings.query_rewriting.preserve_original:
                variants.append(original_variant)
            for sq in decomp_res.subqueries:
                variants.append(
                    QueryVariant(
                        variant_id=sq.subquery_id,
                        kind="subquery",
                        text=sq.text,
                        weight=sq.weight,
                        source_subquery_id=sq.subquery_id,
                    )
                )

        elif primary == "conversational":
            selected_strategy = "conversational_resolve"
            rationale.append("Intent 'conversational' routed to reference-resolving rewrite.")
            if rw_enabled:
                rewrite_res, used = await rewrite_query(
                    analysis,
                    self.settings,
                    self.settings.query_rewriting,
                    history=history,
                    coordinator=coordinator,
                    llm_budget_remaining=max(0, budget.max_planning_llm_calls - llm_calls_used),
                )
                llm_calls_used += used
                if rewrite_res.fallback_reason:
                    fallback_reasons.append(rewrite_res.fallback_reason)
            else:
                fallback_reasons.append("rewriting_disabled")

            if self.settings.query_rewriting.preserve_original:
                variants.append(original_variant)
            if rewrite_res.applied:
                variants.append(
                    QueryVariant(
                        variant_id="q_rw_1",
                        kind="rewrite",
                        text=rewrite_res.rewritten_query,
                        weight=self.settings.multi_query.rewrite_weight,
                    )
                )

        elif primary == "technical_lookup":
            selected_strategy = "technical_exact"
            rationale.append(
                "Intent 'technical_lookup' preserves exact identifiers and avoids rewriting."
            )
            if retrieval_mode == "hybrid":
                sparse_weight_override = max(self.settings.fusion.sparse_weight, 1.5)
                dense_weight_override = self.settings.fusion.dense_weight
            variants.append(original_variant)

        elif primary == "exploratory":
            selected_strategy = "exploratory_expanded"
            rationale.append("Intent 'exploratory' routed to bounded variant expansion.")
            variants.append(original_variant)
            if rw_enabled:
                rewrite_res, used = await rewrite_query(
                    analysis,
                    self.settings,
                    self.settings.query_rewriting,
                    history=history,
                    coordinator=coordinator,
                    llm_budget_remaining=max(0, budget.max_planning_llm_calls - llm_calls_used),
                )
                llm_calls_used += used
                if rewrite_res.fallback_reason:
                    fallback_reasons.append(rewrite_res.fallback_reason)
                if rewrite_res.applied:
                    variants.append(
                        QueryVariant(
                            variant_id="q_rw_1",
                            kind="rewrite",
                            text=rewrite_res.rewritten_query,
                            weight=self.settings.multi_query.rewrite_weight,
                        )
                    )

        else:
            selected_strategy = "simple_factual"
            rationale.append("Intent 'factual' routed to single-query hybrid retrieval.")
            variants.append(original_variant)

        # Deduplicate variants and enforce execution/variant caps.
        cap = min(budget.max_query_variants, budget.max_retrieval_executions)
        if not self.settings.multi_query.enabled:
            cap = 1
        deduped: list[QueryVariant] = []
        seen_texts: set[str] = set()
        for var in variants:
            key = var.text.casefold()
            if key in seen_texts:
                continue
            seen_texts.add(key)
            deduped.append(var)
            if len(deduped) >= cap:
                break
        if not deduped:
            deduped.append(original_variant)

        return RetrievalPlan(
            plan_id=plan_id,
            planning_version=PLANNING_VERSION,
            strategy_mode="adaptive",
            selected_strategy=selected_strategy,
            retrieval_mode=retrieval_mode,
            dense_weight_override=dense_weight_override,
            sparse_weight_override=sparse_weight_override,
            analysis=analysis,
            classification=classification,
            rewrite=rewrite_res,
            decomposition=decomp_res,
            variants=tuple(deduped),
            budget=budget,
            rationale=tuple(rationale),
            llm_planning_calls_used=llm_calls_used,
            additional_planning_tokens=rewrite_res.tokens_used + decomp_res.tokens_used,
            fallback_reasons=tuple(fallback_reasons),
        )
