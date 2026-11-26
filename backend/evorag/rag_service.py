"""End-to-end provider-independent grounded RAG orchestration service."""

import asyncio
import time
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from uuid import uuid4

from .admission import admit_answer
from .citations import OutputParseError, parse_structured_answer, validate_citations
from .config import Settings
from .context_engine import build_context_package, estimate_tokens, select_evidence
from .conversation import ConversationStore
from .coordinator import Coordinator
from .errors import EvoRagError, IngestionError
from .evidence_consolidation import consolidate_evidence
from .hybrid_retrieval import HybridRetriever, RetrievalMode, RetrieveRequest, RetrieveResponse
from .inference_types import GenerationRequest, GenerationUsage, ProviderError
from .multi_query_retrieval import MultiQueryRetriever
from .prompt_engine import get_template, render_messages
from .query_planner import AdaptiveQueryPlanner
from .query_types import (
    EvidenceCoverage,
    QueryExecutionTrace,
    RetrievalPlan,
    StrategyMode,
)
from .rag_types import (
    AnswerAdmissionResult,
    CitationValidationResult,
    EvidenceCandidate,
    PipelineStageMetrics,
    RAGExecutionTrace,
    RAGRequest,
    RAGResponse,
    RAGStatusReport,
    RAGStreamEvent,
    SelectedEvidence,
)
from .reflection_engine import ReflectionEngine
from .reflection_types import ReflectionExecutionTrace, ReflectionOutcome
from .retrieval_cache import RetrievalCache
from .tracing import TraceStore

PIPELINE_VERSION = "0.8.0"


class RAGService:
    """Coordinates baseline and adaptive hybrid retrieval, context, generation and validation."""

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        coordinator: Coordinator,
        traces: TraceStore,
        conversations: ConversationStore,
        *,
        retrieval_cache: RetrievalCache | None = None,
        planner: AdaptiveQueryPlanner | None = None,
        multi_retriever: MultiQueryRetriever | None = None,
        reflection: ReflectionEngine | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.coordinator = coordinator
        self.traces = traces
        self.conversations = conversations
        self.retrieval_cache = retrieval_cache or RetrievalCache(settings.retrieval_cache)
        self.planner = planner or AdaptiveQueryPlanner(settings, retriever.analyzer)
        self.multi_retriever = multi_retriever or MultiQueryRetriever(
            settings, retriever, self.retrieval_cache
        )
        self.reflection = reflection or ReflectionEngine(settings, retriever, coordinator)

    def status(self) -> RAGStatusReport:
        return RAGStatusReport(
            enabled=self.settings.rag.enabled,
            default_retrieval_mode=self.settings.rag.default_retrieval_mode,
            prompt_template=self.settings.prompts.default_template,
            tracing_enabled=self.settings.tracing.enabled,
            stored_traces=self.traces.count(),
            active_conversations=self.conversations.count(),
            adaptive_rag_enabled=self.settings.adaptive_rag.enabled,
            default_strategy=self.settings.adaptive_rag.default_strategy,
            retrieval_cache_stats=self.retrieval_cache.stats(),
        )

    async def plan_query(self, request: RAGRequest) -> RetrievalPlan:
        """Build a bounded retrieval plan without executing answer generation."""
        if not self.settings.rag.enabled:
            raise IngestionError("rag_disabled", 503)
        history = (
            self.conversations.get(request.conversation_id).messages
            if request.conversation_id
            else ()
        )
        return await self.planner.plan(
            request,
            history=history,
            coordinator=None,
            strategy_override=request.strategy,
            rewriting_enabled=request.query_rewriting_enabled,
            decomposition_enabled=request.query_decomposition_enabled,
        )

    def _to_candidates(self, response: RetrieveResponse) -> tuple[EvidenceCandidate, ...]:
        explanations = response.explanations or ()
        exp_by_chunk = {exp.chunk_id: exp for exp in explanations}
        candidates: list[EvidenceCandidate] = []
        for res in response.results:
            exp = exp_by_chunk.get(res.chunk.chunk_id)
            candidates.append(
                EvidenceCandidate(
                    result=res,
                    version=exp.version if exp else None,
                    source_id=None,
                    sources=exp.sources if exp else (),
                    retrieval_mode=response.effective_mode,
                )
            )
        return tuple(candidates)

    async def query(self, request: RAGRequest) -> RAGResponse:
        if not self.settings.rag.enabled:
            raise IngestionError("rag_disabled", 503)

        t0 = time.perf_counter()
        t_mono = time.monotonic()
        trace_id = uuid4().hex
        req_id = request.query_id or uuid4().hex
        mode = request.retrieval_mode or self.settings.rag.default_retrieval_mode
        rerank = (
            request.reranking_enabled
            if request.reranking_enabled is not None
            else self.settings.rag.reranking_enabled
        )
        top_k = request.top_k or self.settings.rag.max_evidence_items
        template = get_template(self.settings.prompts.default_template)
        history = (
            self.conversations.get(request.conversation_id).messages
            if request.conversation_id
            else ()
        )
        strategy_mode: StrategyMode = (
            request.strategy or self.settings.adaptive_rag.default_strategy
        )
        if not self.settings.adaptive_rag.enabled:
            strategy_mode = "baseline"

        plan: RetrievalPlan | None = None
        coverage: EvidenceCoverage | None = None
        query_execution: QueryExecutionTrace | None = None
        request_deadline: float | None = None

        if strategy_mode == "adaptive":
            p_start = time.perf_counter()
            plan = await self.planner.plan(
                request,
                history=history,
                coordinator=self.coordinator,
                strategy_override="adaptive",
                rewriting_enabled=request.query_rewriting_enabled,
                decomposition_enabled=request.query_decomposition_enabled,
            )
            planning_ms = (time.perf_counter() - p_start) * 1000.0
            if not plan.analysis.is_valid:
                raise IngestionError("invalid_query", 422)
            request_deadline = t_mono + plan.budget.request_deadline_seconds

            r_start = time.perf_counter()
            mq_res = await self.multi_retriever.retrieve(
                plan, filters=request.filters, reranking_enabled=rerank
            )
            multi_retrieval_ms = (time.perf_counter() - r_start) * 1000.0
            retrieval_ms = planning_ms + multi_retrieval_ms

            c_start = time.perf_counter()
            selected, coverage = consolidate_evidence(
                mq_res,
                plan,
                max_items=min(top_k, self.settings.rag.max_evidence_items),
                max_evidence_tokens=self.settings.context.evidence_token_budget,
            )
            consolidation_ms = (time.perf_counter() - c_start) * 1000.0
            package = build_context_package(
                request.question, selected, template.system_prompt, history, self.settings.context
            )
            context_ms = (time.perf_counter() - c_start) * 1000.0
            eff_mode: RetrievalMode = mq_res.effective_mode
            degraded = mq_res.degraded
            degradation_reason = mq_res.degradation_reason
            ret_model = mq_res.model
            ret_digest = mq_res.digest
            query_execution = QueryExecutionTrace(
                strategy_mode="adaptive",
                selected_strategy=plan.selected_strategy,
                planning_version=plan.planning_version,
                primary_intent=plan.classification.primary_intent,
                intent_labels=plan.classification.labels,
                complexity=plan.classification.complexity,
                variant_count=len(plan.variants),
                subquery_count=len(plan.decomposition.subqueries),
                retrieval_executions=mq_res.retrieval_executions,
                rewrite_applied=plan.rewrite.applied,
                rewrite_method=plan.rewrite.method,
                decomposition_applied=plan.decomposition.applied,
                decomposition_method=plan.decomposition.method,
                fallback_reasons=plan.fallback_reasons,
                cache_hits=mq_res.cache_hits,
                cache_misses=mq_res.cache_misses,
                coverage=coverage,
                planning_ms=planning_ms,
                multi_query_retrieval_ms=multi_retrieval_ms,
                consolidation_ms=consolidation_ms,
                additional_planning_tokens=plan.additional_planning_tokens,
                resource_admission_ok=True,
                resource_rejection_reason=None,
            )
        else:
            # 1. Baseline retrieval stage (HybridRetriever handles sparse fallback).
            r_start = time.perf_counter()
            ret_resp = await self.retriever.search(
                RetrieveRequest(
                    query_id=req_id,
                    question=request.question,
                    mode=mode,
                    top_k=top_k,
                    filters=request.filters,
                    reranking_enabled=rerank,
                    include_explanations=True,
                )
            )
            retrieval_ms = (time.perf_counter() - r_start) * 1000.0

            # 2. Evidence selection & context budgeting.
            c_start = time.perf_counter()
            candidates = self._to_candidates(ret_resp)
            selected = select_evidence(candidates, self.settings.rag.max_evidence_items)
            package = build_context_package(
                request.question, selected, template.system_prompt, history, self.settings.context
            )
            context_ms = (time.perf_counter() - c_start) * 1000.0
            eff_mode = ret_resp.effective_mode
            degraded = ret_resp.degraded
            degradation_reason = ret_resp.degradation_reason
            ret_model = ret_resp.model
            ret_digest = ret_resp.digest

        # 3. Abstain without invoking the LLM if no evidence remains.
        if not package.evidence:
            v_start = time.perf_counter()
            admission = admit_answer(
                None,
                None,
                has_evidence=False,
                allow_partial=self.settings.rag.allow_partial_answers,
                require_citations=self.settings.rag.require_citations,
            )
            validation_ms = (time.perf_counter() - v_start) * 1000.0
            total_ms = (time.perf_counter() - t0) * 1000.0
            metrics = PipelineStageMetrics(
                retrieval_ms=retrieval_ms,
                context_ms=context_ms,
                generation_ms=None,
                first_token_ms=None,
                validation_ms=validation_ms,
                total_ms=total_ms,
            )
            await self._persist_trace(
                trace_id=trace_id,
                req_id=req_id,
                request=request,
                template_version=template.version,
                requested_mode=mode,
                effective_mode=eff_mode,
                degraded=degraded,
                degradation_reason=degradation_reason,
                ret_model=ret_model,
                ret_digest=ret_digest,
                selected=package.evidence,
                metrics=metrics,
                est_tokens=package.budget.total_estimated_input_tokens,
                usage=None,
                validation=CitationValidationResult(valid=True),
                admission=admission,
                resource_rejection=None,
                query_execution=query_execution,
            )
            return self._build_response(
                request,
                mode,
                eff_mode,
                degraded,
                degradation_reason,
                admission,
                trace_id,
                metrics,
                strategy_mode=strategy_mode,
                plan=plan,
                coverage=coverage,
            )

        # 4. Prompt rendering & resource-aware generation.
        messages = render_messages(template, package)
        gen_req = GenerationRequest(
            messages=messages,
            temperature=self.settings.runtime.temperature,
            context_window=self.settings.runtime.context_window,
            max_output_tokens=self.settings.runtime.max_output_tokens,
        )

        g_start = time.perf_counter()
        raw_output: str | None = None
        provider_usage: GenerationUsage | None = None
        first_token_ms: float | None = None
        gen_error: str | None = None
        if request_deadline is not None and time.monotonic() >= request_deadline:
            gen_error = "request_timeout"
        else:
            try:
                gen_resp = await self.coordinator.generate(gen_req)
                raw_output = gen_resp.content
                provider_usage = gen_resp.usage
                first_token_ms = gen_resp.metrics.first_token_ms
            except ProviderError as exc:
                gen_error = exc.code
            except EvoRagError as exc:
                gen_error = str(exc) or type(exc).__name__
        generation_ms = (time.perf_counter() - g_start) * 1000.0

        # 5. Output parsing, citation validation & admission.
        v_start = time.perf_counter()
        parsed = None
        validation: CitationValidationResult | None = None
        if gen_error is None and raw_output is not None:
            try:
                parsed = parse_structured_answer(
                    raw_output,
                    max_bytes=self.settings.rag.max_output_bytes,
                    max_claims=self.settings.rag.max_claims,
                )
                validation = validate_citations(
                    parsed,
                    package.evidence,
                    validate_identifiers=self.settings.citations.validate_identifiers,
                    reject_unknown_sources=self.settings.citations.reject_unknown_sources,
                )
            except OutputParseError:
                parsed = None
        admission = admit_answer(
            parsed,
            validation,
            has_evidence=True,
            allow_partial=self.settings.rag.allow_partial_answers,
            require_citations=self.settings.rag.require_citations,
            error_code=gen_error,
        )
        reflection_summary: ReflectionOutcome | None = None
        reflection_exec: ReflectionExecutionTrace | None = None
        ref_enabled = (
            request.reflection_enabled
            if request.reflection_enabled is not None
            else self.settings.reflection.enabled
        )
        ref_mode = request.reflection_mode or self.settings.reflection.mode
        if ref_enabled and ref_mode != "off" and gen_error is None:
            baseline_calls = 1 + (
                plan.budget.max_planning_llm_calls
                if plan and plan.additional_planning_tokens
                else 0
            )
            admission, reflection_summary, reflection_exec = await self.reflection.reflect(
                request,
                mode=ref_mode,
                initial_candidates=candidates,
                initial_evidence=package.evidence,
                initial_parsed=parsed,
                initial_validation=validation,
                initial_admission=admission,
                coverage=coverage,
                baseline_model_calls=baseline_calls,
                deadline_epoch_s=request_deadline or (time.monotonic() + 30.0),
            )
        validation_ms = (time.perf_counter() - v_start) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        if (
            admission.accepted
            and request.conversation_id
            and admission.outcome != "insufficient_evidence"
        ):
            self.conversations.append_turn(
                request.conversation_id, package.question, admission.answer
            )

        metrics = PipelineStageMetrics(
            retrieval_ms=retrieval_ms,
            context_ms=context_ms,
            generation_ms=generation_ms,
            first_token_ms=first_token_ms,
            validation_ms=validation_ms,
            total_ms=total_ms,
        )
        est_prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
        if query_execution is not None and admission.outcome == "resource_unavailable":
            query_execution = query_execution.model_copy(
                update={
                    "resource_admission_ok": False,
                    "resource_rejection_reason": gen_error,
                }
            )
        await self._persist_trace(
            trace_id=trace_id,
            req_id=req_id,
            request=request,
            template_version=template.version,
            requested_mode=mode,
            effective_mode=eff_mode,
            degraded=degraded,
            degradation_reason=degradation_reason,
            ret_model=ret_model,
            ret_digest=ret_digest,
            selected=package.evidence,
            metrics=metrics,
            est_tokens=est_prompt_tokens,
            usage=provider_usage,
            validation=validation
            or CitationValidationResult(valid=False, errors=("not_validated",)),
            admission=admission,
            resource_rejection=gen_error if admission.outcome == "resource_unavailable" else None,
            query_execution=query_execution,
            reflection_execution=reflection_exec,
        )
        return self._build_response(
            request,
            mode,
            eff_mode,
            degraded,
            degradation_reason,
            admission,
            trace_id,
            metrics,
            strategy_mode=strategy_mode,
            plan=plan,
            coverage=coverage,
            reflection_summary=reflection_summary,
        )

    async def _persist_trace(
        self,
        *,
        trace_id: str,
        req_id: str,
        request: RAGRequest,
        template_version: str,
        requested_mode: RetrievalMode,
        effective_mode: RetrievalMode,
        degraded: bool,
        degradation_reason: str | None,
        ret_model: str | None,
        ret_digest: str | None,
        selected: tuple[SelectedEvidence, ...],
        metrics: PipelineStageMetrics,
        est_tokens: int,
        usage: GenerationUsage | None,
        validation: CitationValidationResult,
        admission: AnswerAdmissionResult,
        resource_rejection: str | None,
        query_execution: QueryExecutionTrace | None = None,
        reflection_execution: ReflectionExecutionTrace | None = None,
    ) -> None:
        trace = RAGExecutionTrace(
            trace_id=trace_id,
            request_id=req_id,
            conversation_id=request.conversation_id,
            pipeline_version=PIPELINE_VERSION,
            prompt_template_version=template_version,
            retrieval_mode=requested_mode,
            effective_retrieval_mode=effective_mode,
            embedding_model=ret_model,
            generation_model=self.settings.runtime.generation_model,
            model_digest=ret_digest,
            selected_evidence_ids=tuple(item.evidence_id for item in selected),
            stage_metrics=metrics,
            estimated_input_tokens=est_tokens,
            provider_usage=usage,
            citation_validation=validation,
            admission_outcome=admission.outcome,
            degraded=degraded,
            degradation_reason=degradation_reason,
            resource_rejection_reason=resource_rejection,
            query_execution=query_execution,
            reflection_execution=reflection_execution,
            created_at=datetime.now(UTC),
        )
        await asyncio.to_thread(self.traces.save, trace)

    def _build_response(
        self,
        request: RAGRequest,
        requested_mode: RetrievalMode,
        effective_mode: RetrievalMode,
        degraded: bool,
        degradation_reason: str | None,
        admission: AnswerAdmissionResult,
        trace_id: str,
        metrics: PipelineStageMetrics,
        *,
        strategy_mode: StrategyMode = "baseline",
        plan: RetrievalPlan | None = None,
        coverage: EvidenceCoverage | None = None,
        reflection_summary: ReflectionOutcome | None = None,
    ) -> RAGResponse:
        limitations = admission.limitations
        if coverage is not None and coverage.warnings:
            limitations = tuple(dict.fromkeys([*limitations, *coverage.warnings]))
        return RAGResponse(
            answer=admission.answer,
            status=admission.status,
            admission_outcome=admission.outcome,
            citations=admission.citations if request.include_citations else (),
            limitations=limitations,
            retrieval_mode=requested_mode,
            effective_retrieval_mode=effective_mode,
            degraded=degraded,
            degradation_reason=degradation_reason,
            trace_id=trace_id,
            metrics=metrics if request.include_metrics else None,
            strategy_mode=strategy_mode,
            query_plan=plan if request.include_query_plan else None,
            retrieval_diagnostics=coverage if request.include_retrieval_diagnostics else None,
            reflection_summary=reflection_summary if request.include_reflection_summary else None,
        )

    async def stream_query(self, request: RAGRequest) -> AsyncGenerator[RAGStreamEvent, None]:
        """Emit stage progress events and validate the full answer before completing."""
        yield RAGStreamEvent(event="retrieval_started")
        try:
            response = await self.query(request)
        except ProviderError as exc:
            yield RAGStreamEvent(event="failed", error=exc.code)
            return
        except IngestionError as exc:
            yield RAGStreamEvent(event="failed", error=exc.code)
            return
        yield RAGStreamEvent(
            event="retrieval_completed",
            trace_id=response.trace_id,
            detail=response.effective_retrieval_mode,
        )
        yield RAGStreamEvent(
            event="evidence_selected",
            trace_id=response.trace_id,
            detail=f"citations={len(response.citations)}",
        )
        yield RAGStreamEvent(event="generation_started", trace_id=response.trace_id)
        yield RAGStreamEvent(event="validation_started", trace_id=response.trace_id)
        yield RAGStreamEvent(event="completed", trace_id=response.trace_id, response=response)
