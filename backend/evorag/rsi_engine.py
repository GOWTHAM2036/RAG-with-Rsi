"""Bounded Recursive Self-Improvement (RSI) orchestrator with approval and rollback."""

import asyncio
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .config import Settings
from .eval_types import ExperimentConfiguration
from .experiment_runner import ExperimentRunner
from .quality_gates import evaluate_gates
from .rsi_analysis import analyze_failure_patterns, generate_hypotheses, prioritize_opportunities
from .rsi_interfaces import RSIFeedbackInterface
from .rsi_optimizer import compute_composite_utility, generate_candidates, rank_candidates_pareto
from .rsi_state import RSIStateMachine
from .rsi_store import RSIStore
from .rsi_types import (
    CandidateConfiguration,
    CandidateScore,
    ConfigurationVersion,
    ExperimentTrial,
    ImprovementMemoryEntry,
    PromotionApproval,
    PromotionRecommendation,
    PromotionRecord,
    RollbackRecord,
    RSIBudget,
    RSICycle,
    RSIExecutionTrace,
    RSIOutcomeKind,
    RSIReport,
    RSIStatusReport,
)


def apply_parameters_to_experiment(
    base: ExperimentConfiguration, params: dict[str, Any]
) -> ExperimentConfiguration:
    """Apply candidate search-space parameters to an isolated ExperimentConfiguration."""
    return base.model_copy(
        update={
            "bm25_k1": float(params.get("bm25.k1", base.bm25_k1)),
            "bm25_b": float(params.get("bm25.b", base.bm25_b)),
            "rrf_k": int(params.get("fusion.k", base.rrf_k)),
            "dense_top_k": int(params.get("retrieval.dense_top_k", base.dense_top_k)),
            "sparse_top_k": int(params.get("retrieval.sparse_top_k", base.sparse_top_k)),
            "final_top_k": int(params.get("retrieval.final_top_k", base.final_top_k)),
            "strategy": str(params.get("adaptive_rag.default_strategy", base.strategy)),
            "reranking_enabled": bool(params.get("reranking.enabled", base.reranking_enabled)),
            "query_rewriting_enabled": bool(
                params.get("query_rewriting.enabled", base.query_rewriting_enabled)
            ),
            "query_decomposition_enabled": bool(
                params.get("query_decomposition.enabled", base.query_decomposition_enabled)
            ),
        }
    )


class RSIEngine:
    """Coordinates bounded, isolated RSI cycles, owner approval and rollback."""

    def __init__(
        self,
        settings: Settings,
        store: RSIStore,
        feedback: RSIFeedbackInterface,
        runner: ExperimentRunner,
    ) -> None:
        self.settings = settings
        self.store = store
        self.feedback = feedback
        self.runner = runner
        self._lock = asyncio.Lock()
        self._cancelled: set[str] = set()

    def status(self) -> RSIStatusReport:
        active = self.store.active_version()
        return RSIStatusReport(
            enabled=self.settings.rsi.enabled,
            optimization_strategy=self.settings.rsi.optimization_strategy,
            active_config_version=active.version_id,
            total_cycles=self.store.count_cycles(),
            total_recommendations=len(self.store.list_recommendations()),
        )

    def cancel(self, cycle_id: str) -> RSICycle | None:
        self._cancelled.add(cycle_id)
        return self.store.update_cycle_status(cycle_id, "cancelled")

    async def run_cycle(
        self,
        *,
        dataset_id: str = "evorag_comprehensive_v1",
        dataset_version: str = "1.0.0",
        failure_counts: dict[str, int] | None = None,
        max_dev_cases: int = 10,
        doc_id_to_name: dict[str, str] | None = None,
    ) -> RSIReport:
        if self._lock.locked():
            raise RuntimeError("rsi_cycle_already_active")
        async with self._lock:
            t0 = time.perf_counter()
            started_at = datetime.now(UTC)
            cycle_id = uuid4().hex
            cfg = self.settings.rsi
            active_ver = self.store.active_version()
            parent_params = dict(active_ver.parameters)
            baseline_fp = active_ver.fingerprint

            budget = RSIBudget(
                max_cycles=cfg.max_cycles,
                max_experiments=cfg.max_experiments,
                max_llm_calls=cfg.optimization.max_llm_calls_per_cycle,
                max_holdout_evaluations=cfg.validation.max_holdout_finalist_evaluations,
                deadline_epoch_s=time.monotonic() + cfg.max_execution_seconds,
            )
            sm = RSIStateMachine(budget)
            sm.transition("collecting_feedback", "collect_read_only_feedback")
            snap = self.feedback.snapshot(config_fingerprint=baseline_fp)
            counts = failure_counts if failure_counts is not None else snap.failure_counts

            sm.transition("analyzing_failures", "analyze_failure_patterns")
            patterns = analyze_failure_patterns(
                counts,
                minimum_samples=1 if failure_counts is not None else cfg.feedback.minimum_samples,
            )
            opportunities = prioritize_opportunities(patterns)
            if not opportunities:
                sm.transition("inconclusive", "no_eligible_failure_patterns")
                return self._save_and_return(
                    cycle_id=cycle_id,
                    sm=sm,
                    outcome="inconclusive",
                    baseline_fp=baseline_fp,
                    incumbent_fp=None,
                    started_at=started_at,
                    t0=t0,
                    patterns=patterns,
                    hypotheses=(),
                    candidates=(),
                    trials=(),
                    rec=None,
                )

            sm.transition("generating_hypotheses", "generate_deterministic_hypotheses")
            hypotheses = generate_hypotheses(opportunities, max_hypotheses=3)

            base_exp = apply_parameters_to_experiment(
                ExperimentConfiguration(
                    dataset_id=dataset_id,
                    dataset_version=dataset_version,
                    split="dev",
                    execution_mode="offline",
                ),
                parent_params,
            )
            baseline_res = await self.runner.run(
                base_exp, doc_id_to_name=doc_id_to_name, max_cases=max_dev_cases
            )

            all_candidates: list[CandidateConfiguration] = []
            all_trials: list[ExperimentTrial] = []
            incumbent_params = dict(parent_params)
            incumbent_fp = baseline_fp
            best_trial: ExperimentTrial | None = None

            while (
                sm.can_start_cycle()
                and not sm.deadline_exceeded()
                and cycle_id not in self._cancelled
            ):
                cands = generate_candidates(
                    incumbent_params,
                    hypotheses,
                    strategy=cfg.optimization_strategy,
                    max_candidates=cfg.max_candidates_per_cycle,
                    seed=cfg.seed + sm.budget.cycles_used + 1,
                    allow_reindexing=cfg.optimization.allow_reindexing,
                )
                eligible = [
                    c
                    for c in cands
                    if not self.store.has_rejected_memory(c.fingerprint, dataset_version)
                ]
                if not eligible:
                    break
                sm.record_cycle()
                sm.transition("generating_candidates", f"cycle_{sm.budget.cycles_used}")
                all_candidates.extend(eligible)
                sm.transition("validating_candidates", "validated_against_search_space")
                sm.transition("running_experiments", "run_isolated_dev_experiments")

                cycle_trials: list[ExperimentTrial] = []
                for cand in eligible:
                    if not sm.can_run_experiment(cand.fingerprint) or sm.deadline_exceeded():
                        break
                    sm.record_experiment(cand.fingerprint)
                    cand_exp = apply_parameters_to_experiment(base_exp, cand.parameters)
                    res = await self.runner.run(
                        cand_exp, doc_id_to_name=doc_id_to_name, max_cases=max_dev_cases
                    )
                    comp = self.runner.compare(baseline_res.run.run_id, res.run.run_id)
                    gates = evaluate_gates(res, comparison=comp)
                    m_map = {m.name: m.value for m in res.metrics}
                    util = compute_composite_utility(
                        retrieval_quality=m_map.get("recall_at_k"),
                        answer_correctness=m_map.get("answer_token_f1"),
                        faithfulness=m_map.get("faithfulness"),
                        citation_quality=m_map.get("citation_validity"),
                    )
                    passed = all(
                        g.decision in {"pass", "blocked", "inconclusive"} for g in gates
                    ) and (not any(g.decision == "fail" for g in gates))
                    score = CandidateScore(
                        candidate_id=cand.candidate_id,
                        fingerprint=cand.fingerprint,
                        primary_metric=m_map.get("recall_at_k"),
                        composite_utility=util,
                        p95_latency_ms=res.p95_latency_ms,
                        additional_ram_mib=res.additional_peak_ram_mib,
                        passed_gates=passed,
                    )
                    trial = ExperimentTrial(
                        trial_id=uuid4().hex,
                        cycle_index=sm.budget.cycles_used,
                        candidate_id=cand.candidate_id,
                        candidate_fingerprint=cand.fingerprint,
                        run_id=res.run.run_id,
                        score=score,
                        comparison=comp,
                        gate_results=gates,
                    )
                    cycle_trials.append(trial)
                    all_trials.append(trial)
                    await asyncio.to_thread(
                        self.store.save_memory,
                        ImprovementMemoryEntry(
                            entry_id=uuid4().hex,
                            hypothesis_id=cand.hypothesis_id,
                            parent_fingerprint=cand.parent_fingerprint,
                            candidate_fingerprint=cand.fingerprint,
                            changes=cand.changes,
                            dataset_version=dataset_version,
                            accepted=passed and comp.overall_verdict != "regressed",
                            rejection_reason=None if passed else "gate_failed",
                            created_at=datetime.now(UTC),
                        ),
                    )

                sm.transition("comparing_candidates", "rank_candidates_pareto")
                if not cycle_trials:
                    break
                ranked = rank_candidates_pareto([t.score for t in cycle_trials])
                top_score = ranked.ranked[0]
                top_trial = next(
                    t for t in cycle_trials if t.candidate_id == top_score.candidate_id
                )
                if top_score.passed_gates and (
                    top_trial.comparison is None
                    or top_trial.comparison.overall_verdict != "regressed"
                ):
                    best_trial = top_trial
                    top_cand = next(c for c in eligible if c.fingerprint == top_score.fingerprint)
                    incumbent_params = dict(top_cand.parameters)
                    incumbent_fp = top_cand.fingerprint
                else:
                    break

            if best_trial is None:
                sm.transition("inconclusive", "no_improving_candidate")
                return self._save_and_return(
                    cycle_id=cycle_id,
                    sm=sm,
                    outcome="inconclusive",
                    baseline_fp=baseline_fp,
                    incumbent_fp=None,
                    started_at=started_at,
                    t0=t0,
                    patterns=patterns,
                    hypotheses=hypotheses,
                    candidates=tuple(all_candidates),
                    trials=tuple(all_trials),
                    rec=None,
                )

            sm.transition("selecting_finalist", "freeze_finalist")
            sm.transition("validating_finalist", "controlled_finalist_validation")
            best_cand = next(
                c for c in all_candidates if c.fingerprint == best_trial.candidate_fingerprint
            )
            rec = PromotionRecommendation(
                recommendation_id=uuid4().hex,
                baseline_fingerprint=baseline_fp,
                candidate_fingerprint=best_cand.fingerprint,
                changes=best_cand.changes,
                expected_benefit="Improved development metrics within resource bounds",
                observed_regressions=tuple(
                    d.metric_name
                    for d in (best_trial.comparison.deltas if best_trial.comparison else ())
                    if d.verdict == "regressed"
                ),
                gate_results=best_trial.gate_results,
                execution_mode="offline",
                eligible_for_live_promotion=False,
                rollback_target_version=active_ver.version_id,
                known_limitations=(
                    "Offline evaluation only; live promotion requires live verification."
                ),
                created_at=datetime.now(UTC),
            )
            sm.transition("awaiting_approval", "recommendation_ready_for_owner")
            return self._save_and_return(
                cycle_id=cycle_id,
                sm=sm,
                outcome="improvement_recommended",
                baseline_fp=baseline_fp,
                incumbent_fp=incumbent_fp,
                started_at=started_at,
                t0=t0,
                patterns=patterns,
                hypotheses=hypotheses,
                candidates=tuple(all_candidates),
                trials=tuple(all_trials),
                rec=rec,
            )

    def approve_recommendation(
        self,
        recommendation_id: str,
        *,
        approved_by: str,
        activation_scope: str = "offline_sandbox",
    ) -> PromotionApproval:
        rec = self.store.get_recommendation(recommendation_id)
        if rec is None:
            raise KeyError("recommendation_not_found")
        active = self.store.active_version()
        if rec.baseline_fingerprint != active.fingerprint:
            raise ValueError("stale_recommendation_baseline_changed")
        if activation_scope == "live_runtime" and not rec.eligible_for_live_promotion:
            raise ValueError("offline_result_cannot_authorize_live_promotion")
        approval = PromotionApproval(
            approval_id=uuid4().hex,
            recommendation_id=recommendation_id,
            candidate_fingerprint=rec.candidate_fingerprint,
            baseline_version=active.version_id,
            approved_by=approved_by,
            activation_scope=activation_scope,  # type: ignore[arg-type]
            approved_at=datetime.now(UTC),
        )
        self.store.save_approval(approval)
        return approval

    def promote_approved(self, approval_id: str) -> ConfigurationVersion:
        approval = self.store.get_approval(approval_id)
        if approval is None:
            raise KeyError("approval_not_found")
        active = self.store.active_version()
        if approval.baseline_version != active.version_id:
            raise ValueError("stale_approval_active_version_changed")
        params = self.store.get_candidate_params(approval.candidate_fingerprint)
        if params is None:
            raise KeyError("candidate_parameters_not_found")
        new_ver = ConfigurationVersion(
            version_id=f"v_{approval.candidate_fingerprint[:8]}",
            parent_version_id=active.version_id,
            fingerprint=approval.candidate_fingerprint,
            parameters=params,
            active=True,
            approved=True,
            created_at=datetime.now(UTC),
        )
        rec = PromotionRecord(
            promotion_id=uuid4().hex,
            from_version_id=active.version_id,
            to_version_id=new_ver.version_id,
            candidate_fingerprint=new_ver.fingerprint,
            approval_id=approval_id,
            promoted_at=datetime.now(UTC),
        )
        self.store.activate_version_atomic(active, new_ver, rec)
        return new_ver

    def rollback_to(self, target_version_id: str, *, reason: str) -> ConfigurationVersion:
        active = self.store.active_version()
        versions = {v.version_id: v for v in self.store.list_versions()}
        target = versions.get(target_version_id)
        if target is None:
            raise KeyError("target_version_not_found")
        rb = RollbackRecord(
            rollback_id=uuid4().hex,
            from_version_id=active.version_id,
            to_version_id=target.version_id,
            reason=reason,
            rolled_back_at=datetime.now(UTC),
        )
        self.store.activate_version_atomic(active, target, rb)
        return target

    def _save_and_return(
        self,
        *,
        cycle_id: str,
        sm: RSIStateMachine,
        outcome: RSIOutcomeKind,
        baseline_fp: str,
        incumbent_fp: str | None,
        started_at: datetime,
        t0: float,
        patterns: tuple[Any, ...],
        hypotheses: tuple[Any, ...],
        candidates: tuple[CandidateConfiguration, ...],
        trials: tuple[ExperimentTrial, ...],
        rec: PromotionRecommendation | None,
    ) -> RSIReport:
        elapsed = (time.perf_counter() - t0) * 1000.0
        cycle = RSICycle(
            cycle_id=cycle_id,
            status=sm.state.current,
            outcome=outcome,
            strategy=self.settings.rsi.optimization_strategy,
            baseline_fingerprint=baseline_fp,
            incumbent_fingerprint=incumbent_fp,
            cycles_completed=sm.budget.cycles_used,
            experiments_completed=sm.budget.experiments_used,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )
        trace = RSIExecutionTrace(
            cycle_id=cycle_id,
            state_history=(*sm.state.history, sm.state.current),
            hypotheses_generated=len(hypotheses),
            candidates_generated=len(candidates),
            candidates_rejected=0,
            experiments_run=sm.budget.experiments_used,
            recommendation_id=rec.recommendation_id if rec else None,
            outcome=outcome,
            elapsed_ms=elapsed,
        )
        report = RSIReport(cycle=cycle, trace=trace, trials=trials, recommendation=rec)
        self.store.save_cycle_report(
            report,
            patterns=patterns,
            hypotheses=hypotheses,
            candidates=candidates,
            trials=trials,
        )
        return report
