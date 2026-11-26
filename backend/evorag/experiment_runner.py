"""Bounded sequential experiment runner, report exporter and comparison coordinator."""

import asyncio
import statistics
import time
from datetime import UTC, datetime
from uuid import uuid4

import psutil

from .config import Settings
from .dataset_registry import DatasetRegistry
from .eval_metrics import compute_abstention_matrix, score_answer_proxy, score_citations
from .eval_stats import bootstrap_ci, compare_runs
from .eval_types import (
    EvaluationCaseOutcome,
    EvaluationProvenance,
    EvaluationReport,
    EvaluationRun,
    EvaluationRunResult,
    ExperimentComparison,
    ExperimentConfiguration,
    FailureClassification,
    MetricResult,
)
from .experiment_store import ExperimentStore
from .failure_analysis import classify_case_failures, summarize_failures
from .quality_gates import evaluate_gates
from .rag_service import RAGService
from .rag_types import RAGRequest


class ExperimentRunner:
    """Runs bounded, resource-aware evaluations without modifying active config or indexes."""

    def __init__(
        self,
        settings: Settings,
        registry: DatasetRegistry,
        store: ExperimentStore,
        rag: RAGService,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.store = store
        self.rag = rag
        self._lock = asyncio.Lock()
        self._cancelled_runs: set[str] = set()

    def cancel(self, run_id: str) -> EvaluationRun | None:
        self._cancelled_runs.add(run_id)
        return self.store.update_status(run_id, "cancelled")

    async def run(
        self,
        config: ExperimentConfiguration,
        *,
        doc_id_to_name: dict[str, str] | None = None,
        max_cases: int | None = None,
    ) -> EvaluationRunResult:
        if self._lock.locked():
            raise RuntimeError("evaluation_run_already_active")
        async with self._lock:
            run_id = uuid4().hex
            limit = min(
                max_cases or self.settings.evaluation.max_cases_per_run,
                self.settings.evaluation.max_cases_per_run,
            )
            cases = self.registry.cases_for_split(config.dataset_id, config.split, limit=limit)
            started_at = datetime.now(UTC)
            proc = psutil.Process()
            baseline_rss = proc.memory_info().rss
            peak_rss = baseline_rss

            outcomes: list[EvaluationCaseOutcome] = []
            all_failures: list[FailureClassification] = []
            abstention_inputs: list[tuple[bool, bool, bool]] = []
            name_map = doc_id_to_name or {}

            for case in cases:
                if run_id in self._cancelled_runs:
                    break
                t0 = time.perf_counter()
                resp = await self.rag.query(
                    RAGRequest(
                        question=case.question,
                        retrieval_mode=config.retrieval_mode,
                        top_k=config.final_top_k,
                        reranking_enabled=config.reranking_enabled,
                        strategy=config.strategy,
                        query_rewriting_enabled=config.query_rewriting_enabled,
                        query_decomposition_enabled=config.query_decomposition_enabled,
                    )
                )
                lat_ms = (time.perf_counter() - t0) * 1000.0
                peak_rss = max(peak_rss, proc.memory_info().rss)

                expected_docs = frozenset(
                    e.doc_name for e in case.ground_truth.expected_evidence if e.required
                )
                cit_prec, cit_rec = score_citations(resp.citations, expected_docs, name_map)
                cit_valid = (
                    0.0
                    if resp.admission_outcome == "invalid_citations"
                    else (
                        1.0
                        if resp.citations or resp.admission_outcome == "insufficient_evidence"
                        else 0.0
                    )
                )
                em, f1, _ = score_answer_proxy(resp.answer, case.ground_truth)
                abstained = resp.admission_outcome == "insufficient_evidence"
                res_rej = resp.admission_outcome == "resource_unavailable"
                abstention_inputs.append((case.ground_truth.expect_abstain, abstained, res_rej))

                # Recall proxy from cited docs when retrieval traces are evaluated end-to-end.
                rec_k = cit_rec
                case_failures = classify_case_failures(case, resp, rec_k, cit_valid, f1)
                all_failures.extend(case_failures)
                outcomes.append(
                    EvaluationCaseOutcome(
                        case_id=case.case_id,
                        category=case.category,
                        split=case.split,
                        recall_at_k=rec_k,
                        answer_exact_match=em,
                        answer_token_f1=f1,
                        citation_validity=cit_valid,
                        citation_precision=cit_prec,
                        citation_recall=cit_rec,
                        abstained=abstained,
                        expected_abstain=case.ground_truth.expect_abstain,
                        resource_rejected=res_rej,
                        latency_ms=lat_ms,
                        failures=case_failures,
                    )
                )

            status = "cancelled" if run_id in self._cancelled_runs else "completed"
            abstention = compute_abstention_matrix(abstention_inputs)
            latencies = sorted(o.latency_ms for o in outcomes)
            median_lat = statistics.median(latencies) if latencies else 0.0
            p95_lat = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
            add_ram = max(0.0, (peak_rss - baseline_rss) / (1024**2))

            rec_vals = [o.recall_at_k for o in outcomes if o.recall_at_k is not None]
            f1_vals = [o.answer_token_f1 for o in outcomes if o.answer_token_f1 is not None]
            cit_vals = [o.citation_validity for o in outcomes if o.citation_validity is not None]

            metrics = (
                MetricResult(
                    name="recall_at_k",
                    value=statistics.mean(rec_vals) if rec_vals else None,
                    unavailable_reason=None if rec_vals else "no_expected_evidence",
                    sample_count=len(rec_vals),
                    execution_mode=config.execution_mode,
                ),
                MetricResult(
                    name="answer_token_f1",
                    value=statistics.mean(f1_vals) if f1_vals else None,
                    unavailable_reason=None if f1_vals else "no_reference_answers",
                    sample_count=len(f1_vals),
                    execution_mode=config.execution_mode,
                ),
                MetricResult(
                    name="citation_validity",
                    value=statistics.mean(cit_vals) if cit_vals else None,
                    unavailable_reason=None if cit_vals else "no_citations_evaluated",
                    sample_count=len(cit_vals),
                    execution_mode=config.execution_mode,
                ),
                MetricResult(
                    name="abstention_accuracy",
                    value=abstention.accuracy,
                    unavailable_reason=None
                    if abstention.accuracy is not None
                    else "no_non_rejected_cases",
                    sample_count=len(outcomes),
                    execution_mode=config.execution_mode,
                ),
                MetricResult(
                    name="faithfulness",
                    value=None,
                    unavailable_reason="no_live_or_human_faithfulness_labels",
                    sample_count=0,
                    execution_mode=config.execution_mode,
                ),
            )

            stats_cfg = self.settings.evaluation.statistics
            cis = tuple(
                ci
                for ci in (
                    bootstrap_ci(
                        "recall_at_k",
                        rec_vals,
                        confidence_level=stats_cfg.confidence_level,
                        n_bootstrap=stats_cfg.bootstrap_samples,
                        seed=stats_cfg.seed,
                    ),
                    bootstrap_ci(
                        "citation_validity",
                        cit_vals,
                        confidence_level=stats_cfg.confidence_level,
                        n_bootstrap=stats_cfg.bootstrap_samples,
                        seed=stats_cfg.seed,
                    ),
                )
                if ci is not None
            )

            run = EvaluationRun(
                run_id=run_id,
                status=status,  # type: ignore[arg-type]
                config_fingerprint=config.fingerprint(),
                dataset_id=config.dataset_id,
                dataset_version=config.dataset_version,
                split=config.split,
                execution_mode=config.execution_mode,
                total_cases=len(cases),
                completed_cases=len(outcomes),
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )
            result = EvaluationRunResult(
                run=run,
                config=config,
                metrics=metrics,
                abstention=abstention,
                confidence_intervals=cis,
                case_outcomes=tuple(outcomes),
                failures=tuple(all_failures),
                median_latency_ms=median_lat,
                p95_latency_ms=p95_lat,
                additional_peak_ram_mib=add_ram,
            )
            await asyncio.to_thread(self.store.save_run, result)
            gates = evaluate_gates(result)
            await asyncio.to_thread(self.store.save_gates, run_id, gates)
            return result

    def compare(self, baseline_run_id: str, candidate_run_id: str) -> ExperimentComparison:
        base = self.store.get_run(baseline_run_id)
        cand = self.store.get_run(candidate_run_id)
        if base is None or cand is None:
            raise KeyError("evaluation_run_not_found")
        stats_cfg = self.settings.evaluation.statistics
        comp = compare_runs(
            base,
            cand,
            confidence_level=stats_cfg.confidence_level,
            n_bootstrap=stats_cfg.bootstrap_samples,
            seed=stats_cfg.seed,
        )
        self.store.save_comparison(comp)
        return comp

    def build_report(self, run_id: str) -> EvaluationReport:
        result = self.store.get_run(run_id)
        if result is None:
            raise KeyError("evaluation_run_not_found")
        ds = self.registry.get(result.run.dataset_id)
        gates = evaluate_gates(result)
        return EvaluationReport(
            run_id=run_id,
            provenance=EvaluationProvenance(
                dataset_id=ds.metadata.dataset_id,
                dataset_version=ds.metadata.version,
                dataset_fingerprint=ds.metadata.fingerprint,
                split=result.run.split,
                execution_mode=result.run.execution_mode,
                generation_model=result.config.generation_model,
                embedding_model=result.config.embedding_model,
                config_fingerprint=result.run.config_fingerprint,
            ),
            metrics=result.metrics,
            abstention=result.abstention,
            confidence_intervals=result.confidence_intervals,
            gate_results=gates,
            failure_summary=summarize_failures(result.case_outcomes),
            known_limitations=(ds.metadata.known_limitations,),
        )

    def render_markdown_report(self, run_id: str) -> str:
        report = self.build_report(run_id)
        lines = [
            f"# Evaluation Report · {report.run_id}",
            "",
            f"- Dataset: `{report.provenance.dataset_id}` "
            f"v{report.provenance.dataset_version} (`{report.provenance.split}`)",
            f"- Execution mode: `{report.provenance.execution_mode}`",
            f"- Config fingerprint: `{report.provenance.config_fingerprint}`",
            "",
            "## Metrics",
            "| Metric | Value | Samples |",
            "| --- | ---: | ---: |",
        ]
        for m in report.metrics:
            val = (
                f"{m.value:.4f}" if m.value is not None else f"unavailable ({m.unavailable_reason})"
            )
            lines.append(f"| {m.name} | {val} | {m.sample_count} |")
        lines.extend(
            ["", "## Quality Gates", "| Gate | Decision | Reason |", "| --- | --- | --- |"]
        )
        for g in report.gate_results:
            lines.append(f"| {g.gate_id} | `{g.decision}` | {g.reason} |")
        return "\n".join(lines)
