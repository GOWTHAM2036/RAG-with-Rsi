"""Custom deterministic bootstrap statistics, paired comparisons and Pareto analysis."""

from collections.abc import Sequence

import numpy as np

from .eval_types import (
    ConfidenceInterval,
    EvaluationRunResult,
    ExperimentComparison,
    MetricDelta,
    RegressionReport,
)


def bootstrap_ci(
    metric_name: str,
    values: Sequence[float],
    *,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> ConfidenceInterval | None:
    """Compute deterministic percentile bootstrap confidence interval."""
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return ConfidenceInterval(
            metric_name=metric_name,
            mean=mean,
            lower=mean,
            upper=mean,
            confidence_level=confidence_level,
            sample_count=1,
            seed=seed,
        )
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))
    means = np.mean(arr[idx], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.quantile(means, alpha))
    upper = float(np.quantile(means, 1.0 - alpha))
    return ConfidenceInterval(
        metric_name=metric_name,
        mean=mean,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        sample_count=len(arr),
        seed=seed,
    )


def paired_bootstrap_delta(
    metric_name: str,
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    higher_is_better: bool = True,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> MetricDelta:
    """Compare paired per-case metrics using deterministic paired bootstrap."""
    if not baseline or len(baseline) != len(candidate):
        return MetricDelta(metric_name=metric_name, verdict="unavailable")
    b_arr = np.asarray(baseline, dtype=np.float64)
    c_arr = np.asarray(candidate, dtype=np.float64)
    diffs = c_arr - b_arr
    b_mean = float(np.mean(b_arr))
    c_mean = float(np.mean(c_arr))
    diff_mean = float(np.mean(diffs))

    wins = int(np.sum(diffs > 1e-12)) if higher_is_better else int(np.sum(diffs < -1e-12))
    losses = int(np.sum(diffs < -1e-12)) if higher_is_better else int(np.sum(diffs > 1e-12))
    ties = len(diffs) - wins - losses

    if len(diffs) < 2:
        verdict: str = "inconclusive"
        ci_low = ci_high = diff_mean
    else:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(diffs), size=(n_bootstrap, len(diffs)))
        boot_means = np.mean(diffs[idx], axis=1)
        alpha = (1.0 - confidence_level) / 2.0
        ci_low = float(np.quantile(boot_means, alpha))
        ci_high = float(np.quantile(boot_means, 1.0 - alpha))
        if ci_low > 0.0:
            verdict = "improved" if higher_is_better else "regressed"
        elif ci_high < 0.0:
            verdict = "regressed" if higher_is_better else "improved"
        else:
            verdict = "inconclusive"

    return MetricDelta(
        metric_name=metric_name,
        baseline_value=b_mean,
        candidate_value=c_mean,
        absolute_diff=diff_mean,
        ci_lower=ci_low,
        ci_upper=ci_high,
        wins=wins,
        ties=ties,
        losses=losses,
        verdict=verdict,  # type: ignore[arg-type]
    )


def is_pareto_dominated(
    candidate_objectives: Sequence[float], other_objectives: Sequence[float]
) -> bool:
    """Return True if `other` is >= candidate on all objectives and > on at least one."""
    if len(candidate_objectives) != len(other_objectives) or not candidate_objectives:
        return False
    all_ge = all(o >= c for c, o in zip(candidate_objectives, other_objectives, strict=True))
    any_gt = any(o > c for c, o in zip(candidate_objectives, other_objectives, strict=True))
    return all_ge and any_gt


def compare_runs(
    baseline: EvaluationRunResult,
    candidate: EvaluationRunResult,
    *,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> ExperimentComparison:
    """Compare two paired evaluation runs across retrieval, answer, citation and latency."""
    b_by_id = {c.case_id: c for c in baseline.case_outcomes}
    c_by_id = {c.case_id: c for c in candidate.case_outcomes}
    common = [cid for cid in b_by_id if cid in c_by_id]

    specs: tuple[tuple[str, bool], ...] = (
        ("recall_at_k", True),
        ("reciprocal_rank", True),
        ("ndcg_at_k", True),
        ("answer_token_f1", True),
        ("citation_validity", True),
        ("latency_ms", False),
    )
    deltas: list[MetricDelta] = []
    for field, hib in specs:
        b_vals: list[float] = []
        c_vals: list[float] = []
        for cid in common:
            bv = getattr(b_by_id[cid], field)
            cv = getattr(c_by_id[cid], field)
            if bv is not None and cv is not None:
                b_vals.append(float(bv))
                c_vals.append(float(cv))
        deltas.append(
            paired_bootstrap_delta(
                field,
                b_vals,
                c_vals,
                higher_is_better=hib,
                confidence_level=confidence_level,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        )

    b_obj = [d.baseline_value or 0.0 for d in deltas if d.metric_name != "latency_ms"] + [
        -(baseline.p95_latency_ms)
    ]
    c_obj = [d.candidate_value or 0.0 for d in deltas if d.metric_name != "latency_ms"] + [
        -(candidate.p95_latency_ms)
    ]
    dominated = is_pareto_dominated(c_obj, b_obj)

    improved = any(d.verdict == "improved" for d in deltas)
    regressed = any(d.verdict == "regressed" for d in deltas)
    if improved and regressed:
        overall = "mixed"
    elif regressed:
        overall = "regressed"
    elif improved:
        overall = "improved"
    else:
        overall = "inconclusive"

    return ExperimentComparison(
        baseline_run_id=baseline.run.run_id,
        candidate_run_id=candidate.run.run_id,
        deltas=tuple(deltas),
        pareto_dominated=dominated,
        overall_verdict=overall,  # type: ignore[arg-type]
    )


def detect_regressions(
    baseline: EvaluationRunResult,
    candidate: EvaluationRunResult,
    comparison: ExperimentComparison,
) -> RegressionReport:
    regressed = tuple(d.metric_name for d in comparison.deltas if d.verdict == "regressed")
    lat_diff = candidate.p95_latency_ms - baseline.p95_latency_ms
    ram_diff = candidate.additional_peak_ram_mib - baseline.additional_peak_ram_mib
    return RegressionReport(
        baseline_run_id=baseline.run.run_id,
        candidate_run_id=candidate.run.run_id,
        regressed_metrics=regressed,
        latency_regression_ms=round(lat_diff, 3),
        ram_regression_mib=round(ram_diff, 3),
        has_regression=bool(regressed),
    )
