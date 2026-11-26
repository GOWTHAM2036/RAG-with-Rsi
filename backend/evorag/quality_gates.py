"""Configurable quality gate engine with strict missing-metric and mode checks."""

from collections.abc import Sequence

from .eval_types import (
    EvaluationRunResult,
    ExperimentComparison,
    QualityGate,
    QualityGateResult,
)

DEFAULT_GATES: tuple[QualityGate, ...] = (
    QualityGate(gate_id="citation_validity_floor", metric_name="citation_validity", min_value=0.90),
    QualityGate(gate_id="recall_floor", metric_name="recall_at_k", min_value=0.70),
    QualityGate(
        gate_id="abstention_accuracy_floor", metric_name="abstention_accuracy", min_value=0.80
    ),
    QualityGate(
        gate_id="live_semantic_faithfulness",
        metric_name="faithfulness",
        min_value=0.80,
        require_live_mode=True,
    ),
)


def evaluate_gates(
    run_result: EvaluationRunResult,
    gates: Sequence[QualityGate] = DEFAULT_GATES,
    comparison: ExperimentComparison | None = None,
) -> tuple[QualityGateResult, ...]:
    """Evaluate quality gates without allowing missing or mocked scores to pass live gates."""
    metrics_by_name = {m.name: m for m in run_result.metrics}
    deltas_by_name = {d.metric_name: d for d in comparison.deltas} if comparison else {}
    results: list[QualityGateResult] = []

    for gate in gates:
        if gate.require_live_mode and run_result.run.execution_mode != "live":
            results.append(
                QualityGateResult(
                    gate_id=gate.gate_id,
                    decision="blocked",
                    observed_value=None,
                    reason=f"Requires live execution mode (actual={run_result.run.execution_mode})",
                )
            )
            continue

        metric = metrics_by_name.get(gate.metric_name)
        if metric is None or metric.value is None:
            results.append(
                QualityGateResult(
                    gate_id=gate.gate_id,
                    decision="inconclusive",
                    observed_value=None,
                    reason=metric.unavailable_reason
                    if metric and metric.unavailable_reason
                    else "metric_unavailable",
                )
            )
            continue

        if gate.min_value is not None and metric.value < gate.min_value:
            results.append(
                QualityGateResult(
                    gate_id=gate.gate_id,
                    decision="fail",
                    observed_value=metric.value,
                    reason=f"{metric.value:.4f} < min_value {gate.min_value:.4f}",
                )
            )
            continue

        if gate.max_regression is not None and gate.metric_name in deltas_by_name:
            delta = deltas_by_name[gate.metric_name]
            if delta.absolute_diff is not None and delta.absolute_diff < -gate.max_regression:
                results.append(
                    QualityGateResult(
                        gate_id=gate.gate_id,
                        decision="fail",
                        observed_value=metric.value,
                        reason=(
                            f"Regressed by {delta.absolute_diff:.4f} "
                            f"(limit -{gate.max_regression:.4f})"
                        ),
                    )
                )
                continue

        results.append(
            QualityGateResult(
                gate_id=gate.gate_id,
                decision="pass",
                observed_value=metric.value,
                reason="Threshold satisfied",
            )
        )
    return tuple(results)
