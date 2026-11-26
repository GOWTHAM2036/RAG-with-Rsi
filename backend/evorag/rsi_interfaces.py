"""Read-only diagnostic interfaces exposing Phase 08 signals for future Phase 09 RSI."""

from collections import Counter

from .config import Settings
from .eval_types import ExperimentComparison, QualityGateResult
from .reflection_types import RSIFeedbackSnapshot
from .tracing import TraceStore


class RSIFeedbackInterface:
    """Strictly read-only view over traces and experiments without exposing held-out labels."""

    def __init__(self, settings: Settings, traces: TraceStore) -> None:
        self.settings = settings
        self.traces = traces

    def snapshot(
        self,
        *,
        config_fingerprint: str,
        recent_run_ids: tuple[str, ...] = (),
        gate_results: tuple[QualityGateResult, ...] = (),
        comparisons: tuple[ExperimentComparison, ...] = (),
    ) -> RSIFeedbackSnapshot:
        """Return an immutable snapshot without mutating active settings or indexes."""
        counts: Counter[str] = Counter()
        return RSIFeedbackSnapshot(
            config_fingerprint=config_fingerprint,
            total_traces=self.traces.count(),
            failure_counts=dict(counts),
            recent_run_ids=recent_run_ids,
            gate_results=gate_results,
            comparisons=comparisons,
        )
