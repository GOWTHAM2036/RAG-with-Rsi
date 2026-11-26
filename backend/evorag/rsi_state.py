"""Bounded RSI state machine enforcing legal transitions and global cycle budgets."""

import time

from .rsi_types import RSIBudget, RSIState, RSIStateName

TERMINAL_RSI_STATES: frozenset[RSIStateName] = frozenset(
    {"completed", "rejected", "inconclusive", "blocked", "failed", "cancelled", "budget_exhausted"}
)

LEGAL_RSI_TRANSITIONS: dict[RSIStateName, frozenset[RSIStateName]] = {
    "idle": frozenset({"collecting_feedback", "cancelled", "failed"}),
    "collecting_feedback": frozenset(
        {"analyzing_failures", "inconclusive", "blocked", "cancelled", "failed", "budget_exhausted"}
    ),
    "analyzing_failures": frozenset(
        {"generating_hypotheses", "inconclusive", "blocked", "cancelled", "failed"}
    ),
    "generating_hypotheses": frozenset(
        {"generating_candidates", "inconclusive", "cancelled", "failed", "budget_exhausted"}
    ),
    "generating_candidates": frozenset(
        {"validating_candidates", "inconclusive", "cancelled", "failed", "budget_exhausted"}
    ),
    "validating_candidates": frozenset(
        {"running_experiments", "rejected", "inconclusive", "cancelled", "failed"}
    ),
    "running_experiments": frozenset(
        {"comparing_candidates", "cancelled", "failed", "budget_exhausted"}
    ),
    "comparing_candidates": frozenset(
        {
            "selecting_finalist",
            "generating_candidates",
            "inconclusive",
            "rejected",
            "cancelled",
            "failed",
            "budget_exhausted",
        }
    ),
    "selecting_finalist": frozenset(
        {"validating_finalist", "inconclusive", "rejected", "cancelled", "failed"}
    ),
    "validating_finalist": frozenset(
        {"awaiting_approval", "blocked", "rejected", "inconclusive", "cancelled", "failed"}
    ),
    "awaiting_approval": frozenset({"promoting", "completed", "rejected", "cancelled"}),
    "promoting": frozenset({"monitoring", "rolling_back", "completed", "failed"}),
    "monitoring": frozenset({"rolling_back", "completed", "failed"}),
    "rolling_back": frozenset({"completed", "failed"}),
    "completed": frozenset(),
    "rejected": frozenset(),
    "inconclusive": frozenset(),
    "blocked": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "budget_exhausted": frozenset(),
}


class IllegalRSITransitionError(ValueError):
    """Raised when an illegal RSI state transition is attempted."""


class RSIStateMachine:
    """Tracks legal RSI transitions, cycle/experiment budgets and duplicate trials."""

    def __init__(self, budget: RSIBudget) -> None:
        self.budget = budget
        self.state = RSIState(current="idle")
        self._evaluated_fingerprints: set[str] = set()

    def transition(self, target: RSIStateName, reason: str) -> RSIState:
        if target not in LEGAL_RSI_TRANSITIONS[self.state.current]:
            raise IllegalRSITransitionError(
                f"Illegal RSI transition {self.state.current} -> {target}"
            )
        self.state = RSIState(
            current=target,
            history=(*self.state.history, self.state.current),
            reasons=(*self.state.reasons, reason),
        )
        return self.state

    def is_terminal(self) -> bool:
        return self.state.current in TERMINAL_RSI_STATES

    def deadline_exceeded(self) -> bool:
        return (
            self.budget.deadline_epoch_s > 0.0 and time.monotonic() > self.budget.deadline_epoch_s
        )

    def can_run_experiment(self, fingerprint: str) -> bool:
        if self.budget.experiments_used >= self.budget.max_experiments:
            return False
        return fingerprint not in self._evaluated_fingerprints

    def record_experiment(self, fingerprint: str) -> None:
        self._evaluated_fingerprints.add(fingerprint)
        self.budget = self.budget.model_copy(
            update={"experiments_used": self.budget.experiments_used + 1}
        )

    def can_start_cycle(self) -> bool:
        return self.budget.cycles_used < self.budget.max_cycles

    def record_cycle(self) -> None:
        self.budget = self.budget.model_copy(update={"cycles_used": self.budget.cycles_used + 1})
