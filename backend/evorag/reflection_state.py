"""Bounded reflection state machine with legal-transition and budget enforcement."""

import time

from .reflection_types import (
    CorrectionActionKind,
    ReflectionBudget,
    ReflectionState,
    ReflectionStateName,
)

TERMINAL_STATES: frozenset[ReflectionStateName] = frozenset(
    {"accepted", "partially_accepted", "abstained", "failed", "cancelled", "budget_exhausted"}
)

LEGAL_TRANSITIONS: dict[ReflectionStateName, frozenset[ReflectionStateName]] = {
    "initial_answer": frozenset({"deterministic_verification", "abstained", "failed", "cancelled"}),
    "deterministic_verification": frozenset(
        {
            "semantic_verification",
            "correction_planning",
            "accepted",
            "partially_accepted",
            "abstained",
            "failed",
            "cancelled",
            "budget_exhausted",
        }
    ),
    "semantic_verification": frozenset(
        {
            "correction_planning",
            "accepted",
            "partially_accepted",
            "abstained",
            "failed",
            "cancelled",
            "budget_exhausted",
        }
    ),
    "correction_planning": frozenset(
        {
            "targeted_retrieval",
            "regeneration",
            "final_verification",
            "accepted",
            "partially_accepted",
            "abstained",
            "failed",
            "cancelled",
            "budget_exhausted",
        }
    ),
    "targeted_retrieval": frozenset(
        {
            "regeneration",
            "final_verification",
            "abstained",
            "failed",
            "cancelled",
            "budget_exhausted",
        }
    ),
    "regeneration": frozenset(
        {"final_verification", "candidate_comparison", "failed", "cancelled", "budget_exhausted"}
    ),
    "final_verification": frozenset(
        {
            "candidate_comparison",
            "accepted",
            "partially_accepted",
            "abstained",
            "failed",
            "cancelled",
        }
    ),
    "candidate_comparison": frozenset(
        {"accepted", "partially_accepted", "abstained", "failed", "cancelled"}
    ),
    "accepted": frozenset(),
    "partially_accepted": frozenset(),
    "abstained": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "budget_exhausted": frozenset(),
}


class IllegalTransitionError(ValueError):
    """Raised when an illegal reflection state transition is attempted."""


class ReflectionStateMachine:
    """Enforces legal transitions, global budgets, deadlines and repeated-action stopping."""

    def __init__(self, budget: ReflectionBudget) -> None:
        self.budget = budget
        self.state = ReflectionState(current="initial_answer")
        self._seen_actions: set[CorrectionActionKind] = set()

    def transition(self, target: ReflectionStateName, reason: str) -> ReflectionState:
        allowed = LEGAL_TRANSITIONS[self.state.current]
        if target not in allowed:
            raise IllegalTransitionError(f"Illegal transition {self.state.current} -> {target}")
        self.state = ReflectionState(
            current=target,
            history=(*self.state.history, self.state.current),
            transition_reasons=(*self.state.transition_reasons, reason),
        )
        return self.state

    def is_terminal(self) -> bool:
        return self.state.current in TERMINAL_STATES

    def check_deadline(self) -> bool:
        if self.budget.deadline_epoch_s <= 0.0:
            return False
        return time.monotonic() > self.budget.deadline_epoch_s

    def can_consume_model_call(self, kind: str) -> bool:
        if self.budget.total_model_calls_used >= self.budget.max_total_model_calls:
            return False
        if kind == "critique":
            return self.budget.critique_calls_used < self.budget.max_critique_calls
        if kind == "correction":
            return self.budget.correction_calls_used < self.budget.max_correction_calls
        return True

    def consume_model_call(self, kind: str) -> None:
        updates: dict[str, int] = {"total_model_calls_used": self.budget.total_model_calls_used + 1}
        if kind == "critique":
            updates["critique_calls_used"] = self.budget.critique_calls_used + 1
        elif kind == "correction":
            updates["correction_calls_used"] = self.budget.correction_calls_used + 1
        self.budget = self.budget.model_copy(update=updates)

    def can_consume_retrieval(self, count: int = 1) -> bool:
        return (
            self.budget.additional_retrievals_used + count <= self.budget.max_additional_retrievals
        )

    def consume_retrieval(self, count: int = 1) -> None:
        self.budget = self.budget.model_copy(
            update={"additional_retrievals_used": self.budget.additional_retrievals_used + count}
        )

    def register_action(self, action: CorrectionActionKind) -> bool:
        """Record an action and return False if it was already attempted (loop guard)."""
        if action in self._seen_actions and action not in {"accept_answer"}:
            return False
        self._seen_actions.add(action)
        return True
