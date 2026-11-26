"""Typed contracts for Phase 09 bounded Recursive Self-Improvement (RSI)."""

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, Field, FiniteFloat, NonNegativeInt, PositiveInt

from .config import Name
from .contracts import Contract
from .eval_types import ExperimentComparison, QualityGateResult

OptimizationStrategy = Literal[
    "coordinate_search", "seeded_random_search", "local_refinement", "pareto_selection"
]
RSIStateName = Literal[
    "idle",
    "collecting_feedback",
    "analyzing_failures",
    "generating_hypotheses",
    "generating_candidates",
    "validating_candidates",
    "running_experiments",
    "comparing_candidates",
    "selecting_finalist",
    "validating_finalist",
    "awaiting_approval",
    "promoting",
    "monitoring",
    "rolling_back",
    "completed",
    "rejected",
    "inconclusive",
    "blocked",
    "failed",
    "cancelled",
    "budget_exhausted",
]
RSIOutcomeKind = Literal[
    "improvement_recommended",
    "no_improvement",
    "inconclusive",
    "blocked",
    "cancelled",
    "budget_exhausted",
    "failed",
]


class RSIConfiguration(Contract):
    schema_version: Name = "1.0.0"
    enabled: bool = False
    automatic_promotion: Literal[False] = False
    automatic_rollback: Literal[False] = False
    optimization_strategy: OptimizationStrategy = "coordinate_search"
    max_cycles: int = Field(default=2, ge=1, le=5)
    max_hypotheses_per_cycle: int = Field(default=3, ge=1, le=10)
    max_candidates_per_cycle: int = Field(default=3, ge=1, le=10)
    max_experiments: int = Field(default=6, ge=1, le=20)
    max_dev_cases: int = Field(default=30, ge=1, le=200)
    max_elapsed_seconds: float = Field(default=300.0, gt=0.0, le=3600.0)
    seed: int = 42


class RSIBudget(Contract):
    max_cycles: PositiveInt = 2
    cycles_used: NonNegativeInt = 0
    max_experiments: PositiveInt = 6
    experiments_used: NonNegativeInt = 0
    max_llm_calls: NonNegativeInt = 2
    llm_calls_used: NonNegativeInt = 0
    max_holdout_evaluations: PositiveInt = 1
    holdout_evaluations_used: NonNegativeInt = 0
    deadline_epoch_s: float = 0.0


class RSIState(Contract):
    current: RSIStateName = "idle"
    history: tuple[RSIStateName, ...] = ()
    reasons: tuple[str, ...] = ()


class FailurePattern(Contract):
    pattern_id: Name
    category: Name
    sample_count: NonNegativeInt
    severity: Annotated[float, Field(ge=0.0, le=1.0)]
    affected_categories: tuple[str, ...] = ()
    description: str


class ImprovementOpportunity(Contract):
    opportunity_id: Name
    pattern_ids: tuple[str, ...]
    priority_score: Annotated[FiniteFloat, Field(ge=0.0)]
    estimated_cost: Annotated[FiniteFloat, Field(gt=0.0)] = 1.0
    reversible: bool = True
    description: str


class ParameterDefinition(Contract):
    path: Name
    param_type: Literal["int", "float", "bool", "enum"]
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: tuple[str, ...] = ()
    requires_reindexing: bool = False


class SearchSpace(Contract):
    schema_version: Name = "1.0.0"
    parameters: tuple[ParameterDefinition, ...]


class ImprovementHypothesis(Contract):
    hypothesis_id: Name
    pattern_ids: tuple[str, ...]
    proposed_changes: dict[str, Any]
    expected_benefit: str
    possible_regression: str
    required_metrics: tuple[str, ...] = ("recall_at_k", "citation_validity")
    estimated_cost: Annotated[FiniteFloat, Field(gt=0.0)] = 1.0
    reversible: bool = True
    source: Literal["deterministic", "llm_assisted"] = "deterministic"


class CandidateConfiguration(Contract):
    candidate_id: Name
    parent_fingerprint: Name
    fingerprint: Name
    hypothesis_id: Name
    changes: dict[str, Any]
    parameters: dict[str, Any]
    requires_reindexing: bool = False

    @classmethod
    def compute_fingerprint(cls, params: dict[str, Any]) -> str:
        canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class CandidateValidationResult(Contract):
    valid: bool
    errors: tuple[str, ...] = ()


class CandidateScore(Contract):
    candidate_id: Name
    fingerprint: Name
    primary_metric: Annotated[FiniteFloat, Field(ge=0.0)] | None = None
    composite_utility: Annotated[FiniteFloat, Field(ge=0.0)] | None = None
    p95_latency_ms: Annotated[FiniteFloat, Field(ge=0.0)] = 0.0
    additional_ram_mib: Annotated[FiniteFloat, Field(ge=0.0)] = 0.0
    pareto_optimal: bool = False
    passed_gates: bool = False


class ExperimentTrial(Contract):
    trial_id: Name
    cycle_index: NonNegativeInt
    candidate_id: Name
    candidate_fingerprint: Name
    run_id: Name
    score: CandidateScore
    comparison: ExperimentComparison | None = None
    gate_results: tuple[QualityGateResult, ...] = ()


class CandidateRanking(Contract):
    ranked: tuple[CandidateScore, ...]


class OptimizationHistory(Contract):
    strategy: OptimizationStrategy
    trials: tuple[ExperimentTrial, ...] = ()
    incumbent_fingerprint: Name | None = None


class PromotionRecommendation(Contract):
    recommendation_id: Name
    baseline_fingerprint: Name
    candidate_fingerprint: Name
    changes: dict[str, Any]
    expected_benefit: str
    observed_regressions: tuple[str, ...] = ()
    gate_results: tuple[QualityGateResult, ...] = ()
    execution_mode: str = "offline"
    eligible_for_live_promotion: bool = False
    rollback_target_version: Name
    known_limitations: str
    created_at: AwareDatetime


class PromotionApproval(Contract):
    approval_id: Name
    recommendation_id: Name
    candidate_fingerprint: Name
    baseline_version: Name
    approved_by: Name
    activation_scope: Literal["offline_sandbox", "live_runtime"] = "offline_sandbox"
    approved_at: AwareDatetime


class ConfigurationVersion(Contract):
    version_id: Name
    parent_version_id: Name | None = None
    fingerprint: Name
    parameters: dict[str, Any]
    active: bool = False
    approved: bool = False
    created_at: AwareDatetime


class PromotionRecord(Contract):
    promotion_id: Name
    from_version_id: Name
    to_version_id: Name
    candidate_fingerprint: Name
    approval_id: Name
    promoted_at: AwareDatetime


class RollbackRecord(Contract):
    rollback_id: Name
    from_version_id: Name
    to_version_id: Name
    reason: str
    rolled_back_at: AwareDatetime


class ImprovementMemoryEntry(Contract):
    entry_id: Name
    hypothesis_id: Name
    parent_fingerprint: Name
    candidate_fingerprint: Name
    changes: dict[str, Any]
    dataset_version: Name
    accepted: bool
    rejection_reason: str | None = None
    created_at: AwareDatetime


class RSICycle(Contract):
    cycle_id: Name
    status: RSIStateName
    outcome: RSIOutcomeKind | None = None
    strategy: OptimizationStrategy
    baseline_fingerprint: Name
    incumbent_fingerprint: Name | None = None
    cycles_completed: NonNegativeInt = 0
    experiments_completed: NonNegativeInt = 0
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None


class RSIExecutionTrace(Contract):
    cycle_id: Name
    state_history: tuple[RSIStateName, ...]
    hypotheses_generated: NonNegativeInt
    candidates_generated: NonNegativeInt
    candidates_rejected: NonNegativeInt
    experiments_run: NonNegativeInt
    recommendation_id: Name | None = None
    outcome: RSIOutcomeKind
    elapsed_ms: Annotated[FiniteFloat, Field(ge=0.0)]


class RSIReport(Contract):
    cycle: RSICycle
    trace: RSIExecutionTrace
    trials: tuple[ExperimentTrial, ...] = ()
    recommendation: PromotionRecommendation | None = None


class RSIStatusReport(Contract):
    enabled: bool
    automatic_promotion: Literal[False] = False
    automatic_rollback: Literal[False] = False
    optimization_strategy: OptimizationStrategy
    active_config_version: Name
    total_cycles: NonNegativeInt
    total_recommendations: NonNegativeInt
