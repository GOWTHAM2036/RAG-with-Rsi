"""Typed contracts for Phase 08 self-reflective RAG, verification and bounded correction."""

from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, NonNegativeInt, PositiveInt

from .config import Name
from .contracts import Contract
from .eval_types import ExperimentComparison, QualityGateResult
from .rag_types import (
    CitationReference,
    CorrectionActionKind,
    ReflectionExecutionTrace,
    ReflectionMode,
    ReflectionOutcome,
    ReflectionStateName,
    ReflectionTriggerKind,
    StructuredAnswer,
)

__all__ = [
    "AnswerComparison",
    "ClaimAssessment",
    "ClaimSupportCategory",
    "CorrectionAction",
    "CorrectionActionKind",
    "CorrectionPlan",
    "CorrectionResult",
    "CritiqueClaimItem",
    "EvidenceAssessment",
    "RSIFeedbackSnapshot",
    "ReflectionBudget",
    "ReflectionConfiguration",
    "ReflectionCritique",
    "ReflectionEvaluationResult",
    "ReflectionExecutionTrace",
    "ReflectionMode",
    "ReflectionOutcome",
    "ReflectionOutcomeStatus",
    "ReflectionRequest",
    "ReflectionState",
    "ReflectionStateName",
    "ReflectionStatusReport",
    "ReflectionTrigger",
    "ReflectionTriggerKind",
    "VerificationFinding",
    "VerificationReport",
]

ClaimSupportCategory = Literal[
    "supported",
    "partially_supported",
    "contradicted",
    "insufficient_evidence",
    "uncertain",
    "not_assessed",
]
ReflectionOutcomeStatus = Literal[
    "accepted",
    "partially_accepted",
    "abstained",
    "failed",
    "cancelled",
    "budget_exhausted",
]


class ReflectionRequest(Contract):
    question: Annotated[str, Field(min_length=1, max_length=16_384, pattern=r"\S")]
    mode: ReflectionMode = "deterministic"
    explicit_trigger: bool = False


class ReflectionConfiguration(Contract):
    schema_version: Name = "1.0.0"
    enabled: bool = False
    mode: ReflectionMode = "deterministic"
    max_cycles: int = Field(default=1, ge=0, le=3)
    max_critique_calls: int = Field(default=1, ge=0, le=3)
    max_correction_calls: int = Field(default=1, ge=0, le=3)
    max_additional_retrievals: int = Field(default=2, ge=0, le=5)
    max_total_model_calls: int = Field(default=3, ge=1, le=6)


class ReflectionBudget(Contract):
    max_cycles: NonNegativeInt = 1
    cycles_used: NonNegativeInt = 0
    max_critique_calls: NonNegativeInt = 1
    critique_calls_used: NonNegativeInt = 0
    max_correction_calls: NonNegativeInt = 1
    correction_calls_used: NonNegativeInt = 0
    max_additional_retrievals: NonNegativeInt = 2
    additional_retrievals_used: NonNegativeInt = 0
    max_total_model_calls: PositiveInt = 3
    total_model_calls_used: NonNegativeInt = 0
    deadline_epoch_s: float = 0.0


class ReflectionState(Contract):
    current: ReflectionStateName
    history: tuple[ReflectionStateName, ...] = ()
    transition_reasons: tuple[str, ...] = ()


class ReflectionTrigger(Contract):
    kind: ReflectionTriggerKind
    reason: str


class ClaimAssessment(Contract):
    claim_id: Name
    evidence_ids: tuple[str, ...] = ()
    category: ClaimSupportCategory
    method: Literal["deterministic", "semantic_model", "human_annotated"]
    reason: str
    uncertain: bool = False


class EvidenceAssessment(Contract):
    evidence_id: Name
    compatible_revision: bool = True
    compatible_span: bool = True
    notes: str = ""


class VerificationFinding(Contract):
    code: Name
    severity: Literal["info", "warning", "error"]
    observed: bool = True
    detail: str


class VerificationReport(Contract):
    structurally_valid: bool
    citations_valid: bool
    provenance_valid: bool
    semantic_assessed: bool = False
    findings: tuple[VerificationFinding, ...] = ()
    claim_assessments: tuple[ClaimAssessment, ...] = ()
    evidence_assessments: tuple[EvidenceAssessment, ...] = ()
    missing_components: tuple[str, ...] = ()
    duplicate_claim_indices: tuple[NonNegativeInt, ...] = ()


class CritiqueClaimItem(Contract):
    claim_id: Name
    assessment: ClaimSupportCategory
    evidence_ids: tuple[str, ...] = ()
    reason: str = ""


class ReflectionCritique(Contract):
    status: Literal["acceptable", "revision_required", "insufficient_evidence"]
    claim_assessments: tuple[CritiqueClaimItem, ...] = ()
    missing_components: tuple[str, ...] = ()
    recommended_action: CorrectionActionKind = "accept_answer"


class CorrectionAction(Contract):
    action: CorrectionActionKind
    triggers: tuple[ReflectionTriggerKind, ...] = ()
    justification: str
    estimated_model_calls: NonNegativeInt = 0
    estimated_retrievals: NonNegativeInt = 0


class CorrectionPlan(Contract):
    actions: tuple[CorrectionAction, ...]
    corrective_queries: tuple[str, ...] = ()


class CorrectionResult(Contract):
    applied: bool
    action_taken: CorrectionActionKind
    corrected_answer: StructuredAnswer | None = None
    corrected_citations: tuple[CitationReference, ...] = ()
    additional_retrievals: NonNegativeInt = 0
    additional_model_calls: NonNegativeInt = 0
    additional_tokens: NonNegativeInt = 0


class AnswerComparison(Contract):
    selected: Literal["original", "corrected", "partial", "abstained"]
    original_valid: bool
    corrected_valid: bool
    reason: str


class ReflectionEvaluationResult(Contract):
    total_cases: NonNegativeInt
    trigger_precision: FiniteFloat | None = None
    trigger_recall: FiniteFloat | None = None
    error_detection_precision: FiniteFloat | None = None
    error_detection_recall: FiniteFloat | None = None
    correction_success_rate: FiniteFloat | None = None
    regression_introduction_rate: FiniteFloat | None = None
    unnecessary_correction_rate: FiniteFloat | None = None
    original_retention_rate: FiniteFloat | None = None


class ReflectionStatusReport(Contract):
    enabled: bool
    mode: ReflectionMode
    trigger_policy: str
    max_cycles: NonNegativeInt
    max_total_model_calls: PositiveInt
    semantic_enabled: bool


class RSIFeedbackSnapshot(Contract):
    """Read-only snapshot exposing Phase 08 diagnostics for Phase 09 RSI without held-out labels."""

    schema_version: Name = "1.0.0"
    config_fingerprint: Name
    total_traces: NonNegativeInt
    failure_counts: dict[str, int]
    recent_run_ids: tuple[str, ...] = ()
    gate_results: tuple[QualityGateResult, ...] = ()
    comparisons: tuple[ExperimentComparison, ...] = ()
