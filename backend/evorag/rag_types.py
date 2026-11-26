"""Typed contracts for Phase 05 grounded generation, citations, admission and tracing."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, FiniteFloat, NonNegativeInt, PositiveInt

from .config import Name
from .contracts import Contract
from .documents import SourceSpan
from .filtering import MetadataFilter
from .hybrid_retrieval import RetrievalMode
from .inference_types import GenerationUsage
from .query_types import (
    EvidenceCandidate as EvidenceCandidate,
)
from .query_types import (
    EvidenceCoverage,
    QueryExecutionTrace,
    RetrievalPlan,
    StrategyMode,
)

ReflectionMode = Literal["off", "deterministic", "semantic"]
ReflectionStateName = Literal[
    "initial_answer",
    "deterministic_verification",
    "semantic_verification",
    "correction_planning",
    "targeted_retrieval",
    "regeneration",
    "final_verification",
    "candidate_comparison",
    "accepted",
    "partially_accepted",
    "abstained",
    "failed",
    "cancelled",
    "budget_exhausted",
]
ReflectionTriggerKind = Literal[
    "invalid_structured_response",
    "invalid_citations",
    "missing_evidence",
    "missing_components",
    "contradictory_evidence",
    "partial_answer",
    "low_evidence_coverage",
    "explicit_request",
    "complex_multipart",
]
CorrectionActionKind = Literal[
    "accept_answer",
    "remove_unsupported_claims",
    "revise_answer",
    "retrieve_more",
    "refine_query",
    "reconcile_conflicting_evidence",
    "return_partial_answer",
    "abstain",
    "terminate_with_error",
]

RAGStatus = Literal["answered", "partially_answered", "insufficient_evidence"]
AdmissionOutcome = Literal[
    "accepted",
    "partially_accepted",
    "insufficient_evidence",
    "invalid_citations",
    "invalid_model_output",
    "generation_failed",
    "resource_unavailable",
]
StreamEventKind = Literal[
    "retrieval_started",
    "retrieval_completed",
    "evidence_selected",
    "generation_started",
    "validation_started",
    "completed",
    "failed",
]


class RAGRequest(Contract):
    query_id: Name | None = None
    question: Annotated[str, Field(min_length=1, max_length=16_384, pattern=r"\S")]
    retrieval_mode: RetrievalMode | None = None
    top_k: Annotated[int, Field(ge=1, le=100)] | None = None
    filters: MetadataFilter | None = None
    reranking_enabled: bool | None = None
    conversation_id: Name | None = None
    include_citations: bool = True
    include_metrics: bool = True
    strategy: StrategyMode | None = None
    query_rewriting_enabled: bool | None = None
    query_decomposition_enabled: bool | None = None
    include_query_plan: bool = False
    include_retrieval_diagnostics: bool = False
    reflection_enabled: bool | None = None
    reflection_mode: ReflectionMode | None = None
    include_reflection_summary: bool = False


class SelectedEvidence(Contract):
    evidence_id: Name
    chunk_id: Name
    document_id: Name
    version: int | None = None
    source_id: str | None = None
    sources: tuple[SourceSpan, ...] = ()
    text: str
    original_rank: PositiveInt
    retrieval_mode: RetrievalMode
    score: FiniteFloat
    truncated: bool = False
    included_start: NonNegativeInt = 0
    included_end: NonNegativeInt = 0


class ConversationMessage(Contract):
    role: Literal["user", "assistant"]
    content: Annotated[str, Field(min_length=1, max_length=16_384, pattern=r"\S")]
    created_at: AwareDatetime


class ConversationState(Contract):
    conversation_id: Name
    messages: tuple[ConversationMessage, ...] = ()
    persisted: bool = False


class ContextBudget(Contract):
    max_context_tokens: PositiveInt
    system_token_budget: PositiveInt
    question_token_budget: PositiveInt
    history_token_budget: PositiveInt
    evidence_token_budget: PositiveInt
    reserved_output_tokens: PositiveInt
    safety_margin_tokens: PositiveInt
    estimated_system_tokens: NonNegativeInt
    estimated_question_tokens: NonNegativeInt
    estimated_history_tokens: NonNegativeInt
    estimated_evidence_tokens: NonNegativeInt
    total_estimated_input_tokens: NonNegativeInt
    truncations: tuple[str, ...] = ()


class ContextPackage(Contract):
    question: str
    evidence: tuple[SelectedEvidence, ...]
    budget: ContextBudget
    history_messages: tuple[ConversationMessage, ...] = ()


class PromptTemplate(Contract):
    template_id: Name
    version: Name
    fingerprint: Name
    system_prompt: str


class StructuredClaim(Contract):
    text: Annotated[str, Field(min_length=1, pattern=r"\S")]
    evidence_ids: tuple[Name, ...] = ()


class StructuredAnswer(Contract):
    status: RAGStatus
    answer: str
    claims: tuple[StructuredClaim, ...] = ()
    limitations: tuple[str, ...] = ()


class CitationReference(Contract):
    citation_number: PositiveInt
    evidence_id: Name
    document_id: Name
    version: int | None = None
    chunk_id: Name
    source_id: str | None = None
    sources: tuple[SourceSpan, ...] = ()
    excerpt: str
    claim_indices: tuple[NonNegativeInt, ...] = ()


class CitationValidationResult(Contract):
    valid: bool
    citations: tuple[CitationReference, ...] = ()
    invalid_evidence_ids: tuple[str, ...] = ()
    uncited_claim_indices: tuple[NonNegativeInt, ...] = ()
    errors: tuple[str, ...] = ()


class AnswerAdmissionResult(Contract):
    outcome: AdmissionOutcome
    accepted: bool
    status: RAGStatus
    answer: str
    citations: tuple[CitationReference, ...] = ()
    limitations: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


class PipelineStageMetrics(Contract):
    retrieval_ms: Annotated[FiniteFloat, Field(ge=0)]
    context_ms: Annotated[FiniteFloat, Field(ge=0)]
    generation_ms: Annotated[FiniteFloat, Field(ge=0)] | None = None
    first_token_ms: Annotated[FiniteFloat, Field(ge=0)] | None = None
    validation_ms: Annotated[FiniteFloat, Field(ge=0)]
    total_ms: Annotated[FiniteFloat, Field(ge=0)]


class ReflectionOutcome(Contract):
    reflection_applied: bool
    verification_method: Literal["none", "deterministic", "semantic"]
    correction_attempted: bool = False
    correction_count: NonNegativeInt = 0
    final_state: ReflectionStateName
    final_admission_status: AdmissionOutcome
    final_status: RAGStatus
    assessed_claim_count: NonNegativeInt = 0
    uncertain_claim_count: NonNegativeInt = 0
    triggers: tuple[ReflectionTriggerKind, ...] = ()
    actions_taken: tuple[CorrectionActionKind, ...] = ()
    limitations: tuple[str, ...] = ()
    reflection_ms: Annotated[FiniteFloat, Field(ge=0)] = 0.0
    correction_ms: Annotated[FiniteFloat, Field(ge=0)] = 0.0


class ReflectionExecutionTrace(Contract):
    trace_id: Name
    reflection_mode: ReflectionMode
    config_version: Name = "1.0.0"
    triggers: tuple[ReflectionTriggerKind, ...] = ()
    verification_method: Literal["none", "deterministic", "semantic"]
    assessed_claims: NonNegativeInt = 0
    structurally_valid_claims: NonNegativeInt = 0
    supported_claims: NonNegativeInt = 0
    contradicted_claims: NonNegativeInt = 0
    uncertain_or_unassessed_claims: NonNegativeInt = 0
    actions: tuple[CorrectionActionKind, ...] = ()
    additional_retrievals: NonNegativeInt = 0
    additional_model_calls: NonNegativeInt = 0
    additional_tokens: NonNegativeInt = 0
    reflection_ms: Annotated[FiniteFloat, Field(ge=0)] = 0.0
    correction_ms: Annotated[FiniteFloat, Field(ge=0)] = 0.0
    budget_exhausted_reason: str | None = None
    selected_candidate: Literal["original", "corrected", "partial", "abstained"] = "original"
    final_admission_outcome: AdmissionOutcome
    created_at: AwareDatetime


class RAGExecutionTrace(Contract):
    trace_id: Name
    request_id: Name
    conversation_id: Name | None = None
    pipeline_version: Name
    prompt_template_version: Name
    retrieval_mode: RetrievalMode
    effective_retrieval_mode: RetrievalMode
    embedding_model: str | None = None
    generation_model: str | None = None
    model_digest: str | None = None
    selected_evidence_ids: tuple[str, ...] = ()
    stage_metrics: PipelineStageMetrics
    estimated_input_tokens: NonNegativeInt
    provider_usage: GenerationUsage | None = None
    citation_validation: CitationValidationResult
    admission_outcome: AdmissionOutcome
    degraded: bool = False
    degradation_reason: str | None = None
    resource_rejection_reason: str | None = None
    query_execution: QueryExecutionTrace | None = None
    reflection_execution: ReflectionExecutionTrace | None = None
    created_at: AwareDatetime


class RAGResponse(Contract):
    answer: str
    status: RAGStatus
    admission_outcome: AdmissionOutcome
    citations: tuple[CitationReference, ...] = ()
    limitations: tuple[str, ...] = ()
    retrieval_mode: RetrievalMode
    effective_retrieval_mode: RetrievalMode
    degraded: bool = False
    degradation_reason: str | None = None
    trace_id: Name
    metrics: PipelineStageMetrics | None = None
    strategy_mode: StrategyMode = "baseline"
    query_plan: RetrievalPlan | None = None
    retrieval_diagnostics: EvidenceCoverage | None = None
    reflection_summary: ReflectionOutcome | None = None


class RAGStreamEvent(Contract):
    event: StreamEventKind
    trace_id: Name | None = None
    detail: str | None = None
    response: RAGResponse | None = None
    error: str | None = None


class RAGStatusReport(Contract):
    enabled: bool
    default_retrieval_mode: RetrievalMode
    prompt_template: str
    tracing_enabled: bool
    stored_traces: NonNegativeInt
    active_conversations: NonNegativeInt
    adaptive_rag_enabled: bool = True
    default_strategy: StrategyMode = "baseline"
    retrieval_cache_stats: dict[str, int] = Field(default_factory=dict)
