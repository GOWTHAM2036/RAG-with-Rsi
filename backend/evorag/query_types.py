"""Typed contracts for Phase 06 query intelligence, adaptive planning and multi-query retrieval."""

from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, NonNegativeInt, PositiveInt

from .config import Name
from .contracts import Contract, RetrievalResult
from .documents import SourceSpan
from .hybrid_retrieval import RetrievalMode

QUERY_INTELLIGENCE_SCHEMA_VERSION = "1.0.0"

QueryIntent = Literal[
    "factual",
    "technical_lookup",
    "comparative",
    "multi_part",
    "conversational",
    "exploratory",
    "unsupported",
]
QueryComplexity = Literal["simple", "moderate", "complex", "invalid"]
RetrievalStrategy = Literal[
    "baseline",
    "simple_factual",
    "technical_exact",
    "conversational_resolve",
    "decomposed_multi_part",
    "exploratory_expanded",
]
QueryVariantKind = Literal["original", "normalized", "rewrite", "subquery"]
StrategyMode = Literal["baseline", "adaptive"]


class EvidenceCandidate(Contract):
    result: RetrievalResult
    version: int | None = None
    source_id: str | None = None
    sources: tuple[SourceSpan, ...] = ()
    retrieval_mode: RetrievalMode = "hybrid"


class QueryAnalysis(Contract):
    schema_version: str = QUERY_INTELLIGENCE_SCHEMA_VERSION
    original_query: str
    normalized_query: str
    char_length: NonNegativeInt
    estimated_tokens: NonNegativeInt
    question_words: tuple[str, ...] = ()
    has_question_mark: bool = False
    technical_identifiers: tuple[str, ...] = ()
    error_codes: tuple[str, ...] = ()
    quoted_phrases: tuple[str, ...] = ()
    numeric_references: tuple[str, ...] = ()
    comparison_terms: tuple[str, ...] = ()
    conjunctions: tuple[str, ...] = ()
    temporal_references: tuple[str, ...] = ()
    ambiguous_pronouns: tuple[str, ...] = ()
    conversation_references: tuple[str, ...] = ()
    negation_terms: tuple[str, ...] = ()
    has_negation: bool = False
    is_valid: bool = True
    validation_errors: tuple[str, ...] = ()


class QueryClassification(Contract):
    schema_version: str = QUERY_INTELLIGENCE_SCHEMA_VERSION
    primary_intent: QueryIntent
    labels: tuple[QueryIntent, ...]
    complexity: QueryComplexity
    triggered_signals: tuple[str, ...] = ()
    classifier_kind: Literal["heuristic", "llm_assisted"] = "heuristic"
    explanation: str


class QueryVariant(Contract):
    variant_id: Name
    kind: QueryVariantKind
    text: Annotated[str, Field(min_length=1, max_length=16_384, pattern=r"\S")]
    weight: Annotated[FiniteFloat, Field(ge=0.0, le=10.0)]
    source_subquery_id: str | None = None


class QueryRewriteResult(Contract):
    schema_version: str = QUERY_INTELLIGENCE_SCHEMA_VERSION
    original_query: str
    rewritten_query: str
    applied: bool
    method: Literal["none", "deterministic", "llm"]
    prompt_version: str | None = None
    fallback_reason: str | None = None
    preserved_identifiers: tuple[str, ...] = ()
    tokens_used: NonNegativeInt = 0


class SubQuery(Contract):
    subquery_id: Name
    text: Annotated[str, Field(min_length=1, max_length=16_384, pattern=r"\S")]
    component_index: NonNegativeInt
    weight: Annotated[FiniteFloat, Field(ge=0.0, le=10.0)]


class QueryDecomposition(Contract):
    schema_version: str = QUERY_INTELLIGENCE_SCHEMA_VERSION
    original_query: str
    subqueries: tuple[SubQuery, ...] = ()
    applied: bool = False
    method: Literal["none", "deterministic", "llm"] = "none"
    depth: Literal[0, 1] = 0
    prompt_version: str | None = None
    fallback_reason: str | None = None
    tokens_used: NonNegativeInt = 0


class RetrievalBudget(Contract):
    max_query_variants: PositiveInt
    max_subqueries: PositiveInt
    max_retrieval_executions: PositiveInt
    max_aggregate_candidates: PositiveInt
    max_final_evidence_items: PositiveInt
    max_planning_llm_calls: NonNegativeInt
    max_decomposition_depth: Literal[1] = 1
    max_concurrent_generations: PositiveInt = 1
    request_deadline_seconds: Annotated[FiniteFloat, Field(gt=0.0)]


class RetrievalPlan(Contract):
    schema_version: str = QUERY_INTELLIGENCE_SCHEMA_VERSION
    plan_id: Name
    planning_version: Name
    strategy_mode: StrategyMode
    selected_strategy: RetrievalStrategy
    retrieval_mode: RetrievalMode
    dense_weight_override: Annotated[FiniteFloat, Field(ge=0.0)] | None = None
    sparse_weight_override: Annotated[FiniteFloat, Field(ge=0.0)] | None = None
    analysis: QueryAnalysis
    classification: QueryClassification
    rewrite: QueryRewriteResult
    decomposition: QueryDecomposition
    variants: tuple[QueryVariant, ...]
    budget: RetrievalBudget
    rationale: tuple[str, ...] = ()
    llm_planning_calls_used: NonNegativeInt = 0
    additional_planning_tokens: NonNegativeInt = 0
    fallback_reasons: tuple[str, ...] = ()


class FusedMultiQueryCandidate(Contract):
    chunk_id: Name
    document_id: Name
    aggregated_score: FiniteFloat
    final_rank: PositiveInt
    original_rank: PositiveInt | None = None
    variant_ranks: dict[str, PositiveInt] = Field(default_factory=dict)
    per_query_fusion_scores: dict[str, FiniteFloat] = Field(default_factory=dict)
    dense_score: FiniteFloat | None = None
    sparse_score: FiniteFloat | None = None
    contributing_query_ids: tuple[str, ...] = ()


class MultiQueryResult(Contract):
    schema_version: str = QUERY_INTELLIGENCE_SCHEMA_VERSION
    plan_id: Name
    effective_mode: RetrievalMode
    degraded: bool = False
    degradation_reason: str | None = None
    retrieval_executions: NonNegativeInt
    cache_hits: NonNegativeInt = 0
    cache_misses: NonNegativeInt = 0
    candidates: tuple[EvidenceCandidate, ...] = ()
    fused_details: tuple[FusedMultiQueryCandidate, ...] = ()
    subquery_candidate_counts: dict[str, NonNegativeInt] = Field(default_factory=dict)
    model: str | None = None
    digest: str | None = None


class EvidenceCoverage(Contract):
    schema_version: str = QUERY_INTELLIGENCE_SCHEMA_VERSION
    subqueries_with_candidates: tuple[str, ...] = ()
    subqueries_in_selected_evidence: tuple[str, ...] = ()
    missing_subqueries: tuple[str, ...] = ()
    duplicate_evidence_ratio: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    unique_document_count: NonNegativeInt
    document_diversity: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    evidence_budget_utilization: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    retrieval_degraded: bool = False
    degradation_reason: str | None = None
    warnings: tuple[str, ...] = ()


class QueryExecutionTrace(Contract):
    schema_version: str = QUERY_INTELLIGENCE_SCHEMA_VERSION
    strategy_mode: StrategyMode
    selected_strategy: RetrievalStrategy
    planning_version: Name
    primary_intent: QueryIntent
    intent_labels: tuple[QueryIntent, ...]
    complexity: QueryComplexity
    variant_count: NonNegativeInt
    subquery_count: NonNegativeInt
    retrieval_executions: NonNegativeInt
    rewrite_applied: bool
    rewrite_method: Literal["none", "deterministic", "llm"]
    decomposition_applied: bool
    decomposition_method: Literal["none", "deterministic", "llm"]
    fallback_reasons: tuple[str, ...] = ()
    cache_hits: NonNegativeInt = 0
    cache_misses: NonNegativeInt = 0
    coverage: EvidenceCoverage | None = None
    planning_ms: Annotated[FiniteFloat, Field(ge=0.0)]
    multi_query_retrieval_ms: Annotated[FiniteFloat, Field(ge=0.0)]
    consolidation_ms: Annotated[FiniteFloat, Field(ge=0.0)]
    additional_planning_tokens: NonNegativeInt = 0
    resource_admission_ok: bool = True
    resource_rejection_reason: str | None = None


class RetrievalCacheEntry(Contract):
    cache_key: Name
    normalized_query: str
    retrieval_mode: RetrievalMode
    config_fingerprint: Name
    analyzer_version: Name
    index_generation: NonNegativeInt
    embedding_space_id: str | None = None
    document_revision_token: str
    reranking_enabled: bool
    created_at_monotonic: FiniteFloat
    ttl_seconds: Annotated[FiniteFloat, Field(gt=0.0)]
