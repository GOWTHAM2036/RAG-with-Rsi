"""Typed contracts for Phase 07 evaluation, datasets, experiments, stats and quality gates."""

import hashlib
import json
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, FiniteFloat, NonNegativeInt, PositiveInt

from .config import Name
from .contracts import Contract
from .hybrid_retrieval import RetrievalMode

RagStrategy = Literal["baseline", "adaptive"]

EvaluationSplit = Literal["dev", "validation", "held_out"]
ExecutionMode = Literal["offline", "synthetic", "live", "human_annotated"]
GateDecision = Literal["pass", "fail", "inconclusive", "blocked"]
FaithfulnessLabel = Literal["supported", "contradicted", "insufficient_evidence", "uncertain"]
RunStatus = Literal["pending", "running", "completed", "cancelled", "failed"]

FailureCategory = Literal[
    "ingestion_failure",
    "embedding_failure",
    "missing_relevant_documents",
    "incorrect_retrieval_ranking",
    "query_planning_failure",
    "query_rewriting_failure",
    "incomplete_evidence_coverage",
    "context_budget_overflow",
    "relevant_evidence_excluded",
    "invalid_model_output",
    "invalid_citation",
    "unsupported_claim",
    "contradicted_claim",
    "incorrect_answer",
    "incorrect_abstention",
    "resource_admission_failure",
    "provider_timeout",
    "integrity_failure",
    "security_policy_failure",
    "evaluation_uncertainty",
]


class EvidenceAnnotation(Contract):
    doc_name: Name
    relevance_grade: int = Field(default=1, ge=0, le=3)
    required: bool = True


class GroundTruth(Contract):
    expected_answer: str | None = None
    acceptable_answers: tuple[str, ...] = ()
    required_facts: tuple[str, ...] = ()
    forbidden_facts: tuple[str, ...] = ()
    expected_numeric: float | None = None
    numeric_tolerance: float = Field(default=1e-3, ge=0.0)
    expected_evidence: tuple[EvidenceAnnotation, ...] = ()
    expect_abstain: bool = False
    faithfulness_labels: tuple[FaithfulnessLabel, ...] = ()
    provenance: str = "manual_fixture"


class EvaluationCase(Contract):
    case_id: Name
    category: Name
    split: EvaluationSplit
    question: Annotated[str, Field(min_length=1, max_length=16_384, pattern=r"\S")]
    ground_truth: GroundTruth
    adversarial: bool = False
    notes: str | None = None


class DatasetVersion(Contract):
    dataset_id: Name
    version: Name
    fingerprint: Name
    fixture_type: Literal["synthetic", "public_domain", "human_annotated"] = "synthetic"
    created_at: str
    known_limitations: str


class EvaluationDataset(Contract):
    metadata: DatasetVersion
    cases: tuple[EvaluationCase, ...]


class MetricDefinition(Contract):
    name: Name
    version: Name = "1.0.0"
    higher_is_better: bool = True
    description: str


class MetricResult(Contract):
    name: Name
    value: FiniteFloat | None = None
    unavailable_reason: str | None = None
    sample_count: NonNegativeInt = 0
    execution_mode: ExecutionMode = "offline"


class ConfidenceInterval(Contract):
    metric_name: Name
    mean: FiniteFloat
    lower: FiniteFloat
    upper: FiniteFloat
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    sample_count: NonNegativeInt
    seed: int = 42


class AbstentionConfusionMatrix(Contract):
    tp_answered: NonNegativeInt = 0
    fn_false_abstain: NonNegativeInt = 0
    tn_correct_abstain: NonNegativeInt = 0
    fp_false_answer: NonNegativeInt = 0
    resource_rejections: NonNegativeInt = 0
    accuracy: FiniteFloat | None = None
    precision: FiniteFloat | None = None
    recall: FiniteFloat | None = None
    false_abstention_rate: FiniteFloat | None = None
    false_answer_rate: FiniteFloat | None = None
    answerable_coverage: FiniteFloat | None = None


class FailureClassification(Contract):
    case_id: Name
    category: FailureCategory
    observed: bool = True
    detail: str


class ExperimentConfiguration(Contract):
    schema_version: Name = "1.0.0"
    dataset_id: Name
    dataset_version: Name
    split: EvaluationSplit = "dev"
    retrieval_mode: RetrievalMode = "hybrid"
    strategy: RagStrategy = "baseline"
    dense_top_k: PositiveInt = 20
    sparse_top_k: PositiveInt = 20
    final_top_k: PositiveInt = 5
    bm25_k1: float = Field(default=1.5, ge=0.0, le=10.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    rrf_k: PositiveInt = 60
    reranking_enabled: bool = False
    query_rewriting_enabled: bool = False
    query_decomposition_enabled: bool = False
    context_max_tokens: PositiveInt = 4096
    prompt_template: Name = "grounded_v1"
    generation_model: Name = "qwen3:4b"
    embedding_model: Name = "qwen3-embedding:0.6b"
    analyzer_version: Name = "v1"
    execution_mode: ExecutionMode = "offline"
    seed: int = 42

    def fingerprint(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class EvaluationConfiguration(Contract):
    experiment: ExperimentConfiguration
    max_cases: PositiveInt = 200
    judge_enabled: bool = False
    bootstrap_samples: int = Field(default=1000, ge=100, le=10000)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)


class EvaluationCaseOutcome(Contract):
    case_id: Name
    category: Name
    split: EvaluationSplit
    precision_at_k: FiniteFloat | None = None
    recall_at_k: FiniteFloat | None = None
    reciprocal_rank: FiniteFloat | None = None
    ndcg_at_k: FiniteFloat | None = None
    answer_exact_match: FiniteFloat | None = None
    answer_token_f1: FiniteFloat | None = None
    citation_validity: FiniteFloat | None = None
    citation_precision: FiniteFloat | None = None
    citation_recall: FiniteFloat | None = None
    faithfulness_score: FiniteFloat | None = None
    abstained: bool = False
    expected_abstain: bool = False
    resource_rejected: bool = False
    latency_ms: Annotated[FiniteFloat, Field(ge=0)] = 0.0
    failures: tuple[FailureClassification, ...] = ()


class EvaluationRun(Contract):
    run_id: Name
    status: RunStatus
    config_fingerprint: Name
    dataset_id: Name
    dataset_version: Name
    split: EvaluationSplit
    execution_mode: ExecutionMode
    total_cases: NonNegativeInt
    completed_cases: NonNegativeInt = 0
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    error: str | None = None


class EvaluationRunResult(Contract):
    run: EvaluationRun
    config: ExperimentConfiguration
    metrics: tuple[MetricResult, ...]
    abstention: AbstentionConfusionMatrix
    confidence_intervals: tuple[ConfidenceInterval, ...] = ()
    case_outcomes: tuple[EvaluationCaseOutcome, ...] = ()
    failures: tuple[FailureClassification, ...] = ()
    median_latency_ms: Annotated[FiniteFloat, Field(ge=0)] = 0.0
    p95_latency_ms: Annotated[FiniteFloat, Field(ge=0)] = 0.0
    additional_peak_ram_mib: Annotated[FiniteFloat, Field(ge=0)] = 0.0


class MetricDelta(Contract):
    metric_name: Name
    baseline_value: FiniteFloat | None = None
    candidate_value: FiniteFloat | None = None
    absolute_diff: FiniteFloat | None = None
    ci_lower: FiniteFloat | None = None
    ci_upper: FiniteFloat | None = None
    wins: NonNegativeInt = 0
    ties: NonNegativeInt = 0
    losses: NonNegativeInt = 0
    verdict: Literal["improved", "regressed", "inconclusive", "unavailable"] = "inconclusive"


class ExperimentComparison(Contract):
    baseline_run_id: Name
    candidate_run_id: Name
    deltas: tuple[MetricDelta, ...]
    pareto_dominated: bool = False
    overall_verdict: Literal["improved", "regressed", "inconclusive", "mixed"] = "inconclusive"


class RegressionReport(Contract):
    baseline_run_id: Name
    candidate_run_id: Name
    regressed_metrics: tuple[str, ...] = ()
    latency_regression_ms: FiniteFloat = 0.0
    ram_regression_mib: FiniteFloat = 0.0
    has_regression: bool = False


class QualityGate(Contract):
    gate_id: Name
    metric_name: Name
    min_value: FiniteFloat | None = None
    max_regression: FiniteFloat | None = None
    require_live_mode: bool = False


class QualityGateResult(Contract):
    gate_id: Name
    decision: GateDecision
    observed_value: FiniteFloat | None = None
    reason: str


class EvaluationProvenance(Contract):
    dataset_id: Name
    dataset_version: Name
    dataset_fingerprint: Name
    split: EvaluationSplit
    execution_mode: ExecutionMode
    generation_model: Name
    embedding_model: Name
    config_fingerprint: Name
    automatic_promotion_enabled: Literal[False] = False


class EvaluationReport(Contract):
    run_id: Name
    provenance: EvaluationProvenance
    metrics: tuple[MetricResult, ...]
    abstention: AbstentionConfusionMatrix
    confidence_intervals: tuple[ConfidenceInterval, ...]
    gate_results: tuple[QualityGateResult, ...]
    failure_summary: dict[str, int]
    known_limitations: tuple[str, ...]
