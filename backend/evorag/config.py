"""Validated defaults, optional TOML overrides and scoped environment overrides."""

import json
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    PositiveInt,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from .errors import ConfigurationError

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]


class SettingsSection(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, validate_default=True, allow_inf_nan=False
    )


class ModelConfiguration(SettingsSection):
    provider: Literal["ollama", "lm_studio", "openai", "gemini", "openai_compatible"] = "ollama"
    base_url: HttpUrl = HttpUrl("http://127.0.0.1:11550")
    generation_model: Name = "qwen3:4b"
    embedding_model: Name = "qwen3-embedding:0.6b"
    context_window: PositiveInt = 4096
    max_output_tokens: PositiveInt = 768
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_concurrent_requests: PositiveInt = 1

    @field_validator("base_url")
    @classmethod
    def check_endpoint(cls, url: HttpUrl) -> HttpUrl:
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("use a provider endpoint without credentials, query or fragment")
        return url

    @model_validator(mode="after")
    def check_context(self) -> Self:
        if self.provider in {"openai", "gemini"} and self.base_url.scheme != "https":
            raise ValueError("cloud providers require an explicit HTTPS endpoint")
        if self.max_output_tokens >= self.context_window:
            raise ValueError("context_window must leave room for input tokens")
        return self


class Embeddings(SettingsSection):
    normalize: bool = True
    device: Literal["cpu", "gpu", "auto"] = "cpu"
    batch_size: PositiveInt = 8
    dimensions: PositiveInt = 1024


class Chunking(SettingsSection):
    strategy: Literal["fixed", "recursive", "structure"] = "recursive"
    tokenizer: Literal["estimated"] = "estimated"
    chunk_size: int = Field(default=384, ge=1, le=4096)
    overlap: int = Field(default=64, ge=0)

    @model_validator(mode="after")
    def check_overlap(self) -> Self:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self


class Retrieval(SettingsSection):
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    dense_top_k: PositiveInt = 20
    sparse_top_k: PositiveInt = 20
    final_top_k: PositiveInt = 5
    rrf_k: PositiveInt = 60
    allow_sparse_fallback: bool = True

    @model_validator(mode="after")
    def check_candidates(self) -> Self:
        if self.final_top_k > self.dense_top_k + self.sparse_top_k:
            raise ValueError("final_top_k exceeds the combined candidate budget")
        return self


class Sparse(SettingsSection):
    analyzer_version: Literal["v1"] = "v1"
    k1: float = Field(default=1.5, ge=0.0, le=10.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)
    preserve_compound_identifiers: bool = True
    remove_stopwords: bool = False
    max_query_tokens: int = Field(default=128, ge=1, le=1024)


class Fusion(SettingsSection):
    method: Literal["rrf"] = "rrf"
    k: PositiveInt = 60
    dense_weight: float = Field(default=1.0, ge=0.0, le=100.0)
    sparse_weight: float = Field(default=1.0, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def check_weights(self) -> Self:
        if self.dense_weight + self.sparse_weight <= 0.0:
            raise ValueError("at least one fusion weight must be positive")
        return self


class Reranking(SettingsSection):
    enabled: bool = False
    max_candidates: PositiveInt = 40
    phrase_weight: float = Field(default=0.35, ge=0.0, le=10.0)
    coverage_weight: float = Field(default=0.30, ge=0.0, le=10.0)
    identifier_weight: float = Field(default=0.25, ge=0.0, le=10.0)
    rrf_weight: float = Field(default=0.40, ge=0.0, le=10.0)
    duplicate_penalty: float = Field(default=0.20, ge=0.0, le=10.0)
    diversity_penalty: float = Field(default=0.10, ge=0.0, le=10.0)


class ReflectionSemanticSettings(SettingsSection):
    enabled: bool = False
    provider: Literal["ollama"] = "ollama"
    model: Name = "qwen3:4b"
    allow_uncertain_assessment: bool = True


class ReflectionCorrectionSettings(SettingsSection):
    allow_targeted_retrieval: bool = True
    allow_regeneration: bool = True
    allow_partial_answer: bool = True
    require_final_validation: bool = True
    max_query_variants: int = Field(default=2, ge=1, le=3)


class ReflectionTracingSettings(SettingsSection):
    enabled: bool = True
    store_raw_critiques: bool = False


class Reflection(SettingsSection):
    enabled: bool = False
    mode: Literal["off", "deterministic", "semantic"] = "deterministic"
    trigger_policy: Literal["on_failure", "always_complex", "explicit"] = "on_failure"
    max_cycles: int = Field(default=1, ge=0, le=3)
    max_critique_calls: int = Field(default=1, ge=0, le=3)
    max_correction_calls: int = Field(default=1, ge=0, le=3)
    max_additional_retrievals: int = Field(default=2, ge=0, le=5)
    max_total_model_calls: int = Field(default=3, ge=1, le=6)
    max_retries: int = Field(default=1, ge=0)
    semantic: ReflectionSemanticSettings = ReflectionSemanticSettings()
    correction: ReflectionCorrectionSettings = ReflectionCorrectionSettings()
    tracing: ReflectionTracingSettings = ReflectionTracingSettings()


class RSIFeedbackSettings(SettingsSection):
    enabled: bool = True
    minimum_samples: int = Field(default=10, ge=1, le=1000)
    store_raw_queries: bool = False


class RSIOptimizationSettings(SettingsSection):
    llm_enabled: bool = False
    max_llm_calls_per_cycle: int = Field(default=1, ge=0, le=3)
    allow_reindexing: bool = False
    minimum_quality_improvement: float = Field(default=0.03, ge=0.0, le=0.5)


class RSIValidationSettings(SettingsSection):
    require_independent_holdout: bool = True
    require_live_quality_for_promotion: bool = True
    max_holdout_finalist_evaluations: int = Field(default=1, ge=1, le=3)
    max_latency_regression: float = Field(default=0.20, ge=0.0, le=2.0)
    max_memory_regression: float = Field(default=0.15, ge=0.0, le=2.0)


class RSIStorageSettings(SettingsSection):
    max_cycles_retained: int = Field(default=100, ge=1, le=1000)
    max_storage_mb: int = Field(default=256, ge=1, le=4096)


class RSIPromotionSettings(SettingsSection):
    require_owner_approval: Literal[True] = True
    allow_offline_promotion: bool = False


class RSI(SettingsSection):
    enabled: bool = False
    automatic_promotion: Literal[False] = False
    automatic_rollback: Literal[False] = False
    automatic_scheduling: Literal[False] = False
    require_human_approval: Literal[True] = True
    optimization_strategy: Literal[
        "coordinate_search", "seeded_random_search", "local_refinement", "pareto_selection"
    ] = "coordinate_search"
    max_cycles: int = Field(default=2, ge=1, le=5)
    max_candidates_per_cycle: int = Field(default=3, ge=1, le=10)
    max_experiments: int = Field(default=6, ge=1, le=20)
    max_concurrent_experiments: Literal[1] = 1
    max_execution_seconds: float = Field(default=300.0, gt=0.0, le=3600.0)
    seed: int = Field(default=42, ge=0)
    feedback: RSIFeedbackSettings = RSIFeedbackSettings()
    optimization: RSIOptimizationSettings = RSIOptimizationSettings()
    validation: RSIValidationSettings = RSIValidationSettings()
    storage: RSIStorageSettings = RSIStorageSettings()
    promotion: RSIPromotionSettings = RSIPromotionSettings()

    @field_validator("automatic_promotion", "require_human_approval", mode="before")
    @classmethod
    def parse_boolean(cls, value: object) -> object:
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        return value


class Ingestion(SettingsSection):
    max_file_size_mb: int = Field(default=10, ge=1, le=32)
    max_extracted_text_mb: int = Field(default=20, ge=1, le=64)
    max_batch_documents: int = Field(default=10, ge=1, le=100)
    max_chunks_per_document: int = Field(default=10000, ge=1, le=50000)
    max_structure_items: int = Field(default=50000, ge=1, le=100000)
    max_depth: int = Field(default=32, ge=1, le=64)
    allowed_extensions: tuple[str, ...] = (".txt", ".md", ".csv", ".json")
    whitespace: Literal["preserve", "trim_trailing"] = "preserve"

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, value: object) -> object:
        return json.loads(value) if isinstance(value, str) else value


class Storage(SettingsSection):
    database_path: Path = Path("data/evorag.sqlite3")
    sqlite_wal: bool = True
    busy_timeout_ms: int = Field(default=5000, ge=1, le=30000)
    metadata_backend: Literal["sqlite"] = "sqlite"
    vector_dtype: Literal["float32"] = "float32"
    vector_backend: Literal["custom_memmap"] = "custom_memmap"


class Resources(SettingsSection):
    max_parallel_generations: PositiveInt = 1
    max_loaded_llms: PositiveInt = 1
    minimum_free_ram_gb: float = Field(default=3, gt=0)
    project_storage_budget_gb: float = Field(default=20, gt=0)


class ProviderSettings(SettingsSection):
    connect_timeout_seconds: float = Field(default=5, gt=0, le=30)
    read_timeout_seconds: float = Field(default=120, gt=0, le=600)
    max_connections: int = Field(default=2, ge=1, le=8)
    max_retries: int = Field(default=2, ge=0, le=3)
    max_request_bytes: int = Field(default=262144, ge=1024, le=1048576)
    queue_capacity: int = Field(default=1, ge=0, le=8)
    queue_timeout_seconds: float = Field(default=5, gt=0, le=30)


class EmbeddingCache(SettingsSection):
    enabled: bool = True
    max_size_mb: int = Field(default=256, ge=1, le=4096)


class RagSettings(SettingsSection):
    enabled: bool = True
    default_retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    max_evidence_items: int = Field(default=5, ge=1, le=20)
    allow_partial_answers: bool = True
    require_citations: bool = True
    reranking_enabled: bool = False
    max_output_bytes: int = Field(default=16384, ge=256, le=65536)
    max_claims: int = Field(default=16, ge=1, le=64)


class ContextSettings(SettingsSection):
    max_context_tokens: PositiveInt = 4096
    system_token_budget: PositiveInt = 384
    question_token_budget: PositiveInt = 256
    history_token_budget: PositiveInt = 384
    evidence_token_budget: PositiveInt = 2048
    reserved_output_tokens: PositiveInt = 768
    safety_margin_tokens: PositiveInt = 256
    oversized_question_policy: Literal["reject", "truncate"] = "reject"

    @model_validator(mode="after")
    def check_budgets(self) -> Self:
        allocated = (
            self.system_token_budget
            + self.question_token_budget
            + self.history_token_budget
            + self.evidence_token_budget
            + self.reserved_output_tokens
            + self.safety_margin_tokens
        )
        if allocated > self.max_context_tokens:
            raise ValueError("allocated context token budgets exceed max_context_tokens")
        return self


class PromptSettings(SettingsSection):
    default_template: Name = "grounded_v1"


class CitationSettings(SettingsSection):
    validate_identifiers: bool = True
    reject_unknown_sources: bool = True


class ConversationSettings(SettingsSection):
    enabled: bool = True
    max_history_messages: int = Field(default=6, ge=0, le=32)
    persist_history: Literal[False] = False


class TracingSettings(SettingsSection):
    enabled: bool = True
    store_raw_prompts: bool = False
    store_raw_questions: bool = False
    max_records: int = Field(default=10000, ge=1, le=100_000)


class AdaptiveRagSettings(SettingsSection):
    enabled: bool = True
    default_strategy: Literal["baseline", "adaptive"] = "baseline"
    max_query_variants: int = Field(default=3, ge=1, le=10)
    max_subqueries: int = Field(default=3, ge=1, le=10)
    max_retrieval_executions: int = Field(default=4, ge=1, le=16)
    max_aggregate_candidates: int = Field(default=80, ge=5, le=500)
    max_planning_llm_calls: int = Field(default=1, ge=0, le=3)
    request_deadline_seconds: float = Field(default=30.0, gt=0.0, le=300.0)


class QueryAnalysisSettings(SettingsSection):
    enabled: bool = True
    classifier: Literal["heuristic", "hybrid"] = "heuristic"
    llm_classification_enabled: bool = False


class QueryRewritingSettings(SettingsSection):
    enabled: bool = True
    llm_enabled: bool = False
    preserve_original: bool = True
    prompt_template: Name = "rewrite_v1"
    max_rewrite_length: int = Field(default=512, ge=16, le=4096)


class QueryDecompositionSettings(SettingsSection):
    enabled: bool = True
    llm_enabled: bool = False
    max_depth: Literal[1] = 1
    max_subqueries: int = Field(default=3, ge=1, le=5)
    prompt_template: Name = "decompose_v1"


class MultiQuerySettings(SettingsSection):
    enabled: bool = True
    original_weight: float = Field(default=1.0, gt=0.0, le=10.0)
    rewrite_weight: float = Field(default=0.8, ge=0.0, le=10.0)
    subquery_weight: float = Field(default=0.7, ge=0.0, le=10.0)
    normalize_variant_weights: bool = True
    variant_weight_cap: float = Field(default=1.5, gt=0.0, le=10.0)
    rrf_k: PositiveInt = 60


class RetrievalCacheSettings(SettingsSection):
    enabled: bool = True
    max_entries: int = Field(default=256, ge=1, le=10_000)
    ttl_seconds: float = Field(default=300.0, gt=0.0, le=86400.0)


class EvaluationJudgeSettings(SettingsSection):
    enabled: bool = False
    provider: Literal["ollama"] = "ollama"
    model: Name = "qwen3:4b"
    max_calls_per_case: int = Field(default=1, ge=1, le=5)


class EvaluationStatisticsSettings(SettingsSection):
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    bootstrap_samples: int = Field(default=1000, ge=100, le=10000)
    seed: int = Field(default=42, ge=0)


class EvaluationStorageSettings(SettingsSection):
    max_runs: int = Field(default=100, ge=1, le=1000)
    max_storage_mb: int = Field(default=256, ge=1, le=4096)


class EvaluationSettings(SettingsSection):
    enabled: bool = True
    default_execution_mode: Literal["offline", "synthetic", "live", "human_annotated"] = "offline"
    max_cases_per_run: int = Field(default=200, ge=1, le=1000)
    max_concurrent_runs: Literal[1] = 1
    persist_raw_inputs: bool = False
    judge: EvaluationJudgeSettings = EvaluationJudgeSettings()
    statistics: EvaluationStatisticsSettings = EvaluationStatisticsSettings()
    storage: EvaluationStorageSettings = EvaluationStorageSettings()


class ExperimentsSettings(SettingsSection):
    enabled: bool = True
    automatic_promotion: Literal[False] = False
    max_candidate_configurations: int = Field(default=5, ge=1, le=20)


class Settings(SettingsSection):
    provider: ProviderSettings = ProviderSettings()
    embedding_cache: EmbeddingCache = EmbeddingCache()
    ingestion: Ingestion = Ingestion()
    runtime: ModelConfiguration = ModelConfiguration()
    embeddings: Embeddings = Embeddings()
    chunking: Chunking = Chunking()
    retrieval: Retrieval = Retrieval()
    sparse: Sparse = Sparse()
    fusion: Fusion = Fusion()
    reranking: Reranking = Reranking()
    rag: RagSettings = RagSettings()
    context: ContextSettings = ContextSettings()
    prompts: PromptSettings = PromptSettings()
    citations: CitationSettings = CitationSettings()
    conversation: ConversationSettings = ConversationSettings()
    tracing: TracingSettings = TracingSettings()
    adaptive_rag: AdaptiveRagSettings = AdaptiveRagSettings()
    query_analysis: QueryAnalysisSettings = QueryAnalysisSettings()
    query_rewriting: QueryRewritingSettings = QueryRewritingSettings()
    query_decomposition: QueryDecompositionSettings = QueryDecompositionSettings()
    multi_query: MultiQuerySettings = MultiQuerySettings()
    retrieval_cache: RetrievalCacheSettings = RetrievalCacheSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    experiments: ExperimentsSettings = ExperimentsSettings()
    reflection: Reflection = Reflection()
    rsi: RSI = RSI()
    storage: Storage = Storage()
    resources: Resources = Resources()


def load_settings(
    path: Path | None = None, *, environ: Mapping[str, str] | None = None
) -> Settings:
    env = os.environ if environ is None else environ
    config_file = path or (Path(env["EVORAG_CONFIG"]) if env.get("EVORAG_CONFIG") else None)
    values: dict[str, Any] = {}
    if config_file:
        try:
            with config_file.open("rb") as stream:
                values = tomllib.load(stream)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(
                f"Cannot read configuration file ({type(exc).__name__})"
            ) from None
    for key, value in env.items():
        if key == "EVORAG_CONFIG" or not key.startswith("EVORAG_"):
            continue
        parts = key.removeprefix("EVORAG_").lower().split("__")
        if len(parts) != 2 or parts[0] not in Settings.model_fields:
            raise ConfigurationError("Expected EVORAG_<SECTION>__<FIELD>")
        section = values.setdefault(parts[0], {})
        if not isinstance(section, dict):
            raise ConfigurationError(f"Expected a table for {parts[0]}")
        section[parts[1]] = value
    try:
        return Settings.model_validate(values)
    except ValidationError as exc:
        errors = [f"{'.'.join(map(str, e['loc']))} ({e['type']})" for e in exc.errors()]
        raise ConfigurationError("Invalid configuration: " + ", ".join(errors)) from None
