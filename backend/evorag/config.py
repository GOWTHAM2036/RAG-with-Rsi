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
    generation_model: Name = "qwen3:4b-instruct"
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


class Reflection(SettingsSection):
    enabled: bool = False
    max_retries: int = Field(default=1, ge=0)


class RSI(SettingsSection):
    enabled: bool = False
    automatic_promotion: Literal[False] = False
    require_human_approval: Literal[True] = True

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
