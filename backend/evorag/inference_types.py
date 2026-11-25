"""Provider-neutral inference contracts and independently implementable capability ports."""

from collections.abc import AsyncIterator
from typing import Annotated, Literal, Protocol

from pydantic import Field, PositiveInt

from .contracts import Contract
from .errors import EvoRagError


class Message(Contract):
    role: Literal["system", "user", "assistant"]
    content: Annotated[str, Field(min_length=1, max_length=65536, pattern=r"\S")]


class GenerationRequest(Contract):
    messages: Annotated[tuple[Message, ...], Field(min_length=1, max_length=64)]
    temperature: Annotated[float, Field(ge=0, le=2)] | None = None
    context_window: PositiveInt | None = None
    max_output_tokens: PositiveInt | None = None
    stop: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=128)], ...], Field(max_length=8)
    ] = ()


class GenerationUsage(Contract):
    estimated_input_tokens: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class InferenceMetrics(Contract):
    elapsed_ms: float
    first_token_ms: float | None = None
    load_ms: float | None = None
    prompt_tokens_per_second: float | None = None
    generation_tokens_per_second: float | None = None


class GenerationResponse(Contract):
    model: str
    content: str
    finish_reason: str | None
    usage: GenerationUsage
    metrics: InferenceMetrics


class GenerationEvent(Contract):
    model: str
    content: str
    done: bool
    finish_reason: str | None = None
    usage: GenerationUsage | None = None
    metrics: InferenceMetrics | None = None


class ModelInfo(Contract):
    name: str
    digest: str
    size_bytes: int
    vram_bytes: int | None = None
    quantization: str | None = None
    parameters: str | None = None
    context_length: int | None = None


class ModelCapabilities(Contract):
    generation: bool
    embedding: bool
    thinking: bool
    context_length: int | None = None
    embedding_dimensions: int | None = None


class ProviderHealth(Contract):
    reachable: bool
    generation_installed: bool | None = None
    embedding_installed: bool | None = None


class ProviderError(EvoRagError):
    """Only public error codes belong in this exception."""

    def __init__(self, code: str, status: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.status = status or {
            "provider_unavailable": 503,
            "connect_timeout": 504,
            "read_timeout": 504,
            "model_not_installed": 404,
            "unsupported_capability": 422,
            "invalid_request": 422,
            "resource_exhaustion": 503,
            "context_overflow": 422,
            "rate_limited": 429,
            "request_cancelled": 409,
        }.get(code, 502)


class TextGenerator(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResponse: ...


class StreamGenerator(Protocol):
    def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationEvent]: ...


class ModelDiscovery(Protocol):
    async def models(self) -> tuple[ModelInfo, ...]: ...
    async def resident(self) -> tuple[ModelInfo, ...]: ...
    async def show(self, name: str) -> ModelCapabilities: ...


class TextEmbedder(Protocol):
    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]: ...
