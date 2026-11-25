"""Serialize inference against a shared daemon; measure capacity before each model request."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing, asynccontextmanager
from typing import Literal, Protocol

from .config import Settings
from .inference_types import (
    GenerationEvent,
    GenerationRequest,
    GenerationResponse,
    ModelDiscovery,
    ModelInfo,
    ProviderError,
    TextGenerator,
)
from .resources import GIB, available_ram


class Runtime(ModelDiscovery, TextGenerator, Protocol):
    def stream(self, request: GenerationRequest) -> AsyncGenerator[GenerationEvent, None]: ...


class Coordinator:
    def __init__(
        self,
        settings: Settings,
        provider: Runtime,
        ram_probe: Callable[[], int | None] = available_ram,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.ram_probe = ram_probe
        self.lock = asyncio.Lock()
        self.waiting = 0

    async def _inspect(self, name: str) -> tuple[ModelInfo | None, dict[str, object]]:
        installed = await self.provider.models()
        tag = name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"
        model = next((row for row in installed if row.name == tag), None)
        available = self.ram_probe()
        required = None
        reason = "model_not_installed"
        if model is not None:
            # Conservative CPU-fallback estimate: weights, scratch, context and OS headroom.
            required = (
                int(max(3, self.settings.resources.minimum_free_ram_gb) * GIB)
                + model.size_bytes
                + GIB // 2
                + self.settings.runtime.context_window * 131072
            )
            if model.size_bytes <= 0 or available is None or available < required:
                reason = "insufficient_memory"
            elif await self.provider.resident():
                reason = "model_already_resident"
            else:
                reason = "admitted"
        return model, {
            "admitted": reason == "admitted",
            "reason": reason,
            "available_ram_bytes": available,
            "required_ram_bytes": required,
        }

    async def admission(self, model_name: str) -> dict[str, object]:
        _, report = await self._inspect(model_name)
        if self.lock.locked():
            report.update(admitted=False, reason="inference_busy")
        return report

    @asynccontextmanager
    async def reserve(
        self, model_name: str, capability: Literal["generation", "embedding"]
    ) -> AsyncIterator[ModelInfo]:
        if self.lock.locked() and self.waiting >= self.settings.provider.queue_capacity:
            raise ProviderError("inference_queue_full", 429)
        self.waiting += 1
        try:
            async with asyncio.timeout(self.settings.provider.queue_timeout_seconds):
                await self.lock.acquire()
        except TimeoutError:
            raise ProviderError("inference_queue_timeout", 503) from None
        finally:
            self.waiting -= 1
        try:
            model, report = await self._inspect(model_name)
            if model is None:
                raise ProviderError("model_not_installed")
            if not report["admitted"]:
                raise ProviderError(str(report["reason"]), 503)
            info = await self.provider.show(model.name)
            if not (info.embedding if capability == "embedding" else info.generation):
                raise ProviderError("unsupported_capability")
            yield model
        finally:
            self.lock.release()

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        async with self.reserve(self.settings.runtime.generation_model, "generation"):
            return await self.provider.generate(request)

    async def stream(self, request: GenerationRequest) -> AsyncGenerator[GenerationEvent, None]:
        async with self.reserve(self.settings.runtime.generation_model, "generation"):
            async with aclosing(self.provider.stream(request)) as stream:
                async for event in stream:
                    yield event
