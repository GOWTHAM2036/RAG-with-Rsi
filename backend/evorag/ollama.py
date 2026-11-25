"""Direct, bounded Ollama HTTP adapter. The caller owns the HTTP client and admission."""

import asyncio
import ipaddress
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager
from time import perf_counter
from typing import Annotated, Any, Literal, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, ValidationError

from .chunking import EstimatedTokens
from .config import Settings
from .inference_types import (
    GenerationEvent,
    GenerationRequest,
    GenerationResponse,
    GenerationUsage,
    InferenceMetrics,
    ModelCapabilities,
    ModelInfo,
    ProviderError,
)
from .provider_payloads import ProviderResponseError, parse_ollama_embeddings

MAX_RESPONSE = 4 * 1024**2
MAX_EVENT = 128 * 1024
UInt = Annotated[int, Field(strict=True, ge=0)]
Label = Annotated[str, Field(strict=True, min_length=1, max_length=512)]


class _Wire(BaseModel):
    model_config = ConfigDict(strict=True)

    @classmethod
    def read(cls, value: object) -> Self:
        try:
            return cls.model_validate(value)
        except ValidationError:
            raise ProviderError("invalid_response") from None


class _Details(_Wire):
    parameter_size: StrictStr | None = None
    quantization_level: StrictStr | None = None


class _Model(_Wire):
    name: Label
    digest: Label
    size: UInt
    size_vram: UInt | None = None
    context_length: UInt | None = None
    details: _Details = _Details()


class _Inventory(_Wire):
    models: list[_Model]


class _Show(_Wire):
    capabilities: list[StrictStr]
    model_info: dict[str, Any] = {}


class _Message(_Wire):
    role: Literal["assistant"]
    content: StrictStr


class _Chat(_Wire):
    model: Label
    message: _Message
    done: StrictBool
    done_reason: Label | None = None
    prompt_eval_count: UInt | None = None
    eval_count: UInt | None = None
    load_duration: UInt | None = None
    prompt_eval_duration: UInt | None = None
    eval_duration: UInt | None = None


def _classify(status: int, detail: object) -> ProviderError:
    message = detail.lower() if isinstance(detail, str) else ""
    if "memory" in message or status == 507:
        return ProviderError("resource_exhaustion")
    if "context" in message or "input length" in message:
        return ProviderError("context_overflow")
    if "does not support" in message:
        return ProviderError("unsupported_capability")
    return ProviderError(
        {
            400: "invalid_request",
            404: "model_not_installed",
            429: "rate_limited",
            503: "provider_unavailable",
        }.get(status, "internal_provider_failure")
    )


def _decode(raw: bytes | bytearray) -> dict[str, Any]:
    try:
        result = json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError):
        raise ProviderError("invalid_response") from None
    if not isinstance(result, dict):
        raise ProviderError("invalid_response")
    return result


def _rate(count: int | None, nanoseconds: int | None) -> float | None:
    return count * 1e9 / nanoseconds if count is not None and nanoseconds else None


def _same_tag(left: str, right: str) -> bool:
    return left.removesuffix(":latest") == right.removesuffix(":latest")


def is_loopback(host: str | None) -> bool:
    name = (host or "").strip("[]")
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


class Ollama:
    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def _url(self, path: str) -> str:
        config = self.settings.runtime
        if config.provider != "ollama":
            raise ProviderError("unsupported_capability")
        if not is_loopback(config.base_url.host):
            raise ProviderError("invalid_request")
        return str(config.base_url).rstrip("/") + "/api/" + path

    @asynccontextmanager
    async def _open(self, path: str, body: dict[str, Any] | None) -> AsyncIterator[httpx.Response]:
        url = self._url(path)
        try:
            async with self.client.stream(
                "GET" if body is None else "POST",
                url,
                json=body,
                headers={"Accept-Encoding": "identity"},
                follow_redirects=False,
            ) as response:
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    raise ProviderError("invalid_response")
                yield response
        except httpx.ConnectTimeout:
            raise ProviderError("connect_timeout") from None
        except httpx.TimeoutException:
            raise ProviderError("read_timeout") from None
        except httpx.RequestError:
            raise ProviderError("provider_unavailable") from None

    async def _bytes(self, response: httpx.Response, limit: int) -> AsyncIterator[bytes]:
        received = 0
        # aiter_bytes also handles buffered transports used by offline fixtures.
        async for block in response.aiter_bytes():
            received += len(block)
            if received > limit:
                raise ProviderError("invalid_response")
            yield block

    async def _read(self, response: httpx.Response, limit: int) -> bytearray:
        data = bytearray()
        async for block in self._bytes(response, limit):
            data.extend(block)
        return data

    async def _check(self, response: httpx.Response) -> None:
        if not response.is_success:
            raw = await self._read(response, 16384)
            try:
                error = _decode(raw).get("error")
            except ProviderError:
                error = None
            raise _classify(response.status_code, error)

    async def _json(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        retries = self.settings.provider.max_retries if path in {"tags", "ps", "show"} else 0
        for attempt in range(retries + 1):
            delay = 0.05 * (2**attempt)
            try:
                async with asyncio.timeout(self.settings.provider.read_timeout_seconds):
                    async with self._open(path, body) as response:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            # Unknown or long retry windows are returned to the caller immediately.
                            delay = (
                                float(retry_after)
                                if retry_after.isdecimal() and len(retry_after) < 4
                                else 999
                            )
                        await self._check(response)
                        value = _decode(await self._read(response, MAX_RESPONSE))
                        if "error" in value:
                            raise _classify(500, value["error"])
                        return value
            except TimeoutError:
                raise ProviderError("read_timeout") from None
            except ProviderError as exc:
                if (
                    attempt == retries
                    or delay > 2
                    or exc.code not in {"provider_unavailable", "connect_timeout", "rate_limited"}
                ):
                    raise
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def _inventory(self, path: str) -> tuple[ModelInfo, ...]:
        response = _Inventory.read(await self._json(path))
        return tuple(
            ModelInfo(
                name=row.name,
                digest=row.digest,
                size_bytes=row.size,
                vram_bytes=row.size_vram,
                parameters=row.details.parameter_size,
                quantization=row.details.quantization_level,
                context_length=row.context_length,
            )
            for row in response.models
        )

    async def models(self) -> tuple[ModelInfo, ...]:
        return await self._inventory("tags")

    async def resident(self) -> tuple[ModelInfo, ...]:
        return await self._inventory("ps")

    async def show(self, name: str) -> ModelCapabilities:
        row = _Show.read(await self._json("show", {"model": name}))
        arch = row.model_info.get("general.architecture")

        def integer(key: str) -> int | None:
            value = row.model_info.get(f"{arch}.{key}")
            if value is not None and (type(value) is not int or value < 1):
                raise ProviderError("invalid_response")
            return int(value) if value is not None else None

        return ModelCapabilities(
            generation="completion" in row.capabilities,
            embedding="embedding" in row.capabilities,
            thinking="thinking" in row.capabilities,
            context_length=integer("context_length"),
            embedding_dimensions=integer("embedding_length"),
        )

    def _chat_body(self, req: GenerationRequest, stream: bool) -> tuple[dict[str, Any], int]:
        cfg = self.settings.runtime
        context = req.context_window or cfg.context_window
        output = req.max_output_tokens or cfg.max_output_tokens
        estimate = sum(EstimatedTokens().count(m.content) + 4 for m in req.messages) + 2
        if (
            context > cfg.context_window
            or output > cfg.max_output_tokens
            or estimate + output > context
        ):
            raise ProviderError("context_overflow")
        return {
            "model": cfg.generation_model,
            "messages": [m.model_dump() for m in req.messages],
            "stream": stream,
            "think": False,
            "keep_alive": 0,
            "options": {
                "temperature": req.temperature if req.temperature is not None else cfg.temperature,
                "num_ctx": context,
                "num_predict": output,
                "stop": list(req.stop),
            },
        }, estimate

    def _event(
        self, raw: dict[str, Any], estimated: int, start: float, first: float | None
    ) -> GenerationEvent:
        if "error" in raw:
            raise _classify(500, raw["error"])
        chat = _Chat.read(raw)
        if not _same_tag(chat.model, self.settings.runtime.generation_model):
            raise ProviderError("invalid_response")
        content = chat.message.content
        # Dedicated message.thinking and tool_calls fields are intentionally never exposed.
        return GenerationEvent(
            model=chat.model,
            content=content,
            done=chat.done,
            finish_reason=chat.done_reason,
            usage=GenerationUsage(
                estimated_input_tokens=estimated,
                prompt_tokens=chat.prompt_eval_count,
                completion_tokens=chat.eval_count,
            )
            if chat.done
            else None,
            metrics=InferenceMetrics(
                elapsed_ms=(perf_counter() - start) * 1000,
                first_token_ms=first,
                load_ms=chat.load_duration / 1e6 if chat.load_duration is not None else None,
                prompt_tokens_per_second=_rate(chat.prompt_eval_count, chat.prompt_eval_duration),
                generation_tokens_per_second=_rate(chat.eval_count, chat.eval_duration),
            )
            if chat.done
            else None,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        body, estimated = self._chat_body(request, False)
        start = perf_counter()
        event = self._event(await self._json("chat", body), estimated, start, None)
        if not event.done or event.usage is None or event.metrics is None:
            raise ProviderError("invalid_response")
        return GenerationResponse(
            model=event.model,
            content=event.content,
            finish_reason=event.finish_reason,
            usage=event.usage,
            metrics=event.metrics,
        )

    async def _lines(self, response: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
        buffer = bytearray()
        async for block in self._bytes(response, MAX_RESPONSE):
            for i, part in enumerate(block.split(b"\n")):
                if i:
                    if buffer.strip():
                        yield _decode(buffer)
                    buffer.clear()
                if len(buffer) + len(part) > MAX_EVENT:
                    raise ProviderError("invalid_response")
                buffer.extend(part)
        if buffer.strip():
            yield _decode(buffer)

    async def stream(self, request: GenerationRequest) -> AsyncGenerator[GenerationEvent, None]:
        body, estimated = self._chat_body(request, True)
        start = perf_counter()
        first = None
        terminal = None
        async with self._open("chat", body) as response:
            await self._check(response)
            async with aclosing(self._lines(response)) as lines:
                while True:
                    try:
                        async with asyncio.timeout_at(
                            asyncio.get_running_loop().time()
                            + max(
                                0,
                                self.settings.provider.read_timeout_seconds
                                - (perf_counter() - start),
                            )
                        ):
                            raw = await anext(lines)
                    except StopAsyncIteration:
                        break
                    except TimeoutError:
                        raise ProviderError("read_timeout") from None
                    if terminal is not None:
                        raise ProviderError("invalid_response")
                    event = self._event(raw, estimated, start, first)
                    if event.content and first is None:
                        first = (perf_counter() - start) * 1000
                    if event.done:
                        terminal = event
                    elif event.content:
                        yield event
        if terminal is None:
            raise ProviderError("invalid_response")
        yield terminal

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        cfg = self.settings.embeddings
        if not 1 <= len(texts) <= cfg.batch_size or any(not text.strip() for text in texts):
            raise ProviderError("invalid_request")
        if any(
            EstimatedTokens().count(text) > self.settings.runtime.context_window for text in texts
        ):
            raise ProviderError("context_overflow")
        options = {"num_gpu": 0} if cfg.device == "cpu" else {}
        raw = await self._json(
            "embed",
            {
                "model": self.settings.runtime.embedding_model,
                "input": texts,
                "truncate": False,
                "keep_alive": 0,
                "options": options,
            },
        )
        try:
            rows = parse_ollama_embeddings(raw, [str(i) for i in range(len(texts))], self.settings)
        except (ProviderResponseError, ValidationError):
            raise ProviderError("invalid_response") from None
        return tuple(row.values for row in rows)
