"""Synthetic Ollama protocol fixtures; these responses are not live inference results."""

import json
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, asynccontextmanager

import httpx
import pytest
from evorag.config import Settings
from evorag.inference_types import GenerationRequest, Message, ProviderError
from evorag.ollama import Ollama, is_loopback

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@asynccontextmanager
async def adapter(
    handler: Callable[[httpx.Request], httpx.Response], settings: Settings | None = None
) -> AsyncIterator[Ollama]:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        yield Ollama(client, settings or Settings())


REQUEST = GenerationRequest(messages=(Message(role="user", content="test"),))
GEN = "qwen3:4b-instruct"
EMBED = "qwen3-embedding:0.6b"


def frame(content: str, done: bool) -> dict[str, object]:
    return {
        "model": GEN,
        "message": {"role": "assistant", "content": content, "thinking": "SECRET"},
        "done": done,
        "eval_count": 2,
        "prompt_eval_count": 3,
        "eval_duration": 500000000,
        "prompt_eval_duration": 1000000000,
        "load_duration": 500000,
        "done_reason": "stop" if done else None,
    }


class Blocks(httpx.AsyncByteStream):
    def __init__(self, *blocks: bytes) -> None:
        self.blocks = blocks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for block in self.blocks:
            yield block

    async def aclose(self) -> None:
        self.closed = True


async def test_discovery_and_metadata() -> None:
    def handle(r: httpx.Request) -> httpx.Response:
        if r.url.path.endswith("/show"):
            assert json.loads(r.content) == {"model": GEN}
            return httpx.Response(
                200,
                json={
                    "capabilities": ["completion", "thinking"],
                    "model_info": {"general.architecture": "qwen3", "qwen3.context_length": 4096},
                },
            )
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": GEN,
                        "digest": "abc",
                        "size": 100,
                        "details": {"parameter_size": "4B", "quantization_level": "Q4_K_M"},
                        "size_vram": 50,
                    }
                ]
            },
        )

    async with adapter(handle) as p:
        assert (await p.models())[0].quantization == "Q4_K_M"
        assert (await p.resident())[0].vram_bytes == 50
        caps = await p.show(GEN)
        assert caps.generation and caps.thinking and not caps.embedding
        assert caps.context_length == 4096 and caps.embedding_dimensions is None


async def test_generate_usage_and_request() -> None:
    def handle(r: httpx.Request) -> httpx.Response:
        body = json.loads(r.content)
        assert body["messages"] == [{"role": "user", "content": "test"}]
        assert body["think"] is False and body["stream"] is False and body["keep_alive"] == 0
        assert body["options"] == {
            "temperature": 0,
            "num_ctx": 512,
            "num_predict": 32,
            "stop": ["END"],
        }
        return httpx.Response(200, json=frame("visible", True))

    async with adapter(handle) as p:
        response = await p.generate(
            GenerationRequest(
                messages=REQUEST.messages,
                temperature=0,
                context_window=512,
                max_output_tokens=32,
                stop=("END",),
            )
        )
        assert response.content == "visible" and response.finish_reason == "stop"
        assert response.usage.prompt_tokens == 3 and response.usage.estimated_input_tokens == 7
        assert (
            response.metrics.load_ms == 0.5 and response.metrics.generation_tokens_per_second == 4
        )
        assert "SECRET" not in response.model_dump_json()
        with pytest.raises(ProviderError, match="context_overflow"):
            await p.generate(GenerationRequest(messages=REQUEST.messages, context_window=2))


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (httpx.Response(200, content=b"{"), "invalid_response"),
        (httpx.Response(200, json=[]), "invalid_response"),
        (httpx.Response(200, json={}), "invalid_response"),
        (httpx.Response(200, json=frame("x", False)), "invalid_response"),
        (httpx.Response(200, json={**frame("x", True), "model": "other"}), "invalid_response"),
        (httpx.Response(200, json={"error": "out of memory SECRET"}), "resource_exhaustion"),
        (httpx.Response(400, json={"error": "context too long SECRET"}), "context_overflow"),
        (
            httpx.Response(400, json={"error": "model does not support think SECRET"}),
            "unsupported_capability",
        ),
        (httpx.Response(400, json={"error": "invalid SECRET"}), "invalid_request"),
        (httpx.Response(404, content=b"SECRET"), "model_not_installed"),
        (httpx.Response(429, json={}), "rate_limited"),
        (httpx.Response(500, json={}), "internal_provider_failure"),
        (
            httpx.Response(302, headers={"location": "https://example.com"}),
            "internal_provider_failure",
        ),
        (httpx.Response(200, headers={"content-encoding": "custom"}), "invalid_response"),
    ],
)
async def test_bad_responses(response: httpx.Response, error: str) -> None:
    async with adapter(lambda _: response) as p:
        with pytest.raises(ProviderError, match=error) as exc:
            await p.generate(REQUEST)
        assert "SECRET" not in str(exc.value)


async def test_fragmented_stream_and_closure() -> None:
    raw = (
        json.dumps(frame("猫", False), ensure_ascii=False).encode()
        + b"\r\n\n"
        + json.dumps(frame("", True)).encode()
    )
    cut = raw.index("猫".encode()) + 1
    blocks = Blocks(raw[:cut], raw[cut : cut + 1], raw[cut + 1 :])
    async with adapter(lambda r: httpx.Response(200, stream=blocks)) as p:
        events = [e async for e in p.stream(REQUEST)]
        assert [e.content for e in events] == ["猫", ""]
        assert events[-1].metrics and events[-1].metrics.first_token_ms is not None
        assert events[-1].done and blocks.closed


@pytest.mark.parametrize(
    "tail", [b"", b"{", b"\xff\n", json.dumps(frame("", True)).encode() + b"\n{}\n"]
)
async def test_stream_corruption_never_replays(tail: bytes) -> None:
    calls = 0

    def handle(r: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, stream=Blocks(json.dumps(frame("x", False)).encode() + b"\n", tail)
        )

    async with adapter(handle) as p:
        stream = p.stream(REQUEST)
        assert (await anext(stream)).content == "x"
        with pytest.raises(ProviderError, match="invalid_response"):
            await anext(stream)
        assert calls == 1


async def test_partial_consumer_cleanup() -> None:
    blocks = Blocks(json.dumps(frame("x", False)).encode() + b"\n", b"{}\n")
    async with adapter(lambda r: httpx.Response(200, stream=blocks)) as p:
        async with aclosing(p.stream(REQUEST)) as stream:
            await anext(stream)
        assert blocks.closed


async def test_size_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evorag.ollama.MAX_EVENT", 8)
    async with adapter(lambda r: httpx.Response(200, content=b"z" * 10)) as p:
        with pytest.raises(ProviderError, match="invalid_response"):
            await anext(p.stream(REQUEST))
    monkeypatch.setattr("evorag.ollama.MAX_RESPONSE", 8)
    async with adapter(lambda r: httpx.Response(200, content=b"z" * 10)) as p:
        with pytest.raises(ProviderError, match="invalid_response"):
            await p.models()


async def test_retry_only_metadata() -> None:
    calls = 0

    def handle(r: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("private")
        if calls == 2:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"models": []})

    async with adapter(handle) as p:
        assert await p.models() == () and calls == 3
    async with adapter(lambda r: httpx.Response(429, headers={"retry-after": "100"})) as p:
        with pytest.raises(ProviderError, match="rate_limited"):
            await p.models()


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (httpx.ConnectTimeout, "connect_timeout"),
        (httpx.ReadTimeout, "read_timeout"),
        (httpx.ConnectError, "provider_unavailable"),
    ],
)
async def test_timeouts(failure: type[httpx.HTTPError], code: str) -> None:
    def handle(r: httpx.Request) -> httpx.Response:
        raise failure("SECRET")

    async with adapter(handle) as p:
        with pytest.raises(ProviderError, match=code):
            await p.generate(REQUEST)


def handler_for(response: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _: response


async def test_embedding_validation() -> None:
    def handle(r: httpx.Request) -> httpx.Response:
        body = json.loads(r.content)
        assert body["input"] == ["a", "b"] and body["options"] == {"num_gpu": 0}
        assert body["truncate"] is False
        return httpx.Response(200, json={"model": EMBED, "embeddings": [[1, 2], [3, 4]]})

    settings = Settings.model_validate({"embeddings": {"dimensions": 2}})
    async with adapter(handle, settings) as p:
        assert await p.embed(("a", "b")) == ((1, 2), (3, 4))
        with pytest.raises(ProviderError, match="invalid_request"):
            await p.embed(("",))
        with pytest.raises(ProviderError, match="context_overflow"):
            await p.embed(("a" * 20000,))
    for bad in [[[1, 2], [True, 3]], [[1, 2]], [[1, 2], [1]]]:
        response = httpx.Response(200, json={"model": EMBED, "embeddings": bad})
        async with adapter(handler_for(response), settings) as p:
            with pytest.raises(ProviderError, match="invalid_response"):
                await p.embed(("a", "b"))


async def test_show_invalid_metadata_and_trusted_endpoints() -> None:
    assert is_loopback("[::1]") and is_loopback("127.1.2.3") and is_loopback("localhost")
    assert not is_loopback("foo.example") and not is_loopback("8.8.8.8")
    async with adapter(
        lambda r: httpx.Response(
            200,
            json={
                "capabilities": [],
                "model_info": {"general.architecture": "x", "x.context_length": "4096"},
            },
        )
    ) as p:
        with pytest.raises(ProviderError, match="invalid_response"):
            await p.show("x")
    for runtime, code in [
        ({"provider": "lm_studio"}, "unsupported_capability"),
        ({"base_url": "https://example.com"}, "invalid_request"),
    ]:
        async with adapter(
            lambda r: pytest.fail("unexpected network request"),
            Settings.model_validate({"runtime": runtime}),
        ) as p:
            with pytest.raises(ProviderError, match=code):
                await p.models()
