import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import httpx
import pytest
from evorag.config import ModelConfiguration, Settings
from evorag.errors import ConfigurationError
from evorag.main import create_app
from fastapi import FastAPI

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def local_client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@asynccontextmanager
async def running_client(
    handler: Callable[[httpx.Request], httpx.Response], settings: Settings | None = None
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings or Settings(), transport=httpx.MockTransport(handler))
    async with app.router.lifespan_context(app), local_client(app) as client:
        yield client


async def test_lifecycle_and_exact_tags() -> None:
    calls: list[str] = []
    settings = Settings()

    def inventory(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": settings.runtime.generation_model},
                    {"name": "different-embedding:latest"},
                ]
            },
        )

    app = create_app(settings, transport=httpx.MockTransport(inventory))
    async with local_client(app) as client:
        assert (await client.get("/ready")).status_code == 503
        async with app.router.lifespan_context(app):
            assert (await client.get("/health")).json() == {"status": "ok"}
            assert calls == []
            ready = await client.get("/ready")
            assert ready.status_code == 200
            assert ready.json() == {
                "application": "ready",
                "models": {
                    "inventory": "available",
                    "generation_installed": True,
                    "embedding_installed": False,
                },
            }
            assert calls == ["GET /api/tags"]
        assert (await client.get("/ready")).status_code == 503


@pytest.mark.parametrize(
    ("response", "state"),
    [
        (httpx.Response(503), "unreachable"),
        (httpx.Response(302), "unreachable"),
        (httpx.Response(200, content=b"{"), "invalid"),
        (httpx.Response(200, json={"models": [{"tag": "x"}]}), "invalid"),
        (httpx.Response(200, content=b" " * (256 * 1024 + 1)), "invalid"),
        (httpx.Response(200, headers={"content-encoding": "custom"}), "invalid"),
    ],
)
async def test_bad_inventory(response: httpx.Response, state: str) -> None:
    async with running_client(lambda request: response) as client:
        ready = await client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["models"] == {
            "inventory": state,
            "generation_installed": None,
            "embedding_installed": None,
        }


async def test_offline_without_daemon() -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    async with running_client(unavailable) as client:
        assert (await client.get("/ready")).json()["models"]["inventory"] == "unreachable"
        assert (await client.get("/health")).status_code == 200


async def test_alternative_provider_skips_probe() -> None:
    def unexpected(request: httpx.Request) -> httpx.Response:
        pytest.fail("unexpected provider request")

    settings = Settings(runtime=ModelConfiguration(provider="lm_studio"))
    async with running_client(unexpected, settings) as client:
        assert (await client.get("/ready")).json()["models"]["inventory"] == "unsupported"


async def test_invalid_configuration_prevents_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVORAG_RUNTIME__CONTEXT_WINDOW", "0")
    app = create_app()
    with pytest.raises(ConfigurationError):
        async with app.router.lifespan_context(app):
            pytest.fail("invalid configuration must prevent startup")


async def test_parallel_readiness_probes_are_bounded() -> None:
    entered = asyncio.Event()
    resume = asyncio.Event()

    async def delayed(request: httpx.Request) -> httpx.Response:
        entered.set()
        await resume.wait()
        return httpx.Response(200, json={"models": []})

    app = create_app(Settings(), transport=httpx.MockTransport(delayed))
    async with app.router.lifespan_context(app), local_client(app) as client:
        pending = asyncio.create_task(client.get("/ready"))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            assert (await client.get("/ready")).json()["models"]["inventory"] == "busy"
        finally:
            resume.set()
            response = await pending
        assert response.json()["models"]["generation_installed"] is False
