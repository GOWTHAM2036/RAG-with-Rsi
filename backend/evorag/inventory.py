"""A bounded, metadata-only provider probe for readiness."""

import asyncio
from typing import Literal

import httpx
from pydantic import BaseModel, StrictStr, ValidationError

from .config import ModelConfiguration


class ModelAvailability(BaseModel):
    inventory: Literal["available", "unreachable", "invalid", "unsupported", "busy"]
    generation_installed: bool | None = None
    embedding_installed: bool | None = None


class _Tag(BaseModel):
    name: StrictStr


class _Inventory(BaseModel):
    models: list[_Tag]


def _explicit_tag(name: str) -> str:
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


async def inspect_models(
    client: httpx.AsyncClient, config: ModelConfiguration
) -> ModelAvailability:
    if config.provider != "ollama":
        return ModelAvailability(inventory="unsupported")
    endpoint = str(config.base_url).rstrip("/") + "/api/tags"
    try:
        async with (
            asyncio.timeout(2),
            client.stream("GET", endpoint, headers={"Accept-Encoding": "identity"}) as response,
        ):
            response.raise_for_status()
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                return ModelAvailability(inventory="invalid")
            data = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=16384):
                if len(data) + len(chunk) > 256 * 1024:
                    return ModelAvailability(inventory="invalid")
                data.extend(chunk)
        tags = {tag.name for tag in _Inventory.model_validate_json(bytes(data)).models}
    except (httpx.HTTPError, TimeoutError):
        return ModelAvailability(inventory="unreachable")
    except ValidationError:
        return ModelAvailability(inventory="invalid")
    return ModelAvailability(
        inventory="available",
        generation_installed=_explicit_tag(config.generation_model) in tags,
        embedding_installed=_explicit_tag(config.embedding_model) in tags,
    )
