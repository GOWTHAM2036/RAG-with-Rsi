"""Validation boundary for an Ollama /api/embed response, before any index write."""

from collections.abc import Sequence

from pydantic import BaseModel

from .config import Settings
from .contracts import Embedding
from .errors import EvoRagError


class ProviderResponseError(EvoRagError):
    """A provider returned a payload incompatible with the request."""


class _EmbedResponse(BaseModel):
    model: str
    embeddings: list[list[object]]


def parse_ollama_embeddings(
    payload: object, chunk_ids: Sequence[str], settings: Settings
) -> tuple[Embedding, ...]:
    """Pair rows in provider order with the requested IDs and configured width."""
    response = _EmbedResponse.model_validate(payload)
    if response.model.removesuffix(":latest") != settings.runtime.embedding_model.removesuffix(
        ":latest"
    ):
        raise ProviderResponseError("Unexpected embedding model")
    if len(response.embeddings) != len(chunk_ids):
        raise ProviderResponseError("Unexpected number of returned embeddings")
    return tuple(
        Embedding.model_validate(
            {
                "chunk_id": chunk_id,
                "model": response.model,
                "dimensions": settings.embeddings.dimensions,
                "values": row,
            }
        )
        for chunk_id, row in zip(chunk_ids, response.embeddings, strict=True)
    )
