"""Bounded in-memory retrieval cache with corpus-revision invalidation and isolation."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from threading import Lock

from .config import RetrievalCacheSettings, Settings
from .filtering import MetadataFilter
from .hybrid_retrieval import RetrievalMode, RetrieveResponse
from .query_types import RetrievalCacheEntry


def build_config_fingerprint(settings: Settings) -> str:
    """Build a deterministic fingerprint of retrieval, sparse, fusion and reranking settings."""
    payload = {
        "retrieval": settings.retrieval.model_dump(mode="json"),
        "sparse": settings.sparse.model_dump(mode="json"),
        "fusion": settings.fusion.model_dump(mode="json"),
        "reranking": settings.reranking.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_cache_key(
    *,
    normalized_query: str,
    retrieval_mode: RetrievalMode,
    config_fingerprint: str,
    analyzer_version: str,
    index_generation: int,
    embedding_space_id: str | None,
    filters: MetadataFilter,
    document_revision_token: str,
    reranking_enabled: bool,
    dense_weight_override: float | None = None,
    sparse_weight_override: float | None = None,
) -> str:
    """Compute a collision-resistant cache key isolating modes, revisions, and filters."""
    payload = {
        "q": normalized_query,
        "mode": retrieval_mode,
        "cfg": config_fingerprint,
        "analyzer": analyzer_version,
        "idx_gen": index_generation,
        "emb_space": embedding_space_id or "",
        "filters": filters.model_dump(mode="json"),
        "doc_rev": document_revision_token,
        "rerank": reranking_enabled,
        "dw": dense_weight_override,
        "sw": sparse_weight_override,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RetrievalCache:
    """Thread-safe bounded LRU+TTL in-memory cache for non-degraded retrieval responses."""

    def __init__(self, settings: RetrievalCacheSettings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._entries: OrderedDict[str, tuple[RetrievalCacheEntry, RetrieveResponse]] = (
            OrderedDict()
        )
        self._hits = 0
        self._misses = 0

    def get(self, cache_key: str) -> RetrieveResponse | None:
        if not self.settings.enabled:
            return None
        now = time.monotonic()
        with self._lock:
            item = self._entries.get(cache_key)
            if item is None:
                self._misses += 1
                return None
            meta, response = item
            if (now - meta.created_at_monotonic) > meta.ttl_seconds:
                del self._entries[cache_key]
                self._misses += 1
                return None
            self._entries.move_to_end(cache_key)
            self._hits += 1
            return response

    def put(self, entry: RetrievalCacheEntry, response: RetrieveResponse) -> None:
        """Cache only non-degraded responses; never cache degraded sparse fallbacks or errors."""
        if not self.settings.enabled or response.degraded:
            return
        with self._lock:
            self._entries[entry.cache_key] = (entry, response)
            self._entries.move_to_end(entry.cache_key)
            while len(self._entries) > self.settings.max_entries:
                self._entries.popitem(last=False)

    def invalidate(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "entries": len(self._entries),
                "max_entries": self.settings.max_entries,
            }
