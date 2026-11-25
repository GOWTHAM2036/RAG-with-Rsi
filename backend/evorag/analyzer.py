"""Deterministic Unicode-aware lexical analyzer for sparse indexing and querying.

Limitations:
    Tokenization uses Unicode word boundaries and technical-identifier patterns
    (hyphens, underscores, dots, slashes). Languages without whitespace word
    boundaries (such as Chinese, Japanese, or Thai) are not segmented into
    dictionary words by this analyzer.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from .config import Sparse

DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)

_COMPOUND_RE = re.compile(r"/?[^\W_]+(?:[._/-]+[^\W_]+)*")
_SUBTOKEN_RE = re.compile(r"[^\W_]+")
_MAX_INPUT_CHARS = 1_000_000
_MAX_CHUNK_TOKENS = 8192


@dataclass(frozen=True)
class AnalyzedText:
    tokens: tuple[str, ...]
    term_frequencies: dict[str, int]
    length: int
    analyzer_version: str


def is_technical_identifier(token: str) -> bool:
    """Return True if a normalized token looks like a code/version/error identifier."""
    has_digit = any(ch.isdigit() for ch in token)
    has_sep = any(ch in "._/-" for ch in token)
    return has_digit or has_sep


class TextAnalyzer:
    """Versioned, deterministic text analyzer shared by indexing and querying."""

    def __init__(
        self,
        *,
        version: str = "v1",
        preserve_compound_identifiers: bool = True,
        remove_stopwords: bool = False,
        stopwords: frozenset[str] = DEFAULT_STOPWORDS,
        max_query_tokens: int = 128,
        max_chunk_tokens: int = _MAX_CHUNK_TOKENS,
    ) -> None:
        if version != "v1":
            raise ValueError(f"unsupported analyzer version: {version}")
        if max_query_tokens < 1 or max_chunk_tokens < 1:
            raise ValueError("token limits must be positive")
        self.version = version
        self.preserve_compound_identifiers = preserve_compound_identifiers
        self.remove_stopwords = remove_stopwords
        self.stopwords = stopwords
        self.max_query_tokens = max_query_tokens
        self.max_chunk_tokens = max_chunk_tokens

    @classmethod
    def from_config(cls, config: Sparse) -> "TextAnalyzer":
        return cls(
            version=config.analyzer_version,
            preserve_compound_identifiers=config.preserve_compound_identifiers,
            remove_stopwords=config.remove_stopwords,
            max_query_tokens=config.max_query_tokens,
        )

    def normalize(self, text: str) -> str:
        bounded = text[:_MAX_INPUT_CHARS]
        return unicodedata.normalize("NFKC", bounded).casefold()

    def tokenize(self, text: str, *, for_query: bool = False) -> tuple[str, ...]:
        if not text:
            return ()
        normalized = self.normalize(text)
        limit = self.max_query_tokens if for_query else self.max_chunk_tokens
        tokens: list[str] = []
        for match in _COMPOUND_RE.finditer(normalized):
            raw = match.group(0)
            subtokens = _SUBTOKEN_RE.findall(raw)
            is_compound = raw.startswith("/") or len(subtokens) > 1
            if self.preserve_compound_identifiers and is_compound:
                tokens.append(raw)
                if len(tokens) >= limit:
                    break
            for sub in subtokens:
                if self.remove_stopwords and sub in self.stopwords:
                    continue
                tokens.append(sub)
                if len(tokens) >= limit:
                    break
            if len(tokens) >= limit:
                break
        return tuple(tokens[:limit])

    def analyze(self, text: str, *, for_query: bool = False) -> AnalyzedText:
        tokens = self.tokenize(text, for_query=for_query)
        counts = dict(Counter(tokens))
        return AnalyzedText(
            tokens=tokens,
            term_frequencies=counts,
            length=len(tokens),
            analyzer_version=self.version,
        )
