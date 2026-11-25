"""Bounded windows over normalized Unicode text. All offsets are half-open."""

import re
from bisect import bisect_right
from collections.abc import Iterator
from typing import Protocol

from .config import Chunking, Ingestion
from .documents import SourceSpan, StoredChunk, digest
from .errors import IngestionError
from .parsing import Parsed


class TokenBudget(Protocol):
    identifier: str
    estimated: bool

    def end(self, text: str, start: int, stop: int, tokens: int) -> int: ...
    def overlap_start(self, text: str, start: int, end: int, tokens: int) -> int: ...
    def count(self, text: str) -> int: ...


class EstimatedTokens:
    identifier = "four-codepoints-v1"
    estimated = True

    def end(self, text: str, start: int, stop: int, tokens: int) -> int:
        return min(stop, start + 4 * tokens)

    def overlap_start(self, text: str, start: int, end: int, tokens: int) -> int:
        return max(start, end - 4 * tokens)

    def count(self, text: str) -> int:
        return (len(text) + 3) // 4


DEFAULT_TOKENIZER = EstimatedTokens()
NONSPACE = re.compile(r"\S")

BOUNDARIES = tuple(
    re.compile(pattern) for pattern in (r"\n\n+", r"\n", r"[.!?。！？](?:\s+|$)", r"\s+")
)


def _boundary(text: str, lower: int, upper: int) -> int:
    for pattern in BOUNDARIES:
        last = None
        for match in pattern.finditer(text, lower, upper):
            last = match.end()
        if last is not None:
            return last
    return upper


def iter_chunks(
    parsed: Parsed,
    document_id: str,
    version_key: str,
    config: Chunking,
    limits: Ingestion,
    tokenizer: TokenBudget = DEFAULT_TOKENIZER,
) -> Iterator[StoredChunk]:
    text = parsed.text
    ends = [s.end for s in parsed.sections]
    previous = 0
    for section in parsed.sections:
        if not previous <= section.start < section.end <= len(text):
            raise IngestionError("invalid_source_span")
        if text[previous : section.start].strip():
            raise IngestionError("missing_source_span")
        previous = section.end
    if text[previous:].strip():
        raise IngestionError("missing_source_span")
    ranges = (
        ((s.start, s.end) for s in parsed.sections)
        if config.strategy == "structure"
        else iter([(0, len(text))])
    )
    count = output_bytes = 0
    for begin, stop in ranges:
        start = last_end = begin
        while start < stop:
            end = tokenizer.end(text, start, stop, config.chunk_size)
            if not last_end < end <= stop:
                raise IngestionError("tokenizer_cannot_advance")
            if config.strategy != "fixed" and end < stop:
                # Leave enough new material to advance even with high overlap.
                lower = max(last_end, tokenizer.end(text, start, end, config.overlap))
                end = _boundary(text, lower, end)
            piece = text[start:end]
            if not piece.strip() and end < stop:
                # Jump over long blank regions instead of rescanning near-identical windows.
                following = NONSPACE.search(text, end, stop)
                if following is None:
                    break
                start = last_end = following.start()
                continue
            if piece.strip():
                count += 1
                if count > limits.max_chunks_per_document:
                    raise IngestionError("too_many_chunks", 413)
                spans = []
                output_bytes += len(piece.encode()) + 512
                index = bisect_right(ends, start)
                while index < len(ends) and parsed.sections[index].start < end:
                    src = parsed.sections[index]
                    output_bytes += len(src.label.encode()) + 192
                    spans.append(
                        SourceSpan(
                            label=src.label,
                            start=max(start, src.start),
                            end=min(end, src.end),
                            parsed_start=src.parsed_start,
                            parsed_end=src.parsed_end,
                        )
                    )
                    index += 1
                if output_bytes > limits.max_extracted_text_mb * 4 * 1024**2:
                    raise IngestionError("chunk_output_limit", 413)
                tokens = tokenizer.count(piece)
                if not 0 < tokens <= config.chunk_size:
                    raise IngestionError("token_budget_exceeded")
                yield StoredChunk(
                    chunk_id=digest(version_key, tokenizer.identifier, str(start), str(end)),
                    document_id=document_id,
                    start=start,
                    end=end,
                    text=piece,
                    token_count=tokens,
                    estimated=tokenizer.estimated,
                    sources=tuple(spans),
                )
            if end == stop:
                break
            next_start = tokenizer.overlap_start(text, start, end, config.overlap)
            if not start < next_start <= end:
                raise IngestionError("tokenizer_cannot_advance")
            start, last_end = next_start, end
